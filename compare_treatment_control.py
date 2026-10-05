"""
compare_treatment_control.py -- the treatment-vs-control analysis reported in the
paper (Component 3, Section VI-C). Reproduces, from the database and the saved
game files:

  * mean baseline_credit per episode, control vs treatment
  * Welch t-test (episode level), 95% CI for the difference, Cohen's d, and a
    bootstrap CI for the difference
  * paired per-agent test (each agent's own control vs treatment mean, n = 5)
  * per-agent difference and Spearman trend of baseline_credit across episodes
  * good-team win rate and Fisher's exact test
  * a power calculation (episodes per condition needed at the observed effect)
  * robustness to which control block is used

baseline_credit is the primary outcome. final_credit is deliberately NOT compared
across conditions: its cross-episode term sums edges to LATER episodes, so episodes
generated later accumulate less of it regardless of coaching (see paper, Section
VII). It is printed for transparency only, clearly marked.

Episode lists come from _control_ids.json / _treatment_ids.json (regenerate by
globbing avalon_tournament_*/individual_games/*.json per condition if stale).
Database settings come from .env (see .env.example).

Usage:
    python compare_treatment_control.py
"""
import collections
import glob
import json
import os

import numpy as np
import psycopg2
from dotenv import load_dotenv
from scipy import stats

load_dotenv()

CONTROL_FILE = "_control_ids.json"
TREATMENT_FILE = "_treatment_ids.json"
BOOTSTRAP_RESAMPLES = 10000
SEED = 0
# A control block that started after the treatment block finished (used only for
# the robustness check). Control episodes are split by episode-id date.
TREATMENT_START = "avalon_20260731"
TREATMENT_END = "avalon_20260804"


def get_connection():
    return psycopg2.connect(
        host=os.environ.get("DB_HOST", "127.0.0.1"),
        port=os.environ.get("DB_PORT", "5434"),
        dbname=os.environ.get("DB_NAME", "avalon_research"),
        user=os.environ.get("DB_USER", "postgres"),
        password=os.environ["DB_PASSWORD"],
    )


def episode_means(cur, ids, column="baseline_credit"):
    cur.execute(
        f"SELECT episode_id, AVG({column}) FROM messages "
        f"WHERE episode_id = ANY(%s) AND {column} IS NOT NULL GROUP BY episode_id",
        (list(ids),),
    )
    return dict(cur.fetchall())


def agent_episode_means(cur, ids):
    cur.execute(
        "SELECT episode_id, sender_id, AVG(baseline_credit) FROM messages "
        "WHERE episode_id = ANY(%s) AND baseline_credit IS NOT NULL "
        "GROUP BY episode_id, sender_id ORDER BY episode_id",
        (list(ids),),
    )
    out = collections.defaultdict(list)
    for ep, agent, v in cur.fetchall():
        out[agent].append((ep, v))
    return out


def winners(ids):
    ids = set(ids)
    out = {}
    for path in glob.glob("avalon_tournament_*/individual_games/*.json"):
        with open(path, encoding="utf-8") as f:
            game = json.load(f)
        if game.get("game_id") in ids:
            out[game["game_id"]] = game["winner"]
    return out


def welch(t, c):
    res = stats.ttest_ind(t, c, equal_var=False)
    diff = float(np.mean(t) - np.mean(c))
    vt, vc = np.var(t, ddof=1), np.var(c, ddof=1)
    nt, nc = len(t), len(c)
    se = np.sqrt(vt / nt + vc / nc)
    df = (vt / nt + vc / nc) ** 2 / ((vt / nt) ** 2 / (nt - 1) + (vc / nc) ** 2 / (nc - 1))
    half = stats.t.ppf(0.975, df) * se
    pooled_sd = np.sqrt(((nt - 1) * vt + (nc - 1) * vc) / (nt + nc - 2))
    return dict(diff=diff, t=float(res.statistic), p=float(res.pvalue), df=float(df),
                ci=(diff - half, diff + half), d=diff / pooled_sd, pooled_sd=float(pooled_sd))


def bootstrap_ci(t, c):
    rng = np.random.default_rng(SEED)
    t, c = np.asarray(t), np.asarray(c)
    diffs = np.empty(BOOTSTRAP_RESAMPLES)
    for i in range(BOOTSTRAP_RESAMPLES):
        diffs[i] = rng.choice(t, len(t)).mean() - rng.choice(c, len(c)).mean()
    return np.percentile(diffs, [2.5, 97.5])


def episodes_needed(d, power):
    """Episodes per condition for a two-sided alpha = 0.05 two-sample test (normal approximation)."""
    z_a, z_b = stats.norm.ppf(0.975), stats.norm.ppf(power)
    return 2 * ((z_a + z_b) / d) ** 2


def main():
    control_ids = json.load(open(CONTROL_FILE))
    treatment_ids = json.load(open(TREATMENT_FILE))
    conn = get_connection()
    cur = conn.cursor()

    c_map = episode_means(cur, control_ids)
    t_map = episode_means(cur, treatment_ids)
    c = np.array([c_map[e] for e in sorted(c_map)])
    t = np.array([t_map[e] for e in sorted(t_map)])
    print(f"Control: {len(control_ids)} episodes listed, {len(c)} scored | Treatment: {len(treatment_ids)} listed, {len(t)} scored")
    unscored = sorted(set(control_ids) - set(c_map))
    if unscored:
        print(f"  control episodes with no scores (excluded): {unscored}")

    print("\n== Primary outcome: mean baseline_credit per episode ==")
    r = welch(t, c)
    print(f"control {c.mean():.4f} (sd {c.std(ddof=1):.4f}) | treatment {t.mean():.4f} (sd {t.std(ddof=1):.4f})")
    print(f"difference {r['diff']:+.4f}   95% CI [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]   "
          f"bootstrap 95% CI [{bootstrap_ci(t, c)[0]:+.4f}, {bootstrap_ci(t, c)[1]:+.4f}]")
    print(f"Welch t = {r['t']:.3f}, df = {r['df']:.1f}, p = {r['p']:.3f}   Cohen's d = {r['d']:.2f}")

    print("\n== Power ==")
    print(f"observed d = {r['d']:.2f}; episodes per condition for 80% power: {episodes_needed(r['d'], 0.80):.0f}; "
          f"for 50% power: {episodes_needed(r['d'], 0.50):.0f}")

    print("\n== Per agent (mean baseline_credit) and trend across episodes ==")
    pt, pc = agent_episode_means(cur, treatment_ids), agent_episode_means(cur, control_ids)
    diffs = []
    print("agent    treat   ctrl    diff    trend-T rho (p)    trend-C rho (p)")
    for agent in sorted(pt):
        vt = [v for _, v in sorted(pt[agent])]
        vc = [v for _, v in sorted(pc[agent])]
        rt = stats.spearmanr(range(len(vt)), vt)
        rc = stats.spearmanr(range(len(vc)), vc)
        diffs.append(np.mean(vt) - np.mean(vc))
        print(f"{agent:<8} {np.mean(vt):.3f}   {np.mean(vc):.3f}   {diffs[-1]:+.3f}   {rt[0]:+.2f} ({rt[1]:.2f})        {rc[0]:+.2f} ({rc[1]:.2f})")
    paired = stats.ttest_1samp(diffs, 0)
    print(f"paired per-agent test (n = {len(diffs)}): t({len(diffs) - 1}) = {paired.statistic:.3f}, p = {paired.pvalue:.3f}; "
          f"{sum(d > 0.005 for d in diffs)} higher, {sum(abs(d) <= 0.005 for d in diffs)} about equal (|diff| <= 0.005), "
          f"{sum(d < -0.005 for d in diffs)} lower under treatment")

    print("\n== Robustness: which control block ==")
    early = [e for e in sorted(c_map) if e < TREATMENT_START]
    late = [e for e in sorted(c_map) if e > TREATMENT_END]
    ce, cl = np.array([c_map[e] for e in early]), np.array([c_map[e] for e in late])
    print(f"early control block (n={len(ce)}) mean {ce.mean():.3f} | late control block (n={len(cl)}) mean {cl.mean():.3f} | "
          f"early vs late p = {stats.ttest_ind(ce, cl, equal_var=False).pvalue:.3f}")
    print(f"treatment vs early controls only: p = {stats.ttest_ind(t, ce, equal_var=False).pvalue:.3f}")
    print(f"treatment vs late controls only:  p = {stats.ttest_ind(t, cl, equal_var=False).pvalue:.3f}  (not a fair headline; shown for completeness)")

    print("\n== Good-team win rate ==")
    wc, wt = winners(control_ids), winners(treatment_ids)
    gc, gt = sum(v == "good" for v in wc.values()), sum(v == "good" for v in wt.values())
    fisher = stats.fisher_exact([[gt, len(wt) - gt], [gc, len(wc) - gc]])[1]
    print(f"control {gc}/{len(wc)} = {100 * gc / len(wc):.1f}% | treatment {gt}/{len(wt)} = {100 * gt / len(wt):.1f}% | Fisher exact p = {fisher:.3f}")

    print("\n== final_credit (NOT comparable across conditions -- printed for transparency only) ==")
    fc, ft = episode_means(cur, control_ids, "final_credit"), episode_means(cur, treatment_ids, "final_credit")
    print(f"control {np.mean(list(fc.values())):.3f} | treatment {np.mean(list(ft.values())):.3f}. "
          "The cross-episode term depends on how many later episodes exist, so this gap reflects "
          "when episodes were generated, not coaching.")
    conn.close()


if __name__ == "__main__":
    main()
