"""
validity_checks.py -- the validity and robustness checks reported in the paper's
limitations (Sections VI and VII). Everything is computed from the database, the saved
game logs, and the released files (db_export.json, dataset/.../all_games.json); no LLM
calls are made.

    A  Table I row (w1 = 1) regenerated from released files only (no database)
    B  does credit separate messages WITHIN a team and game?   (within-team coefficient)
    C  credit score distribution
    D  correlation of credit with the behavioural delta
    E  rubric adherence: do good-team proposals naming an evil member score lower?
    F  candidate-influence edges: negative controls and how much the target's credit explains
    G  mission success rate, treatment vs control (judge-free)
    H  matched vs mismatched side/role between a note's source game and the current game

Usage:
    python validity_checks.py                 # everything
    python validity_checks.py --skip-embeddings   # skip F (needs sentence-transformers, ~1-2 min)
    python validity_checks.py --only A,G
"""
import argparse
import collections
import glob
import json
import os
import random
import warnings

import numpy as np
import pandas as pd
import psycopg2
import statsmodels.api as sm
from dotenv import load_dotenv
from scipy import stats

from data_paths import DATA_PATH

warnings.filterwarnings("ignore")
load_dotenv()

TUNING_DIR = "dataset/1_cross_game_learning_50g/individual_games"
SIM_THRESHOLD = 0.65          # Gate 1 candidate threshold
EXPOSURE_THRESHOLD = 0.6      # exposure_check default
TOP_K = 10                    # Gate 1 neighbours per target
N_PERMUTATIONS = 200
SEED = 0
AGENTS = ["Alice", "Bob", "Charlie", "Diana", "Eve"]


def get_connection():
    return psycopg2.connect(
        host=os.environ.get("DB_HOST", "127.0.0.1"), port=os.environ.get("DB_PORT", "5434"),
        dbname=os.environ.get("DB_NAME", "avalon_research"), user=os.environ.get("DB_USER", "postgres"),
        password=os.environ["DB_PASSWORD"],
    )


GAMES = json.load(open(DATA_PATH, encoding="utf-8"))["games"]
ORDER = {g["game_id"]: i for i, g in enumerate(GAMES)}
WINNER = {g["game_id"]: g["winner"] for g in GAMES}
TEAM = {(g["game_id"], p["name"]): ("good" if p["is_good"] else "evil") for g in GAMES for p in g["players"]}
ROLE = {(g["game_id"], p["name"]): p["role"] for g in GAMES for p in g["players"]}
MISSION = {(g["game_id"], m["mission_number"]): m["mission_result"] for g in GAMES for m in g["missions"]}


def tuning_ids():
    ids = set()
    for p in glob.glob(os.path.join(TUNING_DIR, "*.json")):
        ids.add(json.load(open(p, encoding="utf-8"))["game_id"])
    return ids


def mission_outcome(ep, mission, team):
    """1 if the mission result favoured `team`, else 0 (None if unknown)."""
    res = MISSION.get((ep, mission))
    if res is None or mission is None:
        return None
    return int(res == "success") if team == "good" else int(res == "fail")


def pearson_ci(x, y):
    r = stats.pearsonr(x, y)[0]
    z, se = np.arctanh(r), 1 / np.sqrt(len(x) - 3)
    return r, np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)


def header(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


# ---------------------------------------------------------------- A
def check_A():
    header("A. Table I row (w1 = 1.0, w2 = 0) regenerated from released files only")
    tune = tuning_ids()
    agg = collections.defaultdict(lambda: {"good": [], "evil": []})
    for m in json.load(open("db_export.json", encoding="utf-8"))["messages"]:
        if m["episode_id"] in tune and m["baseline_credit"] not in (None, "None", ""):
            t = TEAM.get((m["episode_id"], m["sender_id"]))
            if t:
                agg[m["episode_id"]][t].append(float(m["baseline_credit"]))
    eps = [e for e in agg if agg[e]["good"] and agg[e]["evil"]]
    x = np.array([np.mean(agg[e]["good"]) - np.mean(agg[e]["evil"]) for e in eps])
    y = np.array([1.0 if WINNER[e] == "good" else 0.0 for e in eps])
    r, lo, hi = pearson_ci(x, y)
    print(f"games {len(eps)}: r = {r:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  sign accuracy = {np.mean((x > 0) == (y == 1)):.2f}")
    print("(baseline_credit equals the semantic score because the behavioural weight is 0)")


# ---------------------------------------------------------------- B, C, D, E (database)
def load_messages(cur):
    cur.execute(
        "SELECT id, episode_id, sender_id, sender_team, mission_number, phase, global_turn_id, "
        "baseline_credit, behavioral_delta_magnitude FROM messages WHERE baseline_credit IS NOT NULL")
    df = pd.DataFrame(cur.fetchall(), columns=["id", "ep", "agent", "team", "mission", "phase", "turn", "credit", "delta"])
    # proposal messages carry global_turn_id = 100000 + mission * 100 + proposal_id (proposal_round is NULL for the original 50 games)
    df["prop"] = np.where(df.phase == "proposal", df.turn - 100000 - df.mission.fillna(0) * 100, np.nan)
    df["y"] = [mission_outcome(e, m, t) for e, m, t in zip(df.ep, df.mission, df.team)]
    return df


def within_coefficient(d):
    d = d.dropna(subset=["y"]).copy()
    d["g"] = d.ep + "|" + d.team
    d["xc"] = d.credit - d.groupby("g").credit.transform("mean")
    d["yc"] = d.y - d.groupby("g").y.transform("mean")
    sxx = (d.xc ** 2).sum()
    b = (d.xc * d.yc).sum() / sxx
    e = d.yc - b * d.xc
    s = (d.xc * e).groupby(d.ep).sum()
    G = len(s)
    se = np.sqrt((G / (G - 1)) * (s ** 2).sum()) / sxx
    p = 2 * stats.t.sf(abs(b / se), G - 1)
    return b, se, p, len(d), G


def check_B(df, tune):
    header("B. Does credit separate messages WITHIN a team and game? (outcome = mission result favourable to the sender's team)")
    for name, d in [("original 50 games", df[df.ep.isin(tune)]), ("all scored games", df)]:
        b, se, p, n, G = within_coefficient(d)
        print(f"{name:<20} within-team-game coefficient b = {b:+.3f} (SE {se:.3f}, p = {p:.3f}; n = {n} messages, {G} episodes, SE clustered by episode)")
    print("A coefficient near zero means credit separates winning from losing teams but is not shown to discriminate between messages within a team.")


def check_C(df, tune):
    header("C. Credit score distribution")
    for name, d in [("original 50 games", df[df.ep.isin(tune)]), ("all scored games", df)]:
        for team in ["good", "evil", None]:
            s = d[d.team == team].credit if team else d.credit
            print(f"{name:<20} {team or 'both':<5} n={len(s):<5} mean {s.mean():+.3f} sd {s.std():.3f} | share > 0: {np.mean(s > 0):.2f} | "
                  f"p10 {s.quantile(.1):+.2f} median {s.median():+.2f} p90 {s.quantile(.9):+.2f}")


def check_D(df, tune):
    header("D. Credit vs behavioural delta (discussion messages)")
    for name, d in [("original 50 games", df[df.ep.isin(tune)]), ("all scored games", df)]:
        d = d.dropna(subset=["delta"])
        print(f"{name:<20} n={len(d)} Pearson r = {stats.pearsonr(d.credit, d.delta)[0]:+.3f}  Spearman rho = {stats.spearmanr(d.credit, d.delta)[0]:+.3f}"
              f"  | corr of |credit| with delta: {stats.pearsonr(d.credit.abs(), d.delta)[0]:+.3f}")


def check_E(df, tune):
    header("E. Rubric adherence: credit for good-team proposals, by whether the proposed team contains an evil player")
    prop = df[df.phase == "proposal"]
    rows = []
    for g in GAMES:
        for m in g["missions"]:
            for pr in m["proposals"]:
                leader = pr.get("leader")
                if TEAM.get((g["game_id"], leader)) != "good":
                    continue
                has_evil = any(TEAM.get((g["game_id"], x)) == "evil" for x in pr["team_members"])
                rows.append((g["game_id"], m["mission_number"], pr["proposal_id"], has_evil))
    meta = pd.DataFrame(rows, columns=["ep", "mission", "prop", "has_evil"])
    d = meta.merge(prop[["ep", "mission", "prop", "credit"]], on=["ep", "mission", "prop"])
    for name, s in [("original 50 games", d[d.ep.isin(tune)]), ("all scored games", d)]:
        a, b = s[s.has_evil].credit, s[~s.has_evil].credit
        print(f"{name:<20} with an evil member: mean {a.mean():.3f} (n={len(a)}) | without: {b.mean():.3f} (n={len(b)}) | "
              f"Welch p = {stats.ttest_ind(a, b, equal_var=False).pvalue:.3f} | Mann-Whitney p = {stats.mannwhitneyu(a, b).pvalue:.3f}")


# ---------------------------------------------------------------- F (edges)
def check_F(cur):
    header("F. Candidate-influence edges: negative controls and what the target's credit explains")
    from sentence_transformers import SentenceTransformer

    cur.execute("SELECT id, episode_id, sender_id, turn, baseline_credit, embedding::text FROM messages WHERE embedding IS NOT NULL")
    rows = cur.fetchall()
    ids = np.array([r[0] for r in rows])
    ep = [r[1] for r in rows]
    agent = [r[2] for r in rows]
    turn = np.array([r[3] if r[3] is not None else 0 for r in rows])
    E = np.array([np.fromstring(r[5].strip("[]"), sep=",") for r in rows])
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    order = np.array([ORDER.get(e, -1) for e in ep])
    n = len(rows)
    sims = E @ E.T
    np.fill_diagonal(sims, -1)
    nn = np.argsort(-sims, axis=1)[:, :TOP_K]      # Gate 1: the 10 nearest messages in the whole table

    same, cross = [], []                             # valid (source j -> target i) pairs under the REAL order
    for i in range(n):
        for j in nn[i]:
            if sims[i, j] < SIM_THRESHOLD:
                continue
            if ep[j] == ep[i]:
                if turn[j] < turn[i]:
                    same.append((j, i))
            elif order[j] >= 0 and order[i] >= 0 and order[j] < order[i]:
                cross.append((j, i))
    print(f"messages {n}; neighbour pairs >= {SIM_THRESHOLD} among top-{TOP_K}: valid same-episode {len(same)}, valid cross-episode {len(cross)}")

    # F1: shuffle the episode order -- if edges needed real temporal precedence, the real order should beat shuffles
    rng = np.random.default_rng(SEED)
    pairs = [(j, i) for i in range(n) for j in nn[i] if sims[i, j] >= SIM_THRESHOLD and ep[j] != ep[i] and order[j] >= 0 and order[i] >= 0]
    pj = np.array([p[0] for p in pairs]); pi = np.array([p[1] for p in pairs])
    n_eps = len(GAMES)
    counts = []
    for _ in range(N_PERMUTATIONS):
        perm = rng.permutation(n_eps)
        counts.append(int(np.sum(perm[order[pj]] < perm[order[pi]])))
    counts = np.array(counts)
    print(f"F1 cross-episode candidates with REAL episode order: {len(cross)} | with SHUFFLED order ({N_PERMUTATIONS} shuffles): "
          f"mean {counts.mean():.0f}, 95% range [{np.percentile(counts, 2.5):.0f}, {np.percentile(counts, 97.5):.0f}] "
          f"-> real is at the {100 * np.mean(counts <= len(cross)):.0f}th percentile of the shuffles")
    print("   (similar counts mean candidate links arise from similarity alone, not from earlier-to-later structure)")

    # F2: exposure check with the true agent vs a wrong agent
    model = SentenceTransformer("all-MiniLM-L6-v2")
    mem = json.load(open("dataset/1_cross_game_learning_50g/player_memories.json", encoding="utf-8"))["player_memories"]
    refl = {a: [] for a in AGENTS}
    for a, v in mem.items():
        for r in v["reflections"]:
            texts = [r.get("self_assessment", "")] + list(r.get("player_observations", {}).values())
            for t in texts:
                if t:
                    refl[a].append((r["game_number"], t))
    mats = {}
    for a in AGENTS:
        gn = np.array([g for g, _ in refl[a]])
        em = model.encode([t for _, t in refl[a]], normalize_embeddings=True, show_progress_bar=False)
        mats[a] = (gn, em)
    max_game = max(g for a in AGENTS for g, _ in refl[a])

    def passes(agent_name, src_ep, tgt_vec):
        if agent_name not in mats:
            return False
        gn, em = mats[agent_name]
        mask = gn < (ORDER[src_ep] + 1)
        return bool(mask.any() and (em[mask] @ tgt_vec).max() >= EXPOSURE_THRESHOLD)

    random.seed(SEED)
    real_pass = wrong_pass = 0
    for j, i in cross:
        real_pass += passes(agent[j], ep[j], E[i])
        wrong = random.choice([a for a in AGENTS if a != agent[j]])
        wrong_pass += passes(wrong, ep[j], E[i])
    print(f"F2 exposure check on the {len(cross)} real cross-episode candidates: pass with the TRUE agent {100 * real_pass / len(cross):.1f}% vs a WRONG agent {100 * wrong_pass / len(cross):.1f}%")
    beyond = sum(1 for j, i in cross if ORDER[ep[j]] + 1 > max_game)
    print(f"   {beyond} of {len(cross)} candidates ({100 * beyond / len(cross):.1f}%) have a source game numbered above {max_game}, the last game with any reflection in the memory file "
          f"(newer runs' reflections were never merged), so they are checked against the original tournament's reflections only")

    # F3: how much of a confirmed edge's effect is the target's own credit?
    cur.execute("""SELECT ce.causal_effect_size, ce.similarity_score, mt.baseline_credit, mt.behavioral_delta_magnitude
                   FROM causal_edges ce JOIN messages mt ON mt.id = ce.target_message_id
                   WHERE ce.confirmed AND ce.bootstrap_stable AND mt.baseline_credit IS NOT NULL""")
    ed = pd.DataFrame(cur.fetchall(), columns=["eff", "sim", "c", "d"]).astype(float)
    ed["d"] = ed.d.fillna(0.0)

    def r2(y, cols):
        X = sm.add_constant(pd.concat(cols, axis=1))
        return sm.OLS(y, X).fit().rsquared
    print(f"F3 confirmed edges {len(ed)}: R^2 of edge effect on the target's credit alone = {r2(ed.eff, [ed.c]):.3f}; "
          f"on target credit x similarity = {r2(ed.eff, [ed.c * ed.sim]):.3f}; adding credit x delta = {r2(ed.eff, [ed.c * ed.sim, ed.c * ed.d * ed.sim]):.3f}")
    print("   (an edge's effect is the target message's own estimated effect scaled by similarity; the source message enters only through similarity)")


# ---------------------------------------------------------------- G, H (experiment)
def condition_ids():
    return json.load(open("_control_ids.json")), json.load(open("_treatment_ids.json"))


def game_logs(ids):
    ids, out = set(ids), {}
    for p in glob.glob("avalon_tournament_*/individual_games/*.json"):
        g = json.load(open(p, encoding="utf-8"))
        if g.get("game_id") in ids:
            out[g["game_id"]] = g
    return out


def check_G():
    header("G. Mission success rate, treatment vs control (from the game logs; no judge involved)")
    control, treatment = condition_ids()
    res = {}
    for name, ids in [("control", control), ("treatment", treatment)]:
        logs = game_logs(ids)
        succ = np.array([np.mean([m["mission_result"] == "success" for m in g["missions"]]) for g in logs.values()])
        n_m = sum(len(g["missions"]) for g in logs.values())
        n_s = sum(m["mission_result"] == "success" for g in logs.values() for m in g["missions"])
        res[name] = (succ, n_s, n_m)
        print(f"{name:<9} games {len(logs)} | missions succeeded {n_s}/{n_m} = {100 * n_s / n_m:.1f}% | per-game mean {succ.mean():.3f}")
    (sc, cs, cm), (st, ts, tm) = res["control"], res["treatment"]
    print(f"per-game mean success: Welch p = {stats.ttest_ind(st, sc, equal_var=False).pvalue:.3f}, Mann-Whitney p = {stats.mannwhitneyu(st, sc).pvalue:.3f}; "
          f"pooled missions Fisher p = {stats.fisher_exact([[ts, tm - ts], [cs, cm - cs]])[1]:.3f} (missions within a game are not independent; the per-game test is primary)")


def check_H(cur):
    header("H. Matched vs mismatched side/role between the note's source game and the current game (treatment only)")
    _, treatment = condition_ids()
    treatment = sorted(treatment)
    cur.execute("SELECT id, episode_id, agent_id, created_at FROM feedback ORDER BY created_at")
    fb = cur.fetchall()
    own = {(e, a): t for _, e, a, t in fb}
    cur.execute("SELECT episode_id, sender_id, AVG(baseline_credit) FROM messages WHERE episode_id = ANY(%s) AND baseline_credit IS NOT NULL GROUP BY 1, 2", (treatment,))
    credit = {(e, a): v for e, a, v in cur.fetchall()}
    rows = []
    for ep in treatment:
        g = next(x for x in GAMES if x["game_id"] == ep)
        for a in AGENTS:
            t_cur = own.get((ep, a))
            prior = [f for f in fb if f[2] == a and f[1] != ep and (t_cur is None or f[3] < t_cur)]
            if not prior or (prior[-1][1], a) not in TEAM or (ep, a) not in TEAM:
                continue
            src = prior[-1][1]
            team = TEAM[(ep, a)]
            outs = [mission_outcome(ep, m["mission_number"], team) for m in g["missions"]]
            outs = [o for o in outs if o is not None]
            rows.append(dict(ep=ep, agent=a, team=team, side_match=TEAM[(src, a)] == team, role_match=ROLE[(src, a)] == ROLE[(ep, a)],
                             credit=credit.get((ep, a), np.nan), mission_win=np.mean(outs) if outs else np.nan))
    d = pd.DataFrame(rows).dropna(subset=["credit"])
    print(f"agent-episodes with a note: {len(d)} | same side as the note's source game: {d.side_match.mean():.1%} | same exact role: {d.role_match.mean():.1%}")
    for flag in ["side_match", "role_match"]:
        for outcome, label in [("credit", "baseline_credit (judge)"), ("mission_win", "own-team mission success (judge-free)")]:
            a, b = d[d[flag]][outcome].dropna(), d[~d[flag]][outcome].dropna()
            X = sm.add_constant(pd.get_dummies(d[["agent", "team"]], drop_first=True).astype(float).assign(m=d[flag].astype(float)))
            fit = sm.OLS(d[outcome], X, missing="drop").fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.ep)[0]})
            print(f"{flag:<10} {label:<40} matched {a.mean():.3f} (n={len(a)}) vs mismatched {b.mean():.3f} (n={len(b)}) | "
                  f"Welch p = {stats.ttest_ind(a, b, equal_var=False).pvalue:.3f} | adjusted for agent and current side: coef {fit.params['m']:+.3f}, p = {fit.pvalues['m']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated subset of A,B,C,D,E,F,G,H")
    ap.add_argument("--skip-embeddings", action="store_true")
    args = ap.parse_args()
    want = set(args.only.split(",")) if args.only else set("ABCDEFGH")
    if args.skip_embeddings:
        want.discard("F")
    conn = get_connection()
    cur = conn.cursor()
    tune = tuning_ids()
    if "A" in want:
        check_A()
    if want & set("BCDE"):
        df = load_messages(cur)
        for k, fn in [("B", check_B), ("C", check_C), ("D", check_D), ("E", check_E)]:
            if k in want:
                fn(df, tune)
    if "F" in want:
        check_F(cur)
    if "G" in want:
        check_G()
    if "H" in want:
        check_H(cur)
    conn.close()


if __name__ == "__main__":
    main()
