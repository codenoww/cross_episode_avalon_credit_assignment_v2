"""
component2_exposure_audit.py -- re-implements Component 2's Gates 1-3 on the original 50-game tournament
(no database writes, no LLM calls) to answer two questions:

  1. Does this re-implementation reproduce the paper's reported numbers (3,078 candidates, 2,073 confirmed,
     1,970 bootstrap-stable, 879 messages, longest chain 15)?  Variant "as_implemented".
  2. How do those numbers change if the exposure check is applied as Section IV-B-1 SPECIFIES it?

Exposure variants (cosine threshold 0.6 against an agent's recorded reflections):
  as_implemented : agent = the candidate SOURCE's sender; reflections from before the SOURCE's game;
                   compared with the TARGET message's text   (what gate1_pipeline.py does)
  spec_all       : agent = the TARGET's sender; reflections from before the TARGET's game (all earlier games);
                   compared with the SOURCE message's text   (what the paper says)
  spec_window3   : as spec_all, but only the last three games before the target game (what the agent's prompt shows)
  no_exposure    : no exposure filter (reference)

Everything else follows the pipeline: top-10 nearest messages per target, cosine >= 0.65, earlier in time,
effect = ITE(target) * similarity, confirmed if |effect| > tau (tau = 0.5 * sd of ITE over the training rows) and
the dataset-level placebo / random-common-cause checks pass, bootstrap-stable if the 95% interval of the target's
ITE over 100 resamples excludes zero.

Usage:  python component2_exposure_audit.py
"""
import glob
import json
import os
import warnings

import networkx as nx
import numpy as np
import pandas as pd
import psycopg2
import statsmodels.api as sm
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from data_paths import DATA_PATH, MEMORY_PATH

warnings.filterwarnings("ignore")
load_dotenv()

SIM_THRESHOLD, EXPOSURE_THRESHOLD, TOP_K, N_BOOT = 0.65, 0.6, 10, 100
rng = np.random.default_rng(42)


def connect():
    return psycopg2.connect(host=os.environ.get("DB_HOST", "127.0.0.1"), port=os.environ.get("DB_PORT", "5434"),
                            dbname=os.environ.get("DB_NAME", "avalon_research"), user=os.environ.get("DB_USER", "postgres"),
                            password=os.environ["DB_PASSWORD"])


def main():
    games = json.load(open(DATA_PATH, encoding="utf-8"))["games"]
    tune_ids = {json.load(open(p, encoding="utf-8"))["game_id"]
                for p in glob.glob("dataset/1_cross_game_learning_50g/individual_games/*.json")}
    pos = {g["game_id"]: i + 1 for i, g in enumerate(games)}                    # canonical 1-based position (1..50 for the tournament)
    mission = {(g["game_id"], m["mission_number"]): m["mission_result"] for g in games for m in g["missions"]}
    wins = {}
    for g in games:
        if g["game_id"] in tune_ids:
            for p in g["players"]:
                w = wins.setdefault(p["name"], [0, 0])
                w[1] += 1
                w[0] += (g["winner"] == "good") == p["is_good"]

    cur = connect().cursor()
    cur.execute("SELECT id, episode_id, sender_id, sender_team, turn, mission_number, baseline_credit, "
                "behavioral_delta_magnitude, embedding::text FROM messages WHERE episode_id = ANY(%s) AND embedding IS NOT NULL",
                (list(tune_ids),))
    rows = cur.fetchall()
    df = pd.DataFrame([r[:8] for r in rows], columns=["id", "ep", "agent", "team", "turn", "mission", "credit", "delta"])
    E = np.array([np.fromstring(r[8].strip("[]"), sep=",") for r in rows])
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    n = len(df)
    df["pos"] = df.ep.map(pos)
    df["delta0"] = df.delta.fillna(0.0)
    print(f"original-tournament messages: {n} in {df.ep.nunique()} games")

    # ---- Gate 2: one regression (mission-level outcome), as gate2.py
    def outcome(ep, m, team):
        r = mission.get((ep, m))
        if r is None or m is None:
            return np.nan
        return float(r == "success") if team == "good" else float(r == "fail")
    df["y"] = [outcome(e, m, t) for e, m, t in zip(df.ep, df.mission, df.team)]
    df["skill"] = df.agent.map(lambda a: wins[a][0] / wins[a][1])
    train = df[df.delta.notna() & df.y.notna() & df.credit.notna()].copy()

    def design(d):
        return sm.add_constant(pd.DataFrame({"c": d.credit, "s": d.skill, "i": d.credit * d.delta0}))
    model = sm.OLS(train.y, design(train)).fit()
    bc, bi = model.params["c"], model.params["i"]
    ite = bc * df.credit + bi * df.credit * df.delta0
    tau = 0.5 * float((bc * train.credit + bi * train.credit * train.delta0).std())
    print(f"regression rows {len(train)}: credit coef {bc:.4f}, interaction {bi:.4f}, tau = {tau:.4f}")

    # Gate 3 (dataset level): placebo and random common cause
    X, y = design(train), train.y
    placebo_df = train.copy(); placebo_df["credit"] = rng.permutation(placebo_df.credit.values)
    pb = sm.OLS(y, design(placebo_df)).fit().params["c"]
    rcc_df = design(train).assign(fake=rng.normal(size=len(train)))
    rc = sm.OLS(y, rcc_df).fit().params["c"]
    gate3 = abs(bc) > 1.5 * abs(pb) and abs(rc - bc) / abs(bc) < 0.5
    print(f"Gate 3: placebo coef {pb:.4f}, RCC coef {rc:.4f} -> passes: {gate3}")

    # bootstrap stability of every message's ITE (own seeded generator so the result does not depend on earlier draws)
    brng = np.random.default_rng(7)
    coefs = np.empty((N_BOOT, 2))
    for b in range(N_BOOT):
        s_ = train.iloc[brng.integers(0, len(train), len(train))]
        p_ = sm.OLS(s_.y, design(s_)).fit().params
        coefs[b] = (p_["c"], p_["i"])
    boot = coefs[:, [0]] * df.credit.values[None, :] + coefs[:, [1]] * (df.credit.values * df.delta0.values)[None, :]
    lo, hi = np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)
    stable = (lo > 0) | (hi < 0)
    print(f"bootstrap ({N_BOOT} resamples): {stable.mean():.3f} of all messages have a stable ITE")

    # ---- Gate 1 candidates (top-10 nearest among the tournament's messages)
    sims = E @ E.T
    np.fill_diagonal(sims, -1)
    nn = np.argsort(-sims, axis=1)[:, :TOP_K]

    # reflections
    model_st = SentenceTransformer("all-MiniLM-L6-v2")
    mem = json.load(open(MEMORY_PATH, encoding="utf-8"))["player_memories"]
    refl = {}
    for a, v in mem.items():
        items = [(r["game_number"], t) for r in v["reflections"]
                 for t in [r.get("self_assessment", "")] + list(r.get("player_observations", {}).values()) if t]
        gn = np.array([g for g, _ in items])
        refl[a] = (gn, model_st.encode([t for _, t in items], normalize_embeddings=True, show_progress_bar=False))

    def exposed(variant, src, tgt):
        if variant == "no_exposure":
            return True
        if variant == "as_implemented":
            agent, before, vec = df.agent.iat[src], df.pos.iat[src], E[tgt]
            lo_game = 0
        else:
            agent, before, vec = df.agent.iat[tgt], df.pos.iat[tgt], E[src]
            lo_game = before - 3 if variant == "spec_window3" else 0
        gn, em = refl[agent]
        mask = (gn < before) & (gn >= lo_game)
        return bool(mask.any() and (em[mask] @ vec).max() >= EXPOSURE_THRESHOLD)

    results = {}
    for variant in ["as_implemented", "spec_all", "spec_window3", "no_exposure"]:
        valid, rejected = [], 0
        for tgt in range(n):
            for src in nn[tgt]:
                if sims[tgt, src] < SIM_THRESHOLD:
                    continue
                same = df.ep.iat[src] == df.ep.iat[tgt]
                if same:
                    if df.turn.iat[src] < df.turn.iat[tgt]:
                        valid.append((src, tgt))
                    continue
                ok = exposed(variant, src, tgt)
                if not ok:
                    rejected += 1
                elif df.pos.iat[src] < df.pos.iat[tgt]:
                    valid.append((src, tgt))
        edges = []
        for src, tgt in valid:
            eff = ite.iat[tgt] * sims[tgt, src]
            if pd.notna(eff) and abs(eff) > tau and gate3:
                edges.append((src, tgt, float(eff), bool(stable[tgt])))
        st = [e for e in edges if e[3]]
        G = nx.DiGraph()
        G.add_edges_from((df.id.iat[s], df.id.iat[t]) for s, t, _, _ in st)
        hops = nx.dag_longest_path_length(G) if G.number_of_nodes() else 0
        infl = collections_sum(st, df)
        score = {}
        for s, t, e, _ in st:
            score[s] = score.get(s, 0.0) + e
        results[variant] = dict(valid=len(valid), rejected=rejected, confirmed=len(edges), stable=len(st),
                                nodes=G.number_of_nodes(), longest_msgs=hops + 1 if hops else 0, hops=hops,
                                cross=sum(df.ep.iat[s] != df.ep.iat[t] for s, t, _, _ in st), influence=infl,
                                nonzero_scores=len(score), top_score=max(score.values()) if score else 0.0)

    # ---- negative controls on the 50-game run itself (the scope of Table II)
    pairs = [(src, tgt) for tgt in range(n) for src in nn[tgt]
             if sims[tgt, src] >= SIM_THRESHOLD and df.ep.iat[src] != df.ep.iat[tgt]]
    pre = []                                                                 # per pair: reflections of the SOURCE sender and their cosine with the TARGET text
    for src, tgt in pairs:
        gn, em = refl[df.agent.iat[src]]
        pre.append((df.pos.iat[src], df.pos.iat[tgt], gn, em @ E[tgt]))
    def count(newpos):
        order_ok = exp_ok = 0
        for ps, pt, gn, cs in pre:
            if newpos[ps - 1] < newpos[pt - 1]:
                order_ok += 1
                if (cs[newpos[gn - 1] < newpos[ps - 1]] >= EXPOSURE_THRESHOLD).any():
                    exp_ok += 1
        return order_ok, exp_ok
    real = count(np.arange(1, 51))
    prng = np.random.default_rng(11)
    shuf = np.array([count(prng.permutation(np.arange(1, 51))) for _ in range(200)])
    print("50-game run: cross-episode candidate pairs (top-10 neighbours, cosine >= 0.65)")
    print(f"  earlier-in-time:          real order {real[0]}  | shuffled order mean {shuf[:, 0].mean():.0f} (95% range {np.percentile(shuf[:, 0], 2.5):.0f}-{np.percentile(shuf[:, 0], 97.5):.0f})")
    print(f"  ... and passing exposure: real order {real[1]}  | shuffled order mean {shuf[:, 1].mean():.0f} (95% range {np.percentile(shuf[:, 1], 2.5):.0f}-{np.percentile(shuf[:, 1], 97.5):.0f})")
    wrng = np.random.default_rng(5)
    true_pass = wrong_pass = n_ord = 0
    for src, tgt in pairs:
        if df.pos.iat[src] < df.pos.iat[tgt]:
            n_ord += 1
            true_pass += exposed("as_implemented", src, tgt)
            other = [a for a in refl if a != df.agent.iat[src]][wrng.integers(0, 4)]
            gn, em = refl[other]
            mask = gn < df.pos.iat[src]
            wrong_pass += bool(mask.any() and (em[mask] @ E[tgt]).max() >= EXPOSURE_THRESHOLD)
    print(f"  exposure check on the {n_ord} order-valid pairs: true source author {100 * true_pass / n_ord:.1f}% vs a wrong author {100 * wrong_pass / n_ord:.1f}%")
    cur.execute("""SELECT ce.causal_effect_size, mt.baseline_credit, ce.episode_gap FROM causal_edges ce
                   JOIN messages mt ON mt.id = ce.target_message_id JOIN messages ms ON ms.id = ce.source_message_id
                   WHERE ce.confirmed AND ce.bootstrap_stable AND mt.episode_id = ANY(%s) AND ms.episode_id = ANY(%s)""",
                (list(tune_ids), list(tune_ids)))
    st_rows = np.array(cur.fetchall(), dtype=float)
    r2 = np.corrcoef(st_rows[:, 0], st_rows[:, 1])[0, 1] ** 2
    cross_rows = st_rows[st_rows[:, 2] > 0]
    print(f"  stored stable edges in the 50-game run: {len(st_rows)}; R^2 of the edge effect on the target's credit alone = {r2:.3f}; "
          f"{100 * np.mean(cross_rows[:, 2] > 3):.1f}% of the {len(cross_rows)} cross-episode stable edges span more than 3 games")

    paper = dict(valid=3078, rejected=5750, confirmed=2073, stable=1970, nodes=879, longest_msgs=15, hops=14, nonzero_scores=644, top_score=2.79)
    keys = ["valid", "rejected", "confirmed", "stable", "nodes", "longest_msgs", "hops", "cross", "nonzero_scores", "top_score"]
    print(f"\n{'':<16}" + "".join(f"{v:>16}" for v in ["paper"] + list(results)))
    for k in keys:
        print(f"{k:<16}" + f"{paper.get(k, '-'):>16}" + "".join(f"{results[v][k]:>16.2f}" if isinstance(results[v][k], float) else f"{results[v][k]:>16}" for v in results))
    print("\nper-agent outgoing influence (sum of stable edge effects), by variant:")
    print(f"{'':<16}" + "".join(f"{v:>16}" for v in ["paper"] + list(results)))
    paper_infl = dict(Charlie=36.71, Eve=32.84, Alice=31.52, Bob=31.37, Diana=24.48)
    for a in ["Alice", "Bob", "Charlie", "Diana", "Eve"]:
        print(f"{a:<16}{paper_infl[a]:>16.2f}" + "".join(f"{results[v]['influence'].get(a, 0):>16.2f}" for v in results))
    json.dump({v: {k: (x if not isinstance(x, dict) else x) for k, x in r.items()} for v, r in results.items()},
              open("component2_exposure_audit.json", "w"), indent=1, default=float)


def collections_sum(stable_edges, df):
    out = {}
    for s, _, e, _ in stable_edges:
        out[df.agent.iat[s]] = out.get(df.agent.iat[s], 0.0) + e
    return out


if __name__ == "__main__":
    main()
