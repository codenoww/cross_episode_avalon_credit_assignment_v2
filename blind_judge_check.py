"""
blind_judge_check.py -- does the credit still track the winner when the judge is NOT told who won?

The Component 1 judge is told the episode outcome, so the credit margin's correlation with the winner
(r = 0.743) holds partly by construction. This script re-scores a sample of the 50 tuning games with the
SAME model twice -- once outcome-aware (the original behaviour) and once outcome-blind
(component1/evaluator.py with JUDGE_SEES_OUTCOME off) -- and compares each with the winner.

Transcripts are built directly from the saved game files, in the same order the parser/evaluator use
(by mission, discussion before proposal reasoning, then turn id), so no database is needed for the
transcripts. Scores are written to a JSON file as each game finishes.

Usage:
    python blind_judge_check.py --games 25 --model openai/gpt-oss-120b --out blind_check.json
    python blind_judge_check.py --analyze blind_check.json
"""
import argparse
import glob
import json
import os
import random
import time

import numpy as np
from scipy import stats

import component1.evaluator as ev

TUNING_DIR = "dataset/1_cross_game_learning_50g/individual_games"


def load_transcript(path):
    g = json.load(open(path, encoding="utf-8"))
    role = {p["name"]: (p["role"], p["is_good"]) for p in g["players"]}
    rows = []
    for m in g["missions"]:
        mn = m["mission_number"]
        for d in m.get("discussion", []):
            rows.append((mn, 0, d.get("global_turn_id") or 0, d["player"], d.get("phase", "discussion"), d["content"]))
        for pr in m.get("proposals", []):
            reasoning = pr.get("reasoning")
            if reasoning:
                rows.append((mn, 1, 100000 + mn * 100 + pr["proposal_id"], pr["leader"], "proposal", reasoning))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    transcript = [{"index": i, "sender": s, "role": role[s][0], "is_good": role[s][1], "text": t, "phase": ph, "mission": mn}
                  for i, (mn, _, _, s, ph, t) in enumerate(rows)]
    return g["game_id"], g["winner"], transcript, {m["mission_number"]: m["mission_result"] for m in g["missions"]}


def score(transcript, winner, game_id, sees_outcome):
    ev.JUDGE_SEES_OUTCOME = sees_outcome
    rng = random.Random(f"{ev.SHUFFLE_SEED}:{game_id}")
    outcome_line = f"GAME OUTCOME: {winner} team won.\n\n" if sees_outcome else ""
    passes = {}
    for team_is_good, rubric in [(True, ev.rubric_for(ev.GOOD_RUBRIC)), (False, ev.rubric_for(ev.EVIL_RUBRIC))]:
        targets = [m["index"] for m in transcript if m["is_good"] == team_is_good]
        if not targets:
            continue
        for k, name in enumerate(ev.PASSES):
            order = list(transcript)
            if name == "reverse":
                order = list(reversed(order))
            elif name == "shuffled":
                order = rng.sample(order, len(order))
            lines = "\n".join(
                f'[{m["index"]}] {m["sender"]} ({m["role"]}, {"GOOD" if m["is_good"] else "EVIL"}'
                f'{", PROPOSAL" if m["phase"] == "proposal" else ""}): {m["text"]}' for m in order)
            user = (f"{outcome_line}FULL TRANSCRIPT for context (line = [index] Speaker (role, TEAM): message):\n{lines}\n\n"
                    f"Score ONLY these indices: {targets}. Return the JSON object.")
            for idx, s in ev.call_judge(rubric, user).items():
                passes.setdefault(idx, [None, None, None])[k] = s
            time.sleep(ev.SLEEP)
    return {i: float(np.mean([v for v in p if v is not None])) for i, p in passes.items() if any(v is not None for v in p)}


def run(args):
    ev.MODEL = args.model
    ev.SLEEP = 0.5
    paths = sorted(glob.glob(os.path.join(TUNING_DIR, "*.json")))
    random.Random(1).shuffle(paths)
    out = json.load(open(args.out)) if os.path.exists(args.out) else []
    done = {o["game"] for o in out}
    for path in paths[: args.games]:
        gid, winner, transcript, missions = load_transcript(path)
        if gid in done:
            continue
        res = {"game": gid, "winner": winner, "missions": missions,
               "meta": [{"is_good": m["is_good"], "mission": m["mission"], "phase": m["phase"]} for m in transcript]}
        for label, sees in [("aware", True), ("blind", False)]:
            res[label] = score(transcript, winner, gid, sees)
        out.append(res)
        json.dump(out, open(args.out, "w"))
        print(f"done {len(out)} {gid}", flush=True)


def analyze(path):
    data = json.load(open(path))
    print(f"games scored: {len(data)}")
    margins = {"aware": [], "blind": []}
    y = []
    pm = []
    for d in data:
        row = {}
        for label in ["aware", "blind"]:
            s = {int(k): v for k, v in d[label].items()}
            good = [v for i, v in s.items() if d["meta"][i]["is_good"]]
            evil = [v for i, v in s.items() if not d["meta"][i]["is_good"]]
            row[label] = (np.mean(good) - np.mean(evil)) if good and evil else np.nan
        if not np.isnan(row["aware"]) and not np.isnan(row["blind"]):
            for label in margins:
                margins[label].append(row[label])
            y.append(1.0 if d["winner"] == "good" else 0.0)
        common = sorted(set(map(int, d["aware"])) & set(map(int, d["blind"])))
        pm += [(d["aware"][str(i)], d["blind"][str(i)]) for i in common]
    y = np.array(y)
    for label in ["aware", "blind"]:
        x = np.array(margins[label])
        r = stats.pearsonr(x, y)[0]
        z, se = np.arctanh(r), 1 / np.sqrt(len(y) - 3)
        print(f"{label:<6} credit margin vs winner: n={len(y)} r={r:.3f} (95% CI {np.tanh(z - 1.96 * se):.3f} to {np.tanh(z + 1.96 * se):.3f}) sign accuracy {np.mean((x > 0) == (y == 1)):.2f}")
    pm = np.array(pm)
    print(f"per-message correlation aware vs blind: r={stats.pearsonr(pm[:, 0], pm[:, 1])[0]:.3f} over {len(pm)} messages; "
          f"mean aware {pm[:, 0].mean():+.3f}, blind {pm[:, 1].mean():+.3f}")
    # within team-and-game association with mission success, for each condition
    for label in ["aware", "blind"]:
        xs, ys, gs = [], [], []
        for d in data:
            for k, v in d[label].items():
                m = d["meta"][int(k)]
                res = d["missions"].get(str(m["mission"])) or d["missions"].get(m["mission"])
                if res is None:
                    continue
                favourable = (res == "success") if m["is_good"] else (res == "fail")
                xs.append(v); ys.append(float(favourable)); gs.append(f'{d["game"]}|{m["is_good"]}')
        xs, ys, gs = np.array(xs), np.array(ys), np.array(gs)
        xc, yc = xs.copy(), ys.copy()
        for g in set(gs):
            mk = gs == g
            xc[mk] -= xs[mk].mean(); yc[mk] -= ys[mk].mean()
        b = (xc * yc).sum() / (xc ** 2).sum()
        print(f"{label:<6} within team-and-game coefficient of credit on mission success: b = {b:+.3f} (n = {len(xs)} messages)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=25)
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--out", default="blind_check.json")
    ap.add_argument("--analyze")
    a = ap.parse_args()
    analyze(a.analyze) if a.analyze else run(a)
