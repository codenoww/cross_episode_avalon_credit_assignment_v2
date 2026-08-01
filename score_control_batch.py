"""
score_control_batch.py -- post-hoc Component 1 + Component 2 scoring for the
50-game control set (raw gameplay, no live pipeline attached). Mirrors what
run_full_pipeline() does per-game, but batched across pre-existing game files
spread over two tournament directories (the original run died mid-way when
the session ended; a 2-game top-up completed the set to 50).

Calls the real Component 1/2 scripts via subprocess -- no custom scoring
logic. Never touches Component 3 (feedback/injection) -- these are control
games, meant to stay a clean no-coaching baseline.

Usage:
    python score_control_batch.py
"""
import glob
import json
import os
import subprocess
import sys

from dotenv import load_dotenv

load_dotenv()

TOURNAMENT_DIRS = [
    "avalon_tournament_20260728_162620",
    "avalon_tournament_20260730_010647",
]

# component1/database.py reads DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASS via
# os.getenv() at MODULE IMPORT TIME (not inside a function), and
# run_component1_batch.py imports component1.database before anything that
# triggers load_dotenv() -- so those values freeze to hardcoded fallback
# defaults (localhost:5432, wrong password) before .env ever loads. Not
# touching component1/database.py (teammate's file) -- instead, setting the
# real values as actual OS env vars on the subprocess sidesteps the race
# entirely, since os.getenv() picks up a real env var regardless of import
# order or when/whether load_dotenv() runs.
_SUBPROCESS_ENV = {
    **os.environ,
    "PYTHONIOENCODING": "utf-8",
    "DB_HOST": os.environ.get("DB_HOST", ""),
    "DB_PORT": os.environ.get("DB_PORT", ""),
    "DB_NAME": os.environ.get("DB_NAME", ""),
    "DB_USER": os.environ.get("DB_USER", ""),
    "DB_PASS": os.environ.get("DB_PASSWORD") or os.environ.get("DB_PASS", ""),
}


def run(cmd, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'='*60}", flush=True)
    result = subprocess.run(cmd, env=_SUBPROCESS_ENV)
    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode}): {label}")
        return False
    print(f"  done: {label}")
    return True


def main():
    episode_ids = []
    for tdir in TOURNAMENT_DIRS:
        for path in sorted(glob.glob(os.path.join(tdir, "individual_games", "*.json"))):
            with open(path, encoding="utf-8") as f:
                episode_ids.append(json.load(f)["game_id"])
    print(f"Found {len(episode_ids)} control episodes across {len(TOURNAMENT_DIRS)} tournament dirs.")

    # ── Step 1: Component 1 -- baseline credit scoring ──────────
    for tdir in TOURNAMENT_DIRS:
        pattern = os.path.join(tdir, "individual_games", "*.json")
        ok = run(
            [sys.executable, "run_component1_batch.py", "--pattern", pattern, "--delay", "2"],
            f"Component 1 batch scoring: {tdir}",
        )
        if not ok:
            print(f"Stopping -- Component 1 failed on {tdir}, nothing downstream will have real data for it.")
            return

    # ── Step 2: backfill embeddings for all new messages ────────
    run([sys.executable, "generate_embeddings.py", "--all"], "Backfill embeddings")

    # ── Step 3: merge each tournament into canonical dataset files ──
    for tdir in TOURNAMENT_DIRS:
        run([sys.executable, "merge_new_games.py", tdir], f"Merge into canonical dataset: {tdir}")

    # ── Step 4: Component 2 -- causal graph, per new episode ────
    for i, ep in enumerate(episode_ids, start=1):
        print(f"\n[{i}/{len(episode_ids)}] Causal graph for {ep}")
        run([sys.executable, "causal_graph.py", "--episode", ep], f"Causal graph: {ep}")

    # ── Step 5: causal_score + final_credit (global, idempotent) ──
    run([sys.executable, "compute_causal_score.py"], "Compute causal_score + final_credit")

    print("\n\n" + "=" * 60)
    print("CONTROL BATCH SCORING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
