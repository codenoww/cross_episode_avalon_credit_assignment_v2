"""
score_control_topup2.py -- post-hoc Component 1 + Component 2 scoring for the
10-game control top-up (avalon_tournament_20260805_091208), which balances
the control set to 60 games matching the 60-game treatment set.

Same approach as score_control_batch.py: calls the real Component 1/2
scripts via subprocess, never touches Component 3.
"""
import glob
import json
import os
import subprocess
import sys

from dotenv import load_dotenv

load_dotenv()

TOURNAMENT_DIRS = [
    "avalon_tournament_20260805_091208",
]

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
    print(f"Found {len(episode_ids)} control episodes.")

    for tdir in TOURNAMENT_DIRS:
        pattern = os.path.join(tdir, "individual_games", "*.json")
        ok = run(
            [sys.executable, "run_component1_batch.py", "--pattern", pattern, "--delay", "2"],
            f"Component 1 batch scoring: {tdir}",
        )
        if not ok:
            print(f"Stopping -- Component 1 failed on {tdir}.")
            return

    run([sys.executable, "generate_embeddings.py", "--all"], "Backfill embeddings")

    for tdir in TOURNAMENT_DIRS:
        run([sys.executable, "merge_new_games.py", tdir], f"Merge into canonical dataset: {tdir}")

    for i, ep in enumerate(episode_ids, start=1):
        print(f"\n[{i}/{len(episode_ids)}] Causal graph for {ep}")
        run([sys.executable, "causal_graph.py", "--episode", ep], f"Causal graph: {ep}")

    run([sys.executable, "compute_causal_score.py"], "Compute causal_score + final_credit")

    print("\n\n" + "=" * 60)
    print("CONTROL TOP-UP 2 SCORING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
