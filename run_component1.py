"""run_component1_batch.py — run Component 1 end-to-end across every game
in the dataset folder, one at a time, so evaluator.py's built-in 429
retry/backoff can absorb per-minute rate limits along the way.

This does NOT get around a DAILY token cap on your Groq key — if you hit
that, the script will start failing/erroring on every call until the next
day's reset. Check your actual limits first:
https://console.groq.com/settings/limits

Usage:
    python run_component1_batch.py
    python run_component1_batch.py --pattern "dataset/1_cross_game_learning_50g/individual_games/*.json"
    python run_component1_batch.py --delay 5   # extra seconds between games, be gentler on rate limits
"""

import os
import sys
import json
import glob
import time
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from component1.database import get_db_connection
from component1.parser import ingest_game
from component1.evaluator import score_game
from run_component1 import behavioral_for_game, fuse_for_game

DEFAULT_PATTERN = "dataset/1_cross_game_learning_50g/individual_games/*.json"


def run_one_game(file_path, conn):
    with open(file_path) as f:
        game_id = json.load(f)["game_id"]

    ingest_game(file_path, conn)
    behavioral_for_game(game_id, conn)
    n = score_game(game_id, conn)      # this is where 429 retries happen, inside call_judge
    fuse_for_game(game_id, conn)
    return game_id, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default=DEFAULT_PATTERN, help="glob pattern for game JSON files")
    ap.add_argument("--delay", type=float, default=2.0, help="seconds to sleep between games")
    ap.add_argument("--limit", type=int, default=None, help="only run the first N games (for testing)")
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern))
    if args.limit:
        files = files[: args.limit]

    print(f"Found {len(files)} game file(s) matching: {args.pattern}")
    if not files:
        print("No files matched. Check the pattern / run from the repo root.")
        return

    conn = get_db_connection()
    ok, failed = 0, 0
    started = time.time()

    for i, path in enumerate(files, start=1):
        print(f"\n[{i}/{len(files)}] {os.path.basename(path)}")
        try:
            game_id, n = run_one_game(path, conn)
            ok += 1
            print(f"  done: {game_id} — {n} messages scored")
        except Exception as e:
            failed += 1
            conn.rollback()
            print(f"  FAILED: {e}")
            # Heuristic: if this looks like a daily cap (not a per-minute 429
            # that evaluator.py already retried through), stop rather than
            # burn through every remaining game on a dead key.
            msg = str(e).lower()
            if "daily" in msg or "quota" in msg:
                print("  Looks like a daily limit was hit. Stopping the batch run.")
                break

        time.sleep(args.delay)

    conn.close()
    elapsed = time.time() - started
    print(f"\n=== BATCH DONE in {elapsed/60:.1f} min ===")
    print(f"  Succeeded: {ok}")
    print(f"  Failed:    {failed}")


if __name__ == "__main__":
    main()