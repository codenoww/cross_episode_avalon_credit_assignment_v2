"""
Command-line entry point for the causal graph pipeline (Gates 1-3 +
bootstrap stability + graph write-back).

Usage:
    python causal_graph.py --episode avalon_20251130_234947
    python causal_graph.py --all
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

from orchestrator import process_episode
from db import get_connection


def get_all_loaded_episodes():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT episode_id FROM messages ORDER BY episode_id")
    rows = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()
    return rows


def print_gate3_summary():
    from gate3 import _run_placebo_check, _run_random_common_cause_check
    placebo = _run_placebo_check()
    rcc = _run_random_common_cause_check()

    print("\n=== GATE 3: MODEL-LEVEL REFUTATION (one-time, applies to the whole fitted model) ===")
    print(f"  Placebo test: real_coef={placebo['real_coef']:.4f}, placebo_coef={placebo['placebo_coef']:.4f}, passed={placebo['passed']}")
    print(f"  Random common cause test: real_coef={rcc['real_coef']:.4f}, rcc_coef={rcc['rcc_coef']:.4f}, passed={rcc['passed']}")


def print_bootstrap_summary():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*), SUM(CASE WHEN bootstrap_stable THEN 1 ELSE 0 END)
        FROM causal_edges WHERE confirmed = TRUE
    """)
    total_confirmed, bootstrap_stable_count = cur.fetchone()
    cur.close()
    conn.close()

    total_confirmed = total_confirmed or 0
    bootstrap_stable_count = bootstrap_stable_count or 0

    print("\n=== BOOTSTRAP STABILITY ===")
    print(f"  Confirmed edges: {total_confirmed}")
    print(f"  Bootstrap-stable: {bootstrap_stable_count}")
    print(f"  Bootstrap-unstable: {total_confirmed - bootstrap_stable_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the causal graph pipeline for one or all episodes.")
    parser.add_argument("--episode", type=str, help="Episode ID to process, e.g. avalon_20251130_234947")
    parser.add_argument("--all", action="store_true", help="Process every episode currently loaded in the database")
    args = parser.parse_args()

    if args.all:
        episodes = get_all_loaded_episodes()
        print(f"Processing {len(episodes)} loaded episodes...")
        total_funnel = {
            "gate1_candidates_found": 0,
            "rejected_no_exposure": 0,
            "skipped_no_credit_data": 0,
            "gate2_evaluated": 0,
            "gate2_passed": 0,
            "confirmed": 0,
        }
        for ep in episodes:
            print(f"\n--- Processing {ep} ---")
            funnel = process_episode(ep, verbose=True)
            for k in total_funnel:
                total_funnel[k] += funnel.get(k, 0)

        print("\n\n=== TOTAL FUNNEL ACROSS ALL EPISODES ===")
        for k, v in total_funnel.items():
            print(f"  {k}: {v}")
        print(f"\n  Episodes processed: {len(episodes)}")

        print_gate3_summary()
        print_bootstrap_summary()

    elif args.episode:
        print(f"Processing episode {args.episode}...")
        funnel = process_episode(args.episode, verbose=True)
        print("\n=== EPISODE FUNNEL ===")
        for k, v in funnel.items():
            print(f"  {k}: {v}")
    else:
        parser.print_help()
        print("\nExample: python causal_graph.py --episode avalon_20251130_234947")
        print("Example: python causal_graph.py --all") 