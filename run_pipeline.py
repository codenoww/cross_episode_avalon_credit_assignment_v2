"""
Master pipeline orchestrator.
Run this after each game completes.

Usage:
    python run_pipeline.py --file dataset/1_cross_game_learning_50g/individual_games/game_01.json
        Ingests a NEW game (Component 1), then runs the full pipeline for it.

    python run_pipeline.py --episode avalon_20251130_234947
        Re-runs the pipeline (embeddings -> causal graph -> feedback -> export)
        for a game ALREADY ingested/scored in the DB.
"""

import subprocess
import sys
import argparse
import json
import psycopg2


def get_db_connection():
    return psycopg2.connect(
        host="127.0.0.1",
        port=5434,
        database="avalon_research",
        user="postgres",
        password="***REMOVED***"
    )


def get_agents_for_episode(episode_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT sender_id, role
        FROM messages
        WHERE episode_id = %s
    """, (episode_id,))
    rows = cur.fetchall()
    conn.close()
    return [{"agent_id": r[0], "role": r[1]} for r in rows]


def get_episode_outcome(episode_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT winner FROM games WHERE game_id = %s", (episode_id,))
    row = cur.fetchone()
    conn.close()
    if row:
        return "Good team won" if row[0] == "good" else "Evil team won"
    return "Unknown"


def run_step(step_name, cmd):
    print(f"\n[{step_name}] Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print(f"  \u2713 {step_name} complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  \u2717 {step_name} FAILED: {e}")
        return False


def run_pipeline(episode_id, game_file=None):
    print(f"\n{'='*60}")
    print(f"RUNNING FULL PIPELINE FOR EPISODE {episode_id}")
    print(f"{'='*60}")

    # ── Step 0: Baseline credit scorer (Component 1) ──────────
    if game_file:
        if not run_step("Step 0: Component 1 (ingest + score)",
                         ["python", "run_component1.py", "--file", game_file]):
            return
    else:
        print("\n[Step 0] Skipped (--episode given, assuming already ingested/scored)")

    # ── Step 1: Backfill embeddings before Component 2 needs them ──
    if not run_step("Step 1: Embeddings",
                     ["python", "generate_embeddings.py", episode_id]):
        return

    # ── Step 2: Causal graph (Component 2) ─────────────────────
    if not run_step("Step 2: Causal graph",
                     ["python", "causal_graph.py", "--episode", episode_id]):
        return

    # ── Step 2b: causal_score + final_credit ────────────────────
    if not run_step("Step 2b: Compute causal_score + final_credit",
                     ["python", "compute_causal_score.py"]):
        return

    # ── Step 3: Verify final_credit exists before feedback ─────
    print(f"\n[Step 3] Verifying final_credit scores in DB...")
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*) FROM messages
        WHERE episode_id = %s AND final_credit IS NOT NULL
    """, (episode_id,))
    count = cur.fetchone()[0]
    conn.close()

    if count == 0:
        print(f"  WARNING: No final_credit scores found for episode {episode_id}")
        print(f"  Backward propagation may not have run yet \u2014 stopping before feedback.")
        return

    print(f"  {count} messages have final_credit scores \u2014 proceeding")

    # ── Step 4: Feedback generator (your component) ────────────
    print(f"\n[Step 4] Generating feedback for all agents...")
    agents = get_agents_for_episode(episode_id)
    outcome = get_episode_outcome(episode_id)

    if not agents:
        print(f"  WARNING: No agents found for episode {episode_id}")
        return

    print(f"  Outcome: {outcome}")
    print(f"  Agents: {[a['agent_id'] for a in agents]}")

    from feedback_generator import generate_feedback
    for agent in agents:
        generate_feedback(
            episode_id=episode_id,
            agent_id=agent["agent_id"],
            agent_role=agent["role"],
            outcome=outcome,
            episode_number=episode_id
        )

    # ── Step 5: Export for dashboard ────────────────────────────
    if not run_step("Step 5: Export to dashboard", ["python", "analytics_export.py"]):
        return

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE FOR EPISODE {episode_id}")
    print(f"Refresh the dashboard to see updated results.")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="path to a NEW game JSON to ingest and run the full pipeline on")
    group.add_argument("--episode", help="episode_id already ingested/scored, e.g. avalon_20251130_234947")
    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            episode_id = json.load(f)["game_id"]
        run_pipeline(episode_id, game_file=args.file)
    else:
        run_pipeline(args.episode)