    """
compare_treatment_control.py — compares your 50 existing control games
(never had injection) against any new treatment games run tonight via
multi_game_runner.py (injection active from game 2 onward).

IMPORTANT HONESTY NOTE, read before presenting any of this:
With only a handful of treatment games against 50 control games, this is a
PILOT comparison, not a statistically powered result. Report it as "early
signal" / "directional", not as proof. A single win/loss swing in a 5-10
game treatment set can easily be noise, not a real effect. Say this
explicitly if you present these numbers.

Usage:
    python compare_treatment_control.py
"""

import psycopg2

CONTROL_EPISODES = {
    "avalon_20251130_234947", "avalon_20251201_000058", "avalon_20251201_000457",
    "avalon_20251201_000805", "avalon_20251201_001747", "avalon_20251201_002652",
    "avalon_20251201_003138", "avalon_20251201_004734", "avalon_20251201_005546",
    "avalon_20251201_011949", "avalon_20251201_012415", "avalon_20251201_013236",
    "avalon_20251201_014622", "avalon_20251201_015725", "avalon_20251201_020734",
    "avalon_20251201_022119", "avalon_20251201_023003", "avalon_20251201_023827",
    "avalon_20251201_024547", "avalon_20251201_030101", "avalon_20251201_030408",
    "avalon_20251201_030844", "avalon_20251201_031607", "avalon_20251201_033330",
    "avalon_20251201_033821", "avalon_20251201_034434", "avalon_20251201_035439",
    "avalon_20251201_040534", "avalon_20251201_041339", "avalon_20251201_041734",
    "avalon_20251201_042616", "avalon_20251201_043224", "avalon_20251201_043610",
    "avalon_20251201_043844", "avalon_20251201_044512", "avalon_20251201_044722",
    "avalon_20251201_045358", "avalon_20251201_045837", "avalon_20251201_050404",
    "avalon_20251201_050750", "avalon_20251201_051414", "avalon_20251201_051759",
    "avalon_20251201_052141", "avalon_20251201_052452", "avalon_20251201_053824",
    "avalon_20251201_054906", "avalon_20251201_055432", "avalon_20251201_060128",
    "avalon_20251201_060359", "avalon_20251201_060806",
}


def get_connection():
    return psycopg2.connect(
        host="127.0.0.1",
        port=5434,
        database="avalon_research",
        user="postgres",
        password="***REMOVED***"
    )


def get_all_episodes():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT game_id, winner FROM games")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def get_credit_stats(episode_ids, label):
    if not episode_ids:
        print(f"  {label}: no episodes found, skipping")
        return
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT AVG(baseline_credit), AVG(final_credit), COUNT(*)
        FROM messages
        WHERE episode_id = ANY(%s) AND baseline_credit IS NOT NULL
    """, (list(episode_ids),))
    avg_baseline, avg_final, n = cur.fetchone()
    cur.close()
    conn.close()
    print(f"  {label}: avg baseline_credit={avg_baseline:.4f}  avg final_credit={avg_final:.4f}  (n={n} messages)")


def win_rate(episode_winner_pairs, label):
    if not episode_winner_pairs:
        print(f"  {label}: no games found, skipping")
        return
    total = len(episode_winner_pairs)
    good_wins = sum(1 for _, w in episode_winner_pairs if w == "good")
    print(f"  {label}: {good_wins}/{total} good-team wins ({100*good_wins/total:.1f}%)")


def per_agent_trend():
    """
    Within treatment games only: for each agent who appears in more than
    one treatment game, does their average final_credit trend up across
    successive games? This is the cleanest "did coaching help THIS agent"
    signal, since it's a within-subject comparison rather than relying on
    noisy win/loss across a tiny sample.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT g.game_id, g.winner, m.sender_id, AVG(m.final_credit) as avg_credit
        FROM messages m
        JOIN games g ON g.game_id = m.episode_id
        WHERE m.episode_id != ALL(%s) AND m.final_credit IS NOT NULL
        GROUP BY g.game_id, g.winner, m.sender_id
        ORDER BY m.sender_id, g.game_id
    """, (list(CONTROL_EPISODES),))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    by_agent = {}
    for game_id, winner, sender_id, avg_credit in rows:
        by_agent.setdefault(sender_id, []).append((game_id, avg_credit))

    print("\n  Per-agent trend across treatment games (chronological by game_id):")
    for agent, entries in sorted(by_agent.items()):
        if len(entries) < 2:
            continue
        values = [f"{c:.3f}" for _, c in entries]
        direction = "UP" if entries[-1][1] > entries[0][1] else "DOWN" if entries[-1][1] < entries[0][1] else "FLAT"
        print(f"    {agent}: {' -> '.join(values)}  [{direction}]")


def main():
    all_episodes = get_all_episodes()
    control = [(eid, w) for eid, w in all_episodes if eid in CONTROL_EPISODES]
    treatment = [(eid, w) for eid, w in all_episodes if eid not in CONTROL_EPISODES]

    print(f"Control episodes found in DB: {len(control)} (expected 50)")
    print(f"Treatment episodes found in DB: {len(treatment)}")

    if len(treatment) == 0:
        print("\nNo treatment games in the database yet -- run multi_game_runner.py first.")
        return

    print("\n=== WIN RATE ===")
    win_rate(control, "Control  ")
    win_rate(treatment, "Treatment")

    print("\n=== CREDIT SCORES ===")
    get_credit_stats(CONTROL_EPISODES, "Control  ")
    get_credit_stats([eid for eid, _ in treatment], "Treatment")

    per_agent_trend()

    print("\n" + "="*60)
    print("REMINDER: with this few treatment games, treat this as a")
    print("directional pilot signal, not a statistically powered result.")
    print("="*60)


if __name__ == "__main__":
    main()