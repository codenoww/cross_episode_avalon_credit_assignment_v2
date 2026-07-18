"""
compute_causal_score.py — recomputes causal_score and final_credit locally.

Matches Component 3's exact, hand-verified formula:
    causal_score(m) = SUM(causal_effect_size) across m's OUTGOING edges
                       where confirmed = TRUE AND bootstrap_stable = TRUE
    final_credit(m) = baseline_credit(m) + causal_score(m)

No decay, no multi-hop propagation, direct sum only -- intentionally simple,
matching what was verified by hand against the real data.

Safe to re-run any time (fully recomputes from current causal_edges/messages
state each time, doesn't accumulate). Run this after causal_graph.py for
each new episode so the graph -> per-message score stays up to date without
waiting on a manual export/import cycle.

Usage:
    python compute_causal_score.py
"""

import psycopg2


def get_connection():
    return psycopg2.connect(
        host="127.0.0.1",
        port=5434,
        database="avalon_research",
        user="postgres",
        password="***REMOVED***"
    )


def compute():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        UPDATE messages m
        SET causal_score = COALESCE((
            SELECT SUM(ce.causal_effect_size)
            FROM causal_edges ce
            WHERE ce.source_message_id = m.id
              AND ce.confirmed = TRUE
              AND ce.bootstrap_stable = TRUE
        ), 0)
    """)
    score_updated = cur.rowcount

    cur.execute("""
        UPDATE messages
        SET final_credit = COALESCE(baseline_credit, 0) + COALESCE(causal_score, 0)
        WHERE baseline_credit IS NOT NULL OR causal_score IS NOT NULL
    """)
    credit_updated = cur.rowcount

    conn.commit()
    cur.close()
    conn.close()

    print(f"causal_score recomputed for {score_updated} messages")
    print(f"final_credit recomputed for {credit_updated} messages")


if __name__ == "__main__":
    compute()