"""
Bridge script — copies baseline_credit from Component 1's schema
into Component 2's messages table after Component 1 runs.
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

def bridge_baseline_credit(game_id):
    conn = get_connection()
    cur = conn.cursor()

    # Read from Component 1's messages table
    cur.execute("""
        SELECT sender_name, global_turn_id, baseline_credit, phase, message_text
        FROM messages
        WHERE game_id = %s
        AND baseline_credit IS NOT NULL
    """, (game_id,))
    rows = cur.fetchall()
    print(f"Found {len(rows)} scored messages from Component 1")

    updated = 0
    for sender_name, turn_id, baseline_credit, phase, message_text in rows:
        # Match to Component 2's messages by episode_id + sender_id + turn
        cur.execute("""
            UPDATE messages
            SET baseline_credit = %s
            WHERE episode_id = %s
            AND sender_id = %s
            AND turn = %s
        """, (baseline_credit, game_id, sender_name, turn_id))
        updated += cur.rowcount

    conn.commit()
    conn.close()
    print(f"Updated {updated} messages with baseline_credit")

if __name__ == "__main__":
    import sys
    game_id = sys.argv[1] if len(sys.argv) > 1 else "avalon_20251130_234947"
    bridge_baseline_credit(game_id)