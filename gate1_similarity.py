from db import get_connection

def find_similar_messages(target_message_id, threshold=0.70, limit=10):
    """
    Finds prior messages similar to the target message, using pgvector's
    cosine distance operator (<=>). pgvector returns DISTANCE, not similarity,
    so similarity = 1 - distance.

    threshold is a SIMILARITY threshold (higher = more similar).
    NOTE: 0.70 is a TEST threshold for fixture validation, not the final
    production threshold (that's still TBD, pending full-dataset analysis).
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, episode_id, turn, embedding
        FROM messages WHERE id = %s
    """, (target_message_id,))
    target = cur.fetchone()

    if not target:
        cur.close()
        conn.close()
        raise ValueError(f"No message found with id {target_message_id}")

    target_id, target_episode, target_turn, target_embedding = target

    cur.execute("""
        SELECT id, episode_id, sender_id, sender_team, turn,
               1 - (embedding <=> %s) AS similarity
        FROM messages
        WHERE id != %s
        ORDER BY embedding <=> %s
        LIMIT %s
    """, (target_embedding, target_id, target_embedding, limit))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    results = []
    for row in rows:
        msg_id, episode_id, sender_id, sender_team, turn, similarity = row
        if similarity >= threshold:
            is_cross_episode = (episode_id != target_episode)
            results.append({
                "message_id": msg_id,
                "episode_id": episode_id,
                "sender_id": sender_id,
                "sender_team": sender_team,
                "turn": turn,
                "similarity": similarity,
                "is_cross_episode": is_cross_episode,
            })
    return results


if __name__ == "__main__":
    print("Testing against message 178 (Case 3 candidate)...")
    matches = find_similar_messages(178, threshold=0.65)
    for m in matches:
        print(m)

    print("\nDirect check: what does pgvector say about message 198 specifically?")
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT a.id, b.id, 1 - (a.embedding <=> b.embedding) AS similarity
        FROM messages a, messages b
        WHERE a.id = 178 AND b.id = 198
    """)
    print(cur.fetchone())
    cur.close()
    conn.close() 