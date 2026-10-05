"""
generate_embeddings.py — backfill missing embeddings for message(s).
Run this after run_component1.py, before causal_graph.py.

Usage:
    python generate_embeddings.py avalon_20251130_234947   # one episode
    python generate_embeddings.py --all                    # every episode missing embeddings
"""
import os
from dotenv import load_dotenv
load_dotenv()
import sys
import psycopg2
from sentence_transformers import SentenceTransformer


def get_connection():
    return psycopg2.connect(
        host="127.0.0.1",
        port=5434,
        database="avalon_research",
        user="postgres",
        password=os.environ["DB_PASSWORD"]
    )


def get_episodes_missing_embeddings():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT episode_id FROM messages WHERE embedding IS NULL")
    rows = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()
    return rows


def generate_embeddings(episode_id, model):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, content
        FROM messages
        WHERE episode_id = %s
        AND embedding IS NULL
        AND content IS NOT NULL
    """, (episode_id,))
    rows = cur.fetchall()
    print(f"Found {len(rows)} messages missing embeddings for {episode_id}")

    if not rows:
        print("Nothing to do.")
        conn.close()
        return 0

    ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]

    print("Encoding...")
    embeddings = model.encode(texts, show_progress_bar=True)

    updated = 0
    for msg_id, emb in zip(ids, embeddings):
        # pgvector accepts a string like '[0.1,0.2,...]'
        emb_str = "[" + ",".join(str(float(x)) for x in emb) + "]"
        cur.execute(
            "UPDATE messages SET embedding = %s WHERE id = %s",
            (emb_str, msg_id)
        )
        updated += cur.rowcount

    conn.commit()
    conn.close()
    print(f"Updated {updated} messages with embeddings for {episode_id}\n")
    return updated


if __name__ == "__main__":
    print("Loading model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        episodes = get_episodes_missing_embeddings()
        print(f"Found {len(episodes)} episode(s) missing embeddings: {episodes}\n")
        total = 0
        for ep in episodes:
            total += generate_embeddings(ep, model)
        print(f"=== DONE: {total} messages updated across {len(episodes)} episodes ===")
    else:
        episode_id = sys.argv[1] if len(sys.argv) > 1 else "avalon_20251130_234947"
        generate_embeddings(episode_id, model)