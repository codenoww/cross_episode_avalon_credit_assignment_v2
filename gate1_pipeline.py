from gate1_similarity import find_similar_messages
from exposure_check import load_memories, check_exposure
from sentence_transformers import SentenceTransformer
from db import get_connection


def get_message_content(message_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT content, sender_id FROM messages WHERE id = %s", (message_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row  # (content, sender_id)


def get_game_number_for_episode(episode_id, all_games_path):
    import json
    with open(all_games_path, encoding="utf-8") as f:
        data = json.load(f)
    for idx, g in enumerate(data["games"], start=1):
        if g["game_id"] == episode_id:
            return idx
    return None


from data_paths import DATA_PATH


def run_gate1_for_message(target_message_id, model, memories, threshold=0.65, exposure_threshold=0.6):
    """
    Full Gate 1 pass for one message:
    1. Find similar prior messages (pgvector)
    2. For same-episode matches: pass straight through (Option A path)
    3. For cross-episode matches: run exposure check before allowing as a candidate
    """
    target_content, _ = get_message_content(target_message_id)
    candidates = find_similar_messages(target_message_id, threshold=threshold)

    results = []
    for c in candidates:
        if not c["is_cross_episode"]:
            c["path"] = "intra_episode (Option A)"
            c["exposure_checked"] = None
            c["exposure_passed"] = None
            results.append(c)
        else:
            game_number = get_game_number_for_episode(c["episode_id"], DATA_PATH)
            if game_number is None:
                continue
            exposed, score, match_text = check_exposure(
                memories, model, c["sender_id"], before_game_number=game_number,
                origin_message_text=target_content, threshold=exposure_threshold
            )
            c["path"] = "cross_episode (Option B)"
            c["exposure_checked"] = True
            c["exposure_passed"] = exposed
            c["exposure_score"] = score
            results.append(c)

    return results


if __name__ == "__main__":
    print("Loading model and memories...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    memories = load_memories()

    print("\nRunning Gate 1 pipeline for message 178 (Case 3 candidate)...")
    results = run_gate1_for_message(178, model, memories, threshold=0.65)

    for r in results:
        print(r)