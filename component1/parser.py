"""parser.py — ingest the static 8p games into games/players/messages/votes."""
import json
import os
import sys
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from component1.database import get_db_connection

TARGET_PATTERN = "dataset/1_cross_game_learning_50g/individual_games/*.json"


def ingest_game(file_path, conn):
    with open(file_path, "r") as f:
        game = json.load(f)

    game_id = game["game_id"]
    num_players = game.get("config", {}).get("num_players")
    winner = game.get("winner")

    cur = conn.cursor()

    cur.execute(
        "INSERT INTO games (game_id, num_players, winner) VALUES (%s,%s,%s) "
        "ON CONFLICT (game_id) DO NOTHING",
        (game_id, num_players, winner),
    )

    # roles live in the players LIST (not a dict) — this was the old bug
    for p in game.get("players", []):
        cur.execute(
            "INSERT INTO players (game_id, player_name, role, is_good, special_knowledge) "
            "VALUES (%s,%s,%s,%s,%s) ON CONFLICT (game_id, player_name) DO NOTHING",
            (game_id, p["name"], p["role"], p["is_good"],
             json.dumps(p.get("special_knowledge", []))),
        )

    msg_count = vote_count = 0
    for mission in game.get("missions", []):
        m_num = mission["mission_number"]

        # PUBLIC discussion = the scored text
        for d in mission.get("discussion", []):
            cur.execute(
                "INSERT INTO messages "
                "(game_id, mission_number, proposal_round, global_turn_id, "
                " sender_name, phase, message_text) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT (game_id, global_turn_id) DO NOTHING",
                (game_id, m_num, None, d.get("global_turn_id"),
                 d["player"], d.get("phase", "discussion"), d["content"]),
            )
            msg_count += 1

        # votes drive the behavioral delta
        for prop in mission.get("proposals", []):
            p_id = prop["proposal_id"]

            # NEW: the leader's PUBLIC proposal reasoning, stored as a scored message.
            # proposal_round = the proposal number; global_turn_id is synthetic (100000+)
            # so it stays unique and doesn't collide with discussion turn ids.
            reasoning = prop.get("reasoning")
            leader = prop.get("leader")
            if reasoning:
                cur.execute(
                    "INSERT INTO messages "
                    "(game_id, mission_number, proposal_round, global_turn_id, "
                    " sender_name, phase, message_text) "
                    "VALUES (%s,%s,%s,%s,%s,'proposal',%s) "
                    "ON CONFLICT (game_id, global_turn_id) DO NOTHING",
                    (game_id, m_num, p_id, 100000 + m_num * 100 + p_id, leader, reasoning),
                )
                msg_count += 1

            for v in prop.get("votes", []):
                cur.execute(
                    "INSERT INTO votes "
                    "(game_id, mission_number, proposal_id, voter_name, vote_action) "
                    "VALUES (%s,%s,%s,%s,%s) "
                    "ON CONFLICT (game_id, mission_number, proposal_id, voter_name) DO NOTHING",
                    (game_id, m_num, p_id, v["player"], v["vote"]),
                )
                vote_count += 1

    conn.commit()
    cur.close()
    return msg_count, vote_count


def main():
    files = sorted(glob.glob(TARGET_PATTERN))
    print(f"Found {len(files)} static 8p games at: {TARGET_PATTERN}")
    if not files:
        print("No files matched. Run from the repo root and check the path.")
        return

    conn = get_db_connection()
    total_msgs = total_votes = games_ok = 0
    try:
        for path in files:
            try:
                m, v = ingest_game(path, conn)
                games_ok += 1
                total_msgs += m
                total_votes += v
                print(f"  {os.path.basename(path)}: {m} messages, {v} votes")
            except Exception as e:
                conn.rollback()
                print(f"  SKIPPED {os.path.basename(path)}: {e}")
    finally:
        conn.close()

    print("\n=== DONE ===")
    print(f"Games: {games_ok} | messages: {total_msgs} | votes: {total_votes}")


if __name__ == "__main__":
    main()