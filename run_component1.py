"""run_component1.py — score ONE game end to end, using the frozen tuned weights.

  python3 components/component1/run_component1.py --file path/to/game.json   # ingest + score a new game
  python3 components/component1/run_component1.py --game <game_id>            # score a game already in the DB
"""

import os
import sys
import json
import argparse
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from component1.database import get_db_connection
from component1.parser import ingest_game
from component1.evaluator import score_game

# Frozen weights (tuned on 50 games: semantic-only was optimal)
W1, W2 = 1.0, 0.0
CREDIT_COL = "baseline_credit"     # set to "final_credit_score" if you haven't renamed the column


def sign(x):
    return (x > 0) - (x < 0)


def behavioral_for_game(game_id, conn):
    cur = conn.cursor()
    cur.execute(
        "SELECT mission_number, proposal_id, voter_name, vote_action "
        "FROM votes WHERE game_id = %s ORDER BY mission_number, proposal_id",
        (game_id,),
    )
    props = defaultdict(list)
    for m, p, voter, action in cur.fetchall():
        props[(m, p)].append((voter, 1 if action == "approve" else 0))

    def first_of(mission):
        ids = [p for (m, p) in props if m == mission]
        return dict(props[(mission, min(ids))]) if ids else None

    def last_of(mission):
        ids = [p for (m, p) in props if m == mission]
        return dict(props[(mission, max(ids))]) if ids else None

    def l1(a, b):
        players = set(a) | set(b)
        return sum(abs(a.get(p, 0) - b.get(p, 0)) for p in players) / len(players) if players else 0.0

    cur.execute("SELECT DISTINCT mission_number FROM messages "
                "WHERE game_id = %s AND phase = 'discussion'", (game_id,))
    for k in sorted(r[0] for r in cur.fetchall()):
        prev, curr = last_of(k - 1), first_of(k)
        delta = l1(prev, curr) if prev and curr else 0.0
        cur.execute("UPDATE messages SET behavioral_delta = %s "
                    "WHERE game_id = %s AND mission_number = %s AND phase = 'discussion'",
                    (delta, game_id, k))
    conn.commit()
    cur.close()


def fuse_for_game(game_id, conn):
    cur = conn.cursor()
    cur.execute("SELECT message_id, semantic_score, behavioral_delta FROM messages "
                "WHERE game_id = %s AND semantic_score IS NOT NULL", (game_id,))
    rows = cur.fetchall()
    for mid, s, b in rows:
        b = b if b is not None else 0.0
        final = sign(s) * (W1 * abs(s) + W2 * b)
        cur.execute(f"UPDATE messages SET {CREDIT_COL} = %s WHERE message_id = %s", (final, mid))
    conn.commit()
    cur.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="path to a game JSON to ingest and score")
    ap.add_argument("--game", help="game_id already in the DB to score")
    args = ap.parse_args()

    conn = get_db_connection()
    if args.file:
        with open(args.file) as f:
            game_id = json.load(f)["game_id"]
        ingest_game(args.file, conn)
        print(f"ingested {game_id}")
    elif args.game:
        game_id = args.game
    else:
        print("Give --file <path> or --game <game_id>")
        return

    behavioral_for_game(game_id, conn)
    n = score_game(game_id, conn)      # semantic pass — needs GROQ_API_KEY
    fuse_for_game(game_id, conn)
    conn.close()
    print(f"done: {game_id} — scored {n} messages, credit written to {CREDIT_COL}.")


if __name__ == "__main__":
    main()