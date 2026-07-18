"""behavioral.py — the deterministic behavioral delta (team-agnostic).

For each discussion message we measure how much the room's VOTING changed
across that mission's discussion phase. It compares the last vote before the
phase with the first vote after it, using normalized L1 distance = "the
fraction of players who changed their vote." Range [0, 1].

This carries NO direction (help vs hurt) — that comes from the semantic pass.
It is identical for good and evil players; all the asymmetry is elsewhere.
"""

import os
import sys
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from component1.database import get_db_connection


def vote_vector(rows):
    """Turn [(player, 'approve'/'reject'), ...] into {player: 1 or 0}."""
    return {voter: (1 if action == "approve" else 0) for voter, action in rows}


def normalized_l1(v_pre, v_post):
    """Fraction of players who changed their vote between two proposals -> [0, 1]."""
    players = set(v_pre) | set(v_post)
    if not players:
        return 0.0
    diff = sum(abs(v_pre.get(p, 0) - v_post.get(p, 0)) for p in players)
    return diff / len(players)


def compute_deltas():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("SELECT game_id FROM games")
    game_ids = [r[0] for r in cur.fetchall()]

    updated = 0
    for game_id in game_ids:
        # Load every VOTED proposal in this game, grouped by (mission, proposal_id).
        cur.execute(
            "SELECT mission_number, proposal_id, voter_name, vote_action "
            "FROM votes WHERE game_id = %s ORDER BY mission_number, proposal_id",
            (game_id,),
        )
        props = defaultdict(list)                 # (mission, proposal_id) -> [(voter, action)]
        for m, p, voter, action in cur.fetchall():
            props[(m, p)].append((voter, action))

        def first_proposal_of(mission):           # proposal 0 = right after the discussion
            ids = [p for (m, p) in props if m == mission]
            return props[(mission, min(ids))] if ids else None

        def last_proposal_of(mission):            # the final voted proposal of a mission
            ids = [p for (m, p) in props if m == mission]
            return props[(mission, max(ids))] if ids else None

        # Which missions have messages in this game?
        cur.execute(
            "SELECT DISTINCT mission_number FROM messages WHERE game_id = %s", (game_id,)
        )
        missions = sorted(r[0] for r in cur.fetchall())

        # One delta per mission (the phase-level quantity).
        for k in missions:
            prev = last_proposal_of(k - 1)        # room's state before this discussion
            curr = first_proposal_of(k)           # room's state right after it
            if prev is None or curr is None:      # mission 1, or missing data
                delta = 0.0
            else:
                delta = normalized_l1(vote_vector(prev), vote_vector(curr))

            # Every message in mission k inherits this phase-level delta.
            cur.execute(
                "UPDATE messages SET behavioral_delta = %s "
                "WHERE game_id = %s AND mission_number = %s",
                (delta, game_id, k),
            )
            updated += cur.rowcount

    conn.commit()
    cur.close()
    conn.close()
    print(f"Behavioral delta written to {updated} messages.")


if __name__ == "__main__":
    compute_deltas()