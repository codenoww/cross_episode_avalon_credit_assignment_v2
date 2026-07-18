"""evaluator.py — asymmetric semantic judge, two dedicated rubrics.

Each team is scored in its own call with its own detailed rubric, averaged
over 3 transcript orderings. Judge: Groq.   export GROQ_API_KEY=...
"""

import os
import sys
import json
import time
import random
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from component1.database import get_db_connection
from groq_keys import call_groq

MODEL = "llama-3.1-8b-instant"
PASSES = ["forward", "reverse", "shuffled"]
SLEEP = 2

GOOD_RUBRIC = (
    "You are an expert Avalon analyst. You score ONLY the GOOD-team players' public "
    "messages. Good's goal: complete 3 missions, deduce and exclude the evil players, "
    "and keep Merlin's identity hidden (Merlin secretly knows the evil players, but if "
    "the Assassin identifies Merlin at the end, evil steals the win). You are given "
    "every player's true role. Judge only each message's causal contribution.\n\n"
    "Scale -1.0 to +1.0, and USE THE FULL RANGE:\n"
    " +0.7..+1.0 decisive help: correctly fingered an evil player, steered the group "
    "off a bad team, secured a genuinely safe mission team, or (as Merlin) guided the "
    "table while staying hidden.\n"
    " +0.3..+0.6 mild help: reasonable suspicion, useful information, moved consensus "
    "but was not decisive.\n"
    "  0.0 neutral: filler, greeting, or a point everyone ignored.\n"
    " -0.3..-0.6 mild harm: muddied the reasoning or mildly misdirected the group.\n"
    " -0.7..-1.0 serious harm: defended an evil player, pushed a team that failed, or "
    "(as Merlin) exposed their own identity.\n"
    "Reserve the extremes for truly decisive messages; most fall in between.\n"
    "Some messages are team PROPOSALS (a leader justifying their team pick): reward "
    "a clean, safe team with clear reasoning; penalize proposing a team that included "
    "evil players or that later failed.\n"
    'Return ONLY JSON: {"scores": [{"index": <int>, "score": <float>}, ...]} for '
    "exactly the indices requested."
)

EVIL_RUBRIC = (
    "You are an expert Avalon analyst. You score ONLY the EVIL-team players' public "
    "messages. Evil's goal: fail 3 missions OR have the Assassin identify Merlin at the "
    "end, all while staying hidden and trusted. You are given every player's true role. "
    "Judge only each message's causal contribution.\n\n"
    "Scale -1.0 to +1.0, and USE THE FULL RANGE:\n"
    " +0.7..+1.0 masterful: convincingly framed an innocent, engineered onto a mission "
    "to sabotage, sowed effective confusion, or gathered information pinpointing Merlin.\n"
    " +0.3..+0.6 mild help: blended in well, built false trust, or cast mild doubt on a "
    "good player.\n"
    "  0.0 neutral: bland agreement that swayed no one.\n"
    " -0.3..-0.6 mild harm: behaved a little suspiciously without being caught.\n"
    " -0.7..-1.0 blunder: a lie so obvious it united the good team, or a move that "
    "exposed a fellow evil (or themselves).\n"
    "Reserve the extremes for truly decisive messages; most fall in between.\n"
    "Some messages are team PROPOSALS (a leader justifying their team pick): reward "
    "sneaking an evil player onto the mission or framing a good one while sounding "
    "reasonable; penalize a proposal so suspicious it exposed you.\n"
    'Return ONLY JSON: {"scores": [{"index": <int>, "score": <float>}, ...]} for '
    "exactly the indices requested."
)


def call_judge(system_prompt, user_prompt, tries=8):
    for attempt in range(tries):
        try:
            resp = call_groq(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            if isinstance(data, dict):
                data = next((v for v in data.values() if isinstance(v, list)), [])
            out = {}
            for item in data:
                try:
                    out[int(item["index"])] = max(-1.0, min(1.0, float(item["score"])))
                except (KeyError, ValueError, TypeError):
                    continue
            return out
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate" in msg.lower():
                m = re.search(r"([0-9.]+)s", msg)
                wait = min((float(m.group(1)) + 1) if m else 10, 30)
                print(f"    rate limited, waiting {wait:.0f}s...")
                time.sleep(wait)
                continue
            print(f"    call error: {msg[:120]}")
            return {}
    return {}


def score_game(game_id, conn):
    cur = conn.cursor()
    cur.execute("SELECT winner FROM games WHERE game_id = %s", (game_id,))
    winner = cur.fetchone()[0]

    cur.execute(
        "SELECT m.message_id, m.sender_name, p.role, p.is_good, m.message_text, "
        "       m.semantic_score, m.phase "
        "FROM messages m JOIN players p "
        "  ON p.game_id = m.game_id AND p.player_name = m.sender_name "
        "WHERE m.game_id = %s "
        "ORDER BY m.mission_number, (m.phase = 'proposal')::int, m.global_turn_id",
        (game_id,),
    )
    rows = cur.fetchall()
    transcript = [
        {"index": i, "message_id": mid, "sender": s, "role": r, "is_good": g,
         "text": t, "scored": sem is not None, "phase": ph}
        for i, (mid, s, r, g, t, sem, ph) in enumerate(rows)
    ]

    totals, counts = {}, {}
    for team_is_good, rubric in [(True, GOOD_RUBRIC), (False, EVIL_RUBRIC)]:
        # only this team's messages that are NOT already scored
        targets = [m["index"] for m in transcript
                   if m["is_good"] == team_is_good and not m["scored"]]
        if not targets:
            continue
        for pass_name in PASSES:
            order = list(transcript)
            if pass_name == "reverse":
                order = list(reversed(order))
            elif pass_name == "shuffled":
                order = random.sample(order, len(order))

            lines = "\n".join(
                f'[{m["index"]}] {m["sender"]} ({m["role"]}, '
                f'{"GOOD" if m["is_good"] else "EVIL"}'
                f'{", PROPOSAL" if m["phase"] == "proposal" else ""}): {m["text"]}'
                for m in order
            )
            user = (
                f"GAME OUTCOME: {winner} team won.\n\n"
                f"FULL TRANSCRIPT for context (line = [index] Speaker (role, TEAM): message):\n{lines}\n\n"
                f"Score ONLY these indices: {targets}. Return the JSON object."
            )
            result = call_judge(rubric, user)
            for idx, score in result.items():
                totals[idx] = totals.get(idx, 0.0) + score
                counts[idx] = counts.get(idx, 0) + 1
            time.sleep(SLEEP)

    written = 0
    for m in transcript:
        idx = m["index"]
        if counts.get(idx):
            cur.execute(
                "UPDATE messages SET semantic_score = %s WHERE message_id = %s",
                (totals[idx] / counts[idx], m["message_id"]),
            )
            written += 1
    conn.commit()
    cur.close()
    return written


def main():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT DISTINCT game_id FROM messages WHERE semantic_score IS NULL ORDER BY game_id"
    )
    game_ids = [r[0] for r in cur.fetchall()]
    cur.close()

    if not game_ids:
        print("All games already scored. Nothing to do.")
        conn.close()
        return

    print(f"{len(game_ids)} game(s) left to score.")
    total = 0
    for gid in game_ids:
        n = score_game(gid, conn)
        total += n
        print(f"  {gid}: scored {n} messages")
    conn.close()
    print(f"\nDone this run. {total} messages scored.")


if __name__ == "__main__":
    main()