import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    return psycopg2.connect(
        host="127.0.0.1",
        port=5434,
        database="avalon_research",
        user="postgres",
        password=os.environ["DB_PASSWORD"]
    )

EVIL_ROLES = {"evil", "assassin", "morgana", "mordred", "oberon", "minion"}

# Role-aware injection. OFF by default so existing behaviour (and every result
# reported so far) is unchanged. When enabled (INJECTION_ROLE_AWARE=1 or
# role_aware=True), a note is only injected if it was written for an episode in
# which the agent played the same side (INJECTION_ROLE_MATCH=team, default) or the
# same exact role (INJECTION_ROLE_MATCH=role) as in the episode now starting.
# If no such note exists, nothing is injected. Written but not run.
ROLE_AWARE_DEFAULT = os.environ.get("INJECTION_ROLE_AWARE", "0") == "1"
ROLE_MATCH_DEFAULT = os.environ.get("INJECTION_ROLE_MATCH", "team")


def get_latest_one_liner(agent_id, current_episode_id, current_role=None, role_aware=None, role_match=None):
    role_aware = ROLE_AWARE_DEFAULT if role_aware is None else role_aware
    role_match = ROLE_MATCH_DEFAULT if role_match is None else role_match
    conn = get_db_connection()
    cur = conn.cursor()
    if role_aware and current_role:
        role = str(current_role).lower()
        if role_match == "role":
            cur.execute("""
                SELECT f.one_liner, f.id
                FROM feedback f
                JOIN players p ON p.game_id = f.episode_id AND p.player_name = f.agent_id
                WHERE f.agent_id = %s AND f.episode_id != %s AND lower(p.role) = %s
                ORDER BY f.created_at DESC
                LIMIT 1
            """, (agent_id, str(current_episode_id), role))
        else:
            cur.execute("""
                SELECT f.one_liner, f.id
                FROM feedback f
                JOIN players p ON p.game_id = f.episode_id AND p.player_name = f.agent_id
                WHERE f.agent_id = %s AND f.episode_id != %s AND p.is_good = %s
                ORDER BY f.created_at DESC
                LIMIT 1
            """, (agent_id, str(current_episode_id), role not in EVIL_ROLES))
    else:
        cur.execute("""
            SELECT one_liner, id
            FROM feedback
            WHERE agent_id = %s
            AND episode_id != %s
            ORDER BY created_at DESC
            LIMIT 1
        """, (agent_id, str(current_episode_id)))
    row = cur.fetchone()
    conn.close()
    if row:
        return {"one_liner": row[0], "feedback_id": row[1]}
    return None

def build_injected_system_prompt(base_system_prompt, agent_id, current_episode_id, current_role=None):
    result = get_latest_one_liner(agent_id, current_episode_id, current_role=current_role)
    
    if not result:
        return base_system_prompt
    
    one_liner = result["one_liner"]
    
    injected_prompt = f"""[EPISODE {current_episode_id} STARTING]
Performance note from previous episode:
{one_liner}

[GAME BEGINS]
{base_system_prompt}"""
    
    return injected_prompt

def mark_feedback_acted_on(feedback_id, acted_on: bool):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE feedback
        SET was_acted_on = %s
        WHERE id = %s
    """, (acted_on, feedback_id))
    conn.commit()
    conn.close()
    print(f"Feedback ID {feedback_id} marked as acted_on={acted_on}")


if __name__ == "__main__":
    base_prompt = "You are Alice, playing The Resistance: Avalon. You are Merlin."
    injected = build_injected_system_prompt(
        base_system_prompt=base_prompt,
        agent_id="Bob",
        current_episode_id="avalon_20251201_001747"
    )
    print("=== INJECTED SYSTEM PROMPT ===")
    print(injected)
    print("=== END ===")