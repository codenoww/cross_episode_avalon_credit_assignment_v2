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

def get_latest_one_liner(agent_id, current_episode_id):
    conn = get_db_connection()
    cur = conn.cursor()
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

def build_injected_system_prompt(base_system_prompt, agent_id, current_episode_id):
    result = get_latest_one_liner(agent_id, current_episode_id)
    
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