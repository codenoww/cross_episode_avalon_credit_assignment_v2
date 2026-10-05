import os
from dotenv import load_dotenv
import psycopg2
import json
load_dotenv()

def get_db_connection():
    return psycopg2.connect(
        host="127.0.0.1",
        port=5434,
        database="avalon_research",
        user="postgres",
        password=os.environ["DB_PASSWORD"]
    )

def export_data():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT episode_id, sender_id, role, phase, turn_number,
               final_credit, baseline_credit, content
        FROM messages
        WHERE episode_id != '1'
        ORDER BY episode_id, turn_number
    """)
    messages = [
        {
            "episode_id": r[0],
            "sender_id": r[1],
            "role": r[2],
            "phase": r[3],
            "turn_number": r[4],
            "final_credit": r[5],
            "baseline_credit": r[6],
            "content": r[7]
        }
        for r in cur.fetchall()
    ]

    cur.execute("""
        SELECT episode_id, agent_id, feedback_text, one_liner,
               was_acted_on, created_at
        FROM feedback
        ORDER BY episode_id, agent_id
    """)
    feedback = [
        {
            "episode_id": r[0],
            "agent_id": r[1],
            "feedback_text": r[2],
            "one_liner": r[3],
            "was_acted_on": r[4],
            "created_at": str(r[5])
        }
        for r in cur.fetchall()
    ]

    conn.close()

    data = {
        "messages": messages,
        "feedback": feedback
    }

    with open("db_export.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"Exported successfully to db_export.json")
    print(f"  Messages: {len(messages)}")
    print(f"  Feedback records: {len(feedback)}")

if __name__ == "__main__":
    export_data()