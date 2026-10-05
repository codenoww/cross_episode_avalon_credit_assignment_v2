import os
from dotenv import load_dotenv
import psycopg2
load_dotenv()

conn = psycopg2.connect(
    host="localhost",
    database="avalon_research",
    user="postgres",
    password=os.environ["DB_PASSWORD"]
)
cur = conn.cursor()

# Insert dummy messages for episode 1
messages = [
    (1, "Alice", "merlin", "discussion", 1, "I think we should look carefully at Charlie's voting pattern — he rejected both strong proposals.", 0.3, 0.82),
    (1, "Bob", "good", "discussion", 2, "I agree with Alice, Charlie has been suspicious from the start.", 0.2, 0.45),
    (1, "Charlie", "evil", "discussion", 3, "I rejected those proposals because the teams were weak, not because I'm evil.", 0.1, 0.21),
    (1, "Diana", "percival", "discussion", 4, "Charlie's explanation doesn't hold up — both teams he rejected had strong players.", 0.4, 0.61),
    (1, "Eve", "assassin", "discussion", 5, "I think we're all being too suspicious of Charlie, let's focus on the mission.", 0.15, 0.18),
    (1, "Alice", "merlin", "voting", 6, "I'm voting reject on this proposal — Eve is trying to deflect attention from Charlie.", 0.5, 0.75),
    (1, "Bob", "good", "voting", 7, "I'll follow Alice's lead and reject as well.", 0.2, 0.38),
    (1, "Charlie", "evil", "voting", 8, "I approve — this team is perfectly fine.", 0.1, 0.15),
    (1, "Diana", "percival", "voting", 9, "Reject. Alice and I are aligned on this.", 0.3, 0.42),
    (1, "Eve", "assassin", "voting", 10, "I approve this team.", 0.1, 0.12),
]

cur.executemany("""
    INSERT INTO messages 
    (episode_id, sender_id, role, phase, turn_number, content, baseline_credit, final_credit)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
""", messages)

conn.commit()
conn.close()
print("Dummy data inserted successfully")