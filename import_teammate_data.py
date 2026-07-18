import psycopg2

# Connect to Docker PostgreSQL on port 5434
conn = psycopg2.connect(
    "host=127.0.0.1 port=5434 dbname=avalon_research user=postgres password=***REMOVED***"
)
cur = conn.cursor()

with open("avalon_component3_export.sql", "r", encoding="utf-8") as f:
    content = f.read()

lines = content.split('\n')

in_messages_copy = False
message_rows = []

for line in lines:
    if 'COPY public.messages' in line or 'COPY messages' in line:
        in_messages_copy = True
        continue
    if line.strip() == '\\.':
        if in_messages_copy:
            in_messages_copy = False
        continue
    if in_messages_copy and line.strip():
        message_rows.append(line)

print(f"Message rows found: {len(message_rows)}")

inserted = 0
skipped = 0

for row in message_rows:
    parts = row.split('\t')
    if len(parts) < 12:
        skipped += 1
        continue
    try:
        episode_id   = parts[1].strip()
        phase        = parts[2].strip()
        turn         = int(parts[3].strip()) if parts[3].strip() != '\\N' else None
        sender_id    = parts[4].strip()
        sender_team  = parts[5].strip()
        content      = parts[7].strip()
        final_credit_raw = parts[11].strip()
        final_credit = float(final_credit_raw) if final_credit_raw != '\\N' else None

        cur.execute("""
            INSERT INTO messages 
            (episode_id, phase, turn, turn_number, sender_id, sender_team, role, content, final_credit)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (episode_id, phase, turn, turn, sender_id, sender_team, sender_team, content, final_credit))
        inserted += 1
    except Exception as e:
        print(f"  Error: {e}")
        skipped += 1

conn.commit()
conn.close()
print(f"Done — inserted: {inserted}, skipped: {skipped}")