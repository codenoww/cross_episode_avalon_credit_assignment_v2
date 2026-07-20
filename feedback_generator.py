import psycopg2
import json
import os
from jinja2 import Environment, FileSystemLoader
from dotenv import load_dotenv
from groq_keys import call_groq

load_dotenv()

GROQ_MODEL = "llama-3.1-8b-instant"


def get_db_connection():
    return psycopg2.connect(
        host="127.0.0.1",
        port=5434,
        database="avalon_research",
        user="postgres",
        password="***REMOVED***"
    )


def get_agent_messages(episode_id, agent_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, content, phase, turn_number, final_credit
        FROM messages
        WHERE episode_id = %s AND sender_id = %s
        ORDER BY final_credit DESC
    """, (episode_id, agent_id))
    rows = cur.fetchall()
    conn.close()
    return [
        {"id": r[0], "content": r[1], "phase": r[2], "turn_number": r[3], "final_credit": r[4]}
        for r in rows
    ]


def get_delta_evidence(episode_id, top_turn_number):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT sender_id, role, content, turn_number
        FROM messages
        WHERE episode_id = %s
        AND turn_number > %s
        AND turn_number <= %s + 5
        ORDER BY turn_number ASC
    """, (episode_id, top_turn_number, top_turn_number))
    rows = cur.fetchall()
    conn.close()
    return [
        {"sender_id": r[0], "role": r[1], "content": r[2], "turn_number": r[3]}
        for r in rows
    ]


def render_prompt(agent_role, episode_number, outcome, messages, delta_evidence):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    env = Environment(loader=FileSystemLoader(os.path.join(base_dir, "templates")))
    role = agent_role.lower()
    if role in ["assassin", "morgana", "mordred", "oberon", "evil"]:
        role = "minion"
    if role in ["good", "percival"]:
        role = "merlin"
    template = env.get_template(f"feedback_prompt_{role}.j2")
    return template.render(
        agent_role=agent_role, episode_number=episode_number,
        outcome=outcome, messages=messages, delta_evidence=delta_evidence
    )


def generate_draft(prompt):
    response = call_groq(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are a strategic coach for a social deduction game called Avalon. Be specific and grounded in what actually happened."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


def verify_draft(draft, episode_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT content FROM messages WHERE episode_id = %s", (episode_id,))
    all_messages = [r[0] for r in cur.fetchall()]
    conn.close()

    verify_prompt = f"""
You are a fact-checker. Does the feedback broadly reflect what
actually happened in the game messages, without inventing
specific facts that contradict the record?

Minor paraphrasing is acceptable. Only reply FAILED if the
feedback contains a specific factual claim that is clearly wrong
or contradicted by the messages.

Reply with just VERIFIED or FAILED.

FEEDBACK DRAFT:
{draft}

ACTUAL GAME MESSAGES:
{chr(10).join(all_messages)}
"""
    response = call_groq(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": verify_prompt}]
    )
    return response.choices[0].message.content.strip().startswith("VERIFIED")


def compress_draft(draft):
    compress_prompt = f"""
Compress this into two things:
1. A 150-word max report with sections: What worked, What didn't work, Pattern, Next episode
2. A one-liner: "[what you did] caused [what changed] — [do this / avoid this] next game"

Return ONLY this JSON, no markdown, no backticks:
{{"report": "...", "one_liner": "..."}}

DRAFT:
{draft}
"""
    # response_format forces strict JSON mode -- the smaller model was
    # occasionally emitting a raw, unescaped newline/control character
    # inside a JSON string value (e.g. from the multi-line report text),
    # which json.loads() rejects by default. Strict mode reduces how
    # often this happens at the source.
    response = call_groq(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": compress_prompt}],
        response_format={"type": "json_object"},
    )
    text = response.choices[0].message.content.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    # Defensive, second layer: even with strict mode, still allow literal
    # control characters inside string values rather than crash outright --
    # this is the exact failure mode that broke Diana's feedback tonight.
    return json.loads(text, strict=False)


def store_feedback(episode_id, agent_id, feedback_text, one_liner, top_id, bottom_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO feedback
        (episode_id, agent_id, feedback_text, one_liner, top_credited_message_id, bottom_credited_message_id)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id
    """, (episode_id, agent_id, feedback_text, one_liner, top_id, bottom_id))
    feedback_id = cur.fetchone()[0]
    conn.commit()
    conn.close()
    return feedback_id


def generate_feedback(episode_id, agent_id, agent_role, outcome, episode_number):
    print(f"\nGenerating feedback for {agent_id} ({agent_role}) — Episode {episode_number}")

    messages = get_agent_messages(episode_id, agent_id)
    if not messages:
        print("No messages found for this agent.")
        return None

    top_message = messages[0]
    bottom_message = messages[-1]
    delta_evidence = get_delta_evidence(episode_id, top_message["turn_number"])
    prompt = render_prompt(agent_role, episode_number, outcome, messages, delta_evidence)

    print(f"\n--- DRAFT ---")
    draft = None
    for attempt in range(2):
        draft = generate_draft(prompt)
        print(draft)
        print("--- END DRAFT ---\n")
        if verify_draft(draft, episode_id):
            print("Draft verified.")
            break
        else:
            print(f"Verification failed, retrying... (attempt {attempt + 1})")

    if not draft:
        print("Could not generate verified draft.")
        return None

    try:
        compressed = compress_draft(draft)
    except json.JSONDecodeError as e:
        # Last-resort: don't lose a verified, good draft just because
        # compression formatting failed -- log clearly and move on rather
        # than silently killing this agent's feedback for the episode.
        print(f"  ✗ Feedback failed for {agent_id}: {e}")
        return None

    feedback_text = compressed["report"]
    one_liner = compressed["one_liner"]

    feedback_id = store_feedback(
        episode_id, agent_id, feedback_text, one_liner,
        top_message["id"], bottom_message["id"]
    )

    print(f"Feedback stored with ID: {feedback_id}")
    print(f"One-liner: {one_liner}")

    return feedback_id


if __name__ == "__main__":
    generate_feedback(
        episode_id="avalon_20251130_234947",
        agent_id="Bob",
        agent_role="good",
        outcome="Good team won",
        episode_number=1
    )