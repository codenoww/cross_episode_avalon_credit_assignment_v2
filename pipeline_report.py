"""
Pipeline Report Generator
Runs after every game and prints a full analysis of what happened
at each component stage.
"""

import psycopg2
import json
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    return psycopg2.connect(
        host="127.0.0.1",
        port=5434,
        database="avalon_research",
        user="postgres",
        password="***REMOVED***"
    )

def generate_pipeline_report(episode_id, game_number, winner):
    conn = get_db_connection()
    cur = conn.cursor()

    print(f"\n{'='*65}")
    print(f"  PIPELINE REPORT — GAME {game_number} | Episode: {episode_id}")
    print(f"  Outcome: {'✓ GOOD TEAM WON' if winner == 'good' else '✗ EVIL TEAM WON'}")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*65}")

    # ── COMPONENT 1: Baseline Credit ────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  COMPONENT 1 — BASELINE CREDIT SCORER")
    print(f"{'─'*65}")

    cur.execute("""
        SELECT sender_id, role, phase, turn_number, 
               baseline_credit, content
        FROM messages
        WHERE episode_id = %s
        AND baseline_credit IS NOT NULL
        ORDER BY baseline_credit DESC
    """, (episode_id,))
    baseline_rows = cur.fetchall()

    if baseline_rows:
        print(f"  Messages scored: {len(baseline_rows)}")
        print(f"\n  TOP 3 BASELINE CREDITED MESSAGES:")
        for i, row in enumerate(baseline_rows[:3]):
            sender, role, phase, turn, score, content = row
            print(f"\n  #{i+1} [{sender} | {role} | {phase} | turn {turn}]")
            print(f"      Score: {score:.4f}")
            print(f"      Message: \"{content[:100]}{'...' if len(content) > 100 else ''}\"")

        print(f"\n  BOTTOM 3 (least impactful):")
        for row in baseline_rows[-3:]:
            sender, role, phase, turn, score, content = row
            print(f"  [{sender} | turn {turn}] Score: {score:.4f} — \"{content[:80]}...\"")
    else:
        print("  No baseline scores found — Component 1 may not have run yet.")

    # ── COMPONENT 2: Causal Graph + DoWhy ───────────────────────
    print(f"\n{'─'*65}")
    print(f"  COMPONENT 2 — CAUSAL GRAPH + DOWHY VERIFICATION")
    print(f"{'─'*65}")

    cur.execute("""
        SELECT COUNT(*) FROM causal_edges WHERE episode_id = %s
    """, (episode_id,))
    total_candidates = cur.fetchone()[0]

    cur.execute("""
        SELECT COUNT(*) FROM causal_edges 
        WHERE episode_id = %s AND confirmed = TRUE
    """, (episode_id,))
    confirmed_count = cur.fetchone()[0]

    cur.execute("""
        SELECT source_message_id, target_message_id, 
               causal_effect_size, influence_type,
               placebo_passed, random_common_cause_passed
        FROM causal_edges
        WHERE episode_id = %s AND confirmed = TRUE
        ORDER BY causal_effect_size DESC
    """, (episode_id,))
    confirmed_edges = cur.fetchall()

    print(f"  Candidate edges generated: {total_candidates}")
    print(f"  Edges surviving DoWhy refutation: {confirmed_count}")
    if total_candidates > 0:
        survival_rate = (confirmed_count / total_candidates) * 100
        print(f"  Survival rate: {survival_rate:.1f}%")

    if confirmed_edges:
        print(f"\n  INFLUENCE TYPES BREAKDOWN:")
        type_counts = {}
        for edge in confirmed_edges:
            inf_type = edge[3] or "unknown"
            type_counts[inf_type] = type_counts.get(inf_type, 0) + 1
        for inf_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            bar = "█" * count
            print(f"  {inf_type:<35} {bar} ({count})")

        print(f"\n  TOP 3 STRONGEST CAUSAL EDGES:")
        for i, edge in enumerate(confirmed_edges[:3]):
            src, tgt, effect, inf_type, placebo, random_cause = edge

            cur.execute("SELECT sender_id, role, content, turn_number FROM messages WHERE id = %s", (src,))
            src_row = cur.fetchone()
            cur.execute("SELECT sender_id, role, content, turn_number FROM messages WHERE id = %s", (tgt,))
            tgt_row = cur.fetchone()

            if src_row and tgt_row:
                print(f"\n  Edge #{i+1} | Effect size: {effect:.4f} | Type: {inf_type}")
                print(f"  SOURCE  [{src_row[0]} | {src_row[1]} | turn {src_row[3]}]")
                print(f"          \"{src_row[2][:90]}{'...' if len(src_row[2]) > 90 else ''}\"")
                print(f"  TARGET  [{tgt_row[0]} | {tgt_row[1]} | turn {tgt_row[3]}]")
                print(f"          \"{tgt_row[2][:90]}{'...' if len(tgt_row[2]) > 90 else ''}\"")
                print(f"  Placebo passed: {placebo} | Random cause passed: {random_cause}")
    else:
        print("  No confirmed edges found for this episode.")

    # ── COMPONENT 3: Final Credit (Backward Propagation) ────────
    print(f"\n{'─'*65}")
    print(f"  COMPONENT 3 — FINAL CREDIT (BACKWARD PROPAGATION)")
    print(f"{'─'*65}")

    cur.execute("""
        SELECT sender_id, role, phase, turn_number,
               baseline_credit, final_credit, content
        FROM messages
        WHERE episode_id = %s
        AND final_credit IS NOT NULL
        ORDER BY final_credit DESC
    """, (episode_id,))
    final_rows = cur.fetchall()

    if final_rows:
        all_final = [r[5] for r in final_rows]
        avg_final = sum(all_final) / len(all_final)
        max_final = max(all_final)
        min_final = min(all_final)

        print(f"  Messages with final credit: {len(final_rows)}")
        print(f"  Score range: {min_final:.4f} to {max_final:.4f}")
        print(f"  Average final credit: {avg_final:.4f}")

        print(f"\n  CREDIT SHIFT (baseline → final, top messages):")
        cur.execute("""
            SELECT sender_id, role, turn_number, 
                   baseline_credit, final_credit, content
            FROM messages
            WHERE episode_id = %s
            AND baseline_credit IS NOT NULL
            AND final_credit IS NOT NULL
            ORDER BY (final_credit - baseline_credit) DESC
            LIMIT 5
        """, (episode_id,))
        shift_rows = cur.fetchall()

        for row in shift_rows:
            sender, role, turn, base, final, content = row
            if base is not None:
                shift = final - base
                direction = "↑" if shift > 0 else "↓"
                print(f"  [{sender} | {role} | turn {turn}] {base:.4f} → {final:.4f} ({direction}{abs(shift):.4f})")
                print(f"    \"{content[:80]}{'...' if len(content) > 80 else ''}\"")

        print(f"\n  TOP 3 FINAL CREDITED MESSAGES (what agents will be coached on):")
        for i, row in enumerate(final_rows[:3]):
            sender, role, phase, turn, base, final, content = row
            print(f"\n  #{i+1} [{sender} | {role} | {phase} | turn {turn}]")
            print(f"      Final credit: {final:.4f}")
            print(f"      Message: \"{content[:100]}{'...' if len(content) > 100 else ''}\"")
    else:
        print("  No final credit scores found.")

    # ── COMPONENT 4: Feedback Generator ─────────────────────────
    print(f"\n{'─'*65}")
    print(f"  COMPONENT 4 — FEEDBACK GENERATOR (YOUR COMPONENT)")
    print(f"{'─'*65}")

    cur.execute("""
        SELECT agent_id, feedback_text, one_liner, 
               top_credited_message_id, bottom_credited_message_id,
               was_acted_on, created_at
        FROM feedback
        WHERE episode_id = %s
        ORDER BY agent_id
    """, (episode_id,))
    feedback_rows = cur.fetchall()

    if feedback_rows:
        print(f"  Feedback notes generated: {len(feedback_rows)}")
        print(f"\n  COACHING NOTES PER AGENT:")
        for row in feedback_rows:
            agent, fb_text, one_liner, top_id, bot_id, acted_on, created = row
            print(f"\n  ── {agent} ──")
            print(f"  One-liner injected: \"{one_liner}\"")
            print(f"  Top credited msg ID: {top_id} | Bottom credited msg ID: {bot_id}")
            status = "✓ Acted on" if acted_on else ("✗ Not acted on" if acted_on is False else "⏳ Pending")
            print(f"  Status: {status}")
    else:
        print("  No feedback generated yet for this episode.")

    # ── COMPONENT 5: System Prompt Injection ─────────────────────
    print(f"\n{'─'*65}")
    print(f"  COMPONENT 5 — SYSTEM PROMPT INJECTION (YOUR COMPONENT)")
    print(f"{'─'*65}")

    cur.execute("""
        SELECT f.agent_id, f.one_liner
        FROM feedback f
        WHERE f.episode_id = %s
        AND f.one_liner IS NOT NULL
    """, (episode_id,))
    injection_rows = cur.fetchall()

    if injection_rows:
        print(f"  Agents receiving injection in NEXT game: {len(injection_rows)}")
        print(f"\n  WHAT EACH AGENT WILL SEE BEFORE NEXT GAME:")
        for agent, one_liner in injection_rows:
            print(f"\n  {agent}:")
            print(f"  [EPISODE {game_number + 1} STARTING]")
            print(f"  Performance note from episode {game_number}:")
            print(f"  {one_liner}")
            print(f"  [GAME BEGINS]")
            print(f"  {{role briefing follows...}}")
    else:
        print("  No one-liners available for injection yet.")

    # ── SUMMARY ─────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  PIPELINE SUMMARY — GAME {game_number}")
    print(f"{'='*65}")
    print(f"  Outcome:             {'Good team won' if winner == 'good' else 'Evil team won'}")
    print(f"  Messages logged:     {len(final_rows)}")
    print(f"  Candidate edges:     {total_candidates}")
    print(f"  Confirmed edges:     {confirmed_count}")
    print(f"  Feedback generated:  {len(feedback_rows)} agents")
    print(f"  Injected next game:  {len(injection_rows)} agents")
    print(f"\n  → Run next game. Agents are carrying their coaching notes.")
    print(f"{'='*65}\n")

    conn.close()