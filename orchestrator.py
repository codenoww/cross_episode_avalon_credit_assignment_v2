"""
Phase 7.1 -- Full orchestrator.

Uses gate1_pipeline.run_gate1_for_message() for real exposure verification.
Gate 3's placebo/RCC results are computed once, model-level, and no longer
printed per-episode -- see causal_graph.py's end-of-run summary instead.

FIXED: gate2_passed funnel count now imports GATE2_CONFIRMATION_THRESHOLD
from threshold_config.py instead of using a stale hardcoded value -- was
previously desynced from the real threshold used in process_candidate_edge,
causing the funnel to undercount.
"""

import warnings
warnings.filterwarnings("ignore")

from sentence_transformers import SentenceTransformer
from db import get_connection
from gate1_pipeline import run_gate1_for_message
from exposure_check import load_memories
from gate2 import get_message_row
from graph_pipeline import process_candidate_edge, write_edge_to_db, get_episode_order
from threshold_config import GATE2_CONFIRMATION_THRESHOLD

_model = None
_memories = None


def _get_model_and_memories():
    global _model, _memories
    if _model is None:
        print("Loading model and memories (once)...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _memories = load_memories()
    return _model, _memories


def get_episode_messages(episode_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, turn FROM messages WHERE episode_id = %s ORDER BY turn", (episode_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def filter_valid_candidates(target_message_id, target_episode_id, target_turn, raw_candidates):
    target_episode_idx = get_episode_order(target_episode_id)
    valid = []
    for c in raw_candidates:
        if c["episode_id"] == target_episode_id:
            if c["turn"] < target_turn:
                valid.append(c)
        else:
            c_episode_idx = get_episode_order(c["episode_id"])
            is_earlier = (c_episode_idx is not None and target_episode_idx is not None
                          and c_episode_idx < target_episode_idx)
            if is_earlier and c.get("exposure_passed") is True:
                valid.append(c)
    return valid


def process_episode(episode_id, similarity_threshold=0.65, verbose=True):
    model, memories = _get_model_and_memories()
    messages = get_episode_messages(episode_id)

    funnel = {
        "gate1_candidates_found": 0,
        "rejected_no_exposure": 0,
        "skipped_no_credit_data": 0,
        "gate2_evaluated": 0,
        "gate2_passed": 0,
        "confirmed": 0,
    }

    for target_message_id, target_turn in messages:
        raw_candidates = run_gate1_for_message(
            target_message_id, model, memories, threshold=similarity_threshold
        )
        cross_episode_raw = [c for c in raw_candidates if c["is_cross_episode"]]
        rejected_here = [c for c in cross_episode_raw if c.get("exposure_passed") is not True]
        funnel["rejected_no_exposure"] += len(rejected_here)

        valid_candidates = filter_valid_candidates(target_message_id, episode_id, target_turn, raw_candidates)
        funnel["gate1_candidates_found"] += len(valid_candidates)

        for candidate in valid_candidates:
            target_row = get_message_row(target_message_id)
            if target_row["final_credit_score"] is None:
                funnel["skipped_no_credit_data"] += 1
                continue

            funnel["gate2_evaluated"] += 1
            edge = process_candidate_edge(
                source_message_id=candidate["message_id"],
                target_message_id=target_message_id,
                similarity_score=candidate["similarity"],
                exposure_score=candidate.get("exposure_score"),
            )

            gate2_passed = edge["causal_effect_size"] is not None and abs(edge["causal_effect_size"]) > GATE2_CONFIRMATION_THRESHOLD
            if gate2_passed:
                funnel["gate2_passed"] += 1

            write_edge_to_db(edge)
            if edge["confirmed"]:
                funnel["confirmed"] += 1
                if verbose:
                    print(f"  CONFIRMED: {candidate['message_id']} -> {target_message_id} "
                          f"(effect={edge['causal_effect_size']:.3f}, type={edge['influence_type']})")

    print(f"\nEpisode {episode_id} funnel: {funnel}")
    return funnel


if __name__ == "__main__":
    print("Processing episode avalon_20251201_000058...")
    funnel = process_episode("avalon_20251201_000058") 