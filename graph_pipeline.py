"""
Orchestrator: runs a candidate message pair through Gate 1 (similarity +
exposure, already computed upstream), Gate 2 (individual causal effect via
one fitted model), Gate 3 (dataset-level refutation checks), and -- IF
Gates 2/3 both pass -- a bootstrap confidence interval check on the target
message. All results, including bootstrap, are written in ONE pass, so
there's no separate manual step to remember.

causal_effect_size is source-aware: scaled by similarity_score (from
Gate 1), so different sources pointing to the same target produce distinct,
pair-specific effects rather than one shared target-only value.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import networkx as nx
from db import get_connection
from gate2 import get_message_row, get_agent_win_rate
from gate3 import run_gate3_for_message
from bootstrap_ci import bootstrap_effect_interval
from threshold_config import GATE2_CONFIRMATION_THRESHOLD

from data_paths import DATA_PATH

_episode_order_cache = None
_bootstrap_cache = {}  # target_message_id -> (stable, ci_low, ci_high), avoids recomputing per edge

N_BOOTSTRAP = 100


def get_episode_order(episode_id):
    global _episode_order_cache
    if _episode_order_cache is None:
        with open(DATA_PATH, encoding="utf-8") as f:
            data = json.load(f)
        _episode_order_cache = {g["game_id"]: idx for idx, g in enumerate(data["games"])}
    return _episode_order_cache.get(episode_id)


def determine_influence_type(source_msg, target_msg):
    if source_msg["episode_id"] == target_msg["episode_id"]:
        return "intra_episode"
    return "same_team_propagation" if source_msg["sender_team"] == target_msg["sender_team"] else "cross_team_adaptation"


def _get_bootstrap_result(target_message_id):
    """
    Cached per target message -- multiple edges often share the same
    target, so this avoids redundant 100-resample computations.
    """
    if target_message_id in _bootstrap_cache:
        return _bootstrap_cache[target_message_id]

    msg = get_message_row(target_message_id)
    skill = get_agent_win_rate(msg["sender_id"])
    point, low, high, stable = bootstrap_effect_interval(
        msg["final_credit_score"], skill, msg["behavioral_score"],
        n_bootstrap=N_BOOTSTRAP,
    )
    result = (bool(stable), float(low), float(high))
    _bootstrap_cache[target_message_id] = result
    return result


def process_candidate_edge(source_message_id, target_message_id, similarity_score, exposure_score=None):
    """
    Runs the full gate pipeline for one candidate edge (source -> target).
    Returns the edge dict regardless of outcome; 'confirmed' tells you if
    it survived all gates. bootstrap_stable is only computed if Gates 2/3
    both pass (expensive -- 100 resamples -- so skipped for anything that
    wouldn't be confirmed anyway).
    """
    source_msg = get_message_row(source_message_id)
    target_msg = get_message_row(target_message_id)

    is_same_episode = source_msg["episode_id"] == target_msg["episode_id"]
    path = "option_a_intra_episode" if is_same_episode else "option_b_cross_episode"

    episode_gap = 0
    if not is_same_episode:
        src_idx = get_episode_order(source_msg["episode_id"])
        tgt_idx = get_episode_order(target_msg["episode_id"])
        if src_idx is not None and tgt_idx is not None:
            episode_gap = abs(tgt_idx - src_idx)

    influence_type = determine_influence_type(source_msg, target_msg)

    gate3_result = run_gate3_for_message(target_message_id, path=path)

    target_effect = gate3_result["original_effect"]

    if target_effect is not None and similarity_score is not None:
        edge_effect = target_effect * similarity_score
    else:
        edge_effect = target_effect

    gate2_confirmed = bool(edge_effect is not None and abs(edge_effect) > GATE2_CONFIRMATION_THRESHOLD)
    placebo_passed = bool(gate3_result.get("placebo_passed"))
    rcc_passed = bool(gate3_result.get("random_common_cause_passed"))
    gate3_confirmed = placebo_passed and rcc_passed

    confirmed = bool(gate2_confirmed and gate3_confirmed)

    bootstrap_stable, bootstrap_ci_low, bootstrap_ci_high = None, None, None
    if confirmed:
        bootstrap_stable, bootstrap_ci_low, bootstrap_ci_high = _get_bootstrap_result(target_message_id)

    edge = {
        "source_message_id": source_message_id,
        "target_message_id": target_message_id,
        "episode_id": target_msg["episode_id"],
        "path": path,
        "causal_effect_size": float(edge_effect) if edge_effect is not None else None,
        "episode_gap": episode_gap,
        "influence_type": influence_type,
        "similarity_score": similarity_score,
        "exposure_score": exposure_score,
        "gate2_reliable": bool(gate3_result["reliable"]),
        "placebo_passed": placebo_passed,
        "random_common_cause_passed": rcc_passed,
        "confirmed": confirmed,
        "bootstrap_stable": bootstrap_stable,
        "bootstrap_ci_low": bootstrap_ci_low,
        "bootstrap_ci_high": bootstrap_ci_high,
    }
    return edge


def write_edge_to_db(edge):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO causal_edges (episode_id, source_message_id, target_message_id, path, causal_effect_size,
                            episode_gap, influence_type, similarity_score, exposure_score,
                            gate2_reliable, placebo_passed, random_common_cause_passed, confirmed,
                            bootstrap_stable, bootstrap_ci_low, bootstrap_ci_high)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
    """, (
        edge["episode_id"], edge["source_message_id"], edge["target_message_id"], edge["path"], edge["causal_effect_size"],
        edge["episode_gap"], edge["influence_type"], edge["similarity_score"], edge["exposure_score"],
        edge["gate2_reliable"], edge["placebo_passed"], edge["random_common_cause_passed"], edge["confirmed"],
        edge["bootstrap_stable"], edge["bootstrap_ci_low"], edge["bootstrap_ci_high"],
    ))
    edge_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return edge_id


def load_graph_from_db(stable_only=False):
    """
    Builds a NetworkX DiGraph from CONFIRMED edges in Postgres.
    stable_only=True restricts to bootstrap-verified edges only (your
    higher-confidence "headline" subset).
    """
    conn = get_connection()
    cur = conn.cursor()
    query = """
        SELECT source_message_id, target_message_id, causal_effect_size,
               episode_gap, influence_type
        FROM causal_edges WHERE confirmed = TRUE
    """
    if stable_only:
        query += " AND bootstrap_stable = TRUE"
    cur.execute(query)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    G = nx.DiGraph()
    for source, target, effect, gap, influence_type in rows:
        G.add_edge(source, target, effect=effect, episode_gap=gap, influence_type=influence_type)
    return G


if __name__ == "__main__":
    print("Running full gate pipeline for message 137 (Case 1 fixture)...")
    edge = process_candidate_edge(
        source_message_id=136, target_message_id=137,
        similarity_score=0.9,
        exposure_score=None,
    )
    print(edge)