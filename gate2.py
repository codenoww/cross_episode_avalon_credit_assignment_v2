"""
Gate 2 -- Causal estimation

Fits ONE regression model against real scored messages (fast, done once,
cached), then computes each message's INDIVIDUAL treatment effect using the
potential-outcomes formula (Imbens & Rubin, 2015):

    ITE_i = coef(credit_score) * credit_score_i
          + coef(interaction)  * credit_score_i * behavioral_score_i

TRAINING set is restricted to messages with a real behavioral_delta
(discussion messages) -- proposal messages (behavioral_delta always NULL
by design) are excluded from FITTING. Proposals remain fully usable as
Gate 1/2/3 candidates.

UPDATED: outcome is now MISSION-LEVEL, not whole-game-level. Previously
every message in a game was scored against that game's final winner --
noisy, since a message could have a real local effect on its own mission
while the game was lost/won later for unrelated reasons. Now outcome
reflects whether the SPECIFIC MISSION the message was part of succeeded,
relative to the sender's team (success = good for Team Good, bad for
Team Evil).

Column name: baseline_credit (Component 1's naming convention).
"""

import json
import pandas as pd
import statsmodels.api as sm
from db import get_connection

from data_paths import DATA_PATH

_games_data_cache = None
_fitted_model_cache = None
_full_dataset_cache = None
_message_row_cache = {}
_win_rate_cache = {}
_mission_lookup_cache = None


def _load_games_data():
    global _games_data_cache
    if _games_data_cache is None:
        with open(DATA_PATH, encoding="utf-8") as f:
            _games_data_cache = json.load(f)
    return _games_data_cache


def _build_mission_lookup():
    """
    Builds (once, cached) a lookup: (episode_id, mission_number) -> mission_result.
    """
    global _mission_lookup_cache
    if _mission_lookup_cache is not None:
        return _mission_lookup_cache

    data = _load_games_data()
    lookup = {}
    for g in data["games"]:
        for m in g["missions"]:
            lookup[(g["game_id"], m["mission_number"])] = m["mission_result"]
    _mission_lookup_cache = lookup
    return lookup


def get_mission_outcome(episode_id, mission_number, sender_team):
    """
    Returns 1 if the mission's result was favorable to sender_team, else 0.
    A mission "success" favors Good; a mission "fail" favors Evil.
    Returns None if the mission can't be found (e.g. mission_number is None).
    """
    if mission_number is None:
        return None
    lookup = _build_mission_lookup()
    result = lookup.get((episode_id, mission_number))
    if result is None:
        return None
    if sender_team == "good":
        return 1 if result == "success" else 0
    elif sender_team == "evil":
        return 1 if result == "fail" else 0
    return None


def get_message_row(message_id):
    """
    Returns a message's data for CANDIDATE/EVALUATION purposes.
    Cached in memory. behavioral_score is coalesced to 0.0 if NULL
    (e.g. proposals). Now includes mission_number for mission-level
    outcome scoring.
    """
    if message_id in _message_row_cache:
        return _message_row_cache[message_id]

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, episode_id, sender_id, sender_team, turn, mission_number,
               baseline_credit, behavioral_delta_magnitude
        FROM messages WHERE id = %s
    """, (message_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row:
        raise ValueError(f"No message found with id {message_id}")
    result = {
        "id": row[0], "episode_id": row[1], "sender_id": row[2],
        "sender_team": row[3], "turn": row[4], "mission_number": row[5],
        "final_credit_score": row[6],
        "behavioral_score": row[7] if row[7] is not None else 0.0,
    }
    _message_row_cache[message_id] = result
    return result


def get_agent_win_rate(agent_name, up_to_episode_id=None, loaded_game_ids=None):
    """
    Cached -- deterministic given (agent_name, up_to_episode_id,
    loaded_game_ids). Still WHOLE-GAME based (unchanged) -- this measures
    general agent skill across games, not mission-level performance.
    """
    cache_key = (agent_name, up_to_episode_id, tuple(loaded_game_ids) if loaded_game_ids else None)
    if cache_key in _win_rate_cache:
        return _win_rate_cache[cache_key]

    data = _load_games_data()
    games = [g for g in data["games"] if loaded_game_ids is None or g["game_id"] in loaded_game_ids]
    if up_to_episode_id:
        game_ids_in_order = [g["game_id"] for g in games]
        if up_to_episode_id in game_ids_in_order:
            cutoff_idx = game_ids_in_order.index(up_to_episode_id)
            games = games[:cutoff_idx]
    wins, total = 0, 0
    for g in games:
        for p in g["players"]:
            if p["name"] == agent_name:
                total += 1
                won = (g["winner"] == "good" and p["is_good"]) or (g["winner"] == "evil" and not p["is_good"])
                if won:
                    wins += 1
    result = wins / total if total else 0.4
    _win_rate_cache[cache_key] = result
    return result


def _build_full_dataset():
    """
    Builds the TRAINING dataset ONCE -- restricted to messages with a real
    behavioral_delta (discussion messages). Outcome is now MISSION-LEVEL.
    """
    global _full_dataset_cache
    if _full_dataset_cache is not None:
        return _full_dataset_cache

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, episode_id, sender_id, sender_team, mission_number,
               baseline_credit, behavioral_delta_magnitude
        FROM messages
        WHERE baseline_credit IS NOT NULL
          AND behavioral_delta_magnitude IS NOT NULL
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    records = []
    for msg_id, episode_id, sender_id, sender_team, mission_number, credit_score, behavioral_score in rows:
        outcome = get_mission_outcome(episode_id, mission_number, sender_team)
        if outcome is None:
            continue
        skill = get_agent_win_rate(sender_id)
        records.append({
            "message_id": msg_id,
            "credit_score": credit_score,
            "behavioral_score": behavioral_score,
            "agent_skill": skill,
            "outcome": outcome,
        })
    _full_dataset_cache = pd.DataFrame(records)
    print(f"Built full real dataset: {len(_full_dataset_cache)} scored messages (mission-level outcomes).")
    return _full_dataset_cache


def _fit_model_once():
    global _fitted_model_cache
    if _fitted_model_cache is not None:
        return _fitted_model_cache

    df = _build_full_dataset()
    df = df.copy()
    df["interaction"] = df["credit_score"] * df["behavioral_score"]

    X = df[["credit_score", "agent_skill", "interaction"]]
    X = sm.add_constant(X)
    y = df["outcome"]

    model = sm.OLS(y, X).fit()
    _fitted_model_cache = model
    return model


def predict_individual_effect(credit_score, agent_skill, behavioral_score):
    model = _fit_model_once()
    coef_credit = model.params.get("credit_score", 0)
    coef_interaction = model.params.get("interaction", 0)
    contribution = (coef_credit * credit_score) + (coef_interaction * credit_score * behavioral_score)
    return contribution


def run_option_a(message_id):
    msg = get_message_row(message_id)
    skill = get_agent_win_rate(msg["sender_id"])
    df = _build_full_dataset()

    effect = predict_individual_effect(
        credit_score=msg["final_credit_score"],
        agent_skill=skill,
        behavioral_score=msg["behavioral_score"],
    )

    return {
        "message_id": message_id,
        "path": "option_a_intra_episode",
        "estimated_effect": effect,
        "n_rows": len(df),
        "reliable": len(df) >= 8,
    }


def run_option_b(destination_message_id, origin_message_id):
    dest = get_message_row(destination_message_id)
    skill = get_agent_win_rate(dest["sender_id"], up_to_episode_id=dest["episode_id"])
    df = _build_full_dataset()

    effect = predict_individual_effect(
        credit_score=dest["final_credit_score"],
        agent_skill=skill,
        behavioral_score=dest["behavioral_score"],
    )

    return {
        "destination_message_id": destination_message_id,
        "origin_message_id": origin_message_id,
        "path": "option_b_cross_episode",
        "episode_gap": None,
        "estimated_effect": effect,
        "n_rows": len(df),
        "reliable": len(df) >= 8,
    }


if __name__ == "__main__":
    print("Testing individual effects on a few different real messages...")
    for mid in [122, 127, 133, 128]:
        result = run_option_a(mid)
        print(result)