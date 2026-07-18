"""
Exposure verification -- checks whether an agent's memory (reflections)
shows real evidence of exposure to an earlier message, before a cross-
episode edge is allowed to become a candidate.

Two caching layers to avoid repeated re-encoding of the same text:
1. Reflection embeddings (self_assessment + player_observations across all
   agents/games) are built ONCE, upfront, in a single batched call.
2. Target/origin message embeddings are cached the first time they're
   encoded, since the same origin message can be checked against multiple
   candidate agents across a run.
"""

import json
from sentence_transformers import util

from data_paths import MEMORY_PATH

_reflection_index_cache = None
_origin_embedding_cache = {}


def load_memories():
    with open(MEMORY_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data["player_memories"]


def _build_reflection_index(memories, model):
    """
    Builds (once, cached) a flat list of every reflection text (self_assessment
    + player_observations) across every agent and game, with embeddings
    computed in ONE batched call.
    """
    global _reflection_index_cache
    if _reflection_index_cache is not None:
        return _reflection_index_cache

    entries = []
    texts = []

    for agent_name, agent_data in memories.items():
        for r in agent_data["reflections"]:
            game_number = r["game_number"]

            sa_text = r.get("self_assessment", "")
            if sa_text:
                entries.append({"agent": agent_name, "game_number": game_number, "text": sa_text})
                texts.append(sa_text)

            for observed_agent, obs_text in r.get("player_observations", {}).items():
                if obs_text:
                    entries.append({"agent": agent_name, "game_number": game_number, "text": obs_text})
                    texts.append(obs_text)

    print(f"Building reflection embedding index: {len(texts)} texts (one-time batch encode)...")
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
    for i, emb in enumerate(embeddings):
        entries[i]["embedding"] = emb

    _reflection_index_cache = entries
    return _reflection_index_cache


def check_exposure(memories, model, agent_name, before_game_number, origin_message_text, threshold=0.6):
    """
    Returns (exposed: bool, best_score: float, best_match_text: str | None)
    Checks whether agent_name's reflections BEFORE before_game_number show
    real semantic overlap with origin_message_text.
    """
    index = _build_reflection_index(memories, model)

    relevant = [e for e in index if e["agent"] == agent_name and e["game_number"] < before_game_number]
    if not relevant:
        return False, 0.0, None

    if origin_message_text in _origin_embedding_cache:
        origin_emb = _origin_embedding_cache[origin_message_text]
    else:
        origin_emb = model.encode(origin_message_text, convert_to_tensor=True)
        _origin_embedding_cache[origin_message_text] = origin_emb

    best_score = 0.0
    best_text = None
    for entry in relevant:
        score = util.cos_sim(origin_emb, entry["embedding"]).item()
        if score > best_score:
            best_score = score
            best_text = entry["text"]

    return best_score >= threshold, best_score, best_text


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    print("Loading model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    memories = load_memories()

    origin_text = "Given M1 with Alice+Diana was clean and the only change on the failed mission was adding Bob, I'm more comfortable minimizing risk by sticking with Alice and Diana."
    exposed, score, match = check_exposure(memories, model, "Charlie", before_game_number=3, origin_message_text=origin_text)
    print(f"\nExposed: {exposed} | Best score: {score:.3f}")
    if match:
        print(f"Best matching reflection text:\n{match[:300]}")