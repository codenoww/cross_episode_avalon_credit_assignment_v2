"""
merge_new_games.py -- keeps Component 2's canonical dataset files current
with every new tournament game.

gate2.py's DATA_PATH, graph_pipeline.py's DATA_PATH, gate1_pipeline.py's
DATA_PATH, and exposure_check.py's MEMORY_PATH are all read from the same
fixed files. Without this script, a new tournament game's messages are
invisible to mission-outcome lookup (get_mission_outcome), episode
ordering (get_episode_order), and cross-episode exposure verification
(load_memories) -- they can only ever produce intra-episode confirmed
edges, and never contribute to Gate 2's training set, regardless of how
the tournament is run.

This must run BEFORE causal_graph.py in the per-game pipeline, since
Gate 1/2 read these files at that point.

Usage:
    python merge_new_games.py <tournament_dir>
"""
import json
import os
import sys

from gate2 import DATA_PATH
from exposure_check import MEMORY_PATH


def merge_games(tournament_dir):
    new_games_path = os.path.join(tournament_dir, "all_games.json")
    if not os.path.exists(new_games_path):
        print(f"  No all_games.json in {tournament_dir} yet -- skipping game merge.")
        return 0

    with open(new_games_path, encoding="utf-8") as f:
        new_data = json.load(f)

    with open(DATA_PATH, encoding="utf-8") as f:
        canonical_data = json.load(f)

    existing_ids = {g["game_id"] for g in canonical_data["games"]}
    added = 0
    for g in new_data.get("games", []):
        if g["game_id"] not in existing_ids:
            canonical_data["games"].append(g)
            existing_ids.add(g["game_id"])
            added += 1

    if added:
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(canonical_data, f, indent=2)

    print(f"  Merged {added} new game(s) into {DATA_PATH} (now {len(canonical_data['games'])} total)")
    return added


def merge_memories(tournament_dir):
    new_mem_path = os.path.join(tournament_dir, "player_memories.json")
    if not os.path.exists(new_mem_path):
        print(f"  No player_memories.json in {tournament_dir} yet -- skipping memory merge.")
        return 0

    with open(new_mem_path, encoding="utf-8") as f:
        new_mem = json.load(f)

    with open(MEMORY_PATH, encoding="utf-8") as f:
        canonical_mem = json.load(f)

    added = 0
    for player_name, data in new_mem.get("player_memories", {}).items():
        if player_name not in canonical_mem["player_memories"]:
            canonical_mem["player_memories"][player_name] = {
                "player_name": player_name, "reflections": []
            }

        existing_game_numbers = {
            r.get("game_number") for r in canonical_mem["player_memories"][player_name]["reflections"]
        }
        for r in data.get("reflections", []):
            if r.get("game_number") not in existing_game_numbers:
                canonical_mem["player_memories"][player_name]["reflections"].append(r)
                existing_game_numbers.add(r.get("game_number"))
                added += 1

    if added:
        with open(MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump(canonical_mem, f, indent=2)

    print(f"  Merged {added} new reflection(s) into {MEMORY_PATH}")
    return added


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python merge_new_games.py <tournament_dir>")
        sys.exit(1)
    tdir = sys.argv[1]
    print(f"Merging {tdir} into canonical dataset files...")
    merge_games(tdir)
    merge_memories(tdir)