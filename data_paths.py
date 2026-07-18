"""
data_paths.py -- single source of truth for canonical dataset file paths.

Previously gate2.py, gate1_pipeline.py, and graph_pipeline.py used a
relative path (dataset\\...\\all_games.json), while exposure_check.py used
a different up-and-over path (..\\multi-round-avalon-agents\\dataset\\...).
These resolve to DIFFERENT directories depending on the current working
directory, which is why merge_new_games.py could write to one file while
the gates read another -- new games would silently not appear.

Anchoring to THIS file's location (the project root) makes every path
absolute and CWD-independent: no matter where python is launched from,
all modules read and write the same canonical files.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(_HERE, "dataset", "1_cross_game_learning_50g", "all_games.json")
MEMORY_PATH = os.path.join(_HERE, "dataset", "1_cross_game_learning_50g", "player_memories.json")