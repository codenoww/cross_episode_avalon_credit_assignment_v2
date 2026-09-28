"""
run_treatment_network_graph.py -- runs Arundhati's real visualize_graph.py
(build_visualization) UNMODIFIED, but scoped to the treatment set's causal
structure instead of the entire historical dataset.

Why scoping is needed: her build_visualization() calls
graph_pipeline.load_graph_from_db() with no filter and does one DB
round-trip per node with pyvis barnes_hut() physics -- fine at the scale it
was built for, but the live dataset has grown to ~17k edges / ~4k nodes,
which hangs any browser (confirmed earlier at a much smaller ~4,700 edges).

Approach: monkey-patch graph_pipeline.load_graph_from_db (in THIS script
only, not her file) to return a pre-filtered subgraph -- edges strictly
within the treatment set, top-N by |effect| -- then call her real
build_visualization() exactly as written. Her rendering/coloring/pyvis
logic never changes; only the data it's handed is scoped down.
"""
import json
import os

from dotenv import load_dotenv

load_dotenv()

import graph_pipeline  # noqa: E402  (must import before monkeypatching)

TOP_N = 200

_real_load = graph_pipeline.load_graph_from_db


def _scoped_load(stable_only=False):
    import psycopg2

    G = _real_load(stable_only=stable_only)

    treatment_ids = json.load(open("_treatment_ids.json", encoding="utf-8"))
    conn = psycopg2.connect(
        host=os.environ.get("DB_HOST"), port=os.environ.get("DB_PORT"),
        dbname=os.environ.get("DB_NAME"), user=os.environ.get("DB_USER"),
        password=os.environ.get("DB_PASSWORD"),
    )
    cur = conn.cursor()
    cur.execute("SELECT id FROM messages WHERE episode_id = ANY(%s)", (treatment_ids,))
    treatment_msg_ids = {r[0] for r in cur.fetchall()}
    cur.close()
    conn.close()

    # Edges strictly within the treatment set (both ends), ranked by |effect|.
    scoped_edges = [
        (u, v, d) for u, v, d in G.edges(data=True)
        if u in treatment_msg_ids and v in treatment_msg_ids
    ]
    scoped_edges.sort(key=lambda e: abs(e[2].get("effect") or 0), reverse=True)
    top_edges = scoped_edges[:TOP_N]

    import networkx as nx
    G_scoped = nx.DiGraph()
    for u, v, d in top_edges:
        G_scoped.add_edge(u, v, **d)

    print(f"[scoped] full graph: {G.number_of_nodes()} nodes / {G.number_of_edges()} edges")
    print(f"[scoped] treatment-set top-{TOP_N}: {G_scoped.number_of_nodes()} nodes / {G_scoped.number_of_edges()} edges")
    return G_scoped


graph_pipeline.load_graph_from_db = _scoped_load

# Import AFTER the monkeypatch, since visualize_graph.py does
# "from graph_pipeline import load_graph_from_db" at its own import time --
# it must see the patched version.
import visualize_graph  # noqa: E402

if __name__ == "__main__":
    visualize_graph.build_visualization(output_path="graph_output_treatment.html")
