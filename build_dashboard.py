"""
Combines the funnel, network graph, and analytics charts into one HTML page.
"""

import json
import plotly.graph_objects as go
import plotly.io as pio
from pyvis.network import Network
from graph_pipeline import load_graph_from_db
from graph_analytics import compute_pagerank, compute_phase_distribution, compute_agent_influence_ranking
from db import get_connection


def load_funnel():
    try:
        with open("last_funnel.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def funnel_chart_html(funnel):
    if not funnel:
        return "<p style='color:white'>No funnel data yet -- run orchestrator.py first.</p>"

    stages = ["gate1_candidates_found", "skipped_no_credit_data", "gate2_evaluated",
              "gate2_passed", "gate3_evaluated", "gate3_confirmed"]
    labels = ["Gate 1 Candidates", "Skipped (no data)", "Gate 2 Evaluated",
              "Gate 2 Passed", "Gate 3 Evaluated", "Gate 3 Confirmed"]
    values = [funnel.get(s, 0) for s in stages]

    fig = go.Figure(go.Bar(x=labels, y=values, marker_color="#F5A623"))
    fig.update_layout(title="Pipeline Funnel -- Candidates per Gate Stage",
                       yaxis_title="Count", template="plotly_dark", height=400)
    return pio.to_html(fig, include_plotlyjs="cdn", full_html=False)


def pagerank_chart_html():
    G = load_graph_from_db()
    pr = compute_pagerank(G)
    if not pr:
        return "<p style='color:white'>No graph data yet.</p>"
    sorted_items = sorted(pr.items(), key=lambda x: -x[1])
    labels = [f"msg {k}" for k, v in sorted_items]
    values = [v for k, v in sorted_items]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color="#4A90D9"))
    fig.update_layout(title="PageRank -- Most Influential Messages", template="plotly_dark", height=400)
    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def phase_chart_html():
    G = load_graph_from_db()
    phases = compute_phase_distribution(G)
    if not phases:
        return "<p style='color:white'>No graph data yet.</p>"
    fig = go.Figure(go.Pie(labels=list(phases.keys()), values=list(phases.values())))
    fig.update_layout(title="Confirmed Edges by Game Phase", template="plotly_dark", height=400)
    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def agent_chart_html():
    G = load_graph_from_db()
    ranking = compute_agent_influence_ranking(G)
    if not ranking:
        return "<p style='color:white'>No graph data yet.</p>"
    fig = go.Figure(go.Bar(x=list(ranking.keys()), y=list(ranking.values()), marker_color="#2ECC71"))
    fig.update_layout(title="Agent Influence Ranking", template="plotly_dark", height=400)
    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def network_graph_html():
    G = load_graph_from_db()
    net = Network(height="600px", width="100%", directed=True, bgcolor="#111111", font_color="white")
    net.barnes_hut()

    team_colors = {"good": "#4A90D9", "evil": "#D9534F", None: "#999999"}
    conn = get_connection()
    cur = conn.cursor()

    for node_id in G.nodes():
        cur.execute("SELECT sender_id, sender_team, content FROM messages WHERE id = %s", (node_id,))
        row = cur.fetchone()
        sender_id, sender_team, content = row if row else (None, None, None)
        short_content = (content[:40] + "...") if content and len(content) > 40 else content
        label = f"[{node_id}] {sender_id} ({sender_team}): {short_content}"
        color = team_colors.get(sender_team, "#999999")
        net.add_node(node_id, label=str(node_id), title=label, color=color)

    cur.close()
    conn.close()

    for source, target, data in G.edges(data=True):
        effect = data.get("effect", 0) or 0
        influence_type = data.get("influence_type", "")
        edge_color = "#2ECC71" if effect > 0 else "#E74C3C"
        net.add_edge(source, target, value=abs(effect) * 5, color=edge_color,
                     title=f"effect={effect:.3f}, type={influence_type}")

    return net.generate_html()


def build_dashboard(output_path="dashboard.html"):
    funnel = load_funnel()

    parts = [
        "<html><head><title>Causal Graph Pipeline Dashboard</title>",
        "<style>body{background:#111;color:white;font-family:sans-serif;margin:0;padding:20px;}"
        "h1{color:#F5A623;} .section{margin-bottom:40px;} iframe{border:none;width:100%;height:650px;}</style>",
        "</head><body>",
        "<h1>Causal Graph Pipeline Dashboard</h1>",

        "<div class='section'><h2>Pipeline Funnel</h2>", funnel_chart_html(funnel), "</div>",

        "<div class='section'><h2>Network Graph</h2>",
        "<iframe srcdoc='" + network_graph_html().replace("'", "&apos;") + "'></iframe></div>",

        "<div class='section'><h2>PageRank</h2>", pagerank_chart_html(), "</div>",
        "<div class='section'><h2>Phase Distribution</h2>", phase_chart_html(), "</div>",
        "<div class='section'><h2>Agent Influence</h2>", agent_chart_html(), "</div>",

        "</body></html>",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(parts))

    print(f"Dashboard written to {output_path} -- open it in your browser.")


if __name__ == "__main__":
    build_dashboard() 