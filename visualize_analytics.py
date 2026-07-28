"""
Renders your graph analytics as interactive Plotly charts, saved as
standalone HTML files -- open in any browser.
"""

import plotly.graph_objects as go
from graph_pipeline import load_graph_from_db
from graph_analytics import compute_pagerank, compute_phase_distribution, compute_agent_influence_ranking


def chart_pagerank(output_path="chart_pagerank.html"):
    G = load_graph_from_db()
    pr = compute_pagerank(G)
    if not pr:
        print("No data for PageRank chart yet.")
        return
    sorted_items = sorted(pr.items(), key=lambda x: -x[1])
    labels = [f"msg {k}" for k, v in sorted_items]
    values = [v for k, v in sorted_items]

    fig = go.Figure(go.Bar(x=labels, y=values, marker_color="#4A90D9"))
    fig.update_layout(title="PageRank -- Most Influential Messages",
                       xaxis_title="Message", yaxis_title="PageRank Score",
                       template="plotly_dark")
    fig.write_html(output_path)
    print(f"PageRank chart written to {output_path}")


def chart_phase_distribution(output_path="chart_phase_distribution.html"):
    G = load_graph_from_db()
    phases = compute_phase_distribution(G)
    if not phases:
        print("No data for phase distribution chart yet.")
        return
    fig = go.Figure(go.Pie(labels=list(phases.keys()), values=list(phases.values())))
    fig.update_layout(title="Confirmed Edges by Game Phase", template="plotly_dark")
    fig.write_html(output_path)
    print(f"Phase distribution chart written to {output_path}")


def chart_agent_influence(output_path="chart_agent_influence.html"):
    G = load_graph_from_db()
    ranking = compute_agent_influence_ranking(G)
    if not ranking:
        print("No data for agent influence chart yet.")
        return
    fig = go.Figure(go.Bar(x=list(ranking.keys()), y=list(ranking.values()), marker_color="#2ECC71"))
    fig.update_layout(title="Agent Influence Ranking (total effect size)",
                       xaxis_title="Agent", yaxis_title="Total Effect",
                       template="plotly_dark")
    fig.write_html(output_path)
    print(f"Agent influence chart written to {output_path}")


if __name__ == "__main__":
    chart_pagerank()
    chart_phase_distribution()
    chart_agent_influence() 