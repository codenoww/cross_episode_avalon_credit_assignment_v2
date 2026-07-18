"""
Graph analytics -- reads confirmed edges from the graph and computes the
research-facing metrics: PageRank, longest chain, phase distribution,
agent influence ranking.
"""

import networkx as nx
from db import get_connection
from graph_pipeline import load_graph_from_db


def compute_pagerank(G):
    """
    Which messages are the most influential hubs.
    PageRank requires non-negative weights (it models a flow/probability
    process). Real effects can be negative (a message that hurt the
    outcome), which breaks convergence at larger scale. Uses abs(effect)
    as weight -- measures influence STRENGTH, not direction; direction is
    already visible elsewhere (edge color, raw effect value).
    """
    if G.number_of_nodes() == 0:
        return {}

    G_weighted = G.copy()
    for u, v, data in G_weighted.edges(data=True):
        data["abs_effect"] = abs(data.get("effect", 0) or 0)

    return nx.pagerank(G_weighted, weight="abs_effect", max_iter=1000)


def compute_longest_chain(G):
    """Deepest multi-hop propagation chain found in the graph."""
    if G.number_of_nodes() == 0:
        return []
    if not nx.is_directed_acyclic_graph(G):
        return None
    return nx.dag_longest_path(G)


def get_message_phase_and_sender(message_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT phase, sender_id, sender_team FROM messages WHERE id = %s", (message_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row


def compute_phase_distribution(G):
    """Which game phase produced the most propagation (edges)."""
    phase_counts = {}
    for source, target in G.edges():
        phase, _, _ = get_message_phase_and_sender(source)
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
    return phase_counts


def compute_agent_influence_ranking(G):
    """Ranks agents by total outgoing influence (sum of effect sizes of edges they originated)."""
    agent_totals = {}
    for source, target, data in G.edges(data=True):
        _, sender_id, _ = get_message_phase_and_sender(source)
        effect = data.get("effect", 0) or 0
        agent_totals[sender_id] = agent_totals.get(sender_id, 0) + effect
    return dict(sorted(agent_totals.items(), key=lambda x: -x[1]))


def run_all_analytics():
    G = load_graph_from_db()
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

    print("=== PageRank ===")
    pr = compute_pagerank(G)
    for msg_id, score in sorted(pr.items(), key=lambda x: -x[1]):
        print(f"  message {msg_id}: {score:.4f}")

    print("\n=== Longest chain ===")
    chain = compute_longest_chain(G)
    print(f"  {chain}")

    print("\n=== Phase distribution ===")
    phases = compute_phase_distribution(G)
    for phase, count in phases.items():
        print(f"  {phase}: {count} edges")

    print("\n=== Agent influence ranking ===")
    ranking = compute_agent_influence_ranking(G)
    for agent, total_effect in ranking.items():
        print(f"  {agent}: {total_effect:.4f}")


if __name__ == "__main__":
    run_all_analytics() 