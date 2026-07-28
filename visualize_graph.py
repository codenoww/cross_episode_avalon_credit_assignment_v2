"""
Renders the confirmed-edges graph as an interactive HTML file using Pyvis.
Open the output file in any browser -- drag nodes, zoom, click edges for details.
"""

from pyvis.network import Network
from graph_pipeline import load_graph_from_db
from db import get_connection


def get_message_label(message_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT sender_id, sender_team, content FROM messages WHERE id = %s", (message_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row:
        return f"msg {message_id}"
    sender_id, sender_team, content = row
    short_content = (content[:40] + "...") if content and len(content) > 40 else content
    return f"[{message_id}] {sender_id} ({sender_team}): {short_content}"


def build_visualization(output_path="graph_output.html"):
    G = load_graph_from_db()

    net = Network(height="750px", width="100%", directed=True, bgcolor="#111111", font_color="white")
    net.barnes_hut()

    team_colors = {"good": "#4A90D9", "evil": "#D9534F", None: "#999999"}

    for node_id in G.nodes():
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT sender_team FROM messages WHERE id = %s", (node_id,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        team = row[0] if row else None
        color = team_colors.get(team, "#999999")
        net.add_node(node_id, label=str(node_id), title=get_message_label(node_id), color=color)

    for source, target, data in G.edges(data=True):
        effect = data.get("effect", 0) or 0
        influence_type = data.get("influence_type", "")
        edge_color = "#2ECC71" if effect > 0 else "#E74C3C"
        net.add_edge(source, target, value=abs(effect) * 5, color=edge_color,
                     title=f"effect={effect:.3f}, type={influence_type}")

    net.show_buttons(filter_=["physics"])
    net.write_html(output_path)
    print(f"Graph written to {output_path} -- open it in your browser.")


if __name__ == "__main__":
    build_visualization() 