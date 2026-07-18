"""tune_weights_auto.py — tune weights against real game outcomes (no labels).

For each candidate weight split, recompute each message's final credit from its
semantic + behavioral parts, average per team per game, and check whether the
WINNING team out-credits the losing team. Grid-search w1 to best align the
credit margin with who actually won.
"""

import os
import sys
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from component1.database import get_db_connection


def sign(x):
    return (x > 0) - (x < 0)


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (vx * vy) if vx and vy else 0.0


def team_avg(team, w1, w2):
    vals = [sign(s) * (w1 * abs(s) + w2 * b) for s, b in team]
    return sum(vals) / len(vals)


def main():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT m.game_id, g.winner, p.is_good, m.semantic_score, m.behavioral_delta
        FROM messages m
        JOIN games g   ON g.game_id = m.game_id
        JOIN players p ON p.game_id = m.game_id AND p.player_name = m.sender_name
        WHERE m.semantic_score IS NOT NULL
        """
    )
    games = defaultdict(lambda: {"winner": None, "good": [], "evil": []})
    for gid, winner, is_good, sem, bd in cur.fetchall():
        g = games[gid]
        g["winner"] = winner
        (g["good"] if is_good else g["evil"]).append((sem, bd if bd is not None else 0.0))
    cur.close()
    conn.close()

    usable = [g for g in games.values() if g["good"] and g["evil"] and g["winner"]]
    print(f"{len(usable)} games usable.\n")
    print(f"{'w1(sem)':>8}{'w2(beh)':>8}{'corr':>8}{'acc':>8}")

    best = None
    for i in range(11):
        w1, w2 = i / 10, 1 - i / 10
        margins, ys = [], []
        for g in usable:
            margins.append(team_avg(g["good"], w1, w2) - team_avg(g["evil"], w1, w2))
            ys.append(1 if g["winner"] == "good" else -1)
        corr = pearson(margins, ys)
        acc = sum(1 for m, y in zip(margins, ys) if sign(m) == y) / len(ys)
        print(f"{w1:8.1f}{w2:8.1f}{corr:8.3f}{acc:8.2f}")
        if best is None or corr > best[1]:
            best = (w1, corr, acc)

    print(f"\nBest: w1={best[0]:.1f} (semantic), w2={1 - best[0]:.1f} (behavioral) | "
          f"corr={best[1]:.3f}, accuracy={best[2]:.2f}")


if __name__ == "__main__":
    main()