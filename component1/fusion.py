"""fusion.py — combine the two signals into the final signed credit score.

Final = sign(semantic) * ( w1 * |semantic| + w2 * behavioral_delta )

Direction comes from the semantic judge; magnitude blends how strong the judge
felt (|semantic|) with how much the room actually moved (behavioral_delta).
No API calls — pure arithmetic over what's already in the database.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from component1.database import get_db_connection

W1 = 1.0   # weight on the semantic (meaning) signal
W2 = 0.0   # weight on the behavioral (vote-shift) signal


def sign(x):
    return (x > 0) - (x < 0)      # +1, 0, or -1


def fuse():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT message_id, semantic_score, behavioral_delta "
        "FROM messages WHERE semantic_score IS NOT NULL"
    )
    rows = cur.fetchall()

    updated = 0
    for message_id, semantic, behavioral in rows:
        behavioral = behavioral if behavioral is not None else 0.0
        final = sign(semantic) * (W1 * abs(semantic) + W2 * behavioral)
        cur.execute(
            "UPDATE messages SET baseline_credit = %s WHERE message_id = %s",
            (final, message_id),
        )
        updated += 1

    conn.commit()
    cur.close()
    conn.close()
    print(f"Final credit score written to {updated} messages.")


if __name__ == "__main__":
    fuse()