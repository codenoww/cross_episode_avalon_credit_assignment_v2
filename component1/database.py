"""database.py — creates the 4 core tables for Component 1."""
import os
import psycopg2

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "avalon_analytics")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "postgres")

def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST, 
        port=DB_PORT,
        database=DB_NAME, 
        user=DB_USER, 
        password=DB_PASS
    )


SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS games (
        game_id      VARCHAR(100) PRIMARY KEY,
        num_players  INT,
        winner       VARCHAR(10)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS players (
        game_id           VARCHAR(100) REFERENCES games(game_id) ON DELETE CASCADE,
        player_name       VARCHAR(100),
        role              VARCHAR(50),
        is_good           BOOLEAN,
        special_knowledge JSONB,
        PRIMARY KEY (game_id, player_name)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS messages (
        message_id         SERIAL PRIMARY KEY,
        game_id            VARCHAR(100) REFERENCES games(game_id) ON DELETE CASCADE,
        mission_number     INT,
        proposal_round     INT,
        global_turn_id     INT,
        sender_name        VARCHAR(100),
        phase              VARCHAR(30),
        message_text       TEXT,
        semantic_score     FLOAT,
        behavioral_delta   FLOAT,
        final_credit_score FLOAT,
        processed_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (game_id, global_turn_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS votes (
        game_id        VARCHAR(100) REFERENCES games(game_id) ON DELETE CASCADE,
        mission_number INT,
        proposal_id    INT,
        voter_name     VARCHAR(100),
        vote_action    VARCHAR(20),
        PRIMARY KEY (game_id, mission_number, proposal_id, voter_name)
    )
    """,
]


def initialize_database():
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        for ddl in SCHEMA:
            cur.execute(ddl)
        conn.commit()
        cur.close()
        print("Schema ready: games, players, messages, votes.")
    finally:
        conn.close()


if __name__ == "__main__":
    initialize_database()