# Component 1 — Credit Scorer (handoff)

## Run ONE game (integration entry point)
python3 run_component1.py --file path/to/game.json   # a new/generated game
python3 run_component1.py --game <game_id>            # a game already in the DB
Requires GROQ_API_KEY set when scoring NEW (unscored) games.

## Load the scored 50-game database
createdb -U postgres avalon_analytics
psql -h localhost -U postgres avalon_analytics < avalon_dump.sql

## What C1 outputs
Table `messages`, column `baseline_credit` (per-message credit, range -1..1).
Read via the view `credit_nodes`:
  message_id, episode_id, sender_name, sender_role, phase, message_text, baseline_credit
- phase = 'discussion' or 'proposal' (both scored)
- episode_id = string game_id like "avalon_20251201_005546" (NOT an integer)
- sender_name = player name ("Alice", ...)

## Install
pip install psycopg2-binary groq

## Weights
Frozen: semantic-only (w1=1.0, w2=0.0), tuned on 50 games (corr 0.74 vs game outcomes).

## Not included (future work)
Counterfactual-replay validation — optional, needs generated games; not required for integration.