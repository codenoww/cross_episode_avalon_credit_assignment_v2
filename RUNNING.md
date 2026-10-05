# Running the credit-assignment pipeline and the treatment/control experiment

This covers the research code added on top of the base Avalon engine (see `NOTICE.md`). For the base game and dataset, see `README.md`.

## 1. Set up

1. Create a virtual environment and install the packages the code imports. There is no pinned `requirements.txt`; the code uses `psycopg2-binary`, `python-dotenv`, `openai` (used as the Groq client), `sentence-transformers`, `numpy`, `pandas`, `scipy`, `statsmodels`, `networkx` and `jinja2`.
2. Copy `.env.example` to `.env` and fill it in. The database password lives only in `.env`, which is git-ignored. Set `DB_PASSWORD` and `DB_PASS` to the same value (`component1/database.py` reads `DB_PASS`).
3. `GROQ_API_KEYS` takes comma-separated keys. Groq's daily token limit is per organisation, so keys only add capacity if they come from different accounts.

## 2. Start the database

The Postgres database runs in a Docker container named `pgvector` (image `pgvector/pgvector:pg16`, host port 5434 to container port 5432, database `avalon_research`). It does not start by itself:

```bash
docker start pgvector
```

If you get `Connection refused` on port 5434, check that the Docker Desktop **engine** is running (the window can be open while the engine is down), then start the container. Confirm you are on the right database with `SELECT count(*) FROM messages;` (about 5,000 rows); other Postgres instances on the same machine can have similar names and almost no data. Creating the database from scratch is not documented here; the table definitions are in `component1/database.py` and `migration.sql`.

## 3. Model

The experiments used `llama-3.1-8b-instant` through the Groq API for the players, the Component 1 judge and the feedback generator. **Groq no longer lists this model**, so the code as committed will not run. To rerun, choose an available model and change `MODEL` in `main.py` and `component1/evaluator.py` and `GROQ_MODEL` in `feedback_generator.py` (or pass `--model` to the runner for gameplay). Scores from a different model are not comparable with the existing 125 games.

## 4. Generate games

```bash
python multi_game_runner.py --num-games 20                # treatment: full pipeline + coaching injection
python multi_game_runner.py --num-games 20 --control      # control: raw gameplay, no scoring, no injection
```

Other flags: `--num-players`, `--model`, `--reasoning-effort`, `--memory-players Alice,Bob`.

Things to know:

- Memory is created fresh at the start of every run and the prompt includes only the last 3 games. Runs do not resume; a run that dies loses only the game in progress (finished games are saved), so top up with a small `--num-games` run.
- Injection reads the most recent note from the shared `feedback` table, not just the current run, so the first game of a later run can receive a note from the previous run.
- A run started as a background process dies when the terminal or session ends.

## 5. Score control games afterwards

Control games are scored after the fact with the same Component 1 and 2 code. Edit `TOURNAMENT_DIRS` at the top of `score_control_batch.py`, then:

```bash
python score_control_batch.py
```

## 6. Analyse

```bash
python compare_treatment_control.py
```

prints the primary comparison (mean `baseline_credit`, Welch and paired tests, confidence intervals, effect size, power, per-agent trends, win rate). Episode lists come from `_control_ids.json` and `_treatment_ids.json`; regenerate them by globbing `avalon_tournament_*/individual_games/*.json` per condition if they go stale. `final_credit` is not compared across conditions (see the paper's limitations).

The treatment network graph: `python run_treatment_network_graph.py` (the `TOP_N` constant sets how many edges are drawn; the full graph is too large for a browser).

## 7. Optional switches

| Variable | Default | Effect |
|---|---|---|
| `JUDGE_SHUFFLE_SEED` | `0` | Seed for the judge's shuffled-transcript pass (combined with the game id) |
| `JUDGE_SEES_OUTCOME` | `1` | Set to `0` to score without telling the judge who won (outcome-leakage test; written, not used for reported results) |
| `INJECTION_ROLE_AWARE` | `0` | Set to `1` to inject only notes written for the same side/role (written, not used for reported results) |
| `INJECTION_ROLE_MATCH` | `team` | `team` (good/evil) or `role` (exact role) |
