# Component 3 Experiment — Summary (as of 2026-09-28)

**For:** whoever picks this up next (teammate or continued session).
**Owner:** Anushka, Component 3 (feedback generation + system-prompt injection).

## The question

Does injecting coaching notes (Component 3) into an agent's next-game system prompt actually improve how well its messages serve its team, compared to games with no coaching at all?

## The experiment

- **Control**: 65 games, no coaching, no Components 1/2/3 running live during play (mirrors how the original GPT-5.1 dataset was built — played first, scored after).
- **Treatment**: 60 games, same model (`llama-3.1-8b-instant`), full pipeline including coaching injection.
- Both scored after the fact by the same, unmodified Component 1 (LLM judge → `baseline_credit`) and Component 2 (causal graph) code.
- `baseline_credit` is the metric that's actually fair to compare between the two groups — it's scored per-message, independent of anything that happens later. (`causal_score`/`final_credit` are **not** fair here — they measure a message's effect on *later* messages, so they mechanically penalize whichever games happen to be most recent. All 60 treatment games are the newest data in the whole dataset, so that metric structurally disadvantages them regardless of coaching. Confirmed by plotting it across the full timeline — it declines toward the recent end, control included.)

## The result

| | Control (n=65 scored) | Treatment (n=60) |
|---|---|---|
| Mean `baseline_credit` | 0.161 | 0.184 |

- Difference +0.023, 95% CI -0.011 to +0.057. Welch t=1.34, p=0.183 (episode level); paired per-agent t(4)=2.23, p=0.089 (n=5 agents). Neither is significant.
- Per agent: 4 of 5 higher under treatment; Charlie -0.007 (about flat).
- Good-team win rate 35.0% vs 30.3% (Fisher p=0.70); judge-free mission success 52.4% vs 50.4% of missions (per-game p=0.78).
- Power: at the observed effect (d about 0.24) about 270 episodes per condition would be needed for 80% power (about 130 for 50%).

**Honest conclusion**: we detect no effect. The interval includes zero and rules out an improvement much larger than +0.057 in baseline credit. The result is also sensitive to which control block is used (against the first 50 controls alone p=0.41; the later 15-episode control block scored lower than the earlier one with no coaching difference), the design is not randomized, only 42% of injected notes came from an episode on the same side as the receiving episode, and the stored judge scores agree only weakly with a second judge (r=0.055). Report it as a pilot with no detectable effect. Run `python compare_treatment_control.py` and `python validity_checks.py` to regenerate every number.

## What's in this repo you should look at

- `final_comparison.html` — the chart (open in any browser): overall means + per-agent breakdown.
- `graph_output_treatment.html` — an interactive causal-influence network graph (drag nodes, zoom, hover edges for details), scoped to the treatment set's top 200 causal edges by effect size. Uses Component 2's actual rendering code (`visualize_graph.py`), unmodified — scoping was done externally via `run_treatment_network_graph.py` since the unscoped graph (now ~17,000 edges) hangs any browser.
- `_control_ids.json` / `_treatment_ids.json` — the exact episode IDs used in the final comparison, if you want to re-run or extend the analysis.
- `score_control_batch.py` — the post-hoc Component 1+2 scoring driver used for all control games (never touches Component 3).

## Known infra gotcha (will hit this again)

The real Postgres database runs in a Docker container named **`pgvector`** that does **not auto-start**. If DB queries fail with `Connection refused` on port 5434: check Docker Desktop is actually running (the app window can be open while its engine is still down), then `docker start pgvector`. Verify you're on the right instance by checking row counts (`SELECT count(*) FROM messages`) — other Postgres services on this machine have similarly-named but much emptier databases.

## Open items, if anyone wants to push this further

1. Scale to ~270 episodes/group for 80% power at the observed effect (large time/compute commitment), ideally as parallel interleaved arms with a placebo-note arm. Note: the generating model (`llama-3.1-8b-instant`) is no longer offered by Groq, so a rerun needs a different model for every condition.
2. Two feedback rows are still missing for 2 treatment episodes (`avalon_20260802_182804`, `avalon_20260802_200515`) — cosmetic gap, doesn't affect the credit-score numbers above.
3. Substitute these final numbers into the paper's Component 3 results section (VI-C).
