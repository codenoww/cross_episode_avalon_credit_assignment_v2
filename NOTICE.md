# Attribution

This repository builds on **Multi-Round Avalon with LLM Agents** by Suveen Ellawela:

- Code and data: https://github.com/SuveenE/multi-round-avalon-agents
- Paper: S. Ellawela, "Trust, Lies, and Long Memories: Emergent Social Dynamics and Reputation in Multi-Round Avalon with LLM Agents", arXiv:2604.20582, 2026. https://arxiv.org/abs/2604.20582

The game engine (`main.py`), the tournament runner this project extends (`multi_game_runner.py`), the rules and tournament documentation under `docs/`, the figures under `assets/`, and everything under `dataset/` come from that work. The original MIT licence is retained in `LICENSE` (Copyright (c) 2026 Suveen Ellawela).

If you use this repository or its dataset, please cite the paper above:

```bibtex
@misc{ellawela2026trustlieslongmemories,
      title={Trust, Lies, and Long Memories: Emergent Social Dynamics and Reputation in Multi-Round Avalon with LLM Agents},
      author={Suveen Ellawela},
      year={2026},
      eprint={2604.20582},
      archivePrefix={arXiv},
      primaryClass={cs.MA},
      url={https://arxiv.org/abs/2604.20582}
}
```

## What was added here

The credit-assignment pipeline and the treatment/control experiment were added on top of that base:

- Component 1: message-level credit scoring (`component1/`, `run_component1*.py`)
- Component 2: candidate-influence graph (`gate1_*.py`, `gate2.py`, `gate3.py`, `exposure_check.py`, `graph_pipeline.py`, `orchestrator.py`, `causal_graph.py`, `bootstrap_ci.py`, `compute_causal_score.py`)
- Component 3: feedback generation and injection (`feedback_generator.py`, `injection.py`, `templates/`)
- Experiment tooling and analysis (`multi_game_runner.py` control mode, `score_control_batch.py`, `compare_treatment_control.py`)

Games under `avalon_tournament_*/` are new data generated for this study with `llama-3.1-8b-instant`; they are not part of the original dataset.
