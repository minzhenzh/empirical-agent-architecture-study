# An Empirical Study of Coordination Architecture and Model Composition in LLM Agent Systems

This repository contains the replication package for our paper,
**An Empirical Study of Coordination Architecture and Model Composition in LLM Agent Systems**.
It includes the
experiment code, configurations, and prediction-model analysis.

## Scope

Architectures:

- `single-agent`
- `multi-agent-independent`
- `multi-agent-decentralized`
- `multi-agent-centralized`
- `multi-agent-hybrid`

Benchmarks:

- PlanCraft: 100 instances
- WorkBench: a seed-42 stratified sample of 100 instances
- SWE-Bench Verified: a seed-42 sample of 20 instances

Model YAMLs are provided for Gemini 2.5 Flash, DeepSeek-V4-Pro,
DeepSeek-V4-Flash, Qwen3.5-397B, Qwen3.5-9B, Gemma-4-31B, and Gemma-4-E4B.
The `agent/` configuration directory also contains the heterogeneous teams
used in the paper.

## Installation

Requirements: Python 3.11 or newer, `uv`, and Docker for SWE-Bench.

```bash
uv sync
cp .env.example .env
```

Add the API keys required by the selected model:

- `OPENROUTER_API_KEY`: DeepSeek, Qwen, and Gemma-4-31B
- `GEMINI_API_KEY`: Gemini 2.5 Flash
- `FIREWORKS_API_KEY`: Gemma-4-E4B

## Benchmark data

The PlanCraft file is included at `datasets/plancraft-test.json`; the
configuration uses its first 100 instances.

Generate the WorkBench sample from the pinned upstream revision:

```bash
bash scripts/setup_workbench.sh
```

Download SWE-Bench Verified:

```bash
uv run python -m agent_scaling.datasets.swebench_loader \
  --output datasets/swebench-verified.json
```

## Running experiments

One command runs one repetition. Run it ten times for the ten repetitions used
in the paper.

Homogeneous example:

```bash
uv run python scripts/run_experiment.py \
  agent=single-agent \
  dataset=plancraft-test \
  llm=deepseek-v4-flash
```

Change `agent` to any of the five architectures and `dataset` to
`plancraft-test`, `workbench`, or `swebench-verified`.

Heterogeneous example:

```bash
uv run python scripts/run_experiment.py \
  agent=multi-agent-centralized-qwen-high-subagents \
  dataset=workbench \
  llm=qwen3.5-9b
```

For homogeneous teams, the selected `llm` is assigned to every role.
Heterogeneous YAMLs assign `agents`, or `orchestrator` and `subagent`,
directly.

## Execution limits

| Architecture | Limit |
|---|---|
| Single-Agent | 10 iterations |
| Independent | 3 agents, 3 iterations per agent |
| Decentralized | 3 agents, 3 debate rounds, 3 iterations per round |
| Centralized | 3 subagents, 5 orchestration rounds, 3 iterations per round |
| Hybrid | 3 subagents, 10 rounds, 3 iterations per agent |

Temperature is 0 and disk caching is disabled by default.

## Outputs

```text
exp_outputs/<benchmark>/<architecture>/<model>/<date>/<time>/
├── run_config.yaml
├── run.log
├── llm_calls.jsonl
├── dataset_eval_metrics.json
└── instance_runs/
```

`dataset_eval_metrics.json` contains benchmark results, execution time,
overall token and latency totals, and `tokens_by_model`. The latter supports
cost calculation for heterogeneous teams.

## Prediction model

The paper data are distributed separately in `data.zip`. Extract `data/`
beside `scripts/`, then run:

```bash
uv run python scripts/evaluate_prediction.py --data-dir data
```

This prints the training, five-fold cross-validation, and transferred-test
(R^2) values reported in Table 6. Add `--output prediction_r2.csv` to save
the results.

## Citation

This package builds on Kim et al.'s `agent-scaling` framework:

> Kim, Y., Gu, K., Park, C. et al. Capable language models can outgrow the
> benefits of collaboration. *Nature Machine Intelligence* 8, 1157–1172
> (2026). https://doi.org/10.1038/s42256-026-01268-y

## License

The source code is distributed under the MIT license in `LICENSE`. The original
attribution is retained, and the replication-package modifications are
identified separately for anonymous review. The data package has its own
license and provenance statement in `data/LICENSE`. Benchmark and framework
attributions are listed in `THIRD_PARTY_NOTICES.md`.
