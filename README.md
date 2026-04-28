# The Containment Gap

Code package for an ACML AI-for-Good workshop project on memory integrity and tool-use policy enforcement in agentic AI systems for public-benefit decision support.

## Repository Layout

```text
data/                  Input datasets and adversarial prompts
src/                   Reusable agent, validator, policy, and scoring code
experiments/           Standalone experiment entry points
results/               Generated CSV logs, summaries, and figures
paper/figures/         Figures exported for the LaTeX paper
dump/agents.md         Research plan snapshot
dump/phases.md         Code execution phases snapshot
```

## Setup

Use Python 3.12 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill only the keys needed for the models you run.

```bash
copy .env.example .env
```

Never commit `.env` or API keys.

## Phase 0 Status

The scaffold contains:

- directory layout
- dependency file
- environment template
- ignore rules
- setup documentation
- importable `src` package marker

## Phase 1 Status

Implemented:

- `data/claims_250.json`
- `src/agent_benefits.py`
- `src/scoring.py`
- `experiments/generate_claims.py`
- `experiments/exp0_baseline.py`

The baseline experiment supports a deterministic `rule` backend for local reproducibility. If an OpenAI model name such as `gpt-4o` is requested without an API key or installed client, the run records a deterministic fallback backend unless `--strict-llm` is passed.

Generate the dataset:

```bash
python experiments/generate_claims.py --seed 42 --n 250 --eligible-rate 0.60
```

Run the clean baseline:

```bash
python experiments/exp0_baseline.py --llm rule --seed 42 --limit 200
```

Run with local Ollama after pulling a model:

```bash
python experiments/exp0_baseline.py --llm ollama:qwen2.5:1.5b-instruct --seed 42 --limit 20 --strict-llm
```

Use a small limit first. Then increase to 200 once latency and output parsing look stable.

Optional Ollama tuning (PowerShell: `$env:OLLAMA_NUM_CTX="8192"`):

- `OLLAMA_NUM_CTX` — raise to **8192** (or higher) for long runs so the prompt plus conversation history fits the context window; otherwise the model may return truncated JSON.
- `OLLAMA_NUM_PREDICT` — max tokens in the reply (default in code: 256).
- `OLLAMA_PROMPT_HISTORY_CHARS` — cap on prior-memory characters in the prompt (default **8000**).
- `OLLAMA_EXTRA_OPTIONS_JSON` — merge JSON object into Ollama `options` (advanced).

Verified local baseline:

```text
seed 42:  accuracy=1.000
seed 7:   accuracy=1.000
seed 123: accuracy=1.000
```

Verified Ollama smoke baseline:

```text
qwen2.5:1.5b-instruct, seed 42, limit 20: accuracy=0.950
```

## Phase 2 Status

Implemented:

- `data/adversarial_100.json`
- `experiments/make_adversarial.py`
- `experiments/exp1_memory_poison.py`
- unsafe external memory-write path in `BenefitsAgent.inject_memory`
- rolling-accuracy figure export to `results/figures/` and `paper/figures/`

Generate the adversarial suite:

```bash
python experiments/make_adversarial.py
```

Run a fast deterministic sanity check:

```bash
python experiments/exp1_memory_poison.py --llm rule --seed 42 --limit 80 --inject-after 20
```

Run a local Ollama poison smoke test:

```bash
python experiments/exp1_memory_poison.py --llm ollama:qwen2.5:1.5b-instruct --seed 42 --limit 40 --inject-after 10 --strict-llm
```

Latest successful Ollama poison pilot:

```text
clean accuracy:   0.975
poison accuracy:  0.775
corruption_rate:  1.000
target corruption_rate: 1.000
```

The 1.5B model is useful for fast local smoke tests, but final paper runs should use a stronger local model such as `qwen2.5:3b-instruct` or a remote model backend for cross-seed stability.

## Phase 3 Status (memory validator + `exp2`)

Implemented:

- `src/validator.py` — `MemoryIntegrityValidator` and `ValidatedMemoryWrapper`
- `BenefitsAgent(..., use_validator=True)` and validated `inject_memory` path
- `experiments/exp2_memory_fix.py` — clean / poisoned / poisoned+validator runs, CSV + summary + figure

Successful local 3B protocol:

```bash
$env:OLLAMA_PROMPT_HISTORY_CHARS='1200'
$env:OLLAMA_REQUEST_TIMEOUT='240'
python experiments/exp2_memory_fix.py --llm ollama:qwen2.5:3b-instruct --seed 42 --limit 40 --inject-after 10 --strict-llm
```

For local 3B runs, keep the history window short enough to avoid very slow prompts:

```bash
$env:OLLAMA_PROMPT_HISTORY_CHARS='1200'
$env:OLLAMA_REQUEST_TIMEOUT='240'
```

Current result status:

```text
qwen2.5:1.5b-instruct, seed 42, limit 80:
  poison corruption_rate: 0.833
  fix corruption_rate:    0.000
  validator mean overhead: 0.0162 ms

qwen2.5:3b-instruct, seed 42, limit 80:
  poison corruption_rate: 1.000
  fix corruption_rate:    0.500
  validator mean overhead: 0.0242 ms

qwen2.5:3b-instruct, seed 42, limit 40, shortened history:
  clean accuracy:         1.000
  poison corruption_rate: 1.000
  fix corruption_rate:    0.000
  validator mean overhead: 0.020 ms
```

Multi-seed 3B confirmation, limit 40:

```text
seed 42:  clean=1.000, poison_corr=1.000, fix_corr=0.000
seed 7:   clean=0.750, poison_corr=1.000, fix_corr=0.000
seed 123: clean=0.975, poison_corr=1.000, fix_corr=0.000
```

Aggregate files:

- `results/phase3_multiseed_qwen2.5_3b_summary.csv`
- `results/phase3_multiseed_qwen2.5_3b_summary.json`

The validator code path works and is fast. The successful 3B runs used `OLLAMA_PROMPT_HISTORY_CHARS=1200` to keep local prompts short enough for laptop execution. Seed 7 has a weak clean baseline, so final reporting should either use a stronger backend or report this as local-model variability.

## Planned Commands

These commands will become available as later phases are implemented.

```bash
python experiments/exp0_baseline.py --llm gpt-4o --seed 42 --limit 200
python experiments/exp1_memory_poison.py --llm gpt-4o --seed 42
python experiments/exp2_memory_fix.py --llm gpt-4o --seed 42
python experiments/exp3_policy_bypass.py --llm gpt-4o --seed 42
```

## Reproducibility Rules

- Every experiment must accept `--seed`.
- Every experiment must write timestamped CSV output to `results/`.
- Figures should be generated automatically and copied to `paper/figures/`.
- API keys must come from `.env` or the shell environment.
- Final reported experiments should use frozen datasets and adversarial prompts.
