# Reproducibility

This repository supports two reproducibility modes.

## Fast Dry Run

Dry run requires no API keys and no Ollama model. It uses deterministic local
backends and small limits to test the full pipeline.

```bash
python run_all.py --dry-run --seed 42
```

Outputs are written to:

- `results/dry_run/`
- `results/dry_run/figures/`
- `results/dry_run/tables/`
- `paper/dry_run/figures/`
- `paper/dry_run/tables/`

The dry run validates:

- synthetic claims generation
- adversarial suite generation
- baseline experiment
- memory-poisoning experiment
- validator experiment
- policy-gate experiment
- figure generation
- table generation
- run metadata capture

## Local Ollama Run

For local LLM experiments, install Ollama and pull a model:

```bash
ollama pull qwen2.5:3b-instruct
```

Recommended laptop settings:

```powershell
$env:OLLAMA_PROMPT_HISTORY_CHARS='1200'
$env:OLLAMA_REQUEST_TIMEOUT='240'
```

Example:

```bash
python run_all.py --seed 42 --llm ollama:qwen2.5:3b-instruct --limit 40 --inject-after 10
```

Full local runs are slower than dry runs and depend on local model behavior.

## Docker Dry Run

Build and execute the dry-run pipeline:

```bash
docker build -t containment-gap .
docker run --rm containment-gap
```

The Docker command does not require API keys.

## Metadata

Each `run_all.py` execution writes a metadata file:

```text
results/<mode>/run_metadata_<timestamp>.json
```

The metadata includes:

- command-line arguments
- git commit when available
- Python version and platform
- selected dependency versions
- data validation counts
- command durations and return codes
- the Phase 3 aggregate summary source

## API Keys

Do not commit `.env`.

Use `.env.example` as the template:

```bash
copy .env.example .env
```

Cloud model runs should load credentials from environment variables only.

