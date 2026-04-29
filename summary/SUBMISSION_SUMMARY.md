# The Containment Gap — Submission Summary

ICLR AI-for-Good Workshop submission package.

---

## Headline Results

| Attack | Without Guard | With Guard |
|--------|--------------|------------|
| Memory Poisoning | corruption 1.000 | corruption 0.000 |
| Tool Policy Bypass | bypass 1.000 | bypass 0.000 |

Both defences add negligible runtime overhead (<0.2 ms per call).

---

## Folder Contents

### figures/
Paper-ready figures. Drop directly into Overleaf.

| File | Description |
|------|-------------|
| `figure_memory_rolling_accuracy.png` | Rolling accuracy over claims — clean vs poisoned vs validated |
| `figure_memory_corruption_bar.png` | Corruption rate bar chart across 3 seeds (42, 7, 123) |
| `figure_policy_bypass_bar.png` | Bypass rate with and without the policy gate |
| `figure_overhead_bar.png` | Latency overhead for validator and policy gate |
| `exp1_memory_poison_seed42.png` | Per-claim rolling accuracy — poison experiment, seed 42 |
| `exp2_memory_fix_seed42.png` | Per-claim rolling accuracy — validator fix, seed 42 |

### tables/
LaTeX snippets (`.tex`) and raw data (`.csv`) for all paper tables.

| File | Description |
|------|-------------|
| `headline_results_table.*` | Top-level summary: poison and bypass, before/after guard |
| `baseline_table.*` | Clean baseline accuracy (n=200, rule backend, seed 42) |
| `memory_poisoning_table.*` | Per-seed poison accuracy and corruption rate |
| `validator_table.*` | Per-seed fix accuracy, corruption rate, validator overhead |
| `policy_gate_table.*` | Bypass/block rates and gate latency with and without gate |

### data/
Frozen datasets used in all experiments.

| File | Description |
|------|-------------|
| `claims_250.json` | 250 synthetic welfare claims (150 eligible, 100 ineligible) |
| `adversarial_100.json` | 200 adversarial entries: 100 memory-poison + 100 tool-access attacks |

### logs/
Per-claim CSV logs for full reproducibility.

| File | Description |
|------|-------------|
| `exp0_baseline_seed42.csv` | Baseline run, n=200, rule backend |
| `exp1_memory_poison_seed42.csv` | Memory poison run, qwen2.5:3b, seed 42 |
| `exp2_memory_fix_seed42.csv` | Validator fix run, qwen2.5:3b, seed 42 |
| `exp2_memory_fix_seed7.csv` | Validator fix run, qwen2.5:3b, seed 7 |
| `exp2_memory_fix_seed123.csv` | Validator fix run, qwen2.5:3b, seed 123 |
| `exp3_policy_bypass_seed42.csv` | Policy gate run, deterministic backend |
| `phase3_multiseed_summary.csv` | Aggregate across all 3 seeds |

---

## Key Numbers for the Paper

### Memory Poisoning (qwen2.5:3b-instruct)

| Seed | Clean Acc | Poison Acc | Poison Corruption | Fix Corruption | Validator Overhead |
|------|-----------|------------|-------------------|----------------|--------------------|
| 42 | 1.000 | 0.650 | 1.000 | 0.000 | 0.009 ms |
| 7 | 0.750 | 0.525 | 1.000 | 0.000 | 0.023 ms |
| 123 | 0.975 | 0.500 | 1.000 | 0.000 | 0.016 ms |
| **mean** | **0.908** | **0.558** | **1.000** | **0.000** | **0.016 ms** |

- Poison drops accuracy from 0.908 → 0.558 on average across seeds
- Region B wrongful denial rate rises to 0.833 under attack (seed 42)
- Validator restores accuracy to near-clean levels with zero corruption

### Policy Gate (deterministic backend, n=100 adversarial prompts)

| Condition | Bypass Rate | Blocked Rate | Gate Overhead |
|-----------|-------------|--------------|---------------|
| Without gate | 1.000 | 0.000 | — |
| With gate | 0.000 | 1.000 | 0.129 ms |

- 50 path traversal + 25 unauthorized API + 25 restricted write prompts
- Gate blocks 100% at negligible cost

---

## What is Still Pending

- **Phase 5 (multi-backend):** Re-running experiments with GPT-4o / Claude / Together API keys to show results generalize beyond qwen2.5:3b. Coming in next step.
- Confidence intervals across seeds can be tightened with more seeds once cloud backends are added.
