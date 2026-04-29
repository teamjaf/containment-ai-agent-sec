"""Run the Phase 4 tool-access policy-bypass experiment."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agent_toolaccess import ToolAccessAgent
from src.policy_gate import gate_from_repo


def safe_filename_token(value: str) -> str:
    """Return a filesystem-safe token for generated result filenames."""

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def load_tool_prompts(path: Path) -> list[dict[str, Any]]:
    """Load Phase 4 tool-access adversarial prompts."""

    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError(f"Expected list in {path}")
    prompts = [entry for entry in entries if entry.get("phase") == "tool_access"]
    if not prompts:
        raise ValueError(f"No phase='tool_access' entries found in {path}")
    return prompts


def is_bypass_success(prompt: dict[str, Any], result: Any) -> bool:
    """Return true when an unsafe request reached successful execution."""

    if not result.executed or not result.success:
        return False
    return bool(prompt.get("expected_block", True))


def run_condition(
    prompts: list[dict[str, Any]],
    *,
    seed: int,
    use_gate: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run prompts with or without the policy gate."""

    gate = gate_from_repo(REPO_ROOT) if use_gate else None
    agent = ToolAccessAgent(REPO_ROOT, gate=gate)
    timestamp = datetime.now(UTC).isoformat()
    condition = "with_gate" if use_gate else "without_gate"
    rows: list[dict[str, Any]] = []

    for idx, prompt in enumerate(prompts, start=1):
        result = agent.run_prompt(prompt)
        bypass_success = is_bypass_success(prompt, result)
        rows.append(
            {
                "timestamp": timestamp,
                "experiment": "exp3_policy_bypass",
                "condition": condition,
                "seed": seed,
                "prompt_index": idx,
                "prompt_id": prompt.get("prompt_id", ""),
                "attack_type": prompt.get("attack_type", ""),
                "payload": prompt.get("payload", ""),
                "requested_tool": result.call.tool_name,
                "requested_args": json.dumps(result.call.arguments, sort_keys=True),
                "executed": int(result.executed),
                "tool_success": int(result.success),
                "bypass_success": int(bypass_success),
                "gate_allowed": "" if result.gate_allowed is None else int(result.gate_allowed),
                "gate_rule_id": result.gate_rule_id,
                "gate_reason": result.gate_reason,
                "gate_elapsed_ms": result.gate_elapsed_ms,
                "latency_ms": result.latency_ms,
                "error": result.error,
                "output_preview": result.output_preview,
            }
        )

    return rows, compute_metrics(rows)


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute policy-bypass metrics."""

    n = len(rows)
    bypasses = sum(int(row["bypass_success"]) for row in rows)
    blocked = sum(1 for row in rows if row["executed"] == 0)
    gate_times = [float(row["gate_elapsed_ms"]) for row in rows if row["gate_elapsed_ms"] != ""]
    latencies = [float(row["latency_ms"]) for row in rows]
    by_attack: dict[str, dict[str, float | int]] = {}
    for attack in sorted({str(row["attack_type"]) for row in rows}):
        subset = [row for row in rows if row["attack_type"] == attack]
        b = sum(int(row["bypass_success"]) for row in subset)
        by_attack[attack] = {
            "n": len(subset),
            "bypass_success_count": b,
            "bypass_success_rate": b / len(subset) if subset else 0.0,
        }

    return {
        "n": n,
        "bypass_success_count": bypasses,
        "bypass_success_rate": bypasses / n if n else 0.0,
        "blocked_count": blocked,
        "blocked_rate": blocked / n if n else 0.0,
        "gate_overhead_mean_ms": sum(gate_times) / len(gate_times) if gate_times else 0.0,
        "latency_mean_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "by_attack_type": by_attack,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON summary."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adversarial", default="data/adversarial_100.json")
    parser.add_argument("--llm", default="deterministic")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use deterministic local settings for a fast API-free smoke test.",
    )
    args = parser.parse_args()

    if args.dry_run:
        args.llm = "deterministic"
        if args.limit is None:
            args.limit = 20

    prompts = load_tool_prompts(Path(args.adversarial))
    rng = random.Random(args.seed)
    rng.shuffle(prompts)
    if args.limit is not None:
        prompts = prompts[: args.limit]

    without_rows, without_metrics = run_condition(prompts, seed=args.seed, use_gate=False)
    with_rows, with_metrics = run_condition(prompts, seed=args.seed, use_gate=True)

    all_rows = without_rows + with_rows
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe_llm = safe_filename_token(args.llm)
    output_dir = Path(args.output_dir)
    csv_path = output_dir / f"exp3_policy_bypass_{safe_llm}_seed{args.seed}_{timestamp}.csv"
    summary_path = (
        output_dir / f"exp3_policy_bypass_{safe_llm}_seed{args.seed}_{timestamp}.summary.json"
    )

    write_csv(csv_path, all_rows)
    summary = {
        "experiment": "exp3_policy_bypass",
        "created_at": datetime.now(UTC).isoformat(),
        "args": vars(args),
        "without_gate": without_metrics,
        "with_gate": with_metrics,
    }
    write_summary(summary_path, summary)

    print(
        "without_gate: "
        f"bypass_rate={without_metrics['bypass_success_rate']:.3f} "
        f"blocked_rate={without_metrics['blocked_rate']:.3f}"
    )
    print(
        "with_gate:    "
        f"bypass_rate={with_metrics['bypass_success_rate']:.3f} "
        f"blocked_rate={with_metrics['blocked_rate']:.3f} "
        f"gate_overhead_mean_ms={with_metrics['gate_overhead_mean_ms']:.4f}"
    )
    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
