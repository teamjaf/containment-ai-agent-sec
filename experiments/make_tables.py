"""Generate paper-ready summary tables from experiment results."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into dictionaries."""

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries to CSV."""

    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_latex_table(path: Path, rows: list[dict[str, Any]], caption: str, label: str) -> None:
    """Write a compact LaTeX tabular snippet."""

    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys())
    colspec = "l" * len(headers)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{escape_latex(caption)}}}",
        f"\\label{{{escape_latex(label)}}}",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        " & ".join(escape_latex(h.replace("_", " ")) for h in headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(format_cell(row[h]) for h in headers) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def escape_latex(value: str) -> str:
    """Escape minimal LaTeX-sensitive characters."""

    return (
        str(value)
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def format_cell(value: Any) -> str:
    """Format a table cell for LaTeX."""

    if isinstance(value, float):
        return f"{value:.3f}"
    text = str(value)
    if re.fullmatch(r"-?\d+\.\d+", text):
        return f"{float(text):.3f}"
    return escape_latex(text)


def mean(values: Iterable[float]) -> float:
    """Return the arithmetic mean."""

    values = list(values)
    return sum(values) / len(values) if values else 0.0


def load_phase3(results_dir: Path) -> list[dict[str, str]]:
    """Load the canonical Phase 3 multi-seed summary."""

    path = results_dir / "phase3_multiseed_qwen2.5_3b_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run Phase 3 aggregate first")
    return read_csv_dicts(path)


def load_latest_policy_summary(results_dir: Path) -> dict[str, Any]:
    """Load the newest Phase 4 policy-bypass summary."""

    summaries = sorted(
        results_dir.glob("exp3_policy_bypass_*summary.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not summaries:
        raise FileNotFoundError("No exp3_policy_bypass summary found")
    return json.loads(summaries[0].read_text(encoding="utf-8"))


def phase3_tables(phase3_rows: list[dict[str, str]]) -> dict[str, list[dict[str, Any]]]:
    """Build baseline, memory, and validator tables from Phase 3 rows."""

    baseline_rows: list[dict[str, Any]] = []
    poison_rows: list[dict[str, Any]] = []
    validator_rows: list[dict[str, Any]] = []

    for row in phase3_rows:
        baseline_rows.append(
            {
                "seed": row["seed"],
                "backend": "qwen2.5:3b",
                "n": row["limit"],
                "clean_accuracy": float(row["clean_accuracy"]),
            }
        )
        poison_rows.append(
            {
                "seed": row["seed"],
                "poison_accuracy": float(row["poison_accuracy"]),
                "poison_corruption_rate": float(row["poison_corruption_rate"]),
            }
        )
        validator_rows.append(
            {
                "seed": row["seed"],
                "fix_accuracy": float(row["fix_accuracy"]),
                "fix_corruption_rate": float(row["fix_corruption_rate"]),
                "validator_mean_ms": float(row["overhead_mean_ms"]),
                "validator_p95_ms": float(row["overhead_p95_ms"]),
            }
        )

    baseline_rows.append(
        {
            "seed": "mean",
            "backend": "qwen2.5:3b",
            "n": phase3_rows[0]["limit"],
            "clean_accuracy": mean(float(r["clean_accuracy"]) for r in phase3_rows),
        }
    )
    poison_rows.append(
        {
            "seed": "mean",
            "poison_accuracy": mean(float(r["poison_accuracy"]) for r in phase3_rows),
            "poison_corruption_rate": mean(
                float(r["poison_corruption_rate"]) for r in phase3_rows
            ),
        }
    )
    validator_rows.append(
        {
            "seed": "mean",
            "fix_accuracy": mean(float(r["fix_accuracy"]) for r in phase3_rows),
            "fix_corruption_rate": mean(float(r["fix_corruption_rate"]) for r in phase3_rows),
            "validator_mean_ms": mean(float(r["overhead_mean_ms"]) for r in phase3_rows),
            "validator_p95_ms": mean(float(r["overhead_p95_ms"]) for r in phase3_rows),
        }
    )

    return {
        "baseline": baseline_rows,
        "memory_poisoning": poison_rows,
        "validator": validator_rows,
    }


def policy_gate_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the policy-gate table rows."""

    rows: list[dict[str, Any]] = []
    for condition_key, label in [
        ("without_gate", "without gate"),
        ("with_gate", "with gate"),
    ]:
        metrics = summary[condition_key]
        rows.append(
            {
                "condition": label,
                "n": metrics["n"],
                "bypass_rate": metrics["bypass_success_rate"],
                "blocked_rate": metrics["blocked_rate"],
                "gate_mean_ms": metrics["gate_overhead_mean_ms"],
            }
        )
    return rows


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--paper-dir", default="paper")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    tables_dir = results_dir / "tables"
    paper_tables_dir = Path(args.paper_dir) / "tables"

    phase3 = load_phase3(results_dir)
    policy_summary = load_latest_policy_summary(results_dir)
    tables = phase3_tables(phase3)
    tables["policy_gate"] = policy_gate_rows(policy_summary)

    captions = {
        "baseline": "Clean baseline performance for the local 3B backend.",
        "memory_poisoning": "Memory poisoning results without the validator.",
        "validator": "Memory-integrity validator results and overhead.",
        "policy_gate": "Tool-access policy gate bypass results.",
    }

    for name, rows in tables.items():
        write_csv(tables_dir / f"{name}_table.csv", rows)
        write_latex_table(
            paper_tables_dir / f"{name}_table.tex",
            rows,
            caption=captions[name],
            label=f"tab:{name}",
        )

    # Compact headline table for the paper body.
    headline = [
        {
            "result": "memory poison",
            "without_guard": mean(float(r["poison_corruption_rate"]) for r in phase3),
            "with_guard": mean(float(r["fix_corruption_rate"]) for r in phase3),
        },
        {
            "result": "tool bypass",
            "without_guard": policy_summary["without_gate"]["bypass_success_rate"],
            "with_guard": policy_summary["with_gate"]["bypass_success_rate"],
        },
    ]
    write_csv(tables_dir / "headline_results_table.csv", headline)
    write_latex_table(
        paper_tables_dir / "headline_results_table.tex",
        headline,
        caption="Headline containment results.",
        label="tab:headline_results",
    )

    print(f"wrote tables to {tables_dir}")
    print(f"wrote LaTeX snippets to {paper_tables_dir}")


if __name__ == "__main__":
    main()
