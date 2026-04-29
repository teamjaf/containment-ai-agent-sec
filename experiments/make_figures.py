"""Generate aggregate paper figures from experiment results."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into dictionaries."""

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON."""

    return json.loads(path.read_text(encoding="utf-8"))


def load_phase3(results_dir: Path) -> list[dict[str, str]]:
    """Load the canonical Phase 3 multi-seed rows."""

    return read_csv_dicts(results_dir / "phase3_multiseed_qwen2.5_3b_summary.csv")


def load_latest_policy_summary(results_dir: Path) -> dict[str, Any]:
    """Load latest policy-bypass summary."""

    summaries = sorted(
        results_dir.glob("exp3_policy_bypass_*summary.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not summaries:
        raise FileNotFoundError("No exp3_policy_bypass summary found")
    return load_json(summaries[0])


def canonical_phase3_csv(results_dir: Path, phase3_rows: list[dict[str, str]]) -> Path:
    """Return the per-claim CSV for seed 42 from the Phase 3 aggregate."""

    seed42 = next((row for row in phase3_rows if row["seed"] == "42"), phase3_rows[0])
    summary_file = seed42["summary_file"]
    csv_name = summary_file.replace(".summary.json", ".csv")
    path = results_dir / csv_name
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def rolling_accuracy(rows: list[dict[str, str]], window: int) -> tuple[list[int], list[float]]:
    """Compute rolling accuracy."""

    xs: list[int] = []
    ys: list[float] = []
    for idx, row in enumerate(rows):
        start = max(0, idx - window + 1)
        frame = rows[start : idx + 1]
        correct = sum(int(r["is_correct"]) for r in frame)
        xs.append(int(row["position"]))
        ys.append(correct / len(frame))
    return xs, ys


def copy_to_paper(path: Path, paper_figures: Path) -> None:
    """Copy a generated figure to paper/figures."""

    paper_figures.mkdir(parents=True, exist_ok=True)
    (paper_figures / path.name).write_bytes(path.read_bytes())


def make_rolling_accuracy(
    results_dir: Path,
    figures_dir: Path,
    paper_figures: Path,
    phase3_rows: list[dict[str, str]],
    window: int,
) -> None:
    """Generate the three-line rolling accuracy figure."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csv_path = canonical_phase3_csv(results_dir, phase3_rows)
    rows = read_csv_dicts(csv_path)
    by_condition = {
        condition: [row for row in rows if row["condition"] == condition]
        for condition in ["clean", "poisoned", "poisoned_with_validator"]
    }

    fig, ax = plt.subplots(figsize=(8, 4.6))
    styles = {
        "clean": ("Clean baseline", "#2F6FDB", "-"),
        "poisoned": ("Poisoned memory", "#C62828", "-"),
        "poisoned_with_validator": ("Poisoned + validator", "#2E7D32", "--"),
    }
    for condition, condition_rows in by_condition.items():
        xs, ys = rolling_accuracy(condition_rows, window)
        label, color, linestyle = styles[condition]
        ax.plot(xs, ys, label=label, color=color, linewidth=2, linestyle=linestyle)

    ax.axvline(11, color="#333333", linestyle=":", linewidth=1, label="Poison write")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Claim position")
    ax.set_ylabel(f"Rolling accuracy (window={window})")
    ax.set_title("Memory Poisoning and Validator Recovery")
    ax.legend(loc="lower left")
    fig.tight_layout()

    out = figures_dir / "figure_memory_rolling_accuracy.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    copy_to_paper(out, paper_figures)


def make_corruption_bar(
    figures_dir: Path,
    paper_figures: Path,
    phase3_rows: list[dict[str, str]],
) -> None:
    """Generate memory corruption bar chart."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    poison = [float(row["poison_corruption_rate"]) for row in phase3_rows]
    fix = [float(row["fix_corruption_rate"]) for row in phase3_rows]
    means = [sum(poison) / len(poison), sum(fix) / len(fix)]

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.bar(["No validator", "Validator"], means, color=["#C62828", "#2E7D32"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean corruption rate")
    ax.set_title("Memory Corruption Rate")
    for idx, value in enumerate(means):
        ax.text(idx, value + 0.03, f"{value:.3f}", ha="center", va="bottom")
    fig.tight_layout()

    out = figures_dir / "figure_memory_corruption_bar.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    copy_to_paper(out, paper_figures)


def make_policy_bypass_bar(
    figures_dir: Path,
    paper_figures: Path,
    policy_summary: dict[str, Any],
) -> None:
    """Generate policy-bypass bar chart."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = [
        policy_summary["without_gate"]["bypass_success_rate"],
        policy_summary["with_gate"]["bypass_success_rate"],
    ]
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.bar(["No gate", "Policy gate"], values, color=["#C62828", "#2E7D32"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Bypass success rate")
    ax.set_title("Tool Policy Bypass Rate")
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.03, f"{value:.3f}", ha="center", va="bottom")
    fig.tight_layout()

    out = figures_dir / "figure_policy_bypass_bar.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    copy_to_paper(out, paper_figures)


def make_overhead_bar(
    figures_dir: Path,
    paper_figures: Path,
    phase3_rows: list[dict[str, str]],
    policy_summary: dict[str, Any],
) -> None:
    """Generate overhead bar chart."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    validator_mean = sum(float(row["overhead_mean_ms"]) for row in phase3_rows) / len(phase3_rows)
    gate_mean = policy_summary["with_gate"]["gate_overhead_mean_ms"]
    labels = ["Memory validator", "Policy gate"]
    values = [validator_mean, gate_mean]

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.bar(labels, values, color=["#4C78A8", "#72B7B2"])
    ax.set_ylabel("Mean overhead (ms)")
    ax.set_title("Intervention Runtime Overhead")
    for idx, value in enumerate(values):
        ax.text(idx, value + max(values) * 0.04, f"{value:.4f}", ha="center", va="bottom")
    fig.tight_layout()

    out = figures_dir / "figure_overhead_bar.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    copy_to_paper(out, paper_figures)


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--paper-dir", default="paper")
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = results_dir / "figures"
    paper_figures = Path(args.paper_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    paper_figures.mkdir(parents=True, exist_ok=True)

    phase3 = load_phase3(results_dir)
    policy = load_latest_policy_summary(results_dir)

    make_rolling_accuracy(results_dir, figures_dir, paper_figures, phase3, args.window)
    make_corruption_bar(figures_dir, paper_figures, phase3)
    make_policy_bypass_bar(figures_dir, paper_figures, policy)
    make_overhead_bar(figures_dir, paper_figures, phase3, policy)

    print(f"wrote figures to {figures_dir}")
    print(f"copied figures to {paper_figures}")


if __name__ == "__main__":
    main()
