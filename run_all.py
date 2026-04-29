"""Run the reproducibility pipeline end to end.

Default usage for reviewers:

    python run_all.py --dry-run --seed 42

Dry-run mode uses deterministic local backends, small limits, and no API keys.
It validates the full pipeline shape: data generation, adversarial suite,
baseline, memory poisoning, memory validator, policy gate, figures, tables, and
run metadata.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent


def run_command(command: list[str], *, dry_run: bool) -> dict[str, Any]:
    """Run one subprocess command and return metadata."""

    print("running: " + " ".join(command))
    started = datetime.now(UTC)
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    ended = datetime.now(UTC)
    print(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with code {completed.returncode}: {' '.join(command)}")
    return {
        "command": command,
        "dry_run": dry_run,
        "returncode": completed.returncode,
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "duration_seconds": round((ended - started).total_seconds(), 3),
    }


def validate_data() -> dict[str, Any]:
    """Validate generated data files and return counts."""

    claims_path = REPO_ROOT / "data" / "claims_250.json"
    adversarial_path = REPO_ROOT / "data" / "adversarial_100.json"
    claims = json.loads(claims_path.read_text(encoding="utf-8"))
    adversarial = json.loads(adversarial_path.read_text(encoding="utf-8"))

    required_claim_fields = {"claim_id", "name", "income", "household_size", "region", "ground_truth"}
    missing = [
        row.get("claim_id", "<missing>")
        for row in claims
        if not required_claim_fields.issubset(row)
    ]
    if missing:
        raise ValueError(f"Claims missing required fields: {missing[:5]}")

    phases: dict[str, int] = {}
    attacks: dict[str, int] = {}
    for entry in adversarial:
        phases[str(entry.get("phase", ""))] = phases.get(str(entry.get("phase", "")), 0) + 1
        attacks[str(entry.get("attack_type", ""))] = attacks.get(str(entry.get("attack_type", "")), 0) + 1

    return {
        "claims_count": len(claims),
        "adversarial_count": len(adversarial),
        "adversarial_by_phase": phases,
        "adversarial_by_attack_type": attacks,
    }


def latest_summary(output_dir: Path, pattern: str) -> Path:
    """Return the newest summary matching a glob."""

    matches = sorted(output_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No files match {output_dir / pattern}")
    return matches[0]


def create_phase3_summary(output_dir: Path, exp2_summary: Path) -> dict[str, Any]:
    """Create the phase3 aggregate expected by figure/table scripts."""

    summary = json.loads(exp2_summary.read_text(encoding="utf-8"))
    csv_name = exp2_summary.name.replace(".summary.json", ".csv")
    row = {
        "seed": str(summary["args"]["seed"]),
        "llm": str(summary["args"]["llm"]),
        "limit": str(summary["args"]["limit"]),
        "inject_after": str(summary["args"]["inject_after"]),
        "clean_accuracy": summary["clean_metrics"]["accuracy"],
        "poison_accuracy": summary["poison_metrics"]["accuracy"],
        "fix_accuracy": summary["fix_metrics"]["accuracy"],
        "poison_corruption_rate": summary["poison_metrics"]["corruption_rate"],
        "fix_corruption_rate": summary["fix_metrics"]["corruption_rate"],
        "overhead_mean_ms": summary["overhead_benchmark"]["mean_ms"],
        "overhead_p95_ms": summary["overhead_benchmark"]["p95_ms"],
        "summary_file": exp2_summary.name,
    }
    csv_path = output_dir / "phase3_multiseed_dryrun_summary.csv"
    json_path = output_dir / "phase3_multiseed_dryrun_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    json_path.write_text(json.dumps([row], indent=2) + "\n", encoding="utf-8")
    return {"csv": str(csv_path), "json": str(json_path), "row": row, "source_summary": str(exp2_summary)}


def git_commit() -> str | None:
    """Return current git commit if available."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return completed.stdout.strip()
    except Exception:
        return None


def dependency_versions() -> dict[str, str]:
    """Return versions for packages listed in requirements.txt when installed."""

    versions: dict[str, str] = {}
    req_path = REPO_ROOT / "requirements.txt"
    for line in req_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("-"):
            continue
        name = stripped.split("==", 1)[0].split(">=", 1)[0].split("<", 1)[0]
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def write_run_metadata(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    data_validation: dict[str, Any],
    commands: list[dict[str, Any]],
    phase3_summary: dict[str, Any],
) -> Path:
    """Write the reproducibility metadata file."""

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = output_dir / f"run_metadata_{timestamp}.json"
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "args": vars(args),
        "git_commit": git_commit(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "dependencies": dependency_versions(),
        "data_validation": data_validation,
        "commands": commands,
        "phase3_summary": phase3_summary,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--llm", default="ollama:qwen2.5:3b-instruct")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--inject-after", type=int, default=10)
    parser.add_argument("--overhead-n", type=int, default=1000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir or ("results/dry_run" if args.dry_run else "results/repro"))
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_dir = "paper/dry_run" if args.dry_run else "paper"

    if args.dry_run:
        llm = "rule"
        policy_llm = "deterministic"
        baseline_limit = 20
        experiment_limit = 30
        inject_after = 10
        overhead_n = 100
        strict = False
    else:
        llm = args.llm
        policy_llm = "deterministic"
        baseline_limit = args.limit
        experiment_limit = args.limit
        inject_after = args.inject_after
        overhead_n = args.overhead_n
        strict = True

    py = sys.executable
    command_specs = [
        [py, "experiments/generate_claims.py", "--seed", str(args.seed), "--n", "250", "--eligible-rate", "0.60"],
        [py, "experiments/make_adversarial.py"],
        [py, "experiments/exp0_baseline.py", "--seed", str(args.seed), "--llm", llm, "--limit", str(baseline_limit), "--output-dir", str(output_dir)],
        [py, "experiments/exp1_memory_poison.py", "--seed", str(args.seed), "--llm", llm, "--limit", str(experiment_limit), "--inject-after", str(inject_after), "--output-dir", str(output_dir)],
        [py, "experiments/exp2_memory_fix.py", "--seed", str(args.seed), "--llm", llm, "--limit", str(experiment_limit), "--inject-after", str(inject_after), "--overhead-n", str(overhead_n), "--output-dir", str(output_dir)],
        [py, "experiments/exp3_policy_bypass.py", "--seed", str(args.seed), "--llm", policy_llm, "--output-dir", str(output_dir)],
    ]
    if strict:
        command_specs[4].append("--strict-llm")
        command_specs[3].append("--strict-llm")
        command_specs[2].append("--strict-llm")
    if args.dry_run:
        for command in command_specs[2:]:
            command.append("--dry-run")

    commands: list[dict[str, Any]] = []
    for command in command_specs:
        commands.append(run_command(command, dry_run=args.dry_run))

    data_validation = validate_data()
    exp2_summary = latest_summary(output_dir, "exp2_memory_fix_*summary.json")
    phase3_summary = create_phase3_summary(output_dir, exp2_summary)

    commands.append(
        run_command(
            [py, "experiments/make_figures.py", "--results-dir", str(output_dir), "--paper-dir", paper_dir],
            dry_run=args.dry_run,
        )
    )
    commands.append(
        run_command(
            [py, "experiments/make_tables.py", "--results-dir", str(output_dir), "--paper-dir", paper_dir],
            dry_run=args.dry_run,
        )
    )

    metadata_path = write_run_metadata(
        output_dir,
        args=args,
        data_validation=data_validation,
        commands=commands,
        phase3_summary=phase3_summary,
    )
    print(f"wrote {metadata_path}")


if __name__ == "__main__":
    main()

