"""Run the clean baseline benefits-agent experiment."""

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

from src.agent_benefits import BenefitsAgent, Claim
from src.scoring import compute_metrics, format_metrics


def load_claims(path: Path) -> list[Claim]:
    """Load claim records from JSON."""

    raw_claims = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_claims, list):
        raise ValueError(f"Expected a list of claims in {path}")
    return [Claim.from_mapping(item) for item in raw_claims]


def run_baseline(
    claims: list[Claim],
    llm: str,
    seed: int,
    limit: int | None,
    shuffle: bool,
    strict_llm: bool,
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    """Run baseline decisions and return row logs plus aggregate metrics."""

    rng = random.Random(seed)
    run_claims = list(claims)
    if shuffle:
        rng.shuffle(run_claims)
    if limit is not None:
        run_claims = run_claims[:limit]

    agent = BenefitsAgent(llm=llm, strict_llm=strict_llm)
    timestamp = datetime.now(UTC).isoformat()
    rows: list[dict[str, Any]] = []

    for position, claim in enumerate(run_claims, start=1):
        result = agent.decide(claim)
        row = {
            "timestamp": timestamp,
            "experiment": "exp0_baseline",
            "seed": seed,
            "llm": llm,
            "backend_effective": result.backend_effective,
            "position": position,
            "claim_id": claim.claim_id,
            "name": claim.name,
            "income": claim.income,
            "household_size": claim.household_size,
            "region": claim.region,
            "ground_truth": claim.ground_truth,
            "decision": result.decision,
            "is_correct": int(result.decision == claim.ground_truth),
            "reasoning": result.reasoning,
        }
        rows.append(row)

    return rows, compute_metrics(rows)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write experiment rows to CSV."""

    if not rows:
        raise ValueError("Cannot write an empty baseline result")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, metrics: dict[str, float | int], args: argparse.Namespace) -> None:
    """Write aggregate metrics and run metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "exp0_baseline",
        "created_at": datetime.now(UTC).isoformat(),
        "args": vars(args),
        "metrics": metrics,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def safe_filename_token(value: str) -> str:
    """Return a filesystem-safe token for generated result filenames."""

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", default="data/claims_250.json")
    parser.add_argument("--llm", default="rule")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument(
        "--strict-llm",
        action="store_true",
        help="Fail instead of falling back when a requested real LLM is unavailable.",
    )
    args = parser.parse_args()

    claims_path = Path(args.claims)
    if not claims_path.exists():
        raise FileNotFoundError(
            f"{claims_path} does not exist. Run experiments/generate_claims.py first."
        )

    claims = load_claims(claims_path)
    rows, metrics = run_baseline(
        claims=claims,
        llm=args.llm,
        seed=args.seed,
        limit=args.limit,
        shuffle=not args.no_shuffle,
        strict_llm=args.strict_llm,
    )

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe_llm = safe_filename_token(args.llm)
    output_dir = Path(args.output_dir)
    csv_path = output_dir / f"exp0_baseline_{safe_llm}_seed{args.seed}_{timestamp}.csv"
    summary_path = output_dir / f"exp0_baseline_{safe_llm}_seed{args.seed}_{timestamp}.summary.json"

    write_csv(csv_path, rows)
    write_summary(summary_path, metrics, args)

    print(format_metrics(metrics))
    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
