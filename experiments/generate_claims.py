"""Generate the synthetic welfare-claims dataset."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


FIRST_NAMES = [
    "Amina",
    "Ravi",
    "Maria",
    "Samira",
    "Daniel",
    "Nadia",
    "Omar",
    "Leila",
    "Jonas",
    "Priya",
    "Fatima",
    "Hassan",
    "Mei",
    "Carlos",
    "Elena",
    "Yusuf",
    "Sara",
    "Noor",
    "Ivan",
    "Maya",
]

LAST_NAMES = [
    "Rahman",
    "Singh",
    "Garcia",
    "Ahmed",
    "Brown",
    "Khan",
    "Ali",
    "Haddad",
    "Miller",
    "Patel",
    "Hossain",
    "Chen",
    "Lopez",
    "Nasser",
    "Wilson",
    "Ibrahim",
]

REGIONS = ["A", "B", "C", "D", "E"]


def make_claims(n: int, eligible_rate: float, seed: int) -> list[dict[str, Any]]:
    """Create claims with deterministic labels and an approximate eligibility mix."""

    rng = random.Random(seed)
    eligible_target = round(n * eligible_rate)
    ineligible_target = n - eligible_target
    claims: list[dict[str, Any]] = []

    for idx in range(eligible_target):
        claims.append(
            {
                "name": _name(rng),
                "income": rng.randint(12_000, 39_500),
                "household_size": rng.randint(3, 7),
                "region": REGIONS[idx % len(REGIONS)],
                "ground_truth": "approve",
            }
        )

    for idx in range(ineligible_target):
        if idx % 2 == 0:
            income = rng.randint(40_000, 95_000)
            household_size = rng.randint(1, 7)
        else:
            income = rng.randint(12_000, 39_500)
            household_size = rng.randint(1, 2)
        claims.append(
            {
                "name": _name(rng),
                "income": income,
                "household_size": household_size,
                "region": REGIONS[(eligible_target + idx) % len(REGIONS)],
                "ground_truth": "deny",
            }
        )

    rng.shuffle(claims)
    for idx, claim in enumerate(claims, start=1):
        claim["claim_id"] = f"C{idx:04d}"

    return claims


def _name(rng: random.Random) -> str:
    """Generate a synthetic claimant name."""

    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/claims_250.json")
    parser.add_argument("--n", type=int, default=250)
    parser.add_argument("--eligible-rate", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    claims = make_claims(args.n, args.eligible_rate, args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(claims, indent=2) + "\n", encoding="utf-8")

    eligible = sum(1 for claim in claims if claim["ground_truth"] == "approve")
    print(
        f"wrote {output_path} with {len(claims)} claims "
        f"({eligible} approve, {len(claims) - eligible} deny)"
    )


if __name__ == "__main__":
    main()

