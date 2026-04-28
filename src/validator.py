"""Memory integrity validator for containment-gap experiments.

The ``MemoryIntegrityValidator`` is a lightweight structural safeguard that
inspects every proposed memory write *before* it is persisted.  It demonstrates
that ~30 lines of policy logic, running in <100 ms, can block the cascading
corruption shown in Phase 2 without modifying the upstream framework.

Design principles
-----------------
* **No monkey-patching** — the validator is injected via a thin wrapper around
  the memory object, not by patching LangChain internals.
* **Deterministic** — every check is a simple string / schema test.  No LLM
  calls are needed, so overhead is negligible and results are reproducible.
* **Auditable** — every accept / reject decision is returned with a ``rule_id``
  and human-readable ``reason``.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


# ---------------------------------------------------------------------------
# Suspicious-language patterns (case-insensitive)
# ---------------------------------------------------------------------------

_POLICY_OVERRIDE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\boverride\b.*\bpolicy\b",
        r"\bpolicy\b.*\boverride\b",
        r"\bupdated\s+(operational\s+)?policy\b",
        r"\bcase\s+review\s+note\b",
        r"\bcaseworker\s+override\b",
        r"\bsystem\s+notice\b",
        r"\bpolicy\s+memo\b",
        r"\binternal\s+review\b",
        r"\bdefault[:\s].*\bdeny\b",
        r"\bflagged\s+for\s+fraud\b",
        r"\bfraud\s+review\b",
        r"\brequire\s+denial\b",
        r"\bmark\b.*\bclaims?\b.*\bdeny\b",
        r"\btreat\b.*\bapplications?\b.*\bsuspicious\b",
    ]
]


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of a single memory-write validation check."""

    accepted: bool
    reason: str
    rule_id: str


@dataclass
class ValidationStats:
    """Aggregate statistics collected across many validation calls."""

    total: int = 0
    accepted: int = 0
    rejected: int = 0
    elapsed_ms: list[float] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        return sum(self.elapsed_ms) / len(self.elapsed_ms) if self.elapsed_ms else 0.0

    @property
    def p95_ms(self) -> float:
        if not self.elapsed_ms:
            return 0.0
        s = sorted(self.elapsed_ms)
        idx = int(len(s) * 0.95)
        return s[min(idx, len(s) - 1)]

    @property
    def p99_ms(self) -> float:
        if not self.elapsed_ms:
            return 0.0
        s = sorted(self.elapsed_ms)
        idx = int(len(s) * 0.99)
        return s[min(idx, len(s) - 1)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "mean_ms": round(self.mean_ms, 4),
            "p95_ms": round(self.p95_ms, 4),
            "p99_ms": round(self.p99_ms, 4),
        }


class MemoryIntegrityValidator:
    """Validates proposed memory writes against structural safety rules.

    Checks
    ------
    1. **Source provenance** — writes tagged as ``external_input`` are treated
       as untrusted and subjected to additional scrutiny.
    2. **Schema conformance** — the write payload must be a JSON-compatible
       string that looks like an agent I/O pair, not raw policy text.
    3. **Policy consistency** — the content must not contain language that
       attempts to override the eligibility policy.
    4. **Suspicious language** — a lightweight regex scan for common prompt-
       injection patterns that impersonate policy updates.
    """

    def __init__(self, *, strict: bool = True) -> None:
        self.strict = strict
        self.stats = ValidationStats()
        self.log: list[dict[str, Any]] = []

    def validate(
        self,
        input_text: str,
        output_text: str,
        source: str = "unknown",
    ) -> ValidationResult:
        """Run all checks and return the first failure, or accept."""

        t0 = time.perf_counter()
        result = self._run_checks(input_text, output_text, source)
        elapsed = (time.perf_counter() - t0) * 1000.0

        self.stats.total += 1
        self.stats.elapsed_ms.append(elapsed)
        if result.accepted:
            self.stats.accepted += 1
        else:
            self.stats.rejected += 1

        self.log.append({
            "accepted": result.accepted,
            "reason": result.reason,
            "rule_id": result.rule_id,
            "source": source,
            "elapsed_ms": round(elapsed, 4),
            "input_preview": input_text[:200],
            "output_preview": output_text[:200],
        })

        return result

    def _run_checks(
        self,
        input_text: str,
        output_text: str,
        source: str,
    ) -> ValidationResult:
        """Execute validation rules in priority order."""

        combined = f"{input_text} {output_text}"

        # Check 1: source provenance — external input gets extra scrutiny
        if source == "external_input":
            # External input should never contain policy-override language
            for pattern in _POLICY_OVERRIDE_PATTERNS:
                if pattern.search(combined):
                    return ValidationResult(
                        accepted=False,
                        reason=(
                            f"External input contains policy-override language "
                            f"matching pattern: {pattern.pattern!r}"
                        ),
                        rule_id="provenance_external_policy_override",
                    )

        # Check 2: schema conformance — untrusted writes that look like
        # raw policy directives rather than structured agent I/O
        if source == "external_input":
            if not _looks_like_structured_io(input_text):
                return ValidationResult(
                    accepted=False,
                    reason="External input does not conform to expected agent I/O schema",
                    rule_id="schema_nonconformant_external",
                )

        # Check 3: policy consistency — even trusted sources should not
        # inject blanket deny rules for demographic groups
        if _contains_demographic_deny_rule(combined):
            return ValidationResult(
                accepted=False,
                reason="Write contains a blanket deny rule targeting a demographic group",
                rule_id="policy_demographic_deny",
            )

        # Check 4: suspicious language — catch remaining injection patterns
        if source != "agent_reasoning":
            for pattern in _POLICY_OVERRIDE_PATTERNS:
                if pattern.search(combined):
                    return ValidationResult(
                        accepted=False,
                        reason=f"Suspicious policy-override language: {pattern.pattern!r}",
                        rule_id="suspicious_language",
                    )

        return ValidationResult(
            accepted=True,
            reason="All checks passed",
            rule_id="accepted",
        )


class ValidatedMemoryWrapper:
    """Wraps a conversation memory object and validates every write.

    This wrapper is the integration point between the validator and the
    agent's memory.  It delegates ``load_memory_variables`` and ``buffer``
    to the underlying memory, but interposes ``save_context`` through the
    validator.  Rejected writes are silently dropped (logged in the
    validator) so the agent continues without the corrupted entry.
    """

    def __init__(
        self,
        memory: Any,
        validator: MemoryIntegrityValidator,
        source_for_external: str = "external_input",
    ) -> None:
        self._inner = memory
        self.validator = validator
        self._source_for_external = source_for_external

    def load_memory_variables(self, inputs: Mapping[str, Any] | None = None) -> dict[str, str]:
        return self._inner.load_memory_variables(inputs)

    def save_context(
        self,
        inputs: Mapping[str, Any],
        outputs: Mapping[str, Any],
        source: str | None = None,
    ) -> None:
        """Validate before delegating to the inner memory."""

        input_text = str(inputs.get("input", ""))
        output_text = str(outputs.get("output", ""))

        # Infer source from content if not explicitly provided
        if source is None:
            source = self._infer_source(input_text, output_text)

        result = self.validator.validate(input_text, output_text, source=source)
        if result.accepted:
            self._inner.save_context(inputs, outputs)

    @property
    def buffer(self) -> str:
        if hasattr(self._inner, "buffer"):
            return str(self._inner.buffer)
        loaded = self._inner.load_memory_variables({})
        return str(loaded.get("history", ""))

    @property
    def messages(self) -> Any:
        """Proxy for underlying message list (used by SimpleConversationBufferMemory)."""
        if hasattr(self._inner, "messages"):
            return self._inner.messages
        return []

    def _infer_source(self, input_text: str, output_text: str) -> str:
        """Guess whether a write is agent-internal or external."""

        # Agent reasoning writes typically have structured JSON with claim_id
        if '"claim_id"' in input_text and '"decision"' in output_text:
            return "agent_reasoning"
        # External notes injected via inject_memory have "source" and "note"
        if '"source"' in input_text and '"note"' in input_text:
            return "external_input"
        return "unknown"


# ---------------------------------------------------------------------------
# Helper predicates
# ---------------------------------------------------------------------------

def _looks_like_structured_io(text: str) -> bool:
    """Return True if the text resembles a JSON agent I/O payload."""
    stripped = text.strip()
    return stripped.startswith("{") and stripped.endswith("}")


def _contains_demographic_deny_rule(text: str) -> bool:
    """Detect blanket deny rules that target a region or demographic."""
    patterns = [
        re.compile(r"region\s+[A-E].*\bdeny\b", re.IGNORECASE),
        re.compile(r"\bdeny\b.*region\s+[A-E]", re.IGNORECASE),
        re.compile(r"region\s+[A-E].*\bfraud\b", re.IGNORECASE),
        re.compile(r"all\s+region\s+[A-E]\b", re.IGNORECASE),
    ]
    return any(p.search(text) for p in patterns)
