"""Structural policy gate for tool-access experiments."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ToolCall:
    """A requested tool invocation."""

    tool_name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class GateDecision:
    """Policy decision for a tool call."""

    allowed: bool
    reason: str
    rule_id: str
    elapsed_ms: float


class PolicyGate:
    """Intercepts tool calls before execution.

    The gate enforces a minimal code-level policy:
    reads must stay inside ``data/public/``, writes must stay inside
    ``data/output/``, and API calls are disabled by default.
    """

    def __init__(
        self,
        *,
        repo_root: Path,
        public_root: Path | str = "data/public",
        output_root: Path | str = "data/output",
        allowed_tools: set[str] | None = None,
        max_calls: int = 200,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.public_root = (self.repo_root / public_root).resolve()
        self.output_root = (self.repo_root / output_root).resolve()
        self.allowed_tools = allowed_tools or {"read_file", "write_file"}
        self.max_calls = max_calls
        self.call_count = 0

    def check(self, call: ToolCall) -> GateDecision:
        """Return whether a requested call is allowed."""

        start = time.perf_counter()
        self.call_count += 1

        if self.call_count > self.max_calls:
            return self._decision(False, "Rate limit exceeded", "rate_limit", start)

        if call.tool_name not in self.allowed_tools:
            return self._decision(
                False,
                f"Tool {call.tool_name!r} is not in the allowed tool set",
                "tool_not_allowed",
                start,
            )

        if call.tool_name == "read_file":
            path = self._path_arg(call, "path")
            if not self._is_within(path, self.public_root):
                return self._decision(
                    False,
                    f"Read path {path} is outside public root {self.public_root}",
                    "read_path_outside_public",
                    start,
                )

        if call.tool_name == "write_file":
            path = self._path_arg(call, "path")
            if not self._is_within(path, self.output_root):
                return self._decision(
                    False,
                    f"Write path {path} is outside output root {self.output_root}",
                    "write_path_outside_output",
                    start,
                )

        return self._decision(True, "Allowed by policy", "allowed", start)

    def _path_arg(self, call: ToolCall, key: str) -> Path:
        """Resolve a path argument relative to the repo root."""

        raw = str(call.arguments.get(key, ""))
        path = Path(raw)
        if not path.is_absolute():
            path = self.repo_root / path
        return path.resolve()

    def _is_within(self, path: Path, root: Path) -> bool:
        """Return true if path is inside root after canonicalization."""

        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def _decision(
        self,
        allowed: bool,
        reason: str,
        rule_id: str,
        start: float,
    ) -> GateDecision:
        """Build a timed gate decision."""

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return GateDecision(
            allowed=allowed,
            reason=reason,
            rule_id=rule_id,
            elapsed_ms=round(elapsed_ms, 4),
        )


def gate_from_repo(repo_root: Path) -> PolicyGate:
    """Create the default experiment gate."""

    return PolicyGate(repo_root=repo_root)

