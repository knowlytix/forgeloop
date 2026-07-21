"""Budgets and budget tracking.

A Budget caps tokens, wall-clock seconds, tool calls and dollars. A
BudgetTracker accumulates Consumption against a Budget and reports when
any axis is exhausted. Real agents fail on cost before they fail on
safety, so this is a first-class concern, not an afterthought.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Budget:
    """Per-axis upper bounds. None means unlimited on that axis."""

    tokens: int | None = None
    seconds: float | None = None
    tool_calls: int | None = None
    dollars: float | None = None


@dataclass
class Consumption:
    tokens: int = 0
    seconds: float = 0.0
    tool_calls: int = 0
    dollars: float = 0.0


class BudgetTracker:
    def __init__(self, budget: Budget) -> None:
        self._budget = budget
        self._cons = Consumption()
        self._start = time.time()

    def record_tokens(self, n: int) -> None:
        self._cons.tokens += n

    def record_tool_call(self, count: int = 1) -> None:
        self._cons.tool_calls += count

    def record_dollars(self, d: float) -> None:
        self._cons.dollars += d

    def elapsed(self) -> float:
        return time.time() - self._start

    def consumption(self) -> Consumption:
        return Consumption(
            tokens=self._cons.tokens,
            seconds=self.elapsed(),
            tool_calls=self._cons.tool_calls,
            dollars=self._cons.dollars,
        )

    def reason_exhausted(self) -> str | None:
        c = self.consumption()
        b = self._budget
        if b.tokens is not None and c.tokens >= b.tokens:
            return f"tokens exhausted ({c.tokens}/{b.tokens})"
        if b.seconds is not None and c.seconds >= b.seconds:
            return f"time exhausted ({c.seconds:.2f}/{b.seconds:.2f}s)"
        if b.tool_calls is not None and c.tool_calls >= b.tool_calls:
            return f"tool calls exhausted ({c.tool_calls}/{b.tool_calls})"
        if b.dollars is not None and c.dollars >= b.dollars:
            return f"dollars exhausted ({c.dollars:.4f}/{b.dollars:.4f})"
        return None

    def exhausted(self) -> bool:
        return self.reason_exhausted() is not None
