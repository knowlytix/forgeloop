"""Budgets and budget tracking.

A Budget caps tokens, wall-clock seconds, tool calls and dollars. A
BudgetTracker accumulates Consumption against a Budget and reports when
any axis is exhausted. Real agents fail on cost before they fail on
safety, so this is a first-class concern, not an afterthought.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class Budget:
    """Per-axis upper bounds. None means unlimited on that axis.

    Attributes:
        tokens: Maximum tokens allowed, or None for unlimited.
        seconds: Maximum wall-clock seconds allowed, or None for unlimited.
        tool_calls: Maximum number of tool calls allowed, or None for unlimited.
        dollars: Maximum dollar cost allowed, or None for unlimited.
    """

    tokens: int | None = None
    seconds: float | None = None
    tool_calls: int | None = None
    dollars: float | None = None


@dataclass
class Consumption:
    """Accumulated resource usage across the budget axes.

    Attributes:
        tokens: Tokens consumed so far.
        seconds: Wall-clock seconds elapsed.
        tool_calls: Number of tool calls made.
        dollars: Dollar cost accrued.
    """

    tokens: int = 0
    seconds: float = 0.0
    tool_calls: int = 0
    dollars: float = 0.0


class BudgetTracker:
    """Accumulates consumption against a Budget and reports when an axis is exhausted."""

    def __init__(self, budget: Budget) -> None:
        self._budget = budget
        self._cons = Consumption()
        self._start = time.time()

    def record_tokens(self, n: int) -> None:
        """Add n to the token count."""
        self._cons.tokens += n

    def record_tool_call(self, count: int = 1) -> None:
        """Add count to the tool-call count (default 1)."""
        self._cons.tool_calls += count

    def record_dollars(self, d: float) -> None:
        """Add d to the accrued dollar cost."""
        self._cons.dollars += d

    def elapsed(self) -> float:
        """Return wall-clock seconds since the tracker was created."""
        return time.time() - self._start

    def consumption(self) -> Consumption:
        """Return a snapshot of current consumption, with seconds set to elapsed time."""
        return Consumption(
            tokens=self._cons.tokens,
            seconds=self.elapsed(),
            tool_calls=self._cons.tool_calls,
            dollars=self._cons.dollars,
        )

    def reason_exhausted(self) -> str | None:
        """Return a message for the first exhausted axis, or None if within budget.

        Returns:
            A human-readable string naming the first axis whose consumption
            reached or exceeded its budget, or None when no axis is exhausted.
        """
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
        """Return True if any budget axis has been reached or exceeded."""
        return self.reason_exhausted() is not None
