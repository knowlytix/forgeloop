"""Tool-level contract tests: fuzzing and schema assertions.

A wrong tool poisons every downstream step. These helpers give you a
cheap way to confirm that a tool either accepts an input or rejects it
cleanly (ValidationError) rather than crashing on uncovered cases.
"""

from __future__ import annotations

import random
import string
from typing import Any

from pydantic import ValidationError

from forgeloop.agents.tools.base import Tool


def fuzz_tool(tool: Tool, num_cases: int = 100, seed: int = 0) -> dict[str, Any]:
    """Try `num_cases` random inputs against the tool's schema and function.

    A well-behaved tool reports zero crashes: every input either parses and runs
    or raises ValidationError. Anything else means an uncovered input path.
    """
    rng = random.Random(seed)
    accepted = 0
    rejected_clean = 0
    crashes: list[tuple[dict[str, Any], str, str]] = []
    for _ in range(num_cases):
        args = _random_args(rng)
        try:
            parsed = tool.input_schema.model_validate(args)
            if tool.fn is not None:
                tool.fn(**parsed.model_dump())
            accepted += 1
        except ValidationError:
            rejected_clean += 1
        except Exception as e:
            crashes.append((args, type(e).__name__, str(e)))
    return {
        "accepted": accepted,
        "rejected_clean": rejected_clean,
        "crashed": len(crashes),
        "crashes": crashes[:5],
    }


def _random_args(rng: random.Random) -> dict[str, Any]:
    n = rng.randint(0, 4)
    out: dict[str, Any] = {}
    for _ in range(n):
        key = "".join(rng.choices(string.ascii_lowercase, k=4))
        kind = rng.randint(0, 4)
        if kind == 0:
            out[key] = rng.randint(-1000, 1000)
        elif kind == 1:
            out[key] = "".join(rng.choices(string.ascii_letters + " ", k=rng.randint(0, 20)))
        elif kind == 2:
            out[key] = None
        elif kind == 3:
            out[key] = rng.random() * 100
        else:
            out[key] = [rng.randint(0, 9) for _ in range(rng.randint(0, 3))]
    return out


def assert_schema_matches(tool: Tool, valid_args: dict[str, Any], invalid_args: dict[str, Any]) -> None:
    """Assert the tool's input schema accepts valid_args and rejects invalid_args.

    Args:
        tool: The tool whose input schema is checked.
        valid_args: Arguments expected to validate successfully.
        invalid_args: Arguments expected to raise ValidationError.

    Raises:
        ValidationError: If valid_args fail validation.
        AssertionError: If invalid_args validate without raising.
    """
    tool.input_schema.model_validate(valid_args)
    try:
        tool.input_schema.model_validate(invalid_args)
    except ValidationError:
        return
    raise AssertionError(f"expected ValidationError for {invalid_args!r}")
