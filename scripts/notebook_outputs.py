#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Keep executed output out of tracked notebooks.

Committed notebook output bloats the repo, produces unreadable diffs (a one-line
code edit shows up as thousands of changed output lines), and re-renders stale
results on GitHub long after the code moved on. This script is the one tool for
both sides of that policy:

    python scripts/notebook_outputs.py --check    # CI gate: non-zero if any
                                                  # tracked notebook has output
    python scripts/notebook_outputs.py --strip    # clear it in place

"Output" means, per code cell, ``outputs`` and ``execution_count``, plus the
``execution`` timing block some frontends stash in cell metadata and the
notebook-level ``widgets`` state (which can be megabytes on its own).

Stdlib only, so the CI gate needs no pip install. Each notebook is rewritten in
its own serialization style (indent / ensure_ascii / trailing newline are
detected per file) so stripping does not smuggle in unrelated reformatting --
these notebooks came from several different tools and have no single canonical
encoding.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def tracked_notebooks() -> list[Path]:
    """Every .ipynb git knows about (untracked scratch notebooks are ignored)."""
    out = subprocess.check_output(["git", "ls-files", "*.ipynb"], text=True)
    return [Path(p) for p in out.split("\n") if p.strip()]


def _style(raw: bytes, obj: object) -> tuple[int | None, bool, str]:
    """Recover (indent, ensure_ascii, trailing newline) by round-tripping."""
    for indent in (1, 2, 4, None):
        for ensure_ascii in (True, False):
            for newline in ("\n", ""):
                candidate = json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii) + newline
                if candidate.encode("utf-8") == raw:
                    return indent, ensure_ascii, newline
    return 1, True, "\n"


def strip_notebook(obj: dict) -> bool:
    """Clear output from ``obj`` in place. Returns True if anything changed."""
    changed = False
    for cell in obj.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            cell["outputs"] = []
            changed = True
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True
        if cell.get("metadata", {}).pop("execution", None) is not None:
            changed = True
    if obj.get("metadata", {}).pop("widgets", None) is not None:
        changed = True
    return changed


def offending(obj: dict) -> bool:
    """True when a notebook still carries executed output."""
    for cell in obj.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs") or cell.get("execution_count") is not None:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="fail if output is committed")
    mode.add_argument("--strip", action="store_true", help="clear output in place")
    args = ap.parse_args()

    notebooks = tracked_notebooks()
    hits: list[Path] = []

    for path in notebooks:
        raw = path.read_bytes()
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            print(f"ERROR: {path}: invalid notebook JSON: {exc}", file=sys.stderr)
            return 2

        if args.check:
            if offending(obj):
                hits.append(path)
            continue

        indent, ensure_ascii, newline = _style(raw, obj)
        if strip_notebook(obj):
            path.write_text(
                json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii) + newline,
                encoding="utf-8",
            )
            hits.append(path)

    if args.check:
        if hits:
            print(
                f"ERROR: {len(hits)} of {len(notebooks)} tracked notebooks carry executed "
                f"output.\nRun `python scripts/notebook_outputs.py --strip` and commit.\n",
                file=sys.stderr,
            )
            for path in hits:
                print(f"  {path}", file=sys.stderr)
            return 1
        print(f"ok: {len(notebooks)} tracked notebooks carry no output")
        return 0

    print(f"stripped {len(hits)} of {len(notebooks)} tracked notebooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
