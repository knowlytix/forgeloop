"""Diff code blocks across notebook and book chapters.

Usage:
    python scripts/check_consistency.py

For each chapter we extract code cells from the .ipynb and lstlisting blocks
from the corresponding .tex, then report exact mismatches. Intended as a
heads-up signal; the goal is not byte-identity but no semantic drift.
"""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
BOOK_DIR = ROOT / "book"


PAIRS: list[tuple[str, str]] = [
    ("01_what_is_an_agent",          "01_what_is_an_agent"),
    ("02_minimal_agent_loop",        "02_minimal_agent_loop"),
    ("03_reasoning_traces",          "03_reasoning_traces"),
    ("04_tasks_state_actions",       "04_tasks_state_actions"),
    ("05_tools_as_typed_actions",    "05_tools_as_typed_actions"),
    ("06_safe_tool_execution",       "06_safe_tool_execution"),
    ("07_cost_latency_budgets",      "07_cost_latency_budgets"),
    ("08_planning",                  "08_planning"),
    ("09_memory",                    "09_memory"),
    ("10_trajectory_evaluation",     "10_trajectory_evaluation"),
    ("11_failure_modes_doe",         "11_failure_modes_doe"),
    ("12_runtime_governance",        "12_runtime_governance"),
    ("13_escalation",                "13_escalation"),
    ("14_multi_agent",               "14_multi_agent"),
    ("15_capstone",                  "15_capstone"),
    ("16_gms_fortification",         "16_gms_fortification"),
    ("17_testing_agents",            "17_testing_agents"),
    ("appendix_A_frontier",          "appendix_A_frontier"),
    ("appendix_B_framework_comparison", "appendix_B_framework_comparison"),
]


def notebook_code(nb_path: Path) -> list[str]:
    nb = json.loads(nb_path.read_text())
    return [c["source"].rstrip() for c in nb["cells"] if c["cell_type"] == "code"]


_LST_RE = re.compile(r"\\begin\{lstlisting\}\s*\n(.*?)\n\\end\{lstlisting\}", re.DOTALL)


def tex_code(tex_path: Path) -> list[str]:
    return [m.strip("\n") for m in _LST_RE.findall(tex_path.read_text())]


def main() -> None:
    total_drift = 0
    for nb_slug, tex_slug in PAIRS:
        nb_blocks = notebook_code(NB_DIR / f"{nb_slug}.ipynb")
        tex_blocks = tex_code(BOOK_DIR / f"{tex_slug}.tex")
        matched = sum(1 for b in tex_blocks if b in nb_blocks)
        drift = len(tex_blocks) - matched
        total_drift += drift
        print(f"{tex_slug:<40} nb={len(nb_blocks):>2} tex={len(tex_blocks):>2} matched={matched:>2} drift={drift}")
        if drift > 0:
            for b in tex_blocks:
                if b not in nb_blocks:
                    best = difflib.get_close_matches(b, nb_blocks, n=1, cutoff=0.5)
                    if best:
                        diff = list(difflib.unified_diff(best[0].splitlines(), b.splitlines(), lineterm="", n=1))[:6]
                        print(f"    drift block — closest match diff:")
                        for line in diff:
                            print(f"      {line}")
                    else:
                        print(f"    drift block (no close match): {b[:60]!r}")
    print(f"\ntotal drift blocks: {total_drift}")


if __name__ == "__main__":
    main()
