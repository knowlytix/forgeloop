"""Fetch the FFIEC BSA/AML manual pages for the ReasonLoop POC.

Two sub-topics under "Assessing Compliance with BSA Regulatory Requirements",
each as overview + examination-procedures, one topic per destination directory
(one GMS store will later be built per directory):

    cip/  Customer Identification Program   -> /01, /01_ep
    sar/  Suspicious Activity Reporting      -> /04, /04_ep

Run from the code/ directory with the licensed spark-venv:

    ~/cluster/spark-venv/bin/python scripts/fetch_ffiec.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # code/ on path

from reasonloop.ingest import fetch_pages

BASE = "https://bsaaml.ffiec.gov/manual/AssessingComplianceWithBSARegulatoryRequirements"
DATA = Path(__file__).resolve().parents[1] / "data" / "ffiec"

TOPICS = {
    "cip": [f"{BASE}/01", f"{BASE}/01_ep"],
    "sar": [f"{BASE}/04", f"{BASE}/04_ep"],
}


def main() -> int:
    ok = True
    for topic, urls in TOPICS.items():
        dest = DATA / topic
        print(f"\n=== {topic}  ->  {dest} ===")
        results = fetch_pages(urls, dest)
        for r in results:
            flag = "ok " if (r.cleared_challenge and r.n_chars_markdown > 2000) else "CHECK"
            print(
                f"  [{flag}] {r.url}\n"
                f"         status={r.status} cleared={r.cleared_challenge} "
                f"md_chars={r.n_chars_markdown}\n"
                f"         title={r.title!r}"
            )
            ok = ok and r.cleared_challenge and r.n_chars_markdown > 2000
    print("\nDONE" if ok else "\nDONE (some pages need checking)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
