# SPDX-License-Identifier: Apache-2.0
"""Novel-FACT probe set: store facts whose (head, relation) never appeared as a
training target. Tests the axis the main run deliberately matched -- the
geometric route binds any fact the graph holds without retraining, while the
fine-tuned parser can only emit vocabulary it saw in training. Correct behavior
here is to ANSWER (the facts are in the store); Route B is expected to fail or
abstain on relations it never learned (has_division/has_region/has_head/has_value).
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from _bootstrap import REPO_ROOT, use_branch_library  # noqa: E402

use_branch_library()

import torch  # noqa: E402
from knowlytix.core.config import GeometryConfig  # noqa: E402
from knowlytix.knowledge.config import DocGMSConfig  # noqa: E402
from knowlytix.knowledge.store import GMSExpertStore  # noqa: E402

STORE = os.environ.get("GMS_STORE",
                       os.path.join(REPO_ROOT, "data", "gms_annual_report_store"))
DATA = os.environ.get("NL2T_DATA", os.path.join(REPO_ROOT, "data", "nl2triple"))
EXCLUDE = {"in_section", "has_alias", "has_product"}

QTMPL = {
    "has_division": "Which division does {h} belong to?",
    "has_region": "Which region is {h} located in?",
    "has_head": "Who heads the {h} division?",
    "has_revenue": "What was {h}'s revenue?",
    "has_headcount": "How many people work in {h}?",
    "has_fy2024": "What was {h} in fiscal year 2024?",
    "has_fy2025": "What was {h} in fiscal year 2025?",
    "has_amount": "What is the value of {h}?",
}


def question_for(h, r):
    if r == "has_value":
        if h in ("ceo", "cfo", "auditor"):
            return f"Who is the {h}?"
        if "fiscal year" in h:
            return f"When is the {h}?"
        return f"What is the {h}?"
    return QTMPL.get(r, "What is the {rel} of {h}?").format(
        h=h, rel=r.replace("has_", "").replace("_", " "))


def main():
    md = json.load(open(os.path.join(STORE, "model_dims.json")))["geometry"]
    store = GMSExpertStore(DocGMSConfig(store_path=STORE, geometry=GeometryConfig(
        d_v=md["d_v"], d_u=md["d_u"], m=md["m"], d=md["d"])),
        device=torch.device("cpu"))
    assert store.load()

    train = [json.loads(l) for l in open(os.path.join(DATA, "train.jsonl"))]
    trained = {(h, r) for row in train for h, r, _ in row["triples"]}

    rows, seen = [], set()
    for h, r, t in store.triples:
        if r in EXCLUDE or (h, r) in trained or (h, r) in seen:
            continue
        seen.add((h, r))
        rows.append({"question": question_for(h, r), "triples": [[h, r, "?"]],
                     "attribute": r, "expected_answer": str(t)})

    out = os.path.join(DATA, "novel_facts.jsonl")
    with open(out, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    import collections
    print(f"wrote {out}: {len(rows)} novel facts")
    print("by relation:", dict(collections.Counter(r["attribute"] for r in rows)))
    for r in rows:
        print(f"  {r['triples'][0]} -> {r['expected_answer']:16} | {r['question']}")


if __name__ == "__main__":
    main()
