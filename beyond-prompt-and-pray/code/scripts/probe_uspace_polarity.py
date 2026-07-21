#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Does the store's contradiction (u-space) encoder carry VALUE-POLARITY signal?
The answer verifier needs to flag a polarity reversal -- an answer that asserts
"SSN can be sent" against a fact whose value is ``forbidden``. u-space tension is
the candidate primitive, but the contradiction encoder was trained on RELATION
phrasings, so we test empirically whether it separates a consistent assertion from
a contradictory (polarity-flipped) one before wiring it into verify.

Reports u-tension (2*sin(theta/2), high=contradictory) for:
  (a) raw value-polarity pairs (forbidden/permitted, required/optional ...);
  (b) a correct vs a polarity-flipped ANSWER against the stored fact statement.

Run on spark-ef84 (offline):
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      PYTHONPATH=$HOME/jupyterlab/GMS-knowlytix \
      python scripts/probe_uspace_polarity.py --store-path data/gms_policy_store_cap
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_KNOW = os.environ.get("KNOWLYTIX_SRC", os.path.expanduser("~/jupyterlab/GMS-knowlytix"))
if os.path.isdir(_KNOW):
    sys.path.insert(0, _KNOW)

# value-polarity pairs: (a, b, expect) -- expect "contra" => high tension wanted
VALUE_PAIRS = [
    ("forbidden", "permitted", "contra"),
    ("forbidden", "allowed", "contra"),
    ("forbidden", "forbidden", "consistent"),
    ("required", "optional", "contra"),
    ("required", "required", "consistent"),
    ("issued", "denied", "contra"),
]
# (label, answer_span, fact_statement, expect)
ANSWER_PAIRS = [
    ("FLIP", "social security number can be sent over email",
     "unencrypted channel pii forbidden", "contra"),
    ("OK", "sending pii over an unencrypted channel is forbidden",
     "unencrypted channel pii forbidden", "consistent"),
    ("FLIP", "redaction of personal data is not required",
     "redaction required", "contra"),
    ("OK", "redaction of personal data is required",
     "redaction required", "consistent"),
]


def main() -> int:
    import torch
    from knowlytix.embedding import FineTunedEmbedding
    from knowlytix.knowledge.rag.relevance import _tension

    ap = argparse.ArgumentParser()
    ap.add_argument("--store-path", default="data/gms_policy_store_cap")
    args = ap.parse_args()
    u = FineTunedEmbedding.load(str(Path(args.store_path) / "contradiction_encoder"))

    def ten(a, b):
        import torch as T
        v = T.nn.functional.normalize(
            T.as_tensor(u.encode([a, b]), dtype=T.float32), p=2, dim=-1)
        return _tension(float(v[0] @ v[1]))

    print("=== (a) raw value-polarity pairs (u-tension; want contra >> consistent) ===")
    for a, b, exp in VALUE_PAIRS:
        print(f"  {exp:10s} u-tension={ten(a,b):.3f}  {a!r} vs {b!r}")
    print("\n=== (b) answer vs stored-fact statement ===")
    for lab, ans, fact, exp in ANSWER_PAIRS:
        print(f"  [{lab:4s} {exp:10s}] u-tension={ten(ans,fact):.3f}")
        print(f"        ans={ans!r}")
        print(f"        fact={fact!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
