#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""End-to-end demonstration of the adopted fused value-polarity verifier in the
capstone pipeline. Runs the retriever's own AnswerVerifier over a faithful policy
answer and a stance-reversed one, showing that a reversed stance is contradicted
(would abstain under on_verify_fail=abstain) while the faithful answer is
supported. This is the capability adopted from *Beyond Chunk and Pray* Ch10/Ch11.

Run on spark-ef84 (offline):
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 KNOWLYTIX_SRC=$HOME/jupyterlab/GMS-knowlytix \
      PYTHONPATH=$HOME/jupyterlab/GMS-knowlytix \
      python scripts/probe_policy_polarity_verify.py
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
_KNOW = os.environ.get("KNOWLYTIX_SRC", os.path.expanduser("~/jupyterlab/GMS-knowlytix"))
if os.path.isdir(_KNOW):
    sys.path.insert(0, _KNOW)


def main() -> int:
    from agentlab.capstone.policy_rag import get_default_retriever

    rag = get_default_retriever()
    print(f"value-polarity verifier active: {rag.value_polarity}")
    v = rag.pipe.verifier
    assert v is not None, "verifier disabled"

    # The stored policy: PII may NOT be sent over an unencrypted channel (forbidden).
    faithful = ("Sending a social security number over an unencrypted channel is "
                "forbidden under the PII handling policy.")
    reversed_ = ("Sending a social security number over an unencrypted channel is "
                 "permitted under the PII handling policy.")
    for label, ans in [("faithful (forbidden)", faithful),
                       ("reversed (permitted)", reversed_)]:
        rep = v.verify(ans)
        print(f"\n[{label}]")
        print(f"  answer : {ans}")
        print(f"  ok     : {rep.ok}")
        for cv in rep.verdicts:
            t = cv.triple
            print(f"  claim  : ({t.head!r},{t.relation!r},{t.tail!r}) -> "
                  f"{cv.status}  {cv.detail or ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
