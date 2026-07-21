#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Does the head_facts pipeline attach triple provenance and hand it to Qwen?

For a policy question, prints each retrieved fact's (head, relation, tail, location,
raw span) and reconstructs the exact <evidence> block the Assembler sends to the
synthesis LLM, so we can see whether provenance (source span + file:line:char)
reaches the model.
"""
from __future__ import annotations
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
_KNOW = os.environ.get("KNOWLYTIX_SRC", os.path.expanduser("~/jupyterlab/GMS-knowlytix"))
if os.path.isdir(_KNOW):
    sys.path.insert(0, _KNOW)


def main() -> int:
    from agentlab.capstone.policy_rag import get_default_retriever
    from knowlytix.knowledge.rag.assemble import Assembler

    rag = get_default_retriever()
    for q in ["Can my social security number be sent over email?",
              "What is the overdraft fee?"]:
        ans = rag.pipe.query(q)
        print("=" * 78)
        print("Q:", q)
        print("decision:", ans.decision, " route:", ans.route)
        print("-- retrieved facts (with provenance) --")
        for f in ans.sources:
            print(f"  {f.head} | {f.relation} | {f.tail}")
            print(f"      location: {getattr(f, 'location', None)}")
            print(f"      raw     : {getattr(f, 'raw', None)!r}")
        print("-- exact <evidence> block handed to Qwen --")
        print(Assembler._context(q, ans.sources))
        print("-- synthesized answer --")
        print(" ", ans.answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
