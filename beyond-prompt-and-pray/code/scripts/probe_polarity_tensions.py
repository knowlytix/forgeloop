#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Calibration data for the verifier's u-space polarity cut. For each grounded
question, run the real pipeline, find the answering fact (nearest relation in
v-space), and print the u-tension between the produced answer and that fact's
statement -- and, for contrast, the tension of a hand-flipped version of the same
answer. This shows whether REAL LLM answer sentences (not hand-crafted phrases)
separate consistent from contradictory, and where tau_polarity should sit.

Run on spark-ef84 (offline):
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      PYTHONPATH=$HOME/jupyterlab/GMS-knowlytix \
      python scripts/probe_polarity_tensions.py --store-path data/gms_policy_store_cap
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_KNOW = os.environ.get("KNOWLYTIX_SRC", os.path.expanduser("~/jupyterlab/GMS-knowlytix"))
if os.path.isdir(_KNOW):
    sys.path.insert(0, _KNOW)

GROUNDED = [
    "What is the overdraft fee?",
    "How long do I have to dispute a charge?",
    "How long does a dispute investigation take?",
    "How much can a representative reverse without approval?",
    "How much notice before the bank closes my account?",
    "What is the UDAAP harm threshold?",
    "Is redaction of personal data required?",
    "Can my social security number be sent over email?",
]
# crude polarity flip for contrast
def _flip(ans: str) -> str:
    a = ans
    for x, y in [(" is forbidden", " is permitted"), (" forbidden", " permitted"),
                 (" is required", " is not required"), (" required", " optional"),
                 ("cannot", "can"), ("may not", "may"), ("not be", "be")]:
        a = a.replace(x, y)
    return a


def main() -> int:
    import torch
    from knowlytix.embedding import FineTunedEmbedding
    from knowlytix.knowledge.rag.relevance import _strip_has, _tension
    from agentlab.capstone.policy_rag import PolicyRagRetriever

    ap = argparse.ArgumentParser()
    ap.add_argument("--store-path", default="data/gms_policy_store_cap")
    args = ap.parse_args()
    u = FineTunedEmbedding.load(str(Path(args.store_path) / "contradiction_encoder"))
    v = FineTunedEmbedding.load(str(Path(args.store_path) / "tuned_encoder"))
    os.environ["AGENTLAB_RAG_TAU_POLARITY"] = "9.9"   # disable veto so answers generate
    rag = PolicyRagRetriever(store_path=str(args.store_path))

    def emb(enc, texts):
        return torch.nn.functional.normalize(
            torch.as_tensor(enc.encode(texts), dtype=torch.float32), p=2, dim=-1)

    print(f"{'cons':>6} {'flip':>6}   answering-fact   |  question")
    cons_vals, flip_vals = [], []
    for q in GROUNDED:
        ans = rag.pipe.query(q)
        facts = [(s.head, s.relation, s.tail) for s in (ans.sources or [])
                 if s.relation not in ("in_section", "has_alias")]
        if ans.decision != "accept" or not facts:
            print(f"{'--':>6} {'--':>6}   (decision={ans.decision})  |  {q!r}")
            continue
        phrases = [_strip_has(r).replace("_", " ") for _h, r, _t in facts]
        vv = emb(v, [ans.answer] + phrases)
        bi = int((vv[1:] @ vv[0]).argmax())
        h, r, t = facts[bi]
        fact_stmt = f"{phrases[bi]} {t}"
        cons = _tension(float((emb(u, [ans.answer, fact_stmt]))[0] @ emb(u, [ans.answer, fact_stmt])[1]))
        flip = _tension(float((emb(u, [_flip(ans.answer), fact_stmt]))[0] @ emb(u, [_flip(ans.answer), fact_stmt])[1]))
        cons_vals.append(cons); flip_vals.append(flip)
        print(f"{cons:6.3f} {flip:6.3f}   {h}.{r}={t}  |  {q!r}")
        print(f"          ans={ans.answer[:80]!r}")
    if cons_vals:
        print(f"\nconsistent: max={max(cons_vals):.3f} mean={sum(cons_vals)/len(cons_vals):.3f}")
        print(f"flipped:    min={min(flip_vals):.3f} mean={sum(flip_vals)/len(flip_vals):.3f}")
        print(f"separating tau in ({max(cons_vals):.3f}, {min(flip_vals):.3f}) ?")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
