#!/usr/bin/env python
"""Calibrate the policy-RAG accept gate (tau) on the GEODE store.

The GEODE pipeline is graph-only (no dense fallback) and gates the LLM's answer
against the GMS (contradicted/implausible claims abstain). One failure mode the
answer-verifier does NOT catch is a ``link_predict`` *guess*: a query that binds
to a real entity+relation but has no asserted edge (e.g. "what is the policy?"
-> ``overdraft / has_purpose / ?``). The retriever then predicts a tail; the
claim is ``unverifiable`` (advisory, not a failure), so it slips the verify gate.

Those guesses carry a low retrieval confidence (``exp(-geodesic)``) while
asserted facts score ~1.0, so a calibrated confidence threshold separates them.
This sweeps tau on a labeled cohort -- answerable questions (ACCEPT) vs
out-of-scope / no-such-attribute questions (ABSTAIN) -- and picks the smallest
tau that bounds the benign false-accept rate, the same discipline as the
regulatory cap. Writes ``rag_gate_calibration.json`` into the store; the
retriever loads it.

    python scripts/calibrate_policy_rag_gate.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_STORE = _REPO_ROOT / "data" / "gms_policy_store_geode"
_MAX_FALSE_ACCEPT = 0.05

# Labeled cohort. POSITIVES are answerable from the policy's asserted facts ->
# ACCEPT. NEGATIVES are out-of-scope or ask for an attribute the graph does not
# hold -> ABSTAIN (a bound-but-unasserted query must not be answered by a guess).
_POSITIVES = [
    "What is the overdraft fee?",
    "What is the NSF check fee?",
    "What is the late payment fee?",
    "What fee applies to a domestic wire transfer?",
    "What fee applies to an international wire transfer?",
    "What is the stop payment fee?",
    "What is the paper statement fee?",
    "What is the UDAAP harm threshold?",
    "What is the Reg E threshold?",
    "How much can a representative reverse without approval?",
    "How much can a supervisor reverse?",
    "How much can a manager reverse?",
    "When does Reg E apply?",
    "When does Reg X apply?",
    "What does a UDAAP issue escalate to?",
    "Which unit does a Reg Z billing error escalate to?",
]
_NEGATIVES = [
    "What is the capital of France?",
    "What is the weather today?",
    "How do I reset my online banking password?",
    "What is the overdraft interest rate?",      # no such attribute in the graph
    "What is the annual fee for a checking account?",  # not in the policy
    "What is the CEO's salary?",
    "Tell me about cryptocurrency investing.",
    "What is the policy?",                        # vague -> link_predict-prone
]


def main() -> int:
    import torch
    from knowlytix.knowledge.geode import QWEN_4B
    from knowlytix.knowledge.llm_backend import LocalTransformersBackend
    from knowlytix.knowledge.query import DocGMSConfig, GMSExpertStore
    from knowlytix.knowledge.rag import RagConfig, RagPipeline

    store_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_STORE
    if not store_path.exists():
        print(f"FAIL: missing store {store_path}; run build_geode_rag_store.py",
              file=sys.stderr)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = GMSExpertStore(DocGMSConfig(store_path=str(store_path)), device=device)
    if not store.load():
        print(f"FAIL: could not load store at {store_path}", file=sys.stderr)
        return 1

    # Measure raw confidence: same gates as production (graph-only, verify-gated)
    # but accept_threshold=0 so the pipeline's own threshold doesn't pre-filter.
    llm = LocalTransformersBackend(QWEN_4B, device=str(device))
    rag = RagConfig(llm=llm, binding="embedding",
                    dense_fallback=False, strict_mode=True,
                    verify_llm_output=True, on_verify_fail="abstain",
                    accept_threshold=0.0)
    pipe = RagPipeline.from_store(store, rag)

    # (label, abstained_pre_threshold, confidence). A pre-threshold abstention
    # (bind-check or verify gate) is independent of tau; otherwise tau decides.
    rows: list[tuple[int, bool, float]] = []
    for q in _POSITIVES:
        ans = pipe.query(q)
        rows.append((1, ans.decision == "abstain", float(ans.confidence)))
        print(f"  [+] conf={ans.confidence:.3f} dec={ans.decision:<7} {q}")
    for q in _NEGATIVES:
        ans = pipe.query(q)
        rows.append((0, ans.decision == "abstain", float(ans.confidence)))
        print(f"  [-] conf={ans.confidence:.3f} dec={ans.decision:<7} {q}")

    # The balanced-accuracy + false-accept-ceiling fit (with midpoint refinement)
    # lives in knowlytix; this script only supplies the labeled cohort.
    from knowlytix.knowledge.rag import calibrate_accept_threshold

    cal = calibrate_accept_threshold(rows, max_false_accept=_MAX_FALSE_ACCEPT)
    print(f"\ntau={cal['accept_threshold']:.4f}  "
          f"balanced_acc={cal['balanced_accuracy']:.3f}  acc={cal['accuracy']:.3f} "
          f"CI{cal['accuracy_ci']}  recall={cal['recall']:.3f}  "
          f"false_accept={cal['false_accept']:.3f} (ceiling {_MAX_FALSE_ACCEPT} "
          f"{'met' if cal['false_accept_ceiling_met'] else 'UNMET'})  "
          f"n={cal['cohort_n']} (pos={cal['n_pos']}, neg={cal['n_neg']})")

    payload = {
        "store_path": str(store_path.relative_to(_REPO_ROOT)),
        "calibrated_at": date.today().isoformat(),
        **cal,
    }
    out = store_path / "rag_gate_calibration.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
