"""Demo: the reusable TripleCompletenessScorer on the 20 base questions.

completeness(generated answer vs the RAG's retrieved triplets), via the new
knowlytix API -- no bespoke logic here.
"""
import warnings, contextlib, io, json
warnings.filterwarnings("ignore")

from agentlab.capstone.banking_tools import extract_facts
from agentlab.capstone.policy_rag import (
    get_default_retriever, PolicyRagRetriever, _DEFAULT_STORE)
from knowlytix.knowledge.rag import Extraction
from knowlytix.harness.testing import (
    TripleCompletenessScorer, triples_from_rag_answer)

sink = io.StringIO()
with contextlib.redirect_stderr(sink), contextlib.redirect_stdout(sink):
    retr = get_default_retriever()
    # DUAL embedding: v-encoder binds head+relation; u-encoder (GEODE-calibrated
    # contradiction) verifies the value -- entailment covered, contradiction not.
    v_enc = PolicyRagRetriever._load_tuned_encoder(_DEFAULT_STORE)
    u_enc = PolicyRagRetriever._load_contradiction_encoder(_DEFAULT_STORE)
    import json as _json
    _cal = _json.load(open(f"{_DEFAULT_STORE}/relevance_calibration.json"))
    scorer = TripleCompletenessScorer(
        retr.pipe.store, encoder=v_enc, u_encoder=u_enc,
        tau_contra_per_relation=_cal.get("tau_contra_per_relation"),
        default_tau_contra=_cal.get("default_tau_contra"))

cases = json.load(open("data/eval_cases/cases.json"))
tot_exp = tot_cov = 0
for c in cases:
    with contextlib.redirect_stderr(sink), contextlib.redirect_stdout(sink):
        ex = extract_facts.fn(message=c["message"])
        ser = ex.get("extraction")
        ans = retr.pipe.query(c["message"],
                              extraction=Extraction.from_dict(ser) if ser else None)
        gt = triples_from_rag_answer(ans)             # expected = retrieved triplets
        rep = scorer.score(ans.answer, gt)            # produced + recall, one call
    tot_exp += rep.expected
    tot_cov += rep.covered
    tag = "ABSTAIN" if not gt else f"{rep.covered}/{rep.expected}"
    print(f"{c['id']}: {tag}  expected={[m for m in (gt or [])]}")
    if rep.missing:
        print(f"     answer={ans.answer[:70]!r}  MISSING={rep.missing}")

print(f"\nmicro completeness = {tot_cov}/{tot_exp} = "
      f"{(tot_cov/tot_exp if tot_exp else 1.0):.3f}")
