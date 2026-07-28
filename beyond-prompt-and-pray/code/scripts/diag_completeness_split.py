"""Split rag_test completeness into GMS-recall-miss vs LLM-omission.

Reuses knowlytix machinery end to end:
  * the questions + ground truth come from the SAME stand.rag_test() pipeline;
  * the per-question expected/covered/missing atoms come from knowlytix's
    CompletenessEvaluator (captured, not reimplemented);
  * the GMS-served check reruns the EXACT same CompletenessEvaluator against the
    retriever's retrieved context instead of the answer text.

The only glue is feeding the evaluator the retrieved text and differencing:
  in answer            -> covered
  in retrieved, not in answer -> LLM omission (GMS served it, the LLM dropped it)
  in neither           -> GMS recall miss (never retrieved)
"""
import warnings, contextlib, io, json
warnings.filterwarnings("ignore")

from agentlab.testing import CapstoneTestHarness
from agentlab.capstone.policy_rag import PolicyRagRetriever
from knowlytix.harness.testing.completeness import CompletenessEvaluator

sink = io.StringIO()

# what the GMS served, keyed by the answer string the judge will score
RETRIEVED: dict[str, str] = {}


class InstrumentedRetriever(PolicyRagRetriever):
    def search(self, query, k=3, extraction=None):
        res = super().search(query, k, extraction)
        if res:
            r = res[0]
            # generous "served" blob: grounded provenance spans + bound triples +
            # policy names -- anything the GMS surfaced to the generator.
            qf = "\n".join(f"{h} {rel} {t}" for h, rel, t in r.get("query_facts", []))
            pol = " ".join(r.get("policies", []))
            RETRIEVED[r.get("answer", "")] = "\n".join(
                [r.get("text", ""), qf, pol]).strip()
        return res


# capture each question's completeness inputs/report from the knowlytix judge
CAPTURED = []  # (evaluator_self, answer, ground_truth, meta, answer_report)
_orig_eval = CompletenessEvaluator.evaluate


def _patched_eval(self, answer_text, ground_truth, question_metadata=None):
    rep = _orig_eval(self, answer_text, ground_truth, question_metadata)
    CAPTURED.append((self, answer_text, ground_truth, question_metadata, rep))
    return rep


CompletenessEvaluator.evaluate = _patched_eval

with contextlib.redirect_stderr(sink), contextlib.redirect_stdout(sink):
    stand = CapstoneTestHarness(n_runs=2, seed=42)
    res = stand.rag_test(retriever=InstrumentedRetriever())

# ---- difference: answer coverage vs retrieved coverage, per atom ----
covered = llm_omit = gms_miss = unmatched = 0
matched_q = 0
per_q = []
for ev_self, answer, gt, meta, ans_rep in CAPTURED:
    expected = ans_rep.expected
    if expected == 0:
        continue  # vacuous (nothing checkable) -- excluded from the split
    retrieved_text = RETRIEVED.get(answer)
    if retrieved_text is None:
        unmatched += ans_rep.expected
        continue
    matched_q += 1
    # rerun the SAME evaluator against the retrieved context (bypass the patch)
    ret_rep = _orig_eval(ev_self, retrieved_text, gt, meta)
    ans_missing = set(m.lower() for m in ans_rep.missing)
    ret_missing = set(m.lower() for m in ret_rep.missing)
    c = ans_rep.covered                       # in answer
    lo = len(ans_missing - ret_missing)       # missing from answer BUT in retrieved
    gm = len(ans_missing & ret_missing)       # missing from both
    covered += c; llm_omit += lo; gms_miss += gm
    per_q.append({"answer": answer[:70], "expected": expected, "covered": c,
                  "llm_omission": lo, "gms_miss": gm,
                  "gms_missing_atoms": sorted(ans_missing & ret_missing)})

tot = covered + llm_omit + gms_miss
print("=== rag_test completeness split (atom-level) ===")
print(f"matched questions: {matched_q}  (unmatched expected atoms: {unmatched})")
print(f"total scored atoms: {tot}")
if tot:
    print(f"  covered (answer states it)      : {covered:3d}  ({covered/tot:.2%})")
    print(f"  LLM omission (served, not stated): {llm_omit:3d}  ({llm_omit/tot:.2%})")
    print(f"  GMS recall miss (never served)  : {gms_miss:3d}  ({gms_miss/tot:.2%})")
    print(f"\n  answer completeness  = {covered/tot:.3f}  (matches the 0.57 axis)")
    print(f"  retrieved completeness = {(covered+llm_omit)/tot:.3f}  (GMS recall ceiling)")
print("\n=== per-question GMS-recall misses (atoms never served) ===")
for q in per_q:
    if q["gms_miss"]:
        print(f"  [{q['gms_miss']}/{q['expected']}] {q['gms_missing_atoms']}  | {q['answer']}")
