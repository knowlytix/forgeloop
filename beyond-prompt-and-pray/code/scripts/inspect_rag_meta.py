"""Inspect the rag_test questions' metadata + ground_truth so we can derive the
expected GMS triples reliably (from the store) instead of incomplete schema parse.
"""
import warnings, contextlib, io, json
warnings.filterwarnings("ignore")
from agentlab.testing import CapstoneTestHarness

sink = io.StringIO()
CAP = []
# capture what the compiler sees per question
import knowlytix.harness.testing.claims as C
_orig = C.__dict__.get("ClaimCompiler").compile


def _patched(self, llm_answer, question_metadata=None, question_category="",
             run_id="", question_id="", ground_truth="", question_text=""):
    CAP.append({"cat": question_category, "q": question_text,
                "meta": question_metadata, "gt": ground_truth,
                "answer": llm_answer})
    return _orig(self, llm_answer, question_metadata=question_metadata,
                 question_category=question_category, run_id=run_id,
                 question_id=question_id, ground_truth=ground_truth,
                 question_text=question_text)


C.ClaimCompiler.compile = _patched

with contextlib.redirect_stderr(sink), contextlib.redirect_stdout(sink):
    stand = CapstoneTestHarness(n_runs=2, seed=42)
    stand.rag_test()

seen = set()
for c in CAP:
    key = c["q"]
    if key in seen:
        continue
    seen.add(key)
    print(f"### [{c['cat']}] {c['q']}")
    print(f"    meta keys: {sorted((c['meta'] or {}).keys())}")
    print(f"    meta: { {k: v for k, v in (c['meta'] or {}).items() if k in ('head','relation','tail','true_tail','mid','r1','r2','r_direct','entities','relations','entity','value','answer')} }")
    print(f"    ground_truth: {str(c['gt'])[:120]}")
    print(f"    answer: {str(c['answer'])[:90]}")
    print()
