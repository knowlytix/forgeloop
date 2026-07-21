"""Debug the RAG synthesis prompt on ONE case: show the facts served to Qwen,
Qwen's full raw answer, and the API completeness. Usage: python debug_one_case.py <case_id>
"""
import warnings, contextlib, io, json, sys
warnings.filterwarnings("ignore")

from agentlab.capstone.banking_tools import extract_facts
from agentlab.capstone.policy_rag import get_default_retriever
from knowlytix.knowledge.rag import Extraction
from knowlytix.harness.testing import TripleCompletenessScorer, triples_from_rag_answer

cid = sys.argv[1] if len(sys.argv) > 1 else "case-008"
sink = io.StringIO()
with contextlib.redirect_stderr(sink), contextlib.redirect_stdout(sink):
    retr = get_default_retriever()
    scorer = TripleCompletenessScorer(retr.pipe.store)
    cases = json.load(open("data/eval_cases/cases.json"))
    case = next(c for c in cases if c["id"] == cid)
    ex = extract_facts.fn(message=case["message"])
    ser = ex.get("extraction")
    ans = retr.pipe.query(case["message"],
                          extraction=Extraction.from_dict(ser) if ser else None)
    gt = triples_from_rag_answer(ans)
    rep = scorer.score(ans.answer, gt)
    produced = scorer.parse(ans.answer)

print(f"CASE {cid}: {case['message']!r}")
print(f"\nFACTS SERVED TO QWEN (retrieved triplets):")
for f in (ans.sources or []):
    print(f"   FACT: {f.head} | {f.relation} | {f.tail}")
print(f"\nQWEN RAW ANSWER:\n   {ans.answer!r}")
print(f"\nAPI: completeness {rep.covered}/{rep.expected}  produced(parsed)={sorted(produced)}")
print(f"     MISSING: {rep.missing}")
print(f"     decision={ans.decision} route={getattr(ans,'route',None)}")
