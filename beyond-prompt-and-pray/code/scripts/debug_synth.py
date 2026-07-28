"""Debug the synthesis prompt in isolation across fact shapes: feed FIXED
retrieved triplets to the Assembler (deterministic greedy synthesis), read Qwen's
text + API completeness. Retrieval non-determinism is out of the loop.
"""
import warnings, contextlib, io
warnings.filterwarnings("ignore")

from knowlytix.knowledge.rag.retrieve import RetrievedFact
from knowlytix.knowledge.rag.assemble import Assembler
from agentlab.capstone.policy_rag import QwenLLMBackend, get_default_retriever
from knowlytix.harness.testing import TripleCompletenessScorer

SCENARIOS = [
    ("fee waiver (multi-hop)", "Just waive my overdraft fee.",
     [("fee_reversal", "has_above_cap_authorization", "manager"),
      ("manager", "has_max_reversal", "500.0")]),
    ("overdraft fee (numeric)", "I was charged a $35 overdraft fee.",
     [("overdraft", "has_fee_amount", "35.0")]),
    ("statement type (categorical)", "Something seems off with my statement.",
     [("paper_statement", "has_type", "per_month")]),
    ("identity verify (required)", "Call me about my urgent account issue.",
     [("account_closure", "has_identity_verification", "required")]),
    ("dispute credit (issued)", "I need to dispute a charge.",
     [("disputes", "has_provisional_credit", "issued")]),
]

sink = io.StringIO()
with contextlib.redirect_stderr(sink), contextlib.redirect_stdout(sink):
    asm = Assembler(QwenLLMBackend())
    scorer = TripleCompletenessScorer(get_default_retriever().pipe.store)
    out = []
    for name, q, triples in SCENARIOS:
        facts = [RetrievedFact(head=h, relation=r, tail=t, score=0.0,
                               confidence=1.0, source="") for (h, r, t) in triples]
        ans = asm.assemble(q, facts)
        rep = scorer.score(ans, triples)
        out.append((name, triples, ans, rep))

for name, triples, ans, rep in out:
    print(f"### {name}")
    print(f"   served : {triples}")
    print(f"   answer : {ans!r}")
    print(f"   API    : {rep.covered}/{rep.expected}  missing={rep.missing}")
    print()
