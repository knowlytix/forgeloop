"""Geometric RAG completeness on the 20 base questions, wired like the real RAG.

For each base question (eval_cases message):
  1. fact_extraction (GEODE parse-and-bind) -> bound query triplets   [the RAG's
     actual input path: PolicyRagRetriever.extract]
  2. find the EXPECTED GMS triplets from the store for each bound (head, relation)
     via query_triples  [the ground-truth the answer should contain]
  3. run the real RAG (search_policy) -> answer  [reuse the same extraction]
  4. completeness = of those GMS triplets, how many the answer COVERS, decided by
     the SAME calibrated verifiers (plausibility / tension / holonomy) the
     correctness path uses -- via GMSJudge's calibrated router + evaluate_geometric.

No DOEGMSBenchmark, no generator metadata, no string matching.
"""
import warnings, contextlib, io, json
warnings.filterwarnings("ignore")

from agentlab.capstone.policy_rag import PolicyRagRetriever
from knowlytix.harness.testing.judge import GMSJudge
from knowlytix.harness.testing.claims import GeometricClaimExtractor, FactClaimSchema
from knowlytix.harness.testing.completeness import CompletenessEvaluator

sink = io.StringIO()
with contextlib.redirect_stderr(sink), contextlib.redirect_stdout(sink):
    retr = PolicyRagRetriever()
    store = retr.pipe.store
    judge = GMSJudge(store, claim_extraction="geometric")
    judge.calibrate(seed=42)          # calibrate the verifiers on the store's triples
    router = judge._router
    extractor = GeometricClaimExtractor(store)
    comp = CompletenessEvaluator(store)

cases = json.load(open("data/eval_cases/cases.json"))
tot_cov = tot_exp = 0
rows = []
for case in cases:
    msg = case["message"]
    with contextlib.redirect_stderr(sink), contextlib.redirect_stdout(sink):
        ex = retr.extract(msg)                       # 1) fact_extraction
        qf = ex.get("query_facts", [])
        # 2) expected GMS triplets for each bound (head, relation)
        seen, schemas = set(), []
        for trip in qf:
            h, r = str(trip[0]), str(trip[1])
            for sh, sr, st in store.query_triples(head=h, relation=r):
                key = (sh, sr, str(st))
                if key not in seen:
                    seen.add(key)
                    schemas.append(FactClaimSchema(head=sh, relation=sr,
                                                   expected_tail=str(st)))
        res = retr.search(msg, extraction=ex)        # 3) run the real RAG
        answer = res[0]["answer"] if res else ""
        claims = extractor.extract(answer, [], {})   # 4) geometric coverage
        subv = router.route_all(claims)
        rep = comp.evaluate_geometric(schemas, claims, subv)
    if rep.expected:
        tot_cov += rep.covered
        tot_exp += rep.expected
    rows.append((case["id"], rep, len(qf), answer))

print("=== Geometric RAG completeness on 20 base questions (real wiring) ===")
for cid, rep, nqf, ans in rows:
    if rep.expected:
        print(f"{cid}: covered {rep.covered}/{rep.expected}  bound_facts={nqf}")
        if rep.missing:
            print(f"      missing GMS triplets: {rep.missing}")
        print(f"      answer: {(ans or '')[:90]}")
    else:
        print(f"{cid}: no bound GMS triplets (n/a)  bound_facts={nqf}  ans: {(ans or '')[:50]}")
scored = [r for _, r, _, _ in rows if r.expected]
print(f"\nquestions with GMS triplets: {len(scored)}/20")
if tot_exp:
    print(f"micro completeness = {tot_cov}/{tot_exp} = {tot_cov/tot_exp:.3f}")
    macro = sum(r.score for r in scored) / len(scored)
    print(f"macro completeness (mean per-question recall) = {macro:.3f}")
