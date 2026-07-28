import warnings, contextlib, io
warnings.filterwarnings("ignore")
from agentlab.testing import CapstoneTestHarness
sink = io.StringIO()
with contextlib.redirect_stderr(sink), contextlib.redirect_stdout(sink):
    stand = CapstoneTestHarness(n_runs=2, seed=42)
    r = stand.rag_test()
print("=== rag_test (deployed retrieve-and-generate faithfulness) ===")
print(f"n_questions       = {r.n_questions}")
print(f"correctness       = {r.claims_verified}/{r.n_claims} = "
      f"{(r.claims_verified/r.n_claims if r.n_claims else 1.0):.3f}")
print(f"mean_completeness = {r.mean_completeness:.3f}")
print(f"verifier theta    = {getattr(r,'plausibility_threshold',None)}")
print(f"failure_codes     = {dict(r.failure_codes)}")
