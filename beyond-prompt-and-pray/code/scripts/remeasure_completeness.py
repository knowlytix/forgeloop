"""Re-measure rag_test completeness with the GEOMETRIC CompletenessEvaluator.

Coverage of each expected GMS triple is decided by the SAME calibrated
sub-verdicts (plausibility / tension / holonomy) the correctness path uses --
not string presence. Captures per-question covered/expected/missing triples.
"""
import warnings, contextlib, io
warnings.filterwarnings("ignore")

from agentlab.testing import CapstoneTestHarness
from knowlytix.harness.testing.completeness import CompletenessEvaluator

sink = io.StringIO()
REPORTS = []
_orig = CompletenessEvaluator.evaluate_geometric


def _patched(self, schemas, claims, sub_verdicts):
    rep = _orig(self, schemas, claims, sub_verdicts)
    REPORTS.append(rep)
    return rep


CompletenessEvaluator.evaluate_geometric = _patched

with contextlib.redirect_stderr(sink), contextlib.redirect_stdout(sink):
    stand = CapstoneTestHarness(n_runs=2, seed=42)
    res = stand.rag_test()

print("=== GEOMETRIC completeness re-measure ===")
print(f"rag_test mean_completeness = {res.mean_completeness:.3f}")
print(f"  (atom-level string match was 0.574; string-triple hybrid was 0.519)")
print(f"correctness = {res.claims_verified}/{res.n_claims}")
scored = [r for r in REPORTS if r.expected > 0]
exp = sum(r.expected for r in scored)
cov = sum(r.covered for r in scored)
print(f"questions with expected GMS triples: {len(scored)} "
      f"(vacuous/no-triple: {len(REPORTS) - len(scored)})")
if exp:
    print(f"triple micro-recall = {cov}/{exp} = {cov/exp:.3f}")
print("\n=== per-question coverage (covered/expected GMS triples) ===")
for r in REPORTS:
    if r.expected:
        print(f"  {r.covered}/{r.expected}  missing={r.missing}")
