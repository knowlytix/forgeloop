"""Adversarial precision check for the DUAL-embedding value verifier (u-space).

DECOYS state the WRONG tail (antonym/contrast) -- the u-space contradiction gate
must NOT bind. POSITIVES paraphrase the right fact -- it must bind. Thresholds are
GEODE's calibrated per-relation tau_contra (relevance_calibration.json).
"""
import warnings, contextlib, io, json
warnings.filterwarnings("ignore")

from agentlab.capstone.policy_rag import (
    get_default_retriever, PolicyRagRetriever, _DEFAULT_STORE)
from knowlytix.harness.testing import TripleCompletenessScorer

sink = io.StringIO()
with contextlib.redirect_stderr(sink), contextlib.redirect_stdout(sink):
    store = get_default_retriever().pipe.store
    v_enc = PolicyRagRetriever._load_tuned_encoder(_DEFAULT_STORE)
    u_enc = PolicyRagRetriever._load_contradiction_encoder(_DEFAULT_STORE)
    cal = json.load(open(f"{_DEFAULT_STORE}/relevance_calibration.json"))
    scorer = TripleCompletenessScorer(
        store, encoder=v_enc, u_encoder=u_enc,
        tau_contra_per_relation=cal.get("tau_contra_per_relation"),
        default_tau_contra=cal.get("default_tau_contra"))

DECOYS = [
    ("Disputes have provisional credit denied.",
     ("disputes", "has_provisional_credit", "issued"), False),
    ("Account closure identity verification is optional.",
     ("account_closure", "has_identity_verification", "required"), False),
    ("The paper statement type is per year.",
     ("paper_statement", "has_type", "per_month"), False),
    ("Unencrypted channel PII is allowed.",
     ("pii_handling", "has_unencrypted_channel_pii", "forbidden"), False),
]
POSITIVES = [
    ("The account closure requires identity verification.",
     ("account_closure", "has_identity_verification", "required"), True),
    ("Disputes have provisional credit issued.",
     ("disputes", "has_provisional_credit", "issued"), True),
    ("Paper statements are billed per month.",
     ("paper_statement", "has_type", "per_month"), True),
]

print(f"default_tau_contra = {cal.get('default_tau_contra')}\n")
ok = 0
for label, cases in [("DECOY (expect NOT covered)", DECOYS),
                     ("POSITIVE (expect covered)", POSITIVES)]:
    print(f"=== {label} ===")
    for text, trip, want in cases:
        with contextlib.redirect_stderr(sink), contextlib.redirect_stdout(sink):
            rep = scorer.score(text, [trip])
        covered = rep.covered == 1
        good = covered == want
        ok += good
        print(f"  [{'PASS' if good else 'FAIL'}] covered={covered} want={want}  {text!r}")
    print()
print(f"precision check: {ok}/{len(DECOYS) + len(POSITIVES)} correct")
