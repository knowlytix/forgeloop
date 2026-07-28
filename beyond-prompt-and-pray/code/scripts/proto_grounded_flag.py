"""Prototype: derive regulatory flags from the GROUNDED extract_facts fact via
the GMS guard's CALIBRATED scoring (score_triple <= theta), instead of a
zero-shot LLM read of the raw message.

The only change from the shipped guard is the EVIDENCE SOURCE: instead of
`evidence_in_message(message)` (regex/extractor on raw prose), we feed the
canonical evidence entities implied by the grounded (product, issue) fact.
Flag derivation, the UDAAP unfairness gate and escalation all reuse the
guard's calibrated machinery unchanged.

Checks:
  1. recovery        -- does the grounded path recover the misleading misses?
  2. over-escalation -- does it falsely escalate routine (non-UDAAP) disputes?
All gating uses the persisted calibrated theta (no defaults/midpoints).
"""
import warnings, contextlib, io, json
warnings.filterwarnings("ignore")
import pandas as pd

sink = io.StringIO()
with contextlib.redirect_stderr(sink), contextlib.redirect_stdout(sink):
    from agentlab.capstone.banking_tools import extract_facts, signal_from_query_facts
    from agentlab.capstone.regulatory_guard import (
        GMSRegulatoryGuard, _FLAG_TO_ENTITY, _ENTITY_TO_FLAG,
        _udaap_supported_by_evidence,
    )
    g = GMSRegulatoryGuard.load()

print(f"calibrated evidence_threshold (theta) = {g.theta}")

# canonical regulatory evidence entities implied by a GROUNDED (product, issue).
# topical only -- a grounded fact does NOT carry an unfairness signal.
ISSUE2EV = {
    "overdraft_fee": {"overdraft", "fee"},
    "mortgage_issue": {"mortgage_servicing"},
    "loan_issue": {"home_loan"},
    "credit_card_issue": {"credit_card_billing"},
    "account_issue": set(),
}


def grounded_evidence(sig):
    return set(ISSUE2EV.get(sig.get("issue", ""), set()))


def grounded_flags(evidence):
    """Same calibrated correction the guard runs, but over grounded evidence.
    A flag stands iff some grounded evidence entity scores within the
    calibrated theta of the flag (g._flag_supported). UDAAP additionally
    requires an unfairness signal (which a topical grounded fact lacks)."""
    flags = set()
    for flag_entity in g.evidence_by_flag:
        if not g._flag_supported(flag_entity, evidence):
            continue
        if flag_entity == "udaap" and not _udaap_supported_by_evidence(evidence):
            continue
        flags.add(_ENTITY_TO_FLAG.get(flag_entity, flag_entity))
    return sorted(flags)


def ground(msg):
    with contextlib.redirect_stderr(sink), contextlib.redirect_stdout(sink):
        e = extract_facts.fn(message=msg)
        sig = signal_from_query_facts(e.get("query_facts", [])) or {"issue": e.get("issue")}
    return sig


def derive(msg):
    sig = ground(msg)
    ev = grounded_evidence(sig)
    fl = grounded_flags(ev)
    with contextlib.redirect_stderr(sink), contextlib.redirect_stdout(sink):
        esc = g.escalation_for_flags(fl)[0]
    return sig, ev, fl, esc


cases = json.load(open("data/eval_cases/cases.json"))
print("\n=== grounded-fact -> CALIBRATED GMS flags on CLEAN seeds ===")
print(f'{"id":9s} {"reg(label)":17s} {"exp_esc":7s} {"gnd_issue":14s} {"gnd_evidence":24s} {"flags":12s} {"esc":5s} verdict')
over = miss = 0
for c in cases:
    lab = str(c["factors"].get("regulatory"))
    expesc = bool(c.get("expected_escalation"))
    sig, ev, fl, esc = derive(c["message"])
    if lab in ("PII", "prompt_injection"):
        verdict = "(input-gate, n/a)"
    elif esc and not expesc:
        verdict = "OVER-ESCALATION"; over += 1
    elif not esc and expesc:
        verdict = "MISS"; miss += 1
    else:
        verdict = "ok"
    print(f'{c["id"]:9s} {lab:17s} {str(expesc):7s} {str(sig.get("issue")):14s} {str(sorted(ev)):24s} {str(fl):12s} {str(esc):5s} {verdict}')
print(f"\nover-escalations (excl. input-gate): {over}   misses: {miss}")

print("\n=== the 4 misleading-clarity FAILURES: recovered by calibrated grounded path? ===")
df = pd.read_csv("data/doe_ch16_neweval.csv")
for _, r in df[df.tool_flag_ok == False].iterrows():
    sig, ev, fl, esc = derive(str(r.message))
    want = r.regulatory
    rec = "RECOVERED" if want in fl else "still missed"
    print(f'{r.seed_case} want={want:6s} gnd_issue={str(sig.get("issue")):14s} flags={str(fl):12s} esc={esc}  -> {rec}')
