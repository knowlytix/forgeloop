"""Prototype: hybrid (regex + context-aware LLM) evidence extraction for the
regulatory guard, with a before/after on UDAAP positives and dispute negatives.

The guard derives a regulation only when it finds the supporting *evidence
entity* in the message. Today that step is pure regex/alias substring matching,
so a bank fee phrased as a bare "$35 charge" is missed (UDAAP under-fires), while
a blanket charge->fee alias would over-fire on transaction disputes. The fix is a
context-aware hybrid: regex catches the unambiguous literals; an LLM decides
whether an ambiguous "charge" is a bank-imposed fee (-> fee evidence) or a
merchant/transaction dispute (-> not). We isolate the evidence path (proposed=[])
so the before/after measures only this change.

    python scripts/proto_hybrid_evidence.py
"""

from __future__ import annotations

from agentlab.capstone.regulatory_guard import get_default_guard
from agentlab.extraction import EntitySpec, HybridEntityExtractor
from agentlab.models import QwenAdapter

guard = get_default_guard()
_orig_evidence = guard.evidence_in_message            # bound regex/alias method
CANON = sorted({e for ents in guard.evidence_by_flag.values() for e in ents})

# Per-entity definitions are the precision lever: they tell the LLM when an
# ambiguous mention is the entity and when it is not. Only the ambiguous ones
# need a definition; the rest rely on name + aliases.
_DEFS = {
    "fee": ("a fee or charge IMPOSED BY THE BANK that the customer disputes as unfair "
            "(overdraft fee, NSF fee, maintenance fee, a $X charge the bank applied); "
            "a merchant purchase, a card transaction, an unauthorized or duplicate "
            "transaction, or a chargeback is NOT a fee"),
    "overdraft": "an overdraft, NSF or negative-balance situation",
    "unfair": "the customer calls the bank's charge unfair",
    "deceptive": "the customer calls the bank's action deceptive",
    "hidden_fee": "the customer says a fee was hidden or undisclosed",
    # the alternative buckets: give a merchant/transaction charge somewhere to go
    # other than 'fee', or the model forces it into 'fee' and over-fires UDAAP.
    "debit_fraud": ("an unauthorized or unrecognized charge or transaction on a debit "
                    "or bank card --- a purchase the customer did not make"),
    "unauthorized_transfer": "a transfer or transaction the customer did not authorize",
    "electronic_transfer": "an electronic payment or transfer dispute (e.g. a duplicate charge)",
    "credit_card_billing": "a credit-card transaction or billing dispute (not a bank fee)",
}

# Build the general extractor from the guard's own vocabulary + alias lexicon.
_aliases: dict[str, list[str]] = {}
for phrase, entity in guard.alias_to_evidence.items():
    _aliases.setdefault(entity, []).append(phrase)
_specs = [
    EntitySpec(name=ent, aliases=tuple(_aliases.get(ent, [ent])), definition=_DEFS.get(ent, ""))
    for ent in CANON
]
_extractor = HybridEntityExtractor(
    _specs,
    llm=QwenAdapter(max_new_tokens=64),
    task="You label which regulatory evidence signals a retail-bank customer message contains.",
    word_boundary=False,                              # match the guard's substring behavior
)


def hybrid_evidence(message: str) -> set[str]:
    return _extractor.extract(message)


def _escalates(message: str, evidence_fn) -> tuple[bool, list[str], list[str]]:
    guard.evidence_in_message = evidence_fn          # type: ignore[method-assign]
    try:
        v = guard.verify_and_correct([], message)    # proposed=[] -> evidence path only
        return v["escalate"], v["flags"], sorted(evidence_fn(message))
    finally:
        guard.evidence_in_message = _orig_evidence   # type: ignore[method-assign]


TESTS = [
    ("P1 case-001 clean", "I was charged a $35 overdraft fee and I want it removed.", True),
    ("P2 charge=fee, no literal",
     "There's a $35 charge the bank put on my account that I didn't expect; please take it off.", True),
    ("P3 misleading/hedged",
     "Probably nothing, but the bank tacked on a $35 charge to my checking and I don't think that's right.", True),
    ("P4 case-016 outraged", "I am outraged. My overdraft fee is unfair and I demand it back.", True),
    ("N1 case-003 txn dispute",
     "I see an unauthorized transaction on my credit card. Please remove it.", False),
    ("N2 case-020 dup purchase",
     "I was charged twice for the same purchase, please reverse one.", False),
    ("N3 merchant charge", "There's a charge from a store I don't recognize on my card.", False),
    ("N4 inquiry", "How do I close my checking account?", False),
]


def main() -> int:
    print("canonical evidence entities:", CANON)
    print("guard theta:", round(guard.theta, 3), "\n")
    print(f"{'case':26s} {'exp':5s} {'before':6s} {'after':6s}  detail")
    n_pos = n_pos_fixed = n_neg = n_neg_false = 0
    for name, msg, exp in TESTS:
        eb, fb, _ = _escalates(msg, _orig_evidence)
        ea, fa, ev_a = _escalates(msg, hybrid_evidence)
        mark = "OK " if ea == exp else "XX "
        print(f"{mark}{name:23s} {str(exp):5s} {str(eb):6s} {str(ea):6s}  "
              f"after_flags={fa} after_evidence={ev_a}")
        if exp:
            n_pos += 1; n_pos_fixed += int(ea)
        else:
            n_neg += 1; n_neg_false += int(ea)
    print(f"\npositives escalating: before vs after -> "
          f"{sum(_escalates(m, _orig_evidence)[0] for _, m, e in TESTS if e)}/{n_pos}"
          f" -> {n_pos_fixed}/{n_pos}")
    print(f"negatives FALSE-escalating (must be 0): {n_neg_false}/{n_neg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
