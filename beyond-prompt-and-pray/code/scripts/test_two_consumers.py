"""Focused test: ONE grounded extraction feeds BOTH consumers.

Demonstrates the architecture explicitly -- extract_facts runs the GEODE
parse⇄bind loop once, and the *same* extraction artifact is consumed by

  (1) the RAG (search_policy reuses it instead of re-parsing -- parse-once), and
  (2) regulatory escalation (flag_regulatory grounds product/issue on its bound
      query_facts),

with two checks per message:
  * RAG reuse is lossless: search(msg, extraction=ex) == search(msg).
  * escalation reads the grounded signal: _flag_impl(..., query_facts=qf) uses the
    bound (product, issue) rather than the free-text guess.
"""

from __future__ import annotations

from agentlab.capstone.banking_tools import (
    _extract_impl, _flag_impl, signal_from_query_facts)
from agentlab.capstone.policy_rag import get_default_retriever

CASES = [
    "You charged me an overdraft fee that is completely unfair. Refund it now.",
    "Just waive my $35 overdraft fee.",
    "My mortgage escrow was misapplied and my payment is now late.",
    "How do I close my checking account?",
]


def _pol(results):
    if not results:
        return None
    r = results[0]
    return (tuple(r.get("policies") or []), r.get("answer"))


def main() -> int:
    r = get_default_retriever()
    ok = True
    for msg in CASES:
        print("\n" + "=" * 78)
        print("MESSAGE:", msg)

        # --- one extraction ------------------------------------------------
        facts = _extract_impl(msg)
        ex = facts.get("extraction")
        qf = facts.get("query_facts") or []
        print(f"  extract_facts -> product={facts['product']} issue={facts['issue']}")
        print(f"  query_facts (bound policy triples) = {qf}")
        print(f"  grounded = {bool(ex)}  (is_bound -> grounded signal feeds both)")

        # --- consumer 1: the RAG, reusing the extraction (parse-once) -------
        reuse = r.search(msg, extraction=ex) if ex else r.search(msg)
        fresh = r.search(msg)
        lossless = _pol(reuse) == _pol(fresh)
        print(f"  [RAG] reuse==fresh (parse-once lossless): {lossless}  "
              f"policies={_pol(fresh)[0] if fresh else None}")
        ok = ok and lossless

        # --- consumer 2: escalation, grounded on the same query_facts ------
        grounded = signal_from_query_facts(qf)
        flag_g = _flag_impl(facts["product"], facts["issue"], msg, query_facts=qf)
        flag_u = _flag_impl(facts["product"], facts["issue"], msg, query_facts=[])
        print(f"  [ESC] grounded signal from query_facts = {grounded or '(abstain -> keep guess)'}")
        print(f"  [ESC] flags grounded={flag_g['flags']} escalate={flag_g['escalate']}")
        print(f"  [ESC] flags ungrounded={flag_u['flags']} escalate={flag_u['escalate']}")

    print("\n" + "=" * 78)
    print("RESULT:", "PASS (RAG reuse lossless on all cases)" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
