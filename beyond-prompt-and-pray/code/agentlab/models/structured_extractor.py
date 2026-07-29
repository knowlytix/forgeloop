"""Structured complaint-fact extractor (local Qwen) -- the narrow intake layer.

extract_facts' real job is NOT to classify into a policy taxonomy (the GEODE RAG
and the regulatory classifier already do the hard mapping). It is to turn a noisy
complaint into structured FACTS that (1) compose a clean query for the policy RAG
and (2) feed the regulatory-escalation classifier. So this extracts a controlled
fact record, not a label:

    product   : checking_account|credit_card|mortgage|loan|deposit|zelle|debit_card|unknown
    event     : unauthorized_transaction|fee_dispute|account_closure|credit_reporting|servicing_error|inquiry|none
    allegation: short verbatim-style phrase ("I did not authorize", "I was charged")
    harm      : financial_loss|denial_of_access|credit_damage|delay|emotional_distress|none
    remedy    : refund|correction|explanation|account_reopening|reversal|none
    time_facts: free text (transaction/complaint dates) or ""
    evidence  : subset of [amount, merchant, channel, account_type, prior_contact] present

Local Qwen only (book rule: Sonnet is for offline training-data generation, never
the runtime pipeline). The query composer turns facts into a policy-search query.
"""

from __future__ import annotations

import json
import re
from typing import Any

_PRODUCTS = {"checking_account", "credit_card", "mortgage", "loan", "deposit",
             "zelle", "debit_card", "unknown"}
_EVENTS = {"unauthorized_transaction", "fee_dispute", "account_closure",
           "credit_reporting", "servicing_error", "inquiry", "none"}
_HARMS = {"financial_loss", "denial_of_access", "credit_damage", "delay",
          "emotional_distress", "none"}
_REMEDIES = {"refund", "correction", "explanation", "account_reopening",
             "reversal", "none"}
_EVIDENCE = {"amount", "merchant", "channel", "account_type", "prior_contact"}

_SYS = (
    "Extract structured facts from a retail-bank customer complaint. Return ONLY a "
    "JSON object with keys product, event, allegation, harm, remedy, time_facts, "
    "evidence.\n"
    'product: one of ["checking_account","credit_card","mortgage","loan","deposit",'
    '"zelle","debit_card","unknown"] (unknown if no product is identifiable).\n'
    'event: one of ["unauthorized_transaction","fee_dispute","account_closure",'
    '"credit_reporting","servicing_error","inquiry","none"].\n'
    'allegation: a short phrase capturing what the customer asserts, in their terms.\n'
    'harm: one of ["financial_loss","denial_of_access","credit_damage","delay",'
    '"emotional_distress","none"].\n'
    'remedy: one of ["refund","correction","explanation","account_reopening",'
    '"reversal","none"].\n'
    'time_facts: any dates/deadlines mentioned, else "".\n'
    'evidence: a JSON list of which of ["amount","merchant","channel","account_type",'
    '"prior_contact"] the message states.\n'
    "Use the exact strings. Output only the JSON object."
)


def _coerce(obj: dict[str, Any]) -> dict[str, Any]:
    def pick(v, allowed, default):
        v = str(v or "").strip().lower().replace(" ", "_")
        return v if v in allowed else default
    ev = obj.get("evidence") or []
    if isinstance(ev, str):
        ev = [e.strip() for e in re.split(r"[,;]", ev)]
    ev = [e.strip().lower().replace(" ", "_") for e in ev]
    return {
        "product": pick(obj.get("product"), _PRODUCTS, "unknown"),
        "event": pick(obj.get("event"), _EVENTS, "none"),
        "allegation": str(obj.get("allegation") or "").strip()[:120],
        "harm": pick(obj.get("harm"), _HARMS, "none"),
        "remedy": pick(obj.get("remedy"), _REMEDIES, "none"),
        "time_facts": str(obj.get("time_facts") or "").strip()[:80],
        "evidence": [e for e in ev if e in _EVIDENCE],
    }


def parse_facts(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object from the model text and coerce it onto the controlled fact schema.

    Args:
        text: Raw model output expected to contain a JSON object.

    Returns:
        The coerced fact dict, or None when no valid JSON object is found.
    """
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return _coerce(json.loads(m.group(0)))
    except (json.JSONDecodeError, ValueError):
        return None


def compose_query(facts: dict[str, Any]) -> str:
    """A clean policy-search query from the facts: the product + event + the
    customer's allegation + the remedy -- the policy-relevant content, stripped of
    emotional noise. This is what search_policy should retrieve on."""
    parts = []
    if facts.get("product") and facts["product"] != "unknown":
        parts.append(facts["product"].replace("_", " "))
    if facts.get("event") and facts["event"] not in ("none", "inquiry"):
        parts.append(facts["event"].replace("_", " "))
    if facts.get("allegation"):
        parts.append(facts["allegation"])
    if facts.get("remedy") and facts["remedy"] != "none":
        parts.append(f"requesting {facts['remedy'].replace('_', ' ')}")
    return "; ".join(parts) or (facts.get("allegation") or "")
