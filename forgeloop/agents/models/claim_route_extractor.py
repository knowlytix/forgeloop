"""Claim-decomposition fact extractor for the banking complaint agent.

The Qwen extractor and the flat geometric classifier both collapse a complaint
into one (product, issue) label, so they over-commit on vague input and can drift
to a label no evidence supports (a checking dispute mislabeled credit_card). This
extractor instead routes CLAIMS to GMS: it decomposes the message into clauses,
routes each through the SAME calibrated GEODE retriever ``search_policy`` uses
(``PolicyRagRetriever.route`` -- retrieve-only, cap/tension/relevance/accept
gates, no LLM), and reads the grounded policy domain off each. The issue is the
domain mapped onto the coarse taxonomy; the product follows the store's
``has_product`` edge from that domain. A claim that grounds nothing contributes
nothing, so a vague message lands on ``general`` honestly instead of being forced
onto a concrete label -- and a label is emitted only when a policy domain in GMS
actually supports it, which is what prevents the ungrounded credit_card drift.

Product still falls back to the (strong) Qwen product when no claim grounds one.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from forgeloop.agents._paths import data_path

# Policy domain -> coarse issue taxonomy. `disputes` is deposit-side by default
# and re-routed to credit_card_issue under card context (mirrors the dual
# has_product edge). Cross-product domains (udaap/escalation/pii) imply no single
# product issue, so they map to None and do not force an issue.
_DOMAIN_ISSUE: dict[str, str | None] = {
    "overdraft": "overdraft_fee", "nsf_check": "overdraft_fee",
    "fee_reversal": "overdraft_fee",
    "disputes": "account_issue", "account_closure": "account_issue",
    "stop_payment": "account_issue", "wire_domestic": "account_issue",
    "wire_international": "account_issue", "paper_statement": "account_issue",
    "reg_e": "account_issue",
    "late_payment": "credit_card_issue", "reg_z": "credit_card_issue",
    "reg_x": "mortgage_issue", "loan_servicing": "loan_issue",
    "udaap": None, "regulatory_escalation": None, "pii_handling": None,
}
# Priority when several claims ground: the regulatorily-salient fee issue wins,
# then product-specific issues, then the generic account issue.
_ISSUE_RANK = {"overdraft_fee": 1, "mortgage_issue": 2, "loan_issue": 3,
               "credit_card_issue": 4, "account_issue": 5, "general": 6}
_CARD_RE = re.compile(r"credit[\s-]?card|\bvisa\b|mastercard|amex|\bmy card\b", re.I)
_CLAUSE_SPLIT = re.compile(r"[.;!?]|\band\b|\bbut\b|,", re.I)


_DEFAULT_STORE = data_path("gms_policy_store_geode")


def _clauses(message: str) -> list[str]:
    parts = [c.strip() for c in _CLAUSE_SPLIT.split(message)]
    return [c for c in parts if len(c) >= 6]


class ClaimRouteExtractor:
    """Decomposes a complaint into clauses and routes each through the calibrated GEODE retriever to read a grounded (product, issue) off the policy store."""

    def __init__(self, store_path: Path | str | None = None) -> None:
        from forgeloop.agents.capstone.policy_rag import PolicyRagRetriever

        sp = Path(store_path) if store_path is not None else _DEFAULT_STORE
        self.retriever = PolicyRagRetriever(store_path=sp)
        # has_product edges from the store graph (excluded from query parsing, so
        # read structurally here): domain -> {products}.
        self.product_of: dict[str, set[str]] = {}
        for h, r, t in self.retriever.store.doc_graph.triples:
            if r == "has_product":
                self.product_of.setdefault(h, set()).add(str(t))

    def _ground(self, message: str) -> list[str]:
        """All policy domains the message + its clauses ground to (deduped)."""
        domains: list[str] = []
        for q in [message, *_clauses(message)]:
            for d in self.retriever.route(q):
                if d not in domains:
                    domains.append(d)
        return domains

    def route(self, message: str) -> dict[str, Any] | None:
        """Return {product, issue} from grounded claims, or None if nothing
        grounds (caller then treats the message as general / keeps Qwen product)."""
        domains = self._ground(message)
        card = bool(_CARD_RE.search(message))
        issues: list[tuple[str, str]] = []
        for d in domains:
            iss = _DOMAIN_ISSUE.get(d)
            if d == "disputes" and card:
                iss = "credit_card_issue"
            if iss:
                issues.append((iss, d))
        if not issues:
            return None
        issues.sort(key=lambda x: _ISSUE_RANK[x[0]])
        issue, top_domain = issues[0]
        prods = self.product_of.get(top_domain, set())
        if card and "credit_card" in prods:
            product = "credit_card"
        elif len(prods) == 1:
            product = next(iter(prods))
        elif prods:
            # multi-product domain (disputes) without card context -> deposit side
            product = "checking_account" if "checking_account" in prods else sorted(prods)[0]
        else:
            product = None
        return {"product": product, "issue": issue}


_DEFAULT: ClaimRouteExtractor | None = None


def get_default_claim_router() -> ClaimRouteExtractor:
    """Return the process-wide ClaimRouteExtractor singleton, honoring the AGENTLAB_POLICY_STORE store override.

    Returns:
        The lazily built ClaimRouteExtractor bound to the resolved policy store.
    """
    global _DEFAULT
    if _DEFAULT is None:
        # AGENTLAB_POLICY_STORE overrides the store (e.g. the v2 product-augmented
        # store during validation, before it is swapped in as canonical).
        sp = os.environ.get("AGENTLAB_POLICY_STORE")
        if not sp:
            from forgeloop.agents._paths import ensure_default
            ensure_default("gms_policy_store_geode")
        _DEFAULT = ClaimRouteExtractor(store_path=sp or None)
    return _DEFAULT
