"""Entity-linking + GMS link-prediction fact extractor.

The diagnosis: a complaint expresses INTENT ("waive my overdraft fee", "close my
account"), not policy fact-values, so exact triplet/alias matching under-grounds
and nearest-neighbour binding over-grounds. The GMS-native fix is two steps:

1. ENTITY LINKING with the SFT-tuned (GEODE) embedding -- embed the message and
   link it to the nearest policy ENTITY (overdraft, disputes, account_closure,
   reg_x, loan_servicing, ...) by cosine in the document-tuned space, robust to
   colloquial phrasing. A calibrated accept threshold abstains when nothing is
   close enough, so a vague message links no entity and the issue is `general`.

2. GMS LINK PREDICTION from the linked entity -- ``store.link_predict(entity,
   "has_product")`` recovers the product geometrically (type-constrained), and
   the entity's policy domain maps to the coarse issue. Product is grounded in
   the graph, not guessed.

A label is emitted only when an entity links above threshold AND the product
link predicts -- so it cannot drift to an ungrounded label, and it is not forced
onto a concrete value when the message supports none.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from forgeloop.agents._paths import data_path

# Policy entity -> coarse issue taxonomy (disputes re-routed to credit_card under
# card context). Cross-product domains map to None (no specific product issue).
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
_NONE = "__none__"
_CARD_RE = re.compile(r"credit[\s-]?card|\bvisa\b|mastercard|amex|\bmy card\b", re.I)

_DEFAULT_STORE = data_path("gms_policy_store_geode")


class EntityLinkExtractor:
    """Links a complaint to the nearest policy entity in the GEODE embedding space, then reads the product by GMS link prediction and the issue by domain mapping."""

    def __init__(self, store_path: Path | str | None = None,
                 threshold: float = 0.0) -> None:
        from knowlytix.knowledge.rag import GeometricLabelClassifier

        from forgeloop.agents.capstone.policy_rag import PolicyRagRetriever

        sp = Path(store_path) if store_path is not None else _DEFAULT_STORE
        self.retriever = PolicyRagRetriever(store_path=sp)
        self.store = self.retriever.store
        encode = self.retriever.pipe.rag.encoder  # the v2 doc-tuned SFT encoder

        # Entity exemplars = each policy domain's name + its store aliases. Linking
        # in the tuned space maps colloquial mentions ("chargeback", "fee waiver")
        # to the right entity without exact-string matching.
        heads = {h for h, _, _ in self.store.doc_graph.triples}
        aliases: dict[str, list[str]] = {}
        for h, r, t in self.store.doc_graph.triples:
            if r == "has_alias":
                aliases.setdefault(h, []).append(str(t).replace("_", " "))
        self.exemplars = {
            e: [e.replace("_", " "), *aliases.get(e, [])]
            for e in _DOMAIN_ISSUE if e in heads
        }
        self.clf = GeometricLabelClassifier(
            self.exemplars, encoder=encode, threshold=threshold)

    def calibrate(self, abstain_messages: list[str], ceiling: float = 0.10) -> float:
        """Fit the link-accept threshold so vague messages (no policy entity)
        abstain: false-link rate over the abstain cohort <= ceiling."""
        return self.clf.calibrate(
            [(m, _NONE) for m in abstain_messages],
            abstain_label=_NONE, false_accept_ceiling=ceiling)

    def extract(self, message: str) -> dict[str, Any] | None:
        """Link the message to a policy entity and return its grounded product and issue.

        Args:
            message: The customer complaint text.

        Returns:
            A ``{product, issue, entity}`` dict, or None when nothing links above threshold.
        """
        entity, _score = self.clf.classify(message, abstain_label=_NONE)
        if entity == _NONE:
            return None  # nothing linked -> general / keep Qwen product
        card = bool(_CARD_RE.search(message))
        issue = _DOMAIN_ISSUE.get(entity)
        if entity == "disputes" and card:
            issue = "credit_card_issue"
        # GMS link prediction: most-plausible product for (entity, has_product, ?)
        preds = self.store.link_predict(entity, "has_product", top_k=4)
        products = [p for p, _ in preds]
        if card and "credit_card" in products:
            product = "credit_card"
        elif products:
            product = products[0]
        else:
            product = None
        return {"product": product, "issue": issue or "general", "entity": entity}


_DEFAULT: EntityLinkExtractor | None = None


def get_default_entity_linker() -> EntityLinkExtractor:
    """Return the process-wide EntityLinkExtractor singleton, applying any persisted link-accept threshold.

    Returns:
        The lazily built EntityLinkExtractor bound to the resolved policy store.
    """
    global _DEFAULT
    if _DEFAULT is None:
        import json
        sp = os.environ.get("AGENTLAB_POLICY_STORE")
        if not sp:
            from forgeloop.agents._paths import ensure_default
            ensure_default("gms_policy_store_geode")
        ext = EntityLinkExtractor(store_path=sp or None)
        cal = Path(os.environ.get(
            "AGENTLAB_ENTITY_LINK_CAL",
            str(_DEFAULT_STORE.parent / "entity_link_calibration.json")))
        if cal.exists():
            ext.clf.threshold = float(json.loads(cal.read_text())["threshold"])
        _DEFAULT = ext
    return _DEFAULT
