"""Geometric fact extractor for the banking complaint agent.

An alternative to the Qwen ``qwen_extractor`` whose failure mode the capstone
diagnostic exposed: on an underspecified message the instruct model over-commits
to a concrete ``product`` (e.g. ``checking_account``), which cascades -- because
``issue`` is a near-deterministic function of ``product`` -- into a concrete
``issue`` (``account_issue``) where the truth was ``general``. That single trap
("``general`` is unreachable once a product is picked") was ~60% of all issue
errors.

This extractor follows the :class:`GeometricQueryParser` philosophy: instead of
*generating* a label, it scores the message against a CLOSED set of exemplar
phrases for each label by cosine in an embedding space and picks the nearest --
and ABSTAINS (``unknown`` product / ``general`` issue) when nothing clears a
**calibrated** threshold. The abstain is the cure for over-commitment: a vague
message scores low against every concrete exemplar and lands in ``general``
honestly, instead of being forced onto ``account_issue``.

``product`` and ``issue`` are classified INDEPENDENTLY from the message text, so
an eager product guess no longer drags the issue label with it.

Thresholds are never hardcoded -- they are fit by :func:`calibrate_extractor`
on a labeled cohort to a false-accept ceiling and persisted to
``data/extract_geo_calibration.json``; :func:`get_default_extractor` loads them.
The exemplars below are hand-authored from domain knowledge, deliberately not
copied from the evaluation cases, so calibration/benchmark stay honest.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from agentlab._paths import data_path

# Hand-authored exemplar phrases per label. NOT drawn from data/eval_cases.
_PRODUCT_EXEMPLARS: dict[str, list[str]] = {
    "checking_account": [
        "my checking account", "my savings account", "my debit card",
        "overdraft on my account", "my account balance and statement",
        "a charge on my bank account",
    ],
    "credit_card": [
        "my credit card", "my credit card statement", "my card was declined",
        "the APR on my credit card", "my credit card bill and payment",
    ],
    "mortgage": [
        "my mortgage", "my home loan", "my escrow account",
        "my monthly mortgage payment", "refinancing my house",
    ],
    "loan": [
        "my personal loan", "my auto loan", "my car loan payment",
        "the interest on my loan", "my loan balance and repayment",
    ],
    # The abstain class as a reachable label: procedural / unspecified messages
    # that name no product. Making it a real class (not only the below-threshold
    # fallback) is what stops an eager classifier from forcing a vague message
    # onto checking_account.
    "unknown": [
        "how do I update my address on file", "what are your branch hours",
        "I want to file a general complaint", "I need to speak to a representative",
        "a general question about your services",
    ],
}

_ISSUE_EXEMPLARS: dict[str, list[str]] = {
    "overdraft_fee": [
        "I was charged an overdraft fee", "please waive the overdraft fee",
        "reverse this thirty five dollar fee", "refund the fee you charged me",
        "the fee on my account is unfair",
    ],
    "account_issue": [
        "how do I close my account", "I want to dispute a charge I did not make",
        "I was charged twice for the same purchase", "tell me about account options",
        "I cannot access my account",
    ],
    "credit_card_issue": [
        "there is a problem with my credit card", "dispute a credit card charge",
        "my credit card was declined", "an error on my credit card bill",
    ],
    "mortgage_issue": [
        "there is a problem with my mortgage", "a question about my escrow",
        "my mortgage payment was misapplied", "an error in my home loan",
    ],
    "loan_issue": [
        "there is a problem with my loan", "my loan payment was not credited",
        "a question about my loan balance", "the interest charged on my loan is wrong",
    ],
    # Reachable abstain class: vague / unspecified problems that name no concrete
    # issue -- the 60%-of-errors trap was that this label was unreachable.
    "general": [
        "I have a general question", "something seems off but I am not sure what",
        "can you look into my situation", "I need help with something",
        "a procedural question about your process",
    ],
}


_CALIB_PATH = data_path("extract_geo_calibration.json")


def _rule_aux(message: str) -> dict[str, str]:
    """urgency/sentiment/summary -- not scored by the per-tool evaluator, reused
    verbatim from the keyword heuristic so the geometric path returns a complete,
    schema-valid ExtractOutput."""
    m = message.lower()
    sentiment = "negative" if any(k in m for k in (
        "angry", "frustrated", "terrible", "outraged", "unfair")) else "neutral"
    urgency = "high" if any(k in m for k in (
        "urgent", "immediately", "asap", "emergency")) else "normal"
    summary = message[:140] + ("..." if len(message) > 140 else "")
    return {"urgency": urgency, "sentiment": sentiment, "summary": summary}


class GeometricFactExtractor:
    """Classify product and issue geometrically against closed label vocabularies,
    each with its own calibrated abstain. ``owns`` selects which fields the
    geometric path produces; any field not owned is left to the caller (so a
    hybrid 'Qwen product + geometric issue' variant is just ``owns={'issue'}``)."""

    def __init__(
        self,
        encoder: Callable[[list[str]], Any] | None = None,
        *,
        product_threshold: float = 0.0,
        issue_threshold: float = 0.0,
        owns: set[str] | None = None,
    ) -> None:
        from knowlytix.knowledge.rag import GeometricLabelClassifier

        self.owns = owns or {"product", "issue"}
        self.product_clf = GeometricLabelClassifier(
            _PRODUCT_EXEMPLARS, encoder, threshold=product_threshold)
        self.issue_clf = GeometricLabelClassifier(
            _ISSUE_EXEMPLARS, encoder, threshold=issue_threshold)

    def classify_product(self, message: str) -> tuple[str, float]:
        """Classify the product against the closed exemplar set, abstaining to ``unknown``.

        Args:
            message: The customer complaint text.

        Returns:
            A ``(product, score)`` pair.
        """
        return self.product_clf.classify(message, abstain_label="unknown")

    def classify_issue(self, message: str) -> tuple[str, float]:
        """Classify the issue against the closed exemplar set, abstaining to ``general``.

        Args:
            message: The customer complaint text.

        Returns:
            An ``(issue, score)`` pair.
        """
        return self.issue_clf.classify(message, abstain_label="general")

    def extract(self, message: str) -> dict[str, Any] | None:
        """Return a schema-valid fact dict, classifying only the owned fields geometrically.

        Args:
            message: The customer complaint text; blank text returns None.

        Returns:
            A dict with urgency, sentiment, summary and the owned product/issue fields, or None.
        """
        if not message or not message.strip():
            return None
        out: dict[str, Any] = dict(_rule_aux(message))
        if "product" in self.owns:
            out["product"], _ = self.classify_product(message)
        if "issue" in self.owns:
            out["issue"], _ = self.classify_issue(message)
        return out


def _load_encoder(name: str | None):
    """Resolve the encoder by name: 'minilm' (default graph encoder) or 'tuned:
    <store_path>' for the GEODE document-tuned encoder."""
    if not name or name == "minilm":
        return None  # GeometricLabelClassifier falls back to encode_texts
    if name.startswith("tuned:"):
        from knowlytix.embedding import FineTunedEmbedding
        enc_dir = Path(name.split(":", 1)[1]) / "tuned_encoder"
        return FineTunedEmbedding.load(enc_dir).encode
    raise ValueError(f"unknown encoder spec: {name!r}")


# --- task-tuned issue classifier (the benchmark winner: low-rank FineTunedEmbedding
# nearest-prototype, no abstain -- `general` is a trained class). Used by
# banking_tools' geo_issue extract mode to replace the Qwen issue label while
# keeping Qwen for `product`. -----------------------------------------------------
_ISSUE_ENCODER_DIR = data_path("extract_encoder_issue")
_TASK_ISSUE = None


def get_task_issue_classifier():
    """Lazy singleton FineTunedEmbedding for the issue label. Raises if the
    artifact is absent (train it with scripts/finetune_extract_encoder.py)."""
    global _TASK_ISSUE
    if _TASK_ISSUE is None:
        if not (_ISSUE_ENCODER_DIR / "meta.json").exists():
            raise RuntimeError(
                f"task issue encoder missing at {_ISSUE_ENCODER_DIR}; "
                "run scripts/finetune_extract_encoder.py")
        from knowlytix.embedding import FineTunedEmbedding
        _TASK_ISSUE = FineTunedEmbedding.load(_ISSUE_ENCODER_DIR)
    return _TASK_ISSUE


def classify_issue_task(message: str, *, use_threshold: bool = False) -> str:
    """Task-geometric issue label for a message (nearest prototype; abstain off
    by default since the benchmark showed abstain does not help once `general`
    is a trained class). Falls back to ``general`` on an empty/None result."""
    clf = get_task_issue_classifier()
    label = clf.classify([message], use_threshold=use_threshold)[0][0]
    return label or "general"


_DEFAULT: GeometricFactExtractor | None = None


def get_default_extractor() -> GeometricFactExtractor:
    """Lazy singleton with thresholds loaded from the persisted calibration.
    Raises if the calibration file is absent -- a geometric abstain gate must be
    calibrated before use, never run on the 0.0 default in production."""
    global _DEFAULT
    if _DEFAULT is None:
        from agentlab._paths import ensure_default
        for _art in ("extract_geo_calibration.json", "extract_encoder_issue", "extract_encoder_product"):
            try:
                ensure_default(_art)
            except FileNotFoundError:
                pass  # calibration check below raises the actionable error
        if not _CALIB_PATH.exists():
            raise RuntimeError(
                f"geometric extractor is uncalibrated: {_CALIB_PATH} missing. "
                "Run scripts/calibrate_extract_geo.py first.")
        cal = json.loads(_CALIB_PATH.read_text())
        owns_env = os.environ.get("AGENTLAB_GEO_OWNS")  # e.g. "issue" or "product,issue"
        owns = set(owns_env.split(",")) if owns_env else {"product", "issue"}
        _DEFAULT = GeometricFactExtractor(
            encoder=_load_encoder(cal.get("encoder")),
            product_threshold=float(cal["product_threshold"]),
            issue_threshold=float(cal["issue_threshold"]),
            owns=owns,
        )
    return _DEFAULT
