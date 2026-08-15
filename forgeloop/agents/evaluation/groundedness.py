"""Claim extraction and lexical groundedness — a labeled foil.

`check_groundedness` scores a claim against evidence by content-term overlap.
It is a lexical baseline, kept only to make the verification pattern explicit —
extract claims, attach evidence, classify — and to contrast with the check the
rest of the stack actually uses.

The recommended groundedness check is geometric, not lexical and not an
LLM-as-judge. A claim reduced to a triple is scored by its geodesic distance to
a trained GMS store (`GMSMemory.score_triple`, "groundedness as distance",
Chapter 10): a continuous score, small for a claim the store recognizes and
large for a fabricated one. That score is the same primitive that gates a tool
call and a drafted answer at run time, and it is replayable bit-for-bit, which
neither term overlap nor a model judging a model provides.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

_STOP_WORDS = frozenset(
    {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "and", "or", "but", "if", "then", "of", "to", "in", "on", "at", "by",
        "for", "with", "from", "as", "this", "that", "these", "those", "it",
        "its", "into", "about", "over", "under", "up", "down", "out", "do",
        "does", "did", "have", "has", "had", "will", "would", "should", "could",
        "may", "might", "can", "not", "no", "so", "than", "too", "very", "i",
        "you", "he", "she", "we", "they", "them", "his", "her", "their",
    }
)


@dataclass(frozen=True)
class Claim:
    """A single extracted claim.

    Attributes:
        text: The claim sentence.
    """

    text: str


class ClaimVerdict(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class GroundednessResult:
    """The verdict for one claim against the evidence.

    Attributes:
        claim: The claim text that was checked.
        verdict: The groundedness verdict for the claim.
        evidence_id: Identifier of the best-matching evidence, or None.
    """

    claim: str
    verdict: ClaimVerdict
    evidence_id: str | None = None


def extract_claims(text: str) -> list[Claim]:
    """Split text into sentence-level claims.

    Args:
        text: The text to split.

    Returns:
        One Claim per non-empty sentence.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return [Claim(text=s) for s in sentences]


def _content_terms(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z\-']+", text.lower())
    return {w for w in words if w not in _STOP_WORDS and len(w) > 2}


def check_groundedness(claim: Claim, evidence: dict[str, str]) -> GroundednessResult:
    """Classify a claim as supported or unsupported by content-term overlap.

    This is the lexical foil described in the module docstring. The recommended
    groundedness check is geometric (`GMSMemory.score_triple`); this function is
    kept only for the Chapter 10 contrast.

    Args:
        claim: The claim to check.
        evidence: A mapping of evidence id to evidence text.

    Returns:
        A GroundednessResult; SUPPORTED with the best evidence id when overlap
        meets the threshold, UNSUPPORTED otherwise, and UNCERTAIN when the claim
        carries no content terms.
    """
    claim_terms = _content_terms(claim.text)
    if not claim_terms:
        return GroundednessResult(claim=claim.text, verdict=ClaimVerdict.UNCERTAIN)
    best_id = None
    best_overlap = 0
    for ev_id, ev_text in evidence.items():
        ev_terms = _content_terms(ev_text)
        overlap = len(claim_terms & ev_terms)
        if overlap > best_overlap:
            best_overlap = overlap
            best_id = ev_id
    threshold = max(2, len(claim_terms) // 2)
    if best_overlap >= threshold:
        return GroundednessResult(claim=claim.text, verdict=ClaimVerdict.SUPPORTED, evidence_id=best_id)
    return GroundednessResult(claim=claim.text, verdict=ClaimVerdict.UNSUPPORTED, evidence_id=best_id)


def groundedness_report(claims: list[Claim], evidence: dict[str, str]) -> list[GroundednessResult]:
    """Return the groundedness result for each claim against the evidence.

    Args:
        claims: The claims to check.
        evidence: A mapping of evidence id to evidence text.

    Returns:
        One GroundednessResult per claim.
    """
    return [check_groundedness(c, evidence) for c in claims]


def coverage(report: list[GroundednessResult]) -> float:
    """Return the fraction of claims classified as supported.

    Args:
        report: The groundedness results to summarize.

    Returns:
        The supported fraction, or 0.0 for an empty report.
    """
    if not report:
        return 0.0
    supported = sum(1 for r in report if r.verdict == ClaimVerdict.SUPPORTED)
    return supported / len(report)
