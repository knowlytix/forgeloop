"""Claim extraction and evidence checking.

This is a deliberately naive baseline. Real groundedness checking uses an
NLI model or a verifier LM. The point of the chapter is to make the
verification step explicit — extract claims, attach evidence, classify —
not to ship a state-of-the-art verifier.
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
    text: str


class ClaimVerdict(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class GroundednessResult:
    claim: str
    verdict: ClaimVerdict
    evidence_id: str | None = None


def extract_claims(text: str) -> list[Claim]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return [Claim(text=s) for s in sentences]


def _content_terms(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z\-']+", text.lower())
    return {w for w in words if w not in _STOP_WORDS and len(w) > 2}


def check_groundedness(claim: Claim, evidence: dict[str, str]) -> GroundednessResult:
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
    return [check_groundedness(c, evidence) for c in claims]


def coverage(report: list[GroundednessResult]) -> float:
    if not report:
        return 0.0
    supported = sum(1 for r in report if r.verdict == ClaimVerdict.SUPPORTED)
    return supported / len(report)
