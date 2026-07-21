"""Structured scratchpad: typed reasoning entries with evidence pointers.

The point of a scratchpad is to replace free-form chain-of-thought with
records the rest of the system can inspect. Trust levels are enforced in
code so downstream consumers cannot accidentally treat an assumption as
an observation.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TrustLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    HIGHEST = "highest"


class EntryType(str, Enum):
    CLAIM = "claim"
    ASSUMPTION = "assumption"
    OBSERVATION = "observation"
    QUESTION = "question"


class Entry(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    kind: EntryType
    text: str
    evidence: str | None = None
    source: str | None = None
    trust: TrustLevel = TrustLevel.LOW
    timestamp: float = Field(default_factory=time.time)

    @field_validator("text")
    @classmethod
    def _text_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("entry text must be non-empty")
        return v

    @model_validator(mode="after")
    def _check_invariants(self) -> "Entry":
        if self.kind == EntryType.OBSERVATION and not self.source:
            raise ValueError("observations require a source")
        if self.trust != TrustLevel.LOW and not self.evidence:
            raise ValueError("entries with trust above LOW require an evidence pointer")
        return self


class Scratchpad:
    """Append-only collection of typed reasoning entries."""

    def __init__(self) -> None:
        self._entries: list[Entry] = []

    @property
    def entries(self) -> tuple[Entry, ...]:
        return tuple(self._entries)

    def add_claim(
        self,
        text: str,
        evidence: str | None = None,
        trust: TrustLevel = TrustLevel.LOW,
    ) -> Entry:
        e = Entry(kind=EntryType.CLAIM, text=text, evidence=evidence, trust=trust)
        self._entries.append(e)
        return e

    def add_assumption(self, text: str) -> Entry:
        e = Entry(kind=EntryType.ASSUMPTION, text=text, trust=TrustLevel.LOW)
        self._entries.append(e)
        return e

    def add_observation(
        self,
        text: str,
        source: str,
        evidence: str | None = None,
        trust: TrustLevel = TrustLevel.LOW,
    ) -> Entry:
        e = Entry(
            kind=EntryType.OBSERVATION,
            text=text,
            source=source,
            evidence=evidence,
            trust=trust,
        )
        self._entries.append(e)
        return e

    def add_question(self, text: str) -> Entry:
        e = Entry(kind=EntryType.QUESTION, text=text, trust=TrustLevel.LOW)
        self._entries.append(e)
        return e

    def by_type(self, kind: EntryType) -> list[Entry]:
        return [e for e in self._entries if e.kind == kind]

    def unsupported_claims(self) -> list[Entry]:
        return [e for e in self._entries if e.kind == EntryType.CLAIM and not e.evidence]

    def assert_all_claims_have_evidence(self) -> None:
        bad = self.unsupported_claims()
        if bad:
            raise AssertionError(
                f"{len(bad)} claim(s) without evidence: {[e.text for e in bad]}"
            )

    def render_table(self, as_string: bool = False) -> list[dict[str, Any]] | str:
        rows = [
            {
                "id": e.id,
                "kind": e.kind.value,
                "trust": e.trust.value,
                "text": e.text,
                "evidence": e.evidence or "",
                "source": e.source or "",
            }
            for e in self._entries
        ]
        if not as_string:
            return rows
        if not rows:
            return "(empty scratchpad)"
        cols = ["id", "kind", "trust", "text", "evidence", "source"]
        widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
        header = " | ".join(c.ljust(widths[c]) for c in cols)
        sep = "-+-".join("-" * widths[c] for c in cols)
        body = "\n".join(
            " | ".join(str(r[c]).ljust(widths[c]) for c in cols) for r in rows
        )
        return f"{header}\n{sep}\n{body}"
