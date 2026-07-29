"""Hybrid (regex + context-aware LLM) extraction of canonical entities from text.

Pure-regex / alias matching is brittle: it misses an entity phrased in unexpected
words and, if you widen the aliases, it over-fires on look-alikes (a "$35 charge"
that is a bank fee versus one that is a merchant transaction). A pure-LLM pass is
the opposite trade --- flexible but non-deterministic and prone to over-extraction.

``HybridEntityExtractor`` combines both: cheap deterministic literal/alias matching
for the unambiguous cases, plus an LLM that judges the ambiguous mentions *in
context*, constrained by per-entity definitions the caller supplies. The result is
the union --- regex recall on literals, LLM disambiguation on the rest.

The component is domain-agnostic (the caller provides the vocabulary, aliases and
definitions), deterministic when the LLM decodes greedily, and degrades to
regex-only when no LLM is given or the call fails. It performs *extraction* only;
verifying the extracted entities against a knowledge graph (e.g. a GMS
``score_triple`` gate) is the caller's job, so a precision error in the LLM is the
caller's to backstop.

Dependencies are the standard library plus a duck-typed LLM with a
``complete(prompt) -> str`` method, so the module is self-contained and
upstreamable (e.g. into ``knowlytix.knowledge.extract``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable, Protocol


class _LLM(Protocol):
    def complete(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class EntitySpec:
    """One canonical entity in the vocabulary.

    Attributes:
        name: canonical entity id (returned by :meth:`HybridEntityExtractor.extract`).
        aliases: literal surface phrases matched cheaply and deterministically.
        definition: natural-language guidance that tells the LLM when this entity is
            (and is not) present --- the lever that controls precision on ambiguous
            mentions. Empty means "rely on the name and aliases only".
    """

    name: str
    aliases: tuple[str, ...] = ()
    definition: str = ""


class HybridEntityExtractor:
    """Extract the canonical entities a text genuinely contains, regex + LLM.

    Args:
        entities: the vocabulary, as :class:`EntitySpec` records.
        llm: optional object with ``complete(prompt) -> str`` (greedy for
            determinism). When ``None``, extraction is regex-only.
        task: one-line description of what is being extracted, prepended to the
            LLM prompt.
        word_boundary: wrap each alias in word boundaries (``True``, avoids
            ``fee`` matching ``coffee``) or match as a raw substring (``False``).
    """

    def __init__(
        self,
        entities: Iterable[EntitySpec],
        llm: _LLM | None = None,
        *,
        task: str = "",
        word_boundary: bool = True,
    ) -> None:
        self.entities = list(entities)
        self.vocab = {e.name for e in self.entities}
        self.llm = llm
        self.task = task or (
            "Identify which of the listed signals the text genuinely contains, "
            "judging meaning in context."
        )
        self._patterns: list[tuple[re.Pattern[str], str]] = []
        for e in self.entities:
            for alias in e.aliases:
                if not alias:
                    continue
                body = re.escape(alias)
                pat = rf"\b{body}\b" if word_boundary else body
                self._patterns.append((re.compile(pat, re.IGNORECASE), e.name))

    def regex_extract(self, text: str) -> set[str]:
        """Entities whose literal alias appears in the text (deterministic)."""
        return {name for pat, name in self._patterns if pat.search(text)}

    def llm_extract(self, text: str) -> set[str]:
        """Entities the LLM judges present in context, filtered to the vocabulary.

        Returns an empty set (never raises) when no LLM is configured or the call
        or its parse fails, so the hybrid degrades cleanly to regex-only.
        """
        if self.llm is None:
            return set()
        try:
            out = self.llm.complete(self._prompt(text))
        except Exception:
            return set()
        match = re.search(r"\[.*?\]", out, re.S)
        if not match:
            return set()
        try:
            items = json.loads(match.group(0))
        except Exception:
            return set()
        return {str(x).strip().lower() for x in items if str(x).strip().lower() in self.vocab}

    def extract(self, text: str) -> set[str]:
        """The hybrid result: regex (literal) hits unioned with LLM (context) hits."""
        return self.regex_extract(text) | self.llm_extract(text)

    def _prompt(self, text: str) -> str:
        defs = "\n".join(f"- {e.name}: {e.definition}" for e in self.entities if e.definition)
        lines = [self.task, f"Allowed signals (use only these): {sorted(self.vocab)}"]
        if defs:
            lines += ["Definitions:", defs]
        lines += [
            'Include a signal ONLY if the text supports it. Reply with a JSON list, '
            'for example ["a","b"], or [] if none.',
            "",
            f"Text: {text}",
            "Signals:",
        ]
        return "\n".join(lines)
