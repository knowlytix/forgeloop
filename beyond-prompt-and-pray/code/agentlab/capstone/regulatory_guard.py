"""GMS-backed deterministic guard for the flag_regulatory tool.

This is the deterministic half of the hybrid flag_regulatory tool, and a worked
example of *using a trained GMS store as a governance guard*. Qwen proposes
which regulations a complaint implicates; this guard consults the regulatory
knowledge graph (``data/gms_regulatory_store/``, built and calibrated by
``scripts/build_regulatory_guard_store.py`` + ``calibrate_regulatory_guard.py``)
to decide what may actually fire. It does two things:

  * **verify** — a Qwen-proposed flag is kept only if some evidence entity
    present in the message is graph-linked to that flag with a calibrated
    plausibility score (``score_triple <= theta``). Unsupported proposals are
    dropped, so no escalation rests on a signal the text does not carry.
  * **correct** — independently of Qwen, every flag the message's evidence
    supports is derived from the graph. This recovers the right regulation when
    the model mislabels it (e.g. an unauthorized debit-card charge is Reg E, not
    the Reg Z the model may guess).

Evidence detection is itself graph-driven: the customer-vocabulary aliases in
the store (``evidence has_alias phrase``) map message text onto the canonical
evidence entities the graph scores against.

Degrades safely: if the store or calibration is unavailable, ``load`` raises and
``banking_tools`` falls back to its regex evidence guard.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_STORE = _REPO_ROOT / "data" / "gms_regulatory_store"

# Tool-facing flag names <-> lower-case store entity ids.
_FLAG_TO_ENTITY = {"UDAAP": "udaap", "Reg_E": "reg_e", "Reg_Z": "reg_z", "Reg_X": "reg_x", "FCRA": "fcra"}
_ENTITY_TO_FLAG = {v: k for k, v in _FLAG_TO_ENTITY.items()}

# UDAAP is an unfair/deceptive/abusive *practice*, not a fee mention. The
# substring-correction path may assert it only when the message carries an
# in-context unfairness signal -- not from a bare "fee"/"overdraft" token, which
# is a routine reversal. A genuine UDAAP without these literal tokens still
# arrives via the in-context flagger proposal (verified against the graph).
_UDAAP_UNFAIRNESS = {"unfair", "deceptive", "hidden_fee"}
_UDAAP_FEE = {"fee", "overdraft", "hidden_fee"}


def _udaap_supported_by_evidence(evidence: set) -> bool:
    """UDAAP is an unfair/deceptive practice ABOUT A BANK FEE: it needs BOTH an
    unfairness signal AND a fee. An unfairness signal alone (a grievance, anger,
    a threat to sue) is not UDAAP without a fee; a fee alone is a routine
    reversal. Requiring co-occurrence is robust to a small LLM that over-applies
    \"unfair\" to any aggrieved message -- tightening the definition alone did not."""
    return bool(evidence & _UDAAP_UNFAIRNESS) and bool(evidence & _UDAAP_FEE)

# Per-entity definitions are the precision lever for the hybrid evidence step:
# they tell the LLM when an ambiguous mention *is* the entity and when it is not.
# Only the ambiguous entities need one; the rest rely on their name + aliases.
# The critical case is the bare "$35 charge": a bank-imposed fee (-> fee -> UDAAP)
# versus a merchant/transaction dispute (-> a Reg E/Z bucket, no escalation). The
# dispute entities are defined too, so the model has somewhere to route a stray
# charge other than forcing it into `fee` and over-firing UDAAP.
_EVIDENCE_DEFS = {
    "fee": ("a fee or charge IMPOSED BY THE BANK that the customer disputes as unfair "
            "(overdraft fee, NSF fee, maintenance fee, a $X charge the bank applied); "
            "a merchant purchase, a card transaction, an unauthorized or duplicate "
            "transaction, or a chargeback is NOT a fee"),
    "overdraft": "an overdraft, NSF or negative-balance situation",
    "unfair": ("the customer EXPLICITLY characterizes a specific bank fee or charge as "
               "unfair, unjust or wrongful (e.g. \"this overdraft fee is unfair\"); a "
               "general complaint, an angry tone, a demand for a refund or 'my money "
               "back', or a threat to sue is NOT, by itself, calling a bank charge unfair"),
    "deceptive": "the customer calls the bank's action deceptive",
    "hidden_fee": "the customer says a fee was hidden or undisclosed",
    "debit_fraud": ("an unauthorized or unrecognized charge or transaction on a debit "
                    "or bank card --- a purchase the customer did not make"),
    "unauthorized_transfer": "a transfer or transaction the customer did not authorize",
    "electronic_transfer": "an electronic payment or transfer dispute (e.g. a duplicate charge)",
    "credit_card_billing": "a credit-card transaction or billing dispute (not a bank fee)",
}

_EVIDENCE_TASK = (
    "You label which regulatory evidence signals a retail-bank customer message contains."
)


@dataclass
class GMSRegulatoryGuard:
    store: Any
    theta: float
    alias_to_evidence: dict[str, str]  # customer phrase (lower) -> canonical evidence entity
    evidence_by_flag: dict[str, list[str]]  # flag entity -> [evidence entity]
    # Lazily built context-aware evidence extractor (regex + LLM). Cached after
    # first use; not part of identity/repr. Sentinel ``None`` means "not built".
    _extractor: Any = field(default=None, init=False, repr=False, compare=False)

    @classmethod
    def load(
        cls,
        store_path: Path | None = None,
        device: torch.device | None = None,
    ) -> "GMSRegulatoryGuard":
        from knowlytix.knowledge.query import DocGMSConfig, GMSExpertStore

        store_path = Path(store_path) if store_path is not None else _DEFAULT_STORE
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        store = GMSExpertStore(DocGMSConfig(store_path=str(store_path)), device=device)
        if not store.load():
            raise RuntimeError(
                f"failed to load GMS regulatory store at {store_path!s}; "
                "run scripts/build_regulatory_guard_store.py first."
            )
        calib_path = store_path / "calibration.json"
        if not calib_path.exists():
            raise RuntimeError(
                f"missing {calib_path}; run scripts/calibrate_regulatory_guard.py first."
            )
        theta = float(json.loads(calib_path.read_text())["evidence_threshold"])

        alias_to_evidence: dict[str, str] = {}
        evidence_by_flag: dict[str, list[str]] = {}
        for h, r, t in store.triples:
            if r == "has_evidence":
                evidence_by_flag.setdefault(h, []).append(t)
                # The canonical entity is its own alias (spaces for underscores).
                alias_to_evidence.setdefault(t.replace("_", " "), t)
            elif r == "has_alias":
                alias_to_evidence[t.lower()] = h
        return cls(store=store, theta=theta, alias_to_evidence=alias_to_evidence, evidence_by_flag=evidence_by_flag)

    def _regex_evidence(self, message: str) -> set[str]:
        """Canonical evidence entities whose customer-vocabulary alias appears
        literally in the message. Deterministic and dependency-free; the fallback
        when the LLM is disabled or unavailable."""
        text = message.lower()
        found: set[str] = set()
        for phrase, entity in self.alias_to_evidence.items():
            if phrase and phrase in text:
                found.add(entity)
        return found

    def _build_extractor(self) -> Any:
        """Construct the hybrid (regex + context-aware LLM) evidence extractor
        from the store's own evidence vocabulary and alias lexicon.

        The regex half reproduces ``_regex_evidence`` exactly (same aliases, raw
        substring match), so with the LLM disabled the hybrid degrades to the old
        behavior. The LLM half disambiguates the cases regex cannot --- chiefly a
        bare "$35 charge" that is a bank fee versus a merchant transaction, and the
        in-context unfairness/deception signals UDAAP correction now requires ---
        guided by :data:`_EVIDENCE_DEFS`. ``AGENTLAB_USE_LLM_FLAG=0`` (the same
        toggle the Qwen flagger honors) keeps the guard fully offline/regex-only.
        """
        from agentlab.extraction import EntitySpec, HybridEntityExtractor

        aliases: dict[str, set[str]] = {}
        for phrase, ent in self.alias_to_evidence.items():
            aliases.setdefault(ent, set()).add(phrase)
        for ents in self.evidence_by_flag.values():
            for ent in ents:
                aliases.setdefault(ent, set()).add(ent.replace("_", " "))
        specs = [
            EntitySpec(name=ent, aliases=tuple(sorted(al)), definition=_EVIDENCE_DEFS.get(ent, ""))
            for ent, al in sorted(aliases.items())
        ]

        llm = None
        if os.environ.get("AGENTLAB_USE_LLM_FLAG", "1") != "0":
            try:
                from agentlab.models import QwenAdapter

                llm = QwenAdapter(max_new_tokens=64)
            except Exception:
                llm = None  # degrade to regex-only
        return HybridEntityExtractor(
            specs, llm=llm, task=_EVIDENCE_TASK, word_boundary=False
        )

    def evidence_in_message(self, message: str) -> set[str]:
        """Canonical evidence entities the message carries, via the hybrid
        extractor (regex literals unioned with context-aware LLM judgments).

        Degrades safely: any failure building or running the extractor falls back
        to the deterministic regex pass, so evidence detection never raises."""
        try:
            if self._extractor is None:
                self._extractor = self._build_extractor()
            return self._extractor.extract(message)
        except Exception:
            return self._regex_evidence(message)

    def _flag_supported(self, flag_entity: str, evidence: set[str]) -> bool:
        """True if any evidence entity in the message is graph-linked to the flag
        with a calibrated plausibility score (lower = more plausible)."""
        for ent in evidence:
            score = self.store.score_triple(flag_entity, "has_evidence", ent)
            if score is not None and float(score) <= self.theta:
                return True
        return False

    def _manifold_flags(self, message: str) -> set[str]:
        """High-severity flags the message fires *geometrically* --- the customer
        text, projected through the fine-tuned embedding into the GMS semantic
        sphere, falls inside the flag's ``has_evidence`` cap (calibrated per-flag
        threshold). This reads the message directly, recovering UDAAP / Reg_X cases
        the alias lexicon misses (misleading-clarity wording with no canonical
        token). Degrades safely to the empty set: if the cap artifact or its
        encoder is unavailable, or ``AGENTLAB_USE_MANIFOLD=0``, the guard falls back
        to the alias-lexicon evidence path alone.
        """
        if os.environ.get("AGENTLAB_USE_MANIFOLD", "1") == "0":
            return set()
        try:
            from agentlab.capstone.manifold_evidence import get_default_scorer

            scorer = get_default_scorer()
            return scorer.flags_for_message(message) if scorer is not None else set()
        except Exception:
            return set()

    def _manifold_entity_flags(self, message: str) -> set[str]:
        """High-severity flags recovered by the *windowed* manifold entity
        extractor: a local span of the message, projected through the SFT
        embedding, falls within the calibrated geodesic radius of the flag
        entity's own v-point (``entity_calibration.json``).

        OPT-IN, OFF BY DEFAULT (``AGENTLAB_USE_MANIFOLD_ENTITIES=1`` to enable).
        The DoE A/B (scripts/eval_manifold_entity_flags.py) found that, while this
        path lifts UDAAP/Reg_X recall sharply (misleading-clarity 11%->100%), the
        raw entity v-point is not a trained decision boundary the way the
        transported ``has_evidence`` cap in :meth:`_manifold_flags` is: it admits
        ~30% precision at *every* threshold, so unioning it in regresses overall
        escalation accuracy by falsely escalating benign look-alikes. Kept wired
        for experimentation, but the cap path (:meth:`_manifold_flags`) is the
        default geometric recall path. Degrades safely to the empty set.
        """
        if os.environ.get("AGENTLAB_USE_MANIFOLD_ENTITIES", "0") == "0":
            return set()
        try:
            from agentlab.capstone.manifold_entities import get_default_extractor

            ext = get_default_extractor()
            if ext is None:
                return set()
            return {_ENTITY_TO_FLAG[e] for e in ext.extract(message) if e in _ENTITY_TO_FLAG}
        except Exception:
            return set()

    def escalation_for_flags(self, flags: list[str]) -> tuple[bool, list[dict[str, str]]]:
        """Decide escalation by walking the graph, not a hard-coded set.

        For each fired flag, traverse the two-hop severity path
        ``flag -> has_severity -> severity -> has_action -> action``. The case
        escalates iff some path lands on the ``escalate`` action (i.e. a
        ``high``-severity flag). Returns the decision and the traversed paths so
        the audit log records *why* the agent escalated.
        """
        paths: list[dict[str, str]] = []
        escalate = False
        for flag in flags:
            ent = _FLAG_TO_ENTITY.get(flag)
            if ent is None:
                continue
            for _h, _r, sev in self.store.query_triples(head=ent, relation="has_severity"):
                for _h2, _r2, act in self.store.query_triples(head=sev, relation="has_action"):
                    paths.append({"flag": flag, "severity": sev, "action": act})
                    if act == "escalate":
                        escalate = True
        return escalate, paths

    def verify_and_correct(self, proposed: list[str], message: str) -> dict[str, Any]:
        """Return the graph-sanctioned flag set, escalation decision and audit trail.

        ``flags`` = flags the message's evidence supports (correction) unioned
        with any Qwen proposal the graph confirms (verification). Proposals the
        graph cannot support are reported under ``rejected``. ``escalate`` is
        derived by multi-hop traversal of the severity path (see
        ``escalation_for_flags``), so the escalation policy lives in the
        calibrated graph rather than as a constant in code.
        """
        evidence = self.evidence_in_message(message)
        graph_flags: set[str] = set()
        for flag_entity in self.evidence_by_flag:
            if not self._flag_supported(flag_entity, evidence):
                continue
            # UDAAP from substring correction requires an in-context unfairness
            # signal; a bare fee/overdraft mention is a routine dispute, not an
            # unfair/deceptive practice. (A genuine UDAAP still arrives via the
            # verified in-context flagger proposal below.)
            if flag_entity == "udaap" and not _udaap_supported_by_evidence(evidence):
                continue
            graph_flags.add(_ENTITY_TO_FLAG.get(flag_entity, flag_entity))

        verified: list[str] = []
        rejected: list[str] = []
        for p in proposed:
            ent = _FLAG_TO_ENTITY.get(p)
            if ent is not None and self._flag_supported(ent, evidence):
                verified.append(p)
            else:
                rejected.append(p)

        # Geometric recall: high-severity flags the fine-tuned embedding fires by
        # reading the message itself, unioned with the lexicon/graph paths above.
        # Two complementary geometries: whole-message transported cap
        # (_manifold_flags) and windowed entity-point (_manifold_entity_flags).
        manifold_flags = self._manifold_flags(message) | self._manifold_entity_flags(message)
        flags = sorted(graph_flags | set(verified) | manifold_flags)
        # UDAAP from the alias lexicon requires an unfairness signal AND a fee; drop a
        # lexicon-only UDAAP lacking either (e.g. a litigation threat with no fee). A
        # UDAAP the manifold fired is exempt --- the calibrated cap is its precision
        # check, and it is exactly the misleading-clarity case the lexicon cannot match.
        if ("UDAAP" in flags and "UDAAP" not in manifold_flags
                and not _udaap_supported_by_evidence(evidence)):
            flags.remove("UDAAP")
            rejected.append("UDAAP")
        escalate, paths = self.escalation_for_flags(flags)
        return {
            "flags": flags,
            "escalate": escalate,
            "severity_paths": paths,
            "evidence": sorted(evidence),
            "graph_derived": sorted(graph_flags),
            "manifold_flags": sorted(manifold_flags),
            "verified": sorted(set(verified)),
            "rejected": sorted(set(rejected)),
        }


_DEFAULT_GUARD: GMSRegulatoryGuard | None = None


def get_default_guard() -> GMSRegulatoryGuard:
    global _DEFAULT_GUARD
    if _DEFAULT_GUARD is None:
        _DEFAULT_GUARD = GMSRegulatoryGuard.load()
    return _DEFAULT_GUARD
