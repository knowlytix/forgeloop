"""Graph-RAG retriever for search_policy (GEODE triple-mediated RAG).

The retriever drives knowlytix's GEODE graph-RAG pipeline
(``knowlytix.knowledge.rag.RagPipeline``) over a self-corrected GMS built from
``data/banking_policy.md``. Unlike the previous retriever -- which voted a
*single* policy id and synthesized from that one policy's prose (so any question
whose answer spanned several policy fields came back ``INCOMPLETE``) -- the
pipeline is triple-mediated:

  1. the question is parsed into query triples (``overdraft -> has_fee_amount ->
     ?``), bound to the graph's real vocabulary (embedding binding, so customer
     wording maps onto canonical relations),
  2. answered through the GMS -- asserted edges and exact numeric registers
     (ENM) preferred, multi-hop chains resolved -- with each fact carrying its
     provenance span,
  3. synthesized by Qwen from *all* retrieved facts + spans, then self-verified
     against the GMS (claims that contradict the graph downgrade confidence).

It abstains rather than guess when nothing binds. The store is built by
``scripts/build_geode_rag_store.py``; the source doc's wide tables (Fee Schedule,
Regulatory Flags, Reversal Authority) give the regex ingester entity-headed,
typed triples that triple-mediated retrieval can actually bind to.

``search`` keeps the original return shape -- ``[{"id", "text", "answer",
"score", ...}]`` -- so the ``search_policy`` tool and the draft verifier (which
keys fee ENMs off the returned policy id) are unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from knowlytix.knowledge.geode import QWEN_4B
from knowlytix.knowledge.llm_backend import LocalTransformersBackend
from knowlytix.knowledge.query import DocGMSConfig, GMSExpertStore
from knowlytix.knowledge.rag import Extraction

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_STORE = _REPO_ROOT / "data" / "gms_policy_store_cap"

# Query-time LLM for the RAG pipeline (extract / synthesize / verify). The local
# open-weight Qwen3-4B-Instruct, the same backend the grounded-synthesis chapter
# of *Beyond Chunk and Pray* uses. GMS does the retrieval; this model only reads
# the retrieved facts + provenance spans into a grounded answer, constrained by
# the assembler's evidence block. Override the id with AGENTLAB_RAG_LLM.
_RAG_LLM_MODEL = os.environ.get("AGENTLAB_RAG_LLM", QWEN_4B)

# Logical policy ids the draft verifier knows (draft_verifier._POLICY_FEE_ENM).
# Reversal-authority facts map to the "fee_reversal" policy regardless of which
# role entity (representative/supervisor/...) headed the retrieved triple.
_REVERSAL_RELATIONS = {"has_max_reversal"}


class PolicyRagRetriever:
    def __init__(
        self,
        store_path: Path | None = None,
        policies_dir: Path | None = None,  # kept for signature compatibility
        device: torch.device | None = None,
        binding: str = "embedding",
    ) -> None:
        from knowlytix.knowledge.rag import RagConfig, RagPipeline

        store_path = Path(store_path) if store_path is not None else _DEFAULT_STORE
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = DocGMSConfig(store_path=str(store_path), ingest_mode="regex")
        store = GMSExpertStore(config, device=device)
        if not store.load():
            raise RuntimeError(
                f"failed to load GEODE RAG store at {store_path!s}; "
                "run scripts/build_geode_rag_store.py first."
            )
        # Inject the corpus customer-alias table into the served doc_graph as
        # has_alias edges, for OPERATOR-NATIVE head binding only. The parser reads
        # has_alias TAILS as entity-alias NAMES (head_candidates) and excludes
        # has_alias from content retrieval, so "social security number" binds
        # pii_handling without the alias tails ever becoming head candidates or
        # model entities. Injected AFTER load (the GMS model/adapter stay at the
        # trained entity set -- alias tails are never scored), so this needs no
        # rebuild and carries no model mismatch. The reg aliases are de-collided in
        # the corpus (reg_e has_alias "regulation e"), so there is no binding
        # collision. This is the lever that lifts head-bind recall to ~1.0 at zero
        # false-accept. Skipped silently when the corpus is absent.
        self._inject_aliases(store, store_path)
        self.store = store
        # OPERATOR-NATIVE configuration. Embeddings bind only the HEAD ENTITY;
        # relations are geometric operators, never matched by phrase. The pipeline
        # binds the head, retrieves ALL of that head's admissible facts through the
        # operators (top n_heads heads, head_hops expanded for multi-hop), and lets
        # select-and-answer synthesis pick the attribute or abstain. This retires
        # the embedding relevance gate, whose v-accept floor + u-veto could not
        # separate an absent-but-adjacent attribute ("overdraft interest rate")
        # from a held one ("overdraft fee") at the query distribution -- both bind
        # the same nearest relation. Under head_facts that question retrieves all
        # of overdraft's facts and synthesis abstains because none answers it.
        #  * dense_fallback off + strict_mode -> the opt-in dense vector index is
        #    never used; retrieval is GMS-only.
        #  * verify_llm_output + on_verify_fail="abstain" + geometric verifier ->
        #    a GATED, direction-tolerant check of the answer's claims against the
        #    GMS; a contradicted claim abstains rather than ships.
        #  * accept_threshold (tau) -> a CALIBRATED gate on retrieval confidence,
        #    loaded from the store (0.0 until calibrate_policy_rag.py has run).
        #  * head_bind_floor -> the CALIBRATED operating point that REPLACES the
        #    relevance gate: a question whose best head-name cosine is below it
        #    names no policy entity, so the pipeline abstains. Spurious-but-
        #    plausible binds are caught downstream by select-and-answer synthesis.
        tau = self._load_tau(store_path)
        # Document-tuned encoder from the GEODE embed loop (alias-supervised, Ch7):
        # places a colloquial customer phrasing near the policy entity it denotes,
        # so head binding maps "I overdrew my account" -> overdraft. Falls back to
        # the default MiniLM when no tuned encoder is present.
        encoder = self._load_tuned_encoder(store_path)
        head_bind_floor = self._load_head_bind_floor(store_path)
        # u-space verifier POLARITY check: DISABLED by default. The contradiction
        # encoder cleanly separates polarity on short value tokens
        # (forbidden<->permitted u-tension 0.95), but it was trained on short
        # relation phrasings, so on FULL LLM answer sentences the tension is
        # out-of-distribution and uninformative -- consistent answers score as high
        # as contradictory ones (measured: consistent max 1.27 > flipped min 0.63,
        # no separating cut), and it false-flags via v-space mis-selecting the
        # answering fact. A working version must feed the check the EXTRACTED value
        # token (compare the answer's asserted value-polarity to the stored value),
        # not the whole sentence -- left for the proper fix. Opt in for experiments
        # with AGENTLAB_RAG_TAU_POLARITY (else the check stays off).
        verify_u = (self._load_contradiction_encoder(store_path)
                    if "AGENTLAB_RAG_TAU_POLARITY" in os.environ else None)
        tau_polarity = float(os.environ.get("AGENTLAB_RAG_TAU_POLARITY", "9.9"))
        # FUSED value-polarity check (the adopted improvement from *Beyond Chunk and
        # Pray* Ch10/Ch11). When the verifier decomposes the answer into claims, a
        # CATEGORICAL claim's asserted value (a stance: forbidden/permitted,
        # required/optional, issued/denied) is checked three ways against the single
        # stored value -- cap plausibility (v-space), nearest-entity resolution
        # (synonym tolerance), and 3-class u-space tension (polarity) -- and the
        # answer abstains when the stance is reversed or the channel cannot resolve
        # it. This is what the retired naive u-veto above could not do: it separates
        # "prohibited" (synonym of the stored forbidden -> supported) from
        # "permitted" (reversal -> contradicted). The polarity u-encoder and its
        # CV-calibrated cuts are persisted in the store (value_polarity_encoder/ +
        # value_polarity_calibration.json, built by
        # scripts/build_policy_value_polarity.py). Both must be present for the
        # pipeline to build the ValuePolarityChecker; absent, the verifier's value
        # check is cap-only, unchanged.
        value_polarity_encoder = self._load_value_polarity_encoder(store_path)
        polarity_cuts = self._load_polarity_cuts(store_path)
        # Qwen3-4B-Instruct local backend (select-synthesize/verify), loaded once
        # for the process-cached retriever. The select assembler reads the head's
        # retrieved facts + spans, picks the answering fact(s) or abstains; it
        # never decides retrieval or admissibility.
        self.llm = LocalTransformersBackend(_RAG_LLM_MODEL, device=str(device))
        rag = RagConfig(
            llm=self.llm,
            binding=binding,
            encoder=encoder,
            # Operator-native retrieval: bind head, retrieve all its facts.
            retrieve_mode="head_facts",
            head_bind_floor=head_bind_floor,
            n_heads=int(os.environ.get("AGENTLAB_RAG_N_HEADS", "2")),
            head_hops=int(os.environ.get("AGENTLAB_RAG_HEAD_HOPS", "2")),
            # Geometric query parsing (required for head_candidates / head_facts);
            # falls back to the LLM parser only when no tuned encoder is present.
            query_parse_mode="geometric" if encoder is not None else "llm",
            relevance_gate=False,           # retired under operator-native
            dense_fallback=False,
            strict_mode=True,
            # Provenance = the source SENTENCE that states each fact (readable
            # policy prose), not the table cell -- so synthesis restates the policy
            # ("... is forbidden") instead of decoding a relation name from a value.
            provenance_prefer_prose=True,
            verify_llm_output=os.environ.get("AGENTLAB_RAG_VERIFY", "1") != "0",
            on_verify_fail="abstain",
            verify_u_encoder=verify_u,
            verify_tau_polarity=tau_polarity,
            # Fused categorical-value polarity check (built by the pipeline only when
            # both are set; a reversed stance -> contradicted -> abstain).
            verify_value_polarity_encoder=value_polarity_encoder,
            verify_polarity_cuts=polarity_cuts,
            accept_threshold=tau,
        )
        self.value_polarity = (value_polarity_encoder is not None
                               and polarity_cuts is not None)
        self.accept_threshold = tau
        self.tuned_encoder = encoder is not None
        self.pipe = RagPipeline.from_store(store, rag)
        # Self-verify by HYBRID claim decomposition: geometric extraction (walk the
        # store graph) confirms the answer states real edges -- a correct answer
        # binds and verifies -- while a VOCAB-CONSTRAINED LLM decomposition carries
        # each claim's ASSERTED value, so a value/polarity reversal the geometric
        # pass misses (it keys on the stored tail appearing in the text, which a
        # flipped answer never contains) binds to a real (head, relation) with a
        # mismatching tail and returns "contradicted" -> abstain. An invented
        # relation binds to nothing -> "unverifiable" (advisory). Default is
        # "geometric": on this terse banking corpus the LLM decomposition pass
        # occasionally mislabels a paraphrased value and false-fails a correct
        # answer, and the failure it would catch (a clean value/polarity flip) is
        # better fixed at synthesis (the SSN "can be sent ... forbidden" case is a
        # self-contradiction the decomposer reads as supported). Opt into hybrid
        # decomposition with AGENTLAB_RAG_VERIFY_MODE=hybrid where clean flips matter.
        if getattr(self.pipe, "verifier", None) is not None:
            self.pipe.verifier.mode = os.environ.get(
                "AGENTLAB_RAG_VERIFY_MODE", "geometric")

        # Cap (v-space) + tension (u-space) admissibility -- the third orthogonal
        # gate, via knowlytix's reusable filter_admissible_facts (logic stays in
        # knowlytix, we just invoke it). Calibrate the per-head cap margins +
        # tension cutoff ONCE here (amortized across queries); a no-op if the
        # store's model is not cap-trained. Catches the fabrication / cross-tail
        # leaks (a bound query whose retrieved triple is geometrically
        # implausible) that the confidence/relevance gates do not.
        self._cap_margins = self._tension_tau = None
        _model = getattr(store, "model", None)
        if getattr(_model, "cap_enabled", False):
            from knowlytix.core.graph.admissibility import (
                calibrate_cap_margins_per_head,
                calibrate_tension_threshold,
            )
            self._cap_margins = calibrate_cap_margins_per_head(_model, store.adapter)
            self._tension_tau = calibrate_tension_threshold(_model, store.adapter)

    @staticmethod
    def _load_tau(store_path: Path) -> float:
        """Calibrated accept-threshold from the store, or 0.0 if not yet fit.
        AGENTLAB_RAG_ACCEPT_THRESHOLD overrides it -- the calibration loop sets it
        to 0.0 so the gate does not pre-filter the cohort it is being fit on."""
        import json

        tau = 0.0
        cal = Path(store_path) / "rag_gate_calibration.json"
        if cal.exists():
            try:
                tau = float(json.loads(cal.read_text())["accept_threshold"])
            except (ValueError, KeyError, TypeError):
                pass
        return float(os.environ.get("AGENTLAB_RAG_ACCEPT_THRESHOLD", tau))

    @staticmethod
    def _inject_aliases(store, store_path: Path) -> int:
        """Add the corpus Policy Aliases table to the served ``doc_graph`` as
        has_alias edges (entity-alias NAMES for operator-native head binding).
        Reads ``data/banking_policy_full.md`` (env AGENTLAB_RAG_CORPUS overrides;
        else the repo corpus next to the store). Idempotent and best-effort: a
        missing corpus or an already-present edge is skipped. Returns the count
        added."""
        import re

        corpus = os.environ.get("AGENTLAB_RAG_CORPUS")
        cand = [Path(corpus)] if corpus else []
        cand += [_REPO_ROOT / "data" / "banking_policy_full.md",
                 Path(store_path).parent / "banking_policy_full.md"]
        md = next((p for p in cand if p.exists()), None)
        if md is None or getattr(store, "doc_graph", None) is None:
            return 0
        existing = {(h, t) for h, r, t in store.doc_graph.triples if r == "has_alias"}
        in_tbl, added = False, 0
        for ln in md.read_text().splitlines():
            low = ln.strip().lower()
            # The alias glossary heading: the old table corpus used "## Policy
            # Aliases"; the realistic policy corpus titles it "## Appendix A:
            # Customer Terminology and Aliases". Match either -- any level-2
            # heading naming aliases or customer terminology.
            if low.startswith("## ") and ("alias" in low or "customer terminology" in low):
                in_tbl = True
                continue
            if in_tbl and ln.startswith("## "):
                break
            m = re.match(r"\|\s*([a-z0-9_]+)\s*\|\s*([^|]+?)\s*\|", ln)
            if in_tbl and m and m.group(1) != "policy_id":
                h, a = m.group(1), m.group(2).strip()
                if (h, a) not in existing:
                    try:
                        store.doc_graph.add_triple(h, "has_alias", a)
                        added += 1
                    except Exception:  # noqa: BLE001 - aliases are best-effort
                        pass
        return added

    @staticmethod
    def _load_head_bind_floor(store_path: Path) -> float:
        """Calibrated operator-native head-bind floor from the store: the min
        head-name cosine below which a question names no policy entity and the
        pipeline abstains. Read from head_bind_calibration.json (env-overridable),
        0.0 until calibrate_policy_rag.py has run -- so an uncalibrated store
        binds permissively and leans on select-and-answer synthesis to abstain."""
        import json

        floor = 0.0
        cal = Path(store_path) / "head_bind_calibration.json"
        if cal.exists():
            try:
                floor = float(json.loads(cal.read_text())["head_bind_floor"])
            except (ValueError, KeyError, TypeError):
                pass
        return float(os.environ.get("AGENTLAB_RAG_HEAD_BIND_FLOOR", floor))

    @staticmethod
    def _load_bind_params(store_path: Path) -> tuple[float, float]:
        """Calibrated (bind_threshold, bind_margin) from the store, env-overridable,
        falling back to the RagConfig defaults (0.5, 0.05) until calibration."""
        import json

        th, mg = 0.5, 0.05
        cal = Path(store_path) / "rag_gate_calibration.json"
        if cal.exists():
            try:
                d = json.loads(cal.read_text())
                th = float(d.get("bind_threshold", th))
                mg = float(d.get("bind_margin", mg))
            except (ValueError, KeyError, TypeError):
                pass
        th = float(os.environ.get("AGENTLAB_RAG_BIND_THRESHOLD", th))
        mg = float(os.environ.get("AGENTLAB_RAG_BIND_MARGIN", mg))
        return th, mg

    @staticmethod
    def _load_tuned_encoder(store_path: Path):
        """The GEODE embed loop's document-tuned encoder as a ``list[str] ->
        Tensor`` callable for RagConfig.encoder, or None (default MiniLM) when
        the store carries no ``tuned_encoder/``."""
        enc_dir = Path(store_path) / "tuned_encoder"
        if not (enc_dir / "meta.json").exists():
            return None
        try:
            from knowlytix.embedding import FineTunedEmbedding

            return FineTunedEmbedding.load(enc_dir).encode
        except Exception:  # noqa: BLE001 - tuned encoder is an optimization
            return None

    @staticmethod
    def _load_relevance_calibration(store_path: Path) -> dict:
        """Calibrated geometric-relevance operating points from the store, or
        conservative defaults (accept any bound relation, never veto) when not
        yet fit -- so an uncalibrated store never silently over-rejects."""
        import json

        cal = Path(store_path) / "relevance_calibration.json"
        if cal.exists():
            try:
                d = json.loads(cal.read_text())
                return {"tau_accept": float(d.get("tau_accept", 0.30)),
                        "default_tau_contra": float(d.get("default_tau_contra", 2.0)),
                        "tau_contra_per_relation": d.get("tau_contra_per_relation", {})}
            except (ValueError, KeyError, TypeError):
                pass
        return {"tau_accept": 0.30, "default_tau_contra": 2.0,
                "tau_contra_per_relation": {}}

    @staticmethod
    def _load_contradiction_encoder(store_path: Path):
        """The GEODE u-space contradiction encoder for the geometric relevance
        veto, or None (fall back to the LLM relevance judge)."""
        enc_dir = Path(store_path) / "contradiction_encoder"
        if not (enc_dir / "meta.json").exists():
            return None
        try:
            from knowlytix.embedding import FineTunedEmbedding

            return FineTunedEmbedding.load(enc_dir).encode
        except Exception:  # noqa: BLE001 - u-veto is an optimization
            return None

    @staticmethod
    def _load_value_polarity_encoder(store_path: Path):
        """The polarity-tuned u-encoder for the FUSED value check
        (``value_polarity_encoder/``), as a ``list[str] -> Tensor`` callable, or
        None when the store carries no polarity encoder. This is distinct from the
        retired relevance u-veto: it scores an answer's asserted *stance* against
        the stored one, not the question against the head's relations."""
        enc_dir = Path(store_path) / "value_polarity_encoder"
        if not (enc_dir / "meta.json").exists():
            return None
        try:
            from knowlytix.embedding import FineTunedEmbedding

            return FineTunedEmbedding.load(enc_dir).encode
        except Exception:  # noqa: BLE001 - the fused value check is an optimization
            return None

    @staticmethod
    def _load_polarity_cuts(store_path: Path):
        """The calibrated 3-class u-tension cuts ``(tau_ent, tau_contra)`` from
        ``value_polarity_calibration.json``, or None when the channel was found
        degenerate (or never calibrated) -- in which case the fused check falls
        back to cap plausibility + v-space resolution alone. The cuts are a
        persisted, CV-gated operating point (see calibrate)."""
        import json

        cal = Path(store_path) / "value_polarity_calibration.json"
        if not cal.exists():
            return None
        try:
            d = json.loads(cal.read_text())
            return (float(d["tau_ent"]), float(d["tau_contra"]))
        except (ValueError, KeyError, TypeError):
            return None

    @staticmethod
    def _policy_id(facts) -> str:
        """Map the top retrieved fact to a draft-verifier policy id.

        Reversal-authority facts belong to the ``fee_reversal`` policy whatever
        role headed them; otherwise the fact's head entity *is* the policy id
        (``overdraft``, ``udaap``, ...)."""
        for f in facts:
            if f.relation in _REVERSAL_RELATIONS:
                return "fee_reversal"
        return facts[0].head if facts else ""

    @staticmethod
    def _policy_names(facts) -> list[str]:
        """Every policy domain behind the retrieved facts -- each fact's head
        entity is its policy, with reversal-authority facts mapped to
        ``fee_reversal``. This is the faithful signal for evaluating retrieval
        recall ("did the expected governing policy come back among the retrieved
        evidence?"), free of the single-label collapse in :meth:`_policy_id`
        (where one secondary reversal fact would mask the primary policy)."""
        names: list[str] = []
        # Rank by plausibility (ascending geodesic = most plausible first) so the
        # list is the top-k-by-plausibility the agent gates on and the test stand
        # scores recall@k against -- one ranking, shared by agent and evaluator.
        for f in sorted(facts, key=lambda x: getattr(x, "score", 0.0)):
            n = "fee_reversal" if f.relation in _REVERSAL_RELATIONS else f.head
            if n and n not in names:
                names.append(n)
        return names

    @staticmethod
    def _grounded_text(facts) -> str:
        """Join the provenance source spans (the real document text the facts
        came from); fall back to a ``head relation tail`` rendering."""
        spans, seen = [], set()
        for f in facts:
            line = (f.raw or f"{f.head} {f.relation} {f.tail}").strip()
            if line and line not in seen:
                seen.add(line)
                spans.append(line)
        return "\n".join(spans)

    def extract(self, message: str) -> dict:
        """The reusable fact-extraction step: run the GEODE parse⇄bind loop on a
        message and return its grounded extraction. This is the *same* parse the
        RAG runs, isolated so one extraction can feed both retrieval (pass it back
        to :meth:`search` so the message is parsed once) and regulatory escalation
        (each bound head is a policy entity carrying its governing regulation).

        Returns ``{"extraction": <serialized>, "query_facts": [(h, r, t), ...],
        "is_bound": bool}``. ``is_bound`` is ``False`` when nothing grounds -- the
        honest abstain that lets a vague message fall back to a coarse default
        rather than a fabricated label."""
        ex = self.pipe.extract(message)
        return {
            "extraction": ex.to_dict(),
            "query_facts": [list(t) for t in ex.bound_facts],
            "is_bound": ex.is_bound,
        }

    def search(self, query: str, k: int = 3,
               extraction: dict | None = None) -> list[dict]:
        """Route the query through the GEODE RAG pipeline; return the grounded
        answer in the original ``search_policy`` shape. ``k`` is retained for API
        parity (top_k lives on the pipeline's RagConfig). When ``extraction`` (the
        serialized parse⇄bind result -- the ``"extraction"`` field of an
        :meth:`extract` return) is supplied, reuse it so the message is not
        re-parsed -- the parse-once path."""
        from knowlytix.knowledge.rag import filter_admissible_facts

        ex = Extraction.from_dict(extraction) if extraction else None
        ans = self.pipe.query(query, extraction=ex)
        if ans.decision == "abstain" or not ans.sources:
            return []
        # Cap/tension admissibility gate: drop geometrically-implausible retrieved
        # facts (fabrications / cross-tail). If none survive, the answer rests on
        # inadmissible evidence -> abstain.
        kept, _dropped = filter_admissible_facts(
            self.store, ans.sources,
            cap_margins=self._cap_margins, tension_tau=self._tension_tau)
        if not kept:
            return []
        return [{
            "id": self._policy_id(kept),
            # Retrieval-recall signal for the test stand: the top policies the
            # GMS RETRIEVED ranked by plausibility (ans.sources), not just the
            # admissibility-filtered `kept` subset the answer is grounded on.
            # Scoring recall over the plausibility ranking is the agreed
            # definition and can only help (kept subset of ans.sources).
            "policies": self._policy_names(ans.sources),
            "text": self._grounded_text(kept),
            "answer": ans.answer,
            "score": float(ans.confidence),
            "decision": ans.decision,
            "route": ans.route,
            "verified": bool(ans.verified),
            "notice": ans.notice,
            # The query the RAG's iterative parse CONSTRUCTED (question -> grounded
            # (head, relation, tail) triples). This IS the extracted facts -- it is
            # what bound to real policy vocabulary -- so we surface it rather than
            # discard it: it is stored as the extraction and fed to regulatory
            # escalation. `query_facts` = the bound (grounded) triples; the head of
            # each is a policy entity carrying the governing regulation.
            "query_facts": [(b.head, b.relation, b.tail)
                            for b in ans.bound_triples if getattr(b, "bound", False)],
            "query_triples": [t.as_tuple() for t in ans.query_triples],
        }]


    def route(self, query: str, extraction: dict | None = None) -> list[str]:
        """Retrieve-only claim routing: the plausibility-ranked, gate-admissible
        policy domains a query grounds to, with NO LLM synthesis (``generate=
        False``). Used by claim-route extraction to recover which policy domain a
        customer-message claim implicates (and, via the store's ``has_product``
        edge, the product). Returns ``[]`` when nothing survives the
        cap/tension/relevance/accept gates -- an honest abstain, which is what
        makes a vague message land on ``general`` instead of a fabricated label.
        Reuses a prior :meth:`extract` result when ``extraction`` is supplied."""
        from knowlytix.knowledge.rag import filter_admissible_facts

        ex = Extraction.from_dict(extraction) if extraction else None
        ans = self.pipe.query(query, generate=False, extraction=ex)
        if ans.decision == "abstain" or not ans.sources:
            return []
        kept, _ = filter_admissible_facts(
            self.store, ans.sources,
            cap_margins=self._cap_margins, tension_tau=self._tension_tau)
        return self._policy_names(kept) if kept else []


_DEFAULT_RETRIEVER: PolicyRagRetriever | None = None


def get_default_retriever() -> PolicyRagRetriever:
    global _DEFAULT_RETRIEVER
    if _DEFAULT_RETRIEVER is None:
        _DEFAULT_RETRIEVER = PolicyRagRetriever()
    return _DEFAULT_RETRIEVER
