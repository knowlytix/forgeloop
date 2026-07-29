"""Domain tools for the banking complaint agent.

Every model-driven tool is backed by a real local Qwen3-4B-Instruct:

  - classify_complaint : frozen Qwen encoder + trained logit head
                         (agentlab.models.complaint_classifier)
  - extract_facts      : Qwen + deterministic taxonomy guard (qwen_extractor)
  - search_policy      : GMS Graph RAG retrieval + Qwen grounded synthesis
                         (agentlab.capstone.policy_rag)
  - flag_regulatory    : Qwen proposes flags, a trained + calibrated GMS store
                         verifies and corrects them (agentlab.models.qwen_flagger
                         + agentlab.capstone.regulatory_guard)
  - draft_response     : Qwen + LoRA adapter (agentlab.models.draft_response_lm)

Each hybrid pairs the LLM's coverage with a deterministic backstop so a
regulated decision never rests on an unverifiable model inference. The
backstops fail loud, not silent: flag_regulatory's GMS guard is required and a
load error raises (the agent then escalates to a human) rather than downgrading
to a weaker check — a silent guard downgrade is more dangerous than a visible
failure.
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from agentlab._paths import data_path
from agentlab.models.complaint_classifier import get_default_classifier
from agentlab.models.draft_response_lm import get_default_lm
from agentlab.tools.base import RiskLevel, Tool
from agentlab.tools.registry import ToolRegistry

_DEFAULT_POLICIES_DIR = data_path("policies")


class ClassifyInput(BaseModel):
    message: str


class ClassifyOutput(BaseModel):
    category: str
    confidence: float


def _classify_impl(message: str) -> dict[str, Any]:
    label, confidence = get_default_classifier().classify(message)
    return {"category": label, "confidence": confidence}


classify_complaint = Tool(
    name="classify_complaint",
    description="classify a customer message as complaint, inquiry or other",
    input_schema=ClassifyInput,
    output_schema=ClassifyOutput,
    risk=RiskLevel.LOW,
    fn=_classify_impl,
)


class ExtractInput(BaseModel):
    message: str


class ExtractOutput(BaseModel):
    product: str
    issue: str
    urgency: str
    sentiment: str
    summary: str
    # The grounded fact extraction: query_facts are the bound (head, relation,
    # tail) policy triples the message resolved to; extraction is the serialized
    # parse⇄bind result the agent threads into search_policy so the message is
    # parsed exactly once. Both default empty on the offline/rule path.
    query_facts: list = []
    extraction: dict | None = None


# product values the downstream flag_regulatory / draft_response tools expect.
_PRODUCTS = {"checking_account", "credit_card", "mortgage", "loan", "unknown"}


def _rule_extract(message: str) -> dict[str, Any]:
    """Deterministic keyword extractor. Used directly when LLM extraction is
    disabled, and as the safe fallback when the LLM is unavailable or its
    output cannot be parsed."""
    m = message.lower()
    if "overdraft" in m:
        product, issue = "checking_account", "overdraft_fee"
    elif "credit card" in m or "credit-card" in m:
        product, issue = "credit_card", "credit_card_issue"
    elif "mortgage" in m:
        product, issue = "mortgage", "mortgage_issue"
    elif "loan" in m:
        product, issue = "loan", "loan_issue"
    elif "account" in m or "checking" in m or "savings" in m:
        product, issue = "checking_account", "account_issue"
    else:
        product, issue = "unknown", "general"

    sentiment = "negative" if any(k in m for k in ("angry", "frustrated", "terrible", "outraged", "unfair")) else "neutral"
    urgency = "high" if any(k in m for k in ("urgent", "immediately", "asap", "emergency")) else "normal"
    summary = message[:140] + ("..." if len(message) > 140 else "")
    return {"product": product, "issue": issue, "urgency": urgency, "sentiment": sentiment, "summary": summary}


def _canonical_issue(product: str, llm_issue: str) -> str:
    """Map the LLM's free-text issue onto the enumerable issue taxonomy that
    flag_regulatory and draft_response depend on. The overdraft_fee branch is
    what keeps the UDAAP flag firing on fee complaints that never say
    'overdraft' verbatim."""
    if "overdraft" in llm_issue or ("fee" in llm_issue and product == "checking_account"):
        return "overdraft_fee"
    if product == "credit_card":
        return "credit_card_issue"
    if product == "mortgage":
        return "mortgage_issue"
    if product == "loan":
        return "loan_issue"
    if product == "checking_account":
        return "account_issue"
    return "general"


# A regulatory flag may only fire on evidence present in the source text, not
# on a signal the model inferred. These guards are the deterministic backstop
# of the hybrid: the LLM proposes, but UDAAP (overdraft_fee) and Reg-X
# (mortgage / loan) escalations require a corroborating token in the message.
_FEE_SIGNAL = re.compile(r"overdraft|\bfee\b|charge|\$\s*\d|revers", re.IGNORECASE)
_REGX_SIGNAL = re.compile(r"mortgage|escrow|home loan|\bloan\b", re.IGNORECASE)


def _normalize_llm(llm: dict[str, str], message: str, rule: dict[str, Any]) -> dict[str, Any]:
    """Deterministic validation half of the hybrid: coerce the LLM's proposal
    onto the agent's enumerable taxonomy, falling back to the rule-based result
    for any field the model returned off-taxonomy, and refusing to let a
    regulatory escalation fire on a signal the message does not support."""
    product = llm.get("product", "") if llm.get("product") in _PRODUCTS else rule["product"]
    # Reg-X guard: a mortgage/loan product only stands with textual evidence.
    if product in ("mortgage", "loan") and not _REGX_SIGNAL.search(message):
        product = rule["product"]
    issue = _canonical_issue(product, llm.get("issue", ""))
    # UDAAP guard: overdraft_fee only stands if the message shows a fee/charge.
    # Stops the model inventing a fee complaint from "give me my money back".
    if issue == "overdraft_fee" and not _FEE_SIGNAL.search(message):
        issue = "account_issue" if product == "checking_account" else "general"
    urgency = llm["urgency"] if llm.get("urgency") in ("high", "normal") else rule["urgency"]
    sentiment = llm["sentiment"] if llm.get("sentiment") in ("negative", "neutral") else rule["sentiment"]
    return {
        "product": product,
        "issue": issue,
        "urgency": urgency,
        "sentiment": sentiment,
        "summary": rule["summary"],
    }


def _extract_impl(message: str) -> dict[str, Any]:
    rule = _rule_extract(message)
    # AGENTLAB_USE_LLM_EXTRACT=0 keeps extraction fully rule-based (offline,
    # zero-GPU, deterministic test runs): no store load, no grounding.
    if os.environ.get("AGENTLAB_USE_LLM_EXTRACT", "1") == "0":
        return rule

    # Hybrid Qwen+guard base. It supplies urgency/sentiment (recorded downstream
    # as assumptions, not claims) and the product/issue FALLBACK used when the
    # grounded extraction below abstains.
    try:
        from agentlab.models.qwen_extractor import get_default_extractor

        llm = get_default_extractor().extract(message)
    except Exception:
        llm = None
    out = _normalize_llm(llm, message, rule) if llm else dict(rule)
    out["query_facts"] = []
    out["extraction"] = None

    # Grounded fact extraction (default): the GEODE parse⇄bind loop binds the
    # message to real policy entities. This is the SAME parse search_policy runs,
    # isolated here so one extraction feeds both retrieval (threaded on as
    # `extraction`, parsed once) and regulatory escalation (the bound heads are
    # policy entities -> a grounded product/issue, preferred over the free-text
    # guess). An abstain (nothing binds) leaves the hybrid fallback in place, so a
    # vague message stays `general` rather than acquiring a fabricated label.
    # AGENTLAB_USE_GROUNDED_EXTRACT=0 reverts to the pure hybrid (parity A/B).
    if os.environ.get("AGENTLAB_USE_GROUNDED_EXTRACT", "1") != "0":
        try:
            from agentlab.capstone.policy_rag import get_default_retriever

            g = get_default_retriever().extract(message)
            out["query_facts"] = g["query_facts"]
            out["extraction"] = g["extraction"]
            signal = signal_from_query_facts(g["query_facts"]) if g["is_bound"] else {}
            if signal:
                out["product"], out["issue"] = signal["product"], signal["issue"]
        except Exception:
            pass  # grounding unavailable -> keep the hybrid product/issue
    return out


# Map a grounded policy-entity head (from the RAG-constructed query) to the
# escalation signal. The RAG already bound the complaint to these real policy
# entities, so they are a grounded product/issue -- preferred over the extractor's
# free-text guess. Heads may be typed variants ("overdraft/per_occurrence"); the
# stem before "/" is the policy domain.
_QF_ISSUE = {
    "overdraft": "overdraft_fee", "nsf_check": "overdraft_fee", "fee_reversal": "overdraft_fee",
    "disputes": "account_issue", "account_closure": "account_issue",
    "stop_payment": "account_issue", "wire_domestic": "account_issue",
    "wire_international": "account_issue", "paper_statement": "account_issue", "reg_e": "account_issue",
    "late_payment": "credit_card_issue", "reg_z": "credit_card_issue",
    "reg_x": "mortgage_issue", "loan_servicing": "loan_issue",
}
_QF_PRODUCT = {
    "overdraft": "checking_account", "nsf_check": "checking_account", "fee_reversal": "checking_account",
    "disputes": "checking_account", "account_closure": "checking_account",
    "stop_payment": "checking_account", "wire_domestic": "checking_account",
    "wire_international": "checking_account", "paper_statement": "checking_account", "reg_e": "checking_account",
    "late_payment": "credit_card", "reg_z": "credit_card",
    "reg_x": "mortgage", "loan_servicing": "loan",
}


def signal_from_query_facts(query_facts) -> dict[str, str]:
    """Derive a grounded (product, issue) escalation signal from the RAG-constructed
    query's bound triples. Returns {} when no head maps -- the caller then keeps the
    extractor's product/issue. Most-specific (regulatory) heads win."""
    issues, products = [], []
    for h, _r, _t in (query_facts or []):
        stem = str(h).split("/", 1)[0]
        if stem in _QF_ISSUE:
            issues.append(_QF_ISSUE[stem])
            products.append(_QF_PRODUCT[stem])
    if not issues:
        return {}
    # priority: a fee/UDAAP issue outranks a generic account issue
    order = ["overdraft_fee", "mortgage_issue", "loan_issue", "credit_card_issue", "account_issue"]
    issue = min(issues, key=lambda x: order.index(x) if x in order else 99)
    product = products[issues.index(issue)]
    return {"product": product, "issue": issue}


extract_facts = Tool(
    name="extract_facts",
    description="extract product, issue, urgency and sentiment from a complaint",
    input_schema=ExtractInput,
    output_schema=ExtractOutput,
    risk=RiskLevel.LOW,
    fn=_extract_impl,
)


class SearchPolicyInput(BaseModel):
    query: str
    # The grounded extraction from extract_facts. When present the RAG reuses it
    # instead of re-parsing the message -- the parse-once path.
    extraction: dict | None = None


class SearchPolicyOutput(BaseModel):
    results: list[dict[str, Any]]


def make_search_policy_tool(policies_dir: Path | str | None = None) -> Tool:
    """Build the ``search_policy`` Tool wrapping the shared GMS Graph RAG retriever.

    Args:
        policies_dir: Legacy corpus directory kept for signature compatibility; ignored by the GEODE retriever.

    Returns:
        A Tool that returns trimmed policy snippets with grounded answers and query facts.
    """
    # The Chapter-32 Graph RAG retriever replaces the previous keyword matcher.
    # It routes the query through the GMS policy store's entity index
    # (data/gms_policy_store/) so the agent's search_policy returns the right
    # policy_id at ~87.5% top-1 vs ~75% for the keyword baseline. The
    # policies_dir argument is kept on the closure for backward compatibility
    # with build_complaint_harness's existing signature; the retriever is
    # constructed lazily on first call so import-time cost is zero.
    # Share the one process-wide retriever with extract_facts so the GEODE store
    # is loaded once and the extraction threaded from extract_facts reuses the
    # same pipeline. (policies_dir is legacy; the GEODE retriever ignores it.)
    from agentlab.capstone.policy_rag import get_default_retriever

    def _impl(query: str, extraction: dict | None = None) -> dict[str, Any]:
        results = get_default_retriever().search(query, k=3, extraction=extraction)
        # Trim the body so a single retrieval doesn't blow the audit log; carry
        # the Qwen-synthesized grounded answer alongside the policy snippet.
        trimmed: list[dict[str, Any]] = []
        for r in results:
            text = r.get("text", "")
            snippet = text.strip()[:240] + ("..." if len(text) > 240 else "")
            entry: dict[str, Any] = {"id": r["id"], "text": snippet}
            if r.get("policies"):
                # The policy domains behind the retrieved facts -- the faithful
                # signal the test stand scores retrieval recall against.
                entry["policies"] = r["policies"]
            if r.get("answer"):
                entry["answer"] = r["answer"]
            if r.get("query_facts"):
                # The grounded query the RAG constructed = the extracted facts.
                # Carried through so regulatory escalation reads it (the head of
                # each bound triple is a policy entity with its governing reg).
                entry["query_facts"] = r["query_facts"]
            trimmed.append(entry)
        return {"results": trimmed}

    return Tool(
        name="search_policy",
        description="search the bank policy corpus via Graph RAG",
        input_schema=SearchPolicyInput,
        output_schema=SearchPolicyOutput,
        risk=RiskLevel.LOW,
        fn=_impl,
    )


class FlagInput(BaseModel):
    product: str
    issue: str
    message: str = ""
    # The grounded bound triples from extract_facts. When present they ground
    # (product, issue) before the proposer runs -- the same escalation signal the
    # RAG already bound, preferred over the free-text guess.
    query_facts: list = []


class FlagOutput(BaseModel):
    flags: list[str]
    escalate: bool
    severity_paths: list[dict[str, str]] = []
    # The regulatory evidence entities the guard read from the message (the
    # signal the flags rest on). Part of the contract so the decision is
    # auditable: a flag names the evidence that supports it, not just the label.
    evidence: list[str] = []


def _rule_flags(product: str, issue: str) -> list[str]:
    """Deterministic baseline floor. Reg-X is product-based (mortgage/loan) and
    safe to assert structurally from the product alone.

    UDAAP is deliberately NOT asserted here. A bare overdraft-fee dispute is a
    routine reversal (within the representative's $35 authority), not an
    unfair/deceptive/abusive practice -- and that distinction is a *judgment*
    that requires reading the message in context, not a keyword. The in-context
    Qwen flagger makes that call (confirmed by the GMS guard); asserting UDAAP
    from the issue label alone over-escalated every fee complaint."""
    flags: list[str] = []
    p = product.lower()
    if "mortgage" in p or "loan" in p:
        flags.append("Reg_X")
    return flags


def _flag_impl(product: str, issue: str, message: str = "",
               query_facts: list | None = None) -> dict[str, Any]:
    # Ground (product, issue) on the bound policy triples when extract_facts
    # supplied them: the RAG already resolved the complaint to real policy
    # entities, a stronger signal than the free-text label. Abstain leaves the
    # passed product/issue in place. (The GMS guard's evidence path reads the
    # message independently; this only sharpens the baseline floor + proposer.)
    signal = signal_from_query_facts(query_facts) if query_facts else {}
    if signal:
        product, issue = signal["product"], signal["issue"]

    # Deterministic baseline floor from the structured product/issue.
    baseline = set(_rule_flags(product, issue))

    # Qwen proposes candidate flags. If the proposer is unavailable we WARN and
    # proceed — the GMS guard below still derives flags from message evidence on
    # its own, so coverage degrades visibly, not silently.
    proposed: list[str] = []
    if message.strip() and os.environ.get("AGENTLAB_USE_LLM_FLAG", "1") != "0":
        try:
            from agentlab.models.qwen_flagger import get_default_flagger

            proposed = get_default_flagger().propose(message, product, issue) or []
        except Exception as exc:  # noqa: BLE001 - surface, do not fall back
            warnings.warn(
                f"flag_regulatory: Qwen proposer unavailable ({exc}); "
                "proceeding with GMS evidence correction only.",
                RuntimeWarning,
                stacklevel=2,
            )

    # The GMS regulatory guard is REQUIRED: it verifies/corrects the flag set
    # against the calibrated knowledge graph AND decides escalation by multi-hop
    # traversal of the severity path (flag -> severity -> action). There is no
    # weaker fallback — a silent downgrade of a governance guard is more
    # dangerous than a loud failure, so a load error raises and the agent
    # escalates to a human.
    from agentlab.capstone.regulatory_guard import get_default_guard

    guard = get_default_guard()
    if message.strip():
        verdict = guard.verify_and_correct(proposed, message)
        flags = sorted(baseline | set(verdict["flags"]))
        evidence = list(verdict["evidence"])
    else:
        flags = sorted(baseline)
        evidence = []
    escalate, paths = guard.escalation_for_flags(flags)
    return {"flags": flags, "escalate": escalate, "severity_paths": paths,
            "evidence": evidence}


flag_regulatory = Tool(
    name="flag_regulatory",
    description="flag regulatory risk (UDAAP, Reg X/E/Z, FCRA) from the complaint message",
    input_schema=FlagInput,
    output_schema=FlagOutput,
    risk=RiskLevel.LOW,
    fn=_flag_impl,
)


class DraftInput(BaseModel):
    category: str
    issue: str
    policy_evidence: list[dict[str, Any]]
    message: str


class DraftOutput(BaseModel):
    text: str
    corrected: bool = False
    corrections: list[dict[str, Any]] = []
    requires_escalation: bool = False
    reason: str = ""


def _draft_impl(
    category: str,
    issue: str,
    policy_evidence: list[dict[str, Any]],
    message: str,
) -> dict[str, Any]:
    # The LoRA was SFT-trained on complaint examples only (Chapter 31). For
    # inquiries / other we fall back to a rule-based template so the draft
    # stays on-distribution. This split is intentional and the chapter prose
    # should name it; collapsing both into the LoRA produces hallucinated
    # complaint-shaped replies to inquiries.
    if category == "complaint":
        text = get_default_lm().generate(
            category=category,
            issue=issue,
            policy_evidence=policy_evidence,
            message=message,
        )
        if text:
            # GMS verifies the generated draft: correct drifted fee numbers via
            # ENM, escalate an unauthorized fee-waiver promise. The agent reads
            # requires_escalation on its next step.
            policy_id = next((p.get("id") for p in policy_evidence if p.get("id")), "")
            from agentlab.capstone.draft_verifier import get_default_verifier

            v = get_default_verifier().verify(text, policy_id)
            return {
                "text": v["text"],
                "corrected": v["corrected"],
                "corrections": v["corrections"],
                "requires_escalation": v["escalate"],
                "reason": v["reason"],
            }
    cites = ", ".join(p.get("id", "?") for p in policy_evidence[:2]) or "our standard policies"
    text = (
        f"Thank you for reaching out about your {issue.replace('_', ' ')}. "
        f"Per {cites}, we are reviewing your case. "
        "A representative will follow up within 2 business days."
    )
    return {"text": text}


draft_response = Tool(
    name="draft_response",
    description="draft a customer-facing response citing policy evidence",
    input_schema=DraftInput,
    output_schema=DraftOutput,
    risk=RiskLevel.MEDIUM,
    fn=_draft_impl,
)


def register_all(registry: ToolRegistry, policies_dir: Path | str | None = None) -> None:
    """Register the five banking complaint tools on ``registry``.

    Args:
        registry: The tool registry to populate.
        policies_dir: Legacy corpus directory forwarded to the search_policy tool.
    """
    registry.register(classify_complaint)
    registry.register(extract_facts)
    registry.register(make_search_policy_tool(policies_dir))
    registry.register(flag_regulatory)
    registry.register(draft_response)
