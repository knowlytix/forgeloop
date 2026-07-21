"""SFT / fine-tuning emitters (Chapter 15 consumer).

Two emitters, mirroring the two shipped capstone training pipelines:

  emit_classifier_sft  label-preserving (message, label) pairs — the
                       augment_complaint_training_doe.py pattern. Enrichment
                       varies the surface; the label is invariant.
  emit_draft_sft       grounded (prompt, response) pairs with strict golden
                       validation — the build_memo_data.py / MeMo pattern. A
                       pair is kept only if it cites the policy keyword, holds
                       the byte-exact required number, and uses no forbidden
                       unauthorized-commitment phrase.

Text generation is pluggable: `materialize_fn`/`response_fn` turn a scenario
into surface text (a Qwen rewrite in production, identity/template in tests),
so the format + validation logic here is dependency-light and testable.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .compose import Scenario

MaterializeFn = Callable[[Scenario], str]
ResponseFn = Callable[[Scenario], str]


def _default_materialize(s: Scenario) -> str:
    """Identity materialization — the base query, factors recorded as metadata."""
    return s.base.query


def _default_label(s: Scenario) -> Any:
    return s.base.components.get("expected_classification", s.base.answer)


# -- classifier SFT ---------------------------------------------------------
def emit_classifier_sft(
    scenarios: list[Scenario],
    materialize_fn: MaterializeFn = _default_materialize,
    label_fn: Callable[[Scenario], Any] = _default_label,
) -> list[dict[str, Any]]:
    """Label-preserving (message, label) records for classifier SFT."""
    records = []
    for s in scenarios:
        records.append({
            "message": materialize_fn(s),
            "label": label_fn(s),
            "_seed": s.base.query if s.base.base is None else s.base.qid,
            "_factors": dict(s.factor_levels),
        })
    return records


# -- draft / LLM fine-tuning ------------------------------------------------
def draft_prompt(complaint: str, issue: str, summary: str) -> str:
    """The exact inference-time template (Complaint/Issue/Policy/Response)."""
    return f"Complaint: {complaint}\nIssue: {issue}\nPolicy: {summary}\nResponse:"


def passes_contract(response: str, golden: dict[str, Any]) -> bool:
    """MeMo strictness: cite keyword, hold byte-exact numbers, no forbidden phrase.

    Matches the capstone draft_response_goldens.json schema
    (`citation_keywords`, `required_numbers`, `forbidden_phrases`);
    `required_keywords` is accepted as an alias.
    """
    text = response.lower()
    for kw in golden.get("citation_keywords", golden.get("required_keywords", [])):
        if kw.lower() not in text:
            return False
    for num in golden.get("required_numbers", []):
        if str(num) not in response:
            return False
    for bad in golden.get("forbidden_phrases", []):
        if bad.lower() in text:
            return False
    return True


def emit_draft_sft(
    scenarios: list[Scenario],
    response_fn: ResponseFn,
    issues: dict[str, str] | None = None,
    summaries: dict[str, str] | None = None,
    goldens: dict[str, dict] | None = None,
    validate: bool = True,
    source: str = "gmstest",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Grounded (user, assistant) pairs. Returns (kept, dropped).

    Emits the capstone draft corpus schema {user, assistant, policy_id, issue,
    source}; `user` is the exact inference template. issues/summaries are keyed
    by policy id (scenario metadata 'policy_id'); the Policy line falls back to
    the golden's `one_line_summary`. goldens is the per-policy strictness contract.
    """
    issues = issues or {}
    summaries = summaries or {}
    goldens = goldens or {}
    kept, dropped = [], []
    for s in scenarios:
        pid = s.base.metadata.get("policy_id") or s.base.metadata.get("category", "")
        complaint = s.base.query
        issue = issues.get(pid, s.base.metadata.get("issue", ""))
        summary = summaries.get(pid) or goldens.get(pid, {}).get("one_line_summary", "")
        response = response_fn(s)
        rec = {
            "user": draft_prompt(complaint, issue, summary),
            "assistant": response,
            "policy_id": pid,
            "issue": issue,
            "source": source,
        }
        if validate and pid in goldens and not passes_contract(response, goldens[pid]):
            dropped.append(rec)
        else:
            kept.append(rec)
    return kept, dropped


# -- io ---------------------------------------------------------------------
def to_jsonl(records: list[dict[str, Any]], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return len(records)
