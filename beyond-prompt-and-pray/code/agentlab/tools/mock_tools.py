"""Battery of mock tools used across the book.

Each tool has a tiny in-process implementation. Real services live behind
adapters in production code; the point of this module is to give every
chapter a reliable, offline tool set.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from agentlab.tools.base import RiskLevel, Tool
from agentlab.tools.registry import ToolRegistry


class SearchInput(BaseModel):
    query: str


class SearchOutput(BaseModel):
    results: list[dict[str, Any]]


_CORPUS: dict[str, list[dict[str, str]]] = {
    "overdraft": [{"id": "doc-1", "text": "Overdraft fees apply when the account balance is negative."}],
    "fee": [{"id": "doc-2", "text": "Fee schedule effective 2026 across all consumer accounts."}],
    "policy": [{"id": "doc-3", "text": "Policy document on disputes and customer escalation."}],
    "complaint": [{"id": "doc-4", "text": "Complaint handling SOP: classify, route, respond, audit."}],
}


def _search_impl(query: str) -> dict[str, Any]:
    ql = query.lower()
    out: list[dict[str, str]] = []
    for kw, docs in _CORPUS.items():
        if kw in ql:
            out.extend(docs)
    return {"results": out}


search_documents = Tool(
    name="search_documents",
    description="search a small policy corpus for relevant documents",
    input_schema=SearchInput,
    output_schema=SearchOutput,
    risk=RiskLevel.LOW,
    fn=_search_impl,
)


class CalcInput(BaseModel):
    expression: str


class CalcOutput(BaseModel):
    result: float


_SAFE_EXPR = re.compile(r"^[\d+\-*/().\s]+$")


def _calc_impl(expression: str) -> dict[str, Any]:
    if not _SAFE_EXPR.match(expression):
        raise ValueError("only basic arithmetic is permitted")
    return {"result": float(eval(expression))}


calculator = Tool(
    name="calculator",
    description="evaluate a basic arithmetic expression",
    input_schema=CalcInput,
    output_schema=CalcOutput,
    risk=RiskLevel.LOW,
    fn=_calc_impl,
)


class EmailInput(BaseModel):
    to: str
    subject: str
    body: str


class EmailOutput(BaseModel):
    success: bool
    message_id: str


def _email_impl(to: str, subject: str, body: str) -> dict[str, Any]:
    mid = f"msg-{abs(hash((to, subject))) & 0xffffff:06x}"
    return {"success": True, "message_id": mid}


send_email = Tool(
    name="send_email",
    description="send an email to a recipient",
    input_schema=EmailInput,
    output_schema=EmailOutput,
    risk=RiskLevel.HIGH,
    fn=_email_impl,
)


def register_all(registry: ToolRegistry) -> None:
    registry.register(search_documents)
    registry.register(calculator)
    registry.register(send_email)
