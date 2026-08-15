"""Full GMS MCP server: knowlytix model-agnostic store tools + forgeloop tools.

Composes the deterministic store tools shipped by
:func:`knowlytix.knowledge.mcp_tools.register_store_tools` (memory primitives,
geometric grounded retrieval, optional compiler retrieval, plausibility gate)
with forgeloop's governance, verification, testing and ingestion tools drawn
from the companion packages:

- ``gate_plan``        — plan admissibility (agentlab GMSPlanGate)
- ``judge_answer``     — geometric hallucination judge (agentlab GeometricJudge)
- ``check_grounding``  — geometric claim/evidence groundedness (agentlab GeometricJudge)
- ``test_attribute``   — DoE factor attribution (gmstest.evaluate)
- ``crawl_page``       — browser ingest to markdown (reasonloop.ingest)

The forgeloop tools are imported lazily inside each tool, so the server still
starts (with the store tools) even if a companion package is absent. The
geometric paths call no language model; ``rag_retrieve_compiled`` is the one
opt-in model-backed tool.

Usage:
    python -m forgeloop.mcp [--stores-dir DIR] [--device cpu]
Install the deps with:  pip install "forgeloop[mcp]"
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import sys

from mcp.server.fastmcp import FastMCP

from knowlytix.knowledge.mcp_tools import ActiveStore, register_store_tools

_INSTRUCTIONS = (
    "Deterministic tools over a geometric-memory store (no language model on the "
    "geometric paths). Answer from rag_retrieve's grounded facts and abstain when "
    "it abstains; rag_retrieve_compiled (model-backed) binds more phrasings when "
    "the store ships a compiler. Check answers with judge_answer / check_grounding, "
    "clear actions with gate_plausibility / gate_plan, attribute test failures with "
    "test_attribute, and ingest pages with crawl_page."
)


def register_forgeloop_tools(mcp: FastMCP, active: ActiveStore) -> None:
    """Register forgeloop's governance/testing/ingest tools on ``mcp``."""

    def _tool(fn):
        @functools.wraps(fn)
        def wrapped(*a, **k):
            with contextlib.redirect_stdout(sys.stderr):
                return fn(*a, **k)
        return mcp.tool()(wrapped)

    def _dumps(d):
        return json.dumps(d, indent=2, default=str)

    @_tool
    def gate_plan(steps: list[str], drift_budget: float = 1.35,
                  relation: str = "has_enables") -> str:
        """Admissibility of a multi-step plan by mean geometric transition drift."""
        err = active.ensure()
        if err:
            return _dumps({"error": err})
        from agentlab.gms_backend.plan_gate import GMSPlanGate
        v = GMSPlanGate(active.tools.store, drift_budget=drift_budget,
                        relation=relation).validate(steps)
        return _dumps({"admissible": v.admissible, "drift": v.drift,
                       "budget": drift_budget,
                       "transitions": [[a, b, s] for a, b, s in v.transitions]})

    @_tool
    def judge_answer(answer: str, ground_truth: str) -> str:
        """Geometric hallucination judgment of an answer vs a ground truth."""
        err = active.ensure()
        if err:
            return _dumps({"error": err})
        from agentlab.testing.judge import GeometricJudge
        v = GeometricJudge(active.tools.store).judge(answer, ground_truth)
        return _dumps({"passed": v.passed, "confidence": v.confidence,
                       "geodesic": v.geodesic, "tension": v.tension,
                       "holonomy": v.holonomy, "label": str(v.label), "detail": v.detail})

    @_tool
    def check_grounding(text: str, evidence: dict[str, str]) -> str:
        """Geometric groundedness: for each claim, the geodesic distance to its
        nearest evidence span (agentlab GeometricJudge, no language model).

        coverage is the fraction of claims the judge labels grounded. Each claim
        reports its geodesic distance and hallucination label against the nearest
        span, so the caller can route by margin and reproduce the verdict exactly.
        """
        err = active.ensure()
        if err:
            return _dumps({"error": err})
        from agentlab.evaluation.groundedness import extract_claims
        from agentlab.testing.judge import GeometricJudge
        judge = GeometricJudge(active.tools.store)
        claims_out = []
        n_grounded = 0
        for claim in extract_claims(text):
            best = None  # (geodesic, evidence_id, verdict)
            for ev_id, ev_text in evidence.items():
                v = judge.judge(claim.text, ev_text)
                if best is None or v.geodesic < best[0]:
                    best = (v.geodesic, ev_id, v)
            if best is None:
                claims_out.append({"claim": claim.text, "grounded": False,
                                   "label": "uncertain", "geodesic": None,
                                   "evidence_id": None})
                continue
            geo, ev_id, v = best
            n_grounded += int(v.passed)
            claims_out.append({"claim": claim.text, "grounded": v.passed,
                               "label": str(v.label), "geodesic": geo,
                               "evidence_id": ev_id})
        coverage = n_grounded / len(claims_out) if claims_out else 0.0
        return _dumps({"coverage": coverage, "claims": claims_out})

    @_tool
    def test_attribute(rows: list[dict], factor_names: list[str],
                       metric: str = "correct", alpha: float = 0.05) -> str:
        """Logistic factor attribution (+ Benjamini-Hochberg) over test-result rows."""
        from gmstest import evaluate
        return _dumps({"n_rows": len(rows),
                       "attribution": evaluate.attribute(rows, factor_names,
                                                         metric=metric, alpha=alpha)})

    @_tool
    def crawl_page(url: str, dest_dir: str) -> str:
        """Render a web page in a browser engine and save markdown + provenance."""
        from reasonloop.ingest import fetch_page
        r = fetch_page(url, dest_dir)
        return _dumps({"url": getattr(r, "url", url), "title": getattr(r, "title", None),
                       "markdown_path": str(getattr(r, "md_path",
                                                     getattr(r, "markdown_path", ""))),
                       "chars": getattr(r, "n_chars", None)})


def build_server(stores_dir: str, device=None) -> FastMCP:
    active = ActiveStore(stores_dir, device)
    mcp = FastMCP("GMS (model-agnostic core + forgeloop tools)", instructions=_INSTRUCTIONS)
    register_store_tools(mcp, active)
    register_forgeloop_tools(mcp, active)
    return mcp


def main() -> None:
    # Real parsing, not parse_known_args(): a misspelled flag (--stores_dir,
    # --store-dir) used to be discarded silently, the server started on the
    # default gms_stores, ActiveStore os.makedirs'd it, and every store tool
    # answered "no active store" with nothing pointing at the typo. -h now works
    # too, which add_help=False had disabled despite the docstring advertising it.
    p = argparse.ArgumentParser(
        prog="python -m forgeloop.mcp",
        description="MCP stdio server over a GMS store directory.")
    p.add_argument("--stores-dir", default="gms_stores",
                   help="directory holding the GMS stores (default: gms_stores)")
    p.add_argument("--device", default=None,
                   help="torch device for the store, e.g. cpu or cuda (default: auto)")
    args = p.parse_args()
    build_server(args.stores_dir, args.device).run(transport="stdio")


if __name__ == "__main__":
    main()
