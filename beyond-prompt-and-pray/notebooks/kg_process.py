"""Render what the capstone agent actually did, straight from its trajectory.

The agent (``build_complaint_harness``) runs the workflow classify -> extract_facts
-> search_policy -> flag_regulatory -> draft / escalate and records every step. These
helpers read that recorded trajectory and draw it. Nothing here re-implements or
re-runs a tool: ``process_figure`` annotates each step with the tool's recorded output,
and ``escalation_subgraph`` draws the stored triplets named by ``flag_regulatory``'s own
``severity_paths``.

    from kg_process import records_from_trajectory, process_figure, escalation_subgraph
    recs = records_from_trajectory(traj)     # traj from harness.run(...)
    process_figure(recs)                      # the workflow the agent executed
    escalation_subgraph(kg, recs)             # the triplets it walked to escalate
"""

from __future__ import annotations

import textwrap

# workflow-step colors + the terminal action
_COL = {"message": "#5b6b9e", "classify_complaint": "#6a8caf", "extract_facts": "#c98a3c",
        "search_policy": "#7a9a5a", "flag_regulatory": "#4477AA", "draft_response": "#7a9a5a",
        "escalate": "#EE6677", "finish": "#228833"}


def _wrap(s: str, w: int = 24) -> str:
    return "<br>".join(textwrap.wrap(str(s), w)) or str(s)


def records_from_trajectory(traj) -> list[dict]:
    """Flatten a harness trajectory into a list of recorded steps (verbatim)."""
    recs = []
    for i, r in enumerate(traj.records):
        a = r.action
        kind = getattr(a, "kind", type(a).__name__).lower()
        d = {"i": i, "kind": kind}
        if kind == "tool_call":
            obs = r.observation or {}
            d.update(tool=a.tool_name, args=dict(a.arguments),
                     output=obs.get("output"), success=obs.get("success"))
        else:
            d.update(reason=getattr(a, "reason", None), context=getattr(a, "context", None))
        recs.append(d)
    return recs


def _summarize(rec: dict) -> str:
    """One-line-ish summary of a step's RECORDED output (no inference)."""
    if rec["kind"] != "tool_call":
        return _wrap(rec.get("reason") or rec["kind"], 26)
    out = rec.get("output") or {}
    t = rec["tool"]
    if t == "classify_complaint":
        return f"{out.get('category', '?')} (conf {float(out.get('confidence', 0)):.2f})"
    if t == "extract_facts":
        qf = out.get("query_facts") or []
        q = "; ".join(f"({h}, {r}, {t2})" for h, r, t2 in qf) or "(none)"
        return _wrap(f"product={out.get('product')}", 26) + "<br>" + \
               _wrap(f"issue={out.get('issue')}", 26) + "<br>query: " + _wrap(q, 24)
    if t == "search_policy":
        ids = [p.get("id") for p in (out.get("results") or []) if p.get("id")]
        return _wrap("policy: " + (", ".join(ids) or "none"), 26)
    if t == "flag_regulatory":
        walk = "; ".join(f"{p['flag']}→{p['severity']}→{p['action']}"
                         for p in out.get("severity_paths", []))
        ev = out.get("evidence") or []
        return (_wrap(f"evidence={ev}", 26)
                + f"<br>flags={out.get('flags')}<br>escalate={out.get('escalate')}"
                + ("<br>" + _wrap(walk, 26) if walk else ""))
    if t == "draft_response":
        return _wrap("draft produced", 26)
    return _wrap(str(out), 26)


def process_figure(records: list[dict], title: str = "agent trajectory", dark: bool = True):
    """Draw the workflow the agent executed, each node annotated with its recorded output."""
    import plotly.graph_objects as go

    steps = [{"kind": "message", "label": "MESSAGE",
              "detail": _wrap(next((r["args"].get("message", "") for r in records
                                    if r["kind"] == "tool_call"), ""), 26)}]
    for r in records:
        if r["kind"] == "tool_call":
            steps.append({"kind": r["tool"], "label": r["tool"], "detail": _summarize(r)})
        else:
            steps.append({"kind": r["kind"], "label": r["kind"].upper(), "detail": _summarize(r)})

    fg = "#e8eef6" if dark else "#141a22"
    bg = "#0b0f16" if dark else "#f7f9fc"
    dx = 2.7
    ann = []
    for k in range(len(steps) - 1):                       # forward flow arrows
        ann.append(dict(x=(k + 1) * dx, y=0, ax=k * dx, ay=0,
                        xref="x", yref="y", axref="x", ayref="y",
                        showarrow=True, arrowhead=3, arrowwidth=2.0,
                        arrowcolor="#c8d3e2" if dark else "#333",
                        standoff=42, startstandoff=42))
    for k, s in enumerate(steps):
        c = _COL.get(s["kind"], "#888")
        ann.append(dict(x=k * dx, y=0, text=f"<b>{s['label']}</b>", showarrow=False,
                        font=dict(size=11, color="white"), bgcolor=c, bordercolor=c,
                        borderpad=6, xref="x", yref="y"))
        ann.append(dict(x=k * dx, y=-1.0, text=s["detail"], showarrow=False,
                        font=dict(size=9, color=fg), align="center",
                        bgcolor="rgba(255,255,255,0.04)", bordercolor=c, borderwidth=1,
                        borderpad=4, xref="x", yref="y"))

    fig = go.Figure(go.Scatter(x=[k * dx for k in range(len(steps))],
                               y=[0] * len(steps), mode="markers",
                               marker=dict(size=1, color="rgba(0,0,0,0)"),
                               hoverinfo="skip", showlegend=False))
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(color=fg, size=18)),
        annotations=ann, paper_bgcolor=bg, plot_bgcolor=bg,
        xaxis=dict(visible=False, range=[-1.6, (len(steps) - 1) * dx + 1.6]),
        yaxis=dict(visible=False, range=[-2.2, 1.4]),
        margin=dict(l=10, r=10, t=60, b=10), height=420)
    return fig


def escalation_subgraph(kg, records: list[dict], store_path: str | None = None):
    """KGData of the full escalation path flag_regulatory recorded: the message
    evidence that supports each fired flag (``flag has_evidence <ev>``), the severity
    walk (``flag has_severity <sev>``, ``<sev> has_action <act>``) from its
    ``severity_paths``, and the flag's name/statute. All stored triples (truthful);
    the evidence set is the tool's recorded ``evidence`` field.

    ``store_path`` is the GMS store directory (same path passed to
    ``load_gms_store``). It is used as a fallback: when ``kg`` is loaded with
    ``source='v'`` its labels are vocabulary terms, not entity names; passing
    ``store_path`` lets the function reload the u-space entity embeddings so the
    regulatory entities (e.g. "udaap", "high", "escalate") can be plotted."""
    from agentlab.capstone.regulatory_guard import _FLAG_TO_ENTITY

    flag_rec = next((r for r in records if r.get("tool") == "flag_regulatory"), None)
    out = (flag_rec or {}).get("output", {}) or {}
    paths = out.get("severity_paths", [])
    evidence = set(out.get("evidence", []))
    ev_flag = {(h, t) for (h, r, t) in kg.triples if r == "has_evidence"}
    name_of = {h: t for (h, r, t) in kg.triples if r == "has_regulation_name"}
    stat_of = {h: t for (h, r, t) in kg.triples if r == "has_statute"}

    triples = []
    for p in paths:
        f = _FLAG_TO_ENTITY.get(p["flag"])
        if f is None:
            continue
        # evidence hop: message evidence that supports this flag (stored has_evidence)
        triples += [(f, "has_evidence", e) for e in sorted(evidence) if (f, e) in ev_flag]
        triples.append((f, "has_severity", p["severity"]))
        triples.append((p["severity"], "has_action", p["action"]))
        if f in name_of: triples.append((f, "has_regulation_name", name_of[f]))
        if f in stat_of: triples.append((f, "has_statute", stat_of[f]))
    seen, trip = set(), []
    for t in triples:                                  # dedup, preserve order
        if t not in seen:
            seen.add(t); trip.append(t)
    S = {x for (h, _, t) in trip for x in (h, t)}
    labels = [l for l in kg.labels if l in S]
    if not labels and store_path is not None:
        # kg was loaded from v-space (vocabulary terms); reload entity embeddings
        # from u-space so regulatory entity names ("udaap", "high", …) can be found.
        from knowlytix.knowledge.viz import from_store_dir as _fsd
        kg_u = _fsd(store_path, source="u")
        labels = [l for l in kg_u.labels if l in S]
        emb = kg_u.embeddings[[kg_u.index[l] for l in labels]]
    else:
        emb = kg.embeddings[[kg.index[l] for l in labels]]
    return kg.__class__(labels=labels, embeddings=emb, triples=trip)
