"""Probe: can GMS's two geometric channels --- cap PLAUSIBILITY (v / similarity)
and TENSION (u / logical) --- separate UDAAP from `none` by projecting the raw
MESSAGE into the manifold, with no discrete entity extraction?

The guard today does message --[regex/LLM]--> entities --> score_triple. This
tests the principled alternative the dual embedding enables:

    msg_v = normalize( MiniLM(msg)|d_v  @ P_v.T )          # project to shared m-sphere
    msg_u = normalize( nli-mpnet(msg)|d_u @ P_u.T )
    plaus = arccos<cap_center(flag,has_evidence), msg_v>   # within cap_radius?  (right region)
    tens  = tension_energy(msg_u, project_u(predicate))    # 0 agree .. sqrt2 irrelevant .. 2 contra

Both channels need the message in TRAINED space; trained embeddings drift from
their encoder init, so we measure drift on BOTH first. If drift is high the
projection is mis-aligned (AUC ~0.5) --- itself the signal that a bridge/rebuild
is needed before manifold scoring can replace extraction.

    python scripts/gms_cap_separation_probe.py
"""

from __future__ import annotations

import csv
import statistics as st
from pathlib import Path

import torch
import torch.nn.functional as F

from agentlab.capstone.regulatory_guard import get_default_guard

_V_MODEL = "sentence-transformers/all-MiniLM-L6-v2"      # similarity (v)
_U_MODEL = "sentence-transformers/nli-mpnet-base-v2"      # logical (u)


def encode_texts(texts: list[str], model_name: str, device) -> torch.Tensor:
    """Mean-pooled, L2-normalized embeddings (mirrors gms.graph.encoders)."""
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device).eval()
    out = []
    for i in range(0, len(texts), 64):
        b = tok(texts[i:i + 64], padding=True, truncation=True, max_length=128,
                return_tensors="pt").to(device)
        with torch.no_grad():
            h = mdl(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            emb = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            out.append(F.normalize(emb, p=2, dim=-1).cpu())
    del mdl
    return torch.cat(out, 0)


def _match(v: torch.Tensor, d: int) -> torch.Tensor:
    if v.shape[1] == d:
        return v
    return v[:, :d] if v.shape[1] > d else F.pad(v, (0, d - v.shape[1]))


def _auc(pos: list[float], neg: list[float], smaller_is_pos: bool = True) -> float:
    """P[a positive scores more 'positive' than a negative]; 0.5 = chance."""
    wins = tot = 0.0
    for p in pos:
        for n in neg:
            tot += 1
            d = (p < n) if smaller_is_pos else (p > n)
            wins += 1.0 if d else (0.5 if p == n else 0.0)
    return wins / tot if tot else float("nan")


def main() -> int:
    guard = get_default_guard()
    store = guard.store
    model = getattr(store, "model", None)
    adapter = getattr(store, "adapter", None)
    dev = getattr(store, "device", torch.device("cpu"))
    dual = getattr(model, "dual_emb", None)
    ec = getattr(model, "edge_computer", None)

    print("=== runtime cap/dual API surface ===")
    cap_trained = bool(getattr(model, "cap_enabled", False))
    need = {"model.apply_relation": hasattr(model, "apply_relation"),
            "model.edge_computer": ec is not None,
            "dual.project_v": hasattr(dual, "project_v"),
            "dual.project_u": hasattr(dual, "project_u"),
            "dual.P_v": hasattr(dual, "P_v"), "dual.P_u": hasattr(dual, "P_u")}
    print(f"  model.cap_enabled: {cap_trained}"
          f"  ({'cap radius available' if cap_trained else 'NOT cap-trained -> using operator center, no learned rho'})")
    for k, v in need.items():
        print(f"  {k}: {v}")
    if not all(need.values()):
        print("\nDual-channel geometry not fully exposed on the runtime store; "
              "cannot run the probe here.")
        return 0

    P_v, P_u = dual.P_v.detach().to(dev), dual.P_u.detach().to(dev)
    d_v, d_u = P_v.shape[1], P_u.shape[1]
    m = P_v.shape[0]
    print(f"  d_v={d_v} d_u={d_u} shared m={m}")

    def proj(raw: torch.Tensor, P: torch.Tensor, d: int) -> torch.Tensor:
        return F.normalize(_match(raw, d).to(dev) @ P.T, p=2, dim=-1)

    e2i, r2i = adapter.entity_to_idx, adapter.relation_to_idx
    rel = "has_evidence"
    r_idx, udaap_idx = r2i.get(rel), e2i.get("udaap")
    rho = float(model.cap_radii()[r_idx].item()) if cap_trained else None
    preds = [e for e in ("unfair", "deceptive", "hidden_fee", "udaap") if e in e2i]
    print(f"  rel='{rel}' rho={rho}  udaap_idx={udaap_idx}  predicates={preds}\n")

    # ---- drift: trained project_{v,u}(idx) vs raw-encoder projection of the name
    names = list(e2i)
    idxs = torch.tensor([e2i[n] for n in names], device=dev)
    with torch.no_grad():
        tv = F.normalize(dual.project_v(idxs), p=2, dim=-1).cpu()
        tu = F.normalize(dual.project_u(idxs), p=2, dim=-1).cpu()
    rv = proj(encode_texts(names, _V_MODEL, dev), P_v, d_v).cpu()
    ru = proj(encode_texts(names, _U_MODEL, dev), P_u, d_u).cpu()
    cv = (tv * rv).sum(-1)
    cu = (tu * ru).sum(-1)
    print("=== drift  (trained projection  vs  raw-encoder projection of name) ===")
    print(f"  v-channel: cos mean={cv.mean():.3f} min={cv.min():.3f} median={cv.median():.3f}")
    print(f"  u-channel: cos mean={cu.mean():.3f} min={cu.min():.3f} median={cu.median():.3f}")
    print("  (low cos => raw message lands off the trained manifold => need bridge/rebuild)\n")

    # ---- both channels on the DoE messages (UDAAP vs none) ----
    rows = [r for r in csv.DictReader(Path("data/capstone_doe_results.csv").open())
            if r["regulatory"] in ("UDAAP", "none")]
    msgs = [r["message"] for r in rows]
    is_u = [r["regulatory"] == "UDAAP" for r in rows]

    msg_v = proj(encode_texts(msgs, _V_MODEL, dev), P_v, d_v)
    msg_u = proj(encode_texts(msgs, _U_MODEL, dev), P_u, d_u)
    with torch.no_grad():
        h_v = dual.project_v(torch.tensor([udaap_idx], device=dev))          # (1,m)
        r_t = torch.tensor([r_idx], device=dev)
        c = model.cap_center(torch.tensor([udaap_idx], device=dev), r_t) if cap_trained \
            else model.apply_relation(h_v, r_t)                             # operator-transported center
        c = F.normalize(c, p=2, dim=-1)
        plaus = torch.arccos((c * msg_v).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)).cpu().tolist()
        pred_u = torch.stack([dual.project_u(torch.tensor([e2i[p]], device=dev)).squeeze(0)
                              for p in preds])                       # (P, m)
        tens_all = torch.stack([ec.compute_tension_energy(msg_u, pu.unsqueeze(0).expand_as(msg_u))
                                for pu in pred_u])                   # (P, N)
        tens = tens_all.min(0).values.cpu().tolist()                 # best-entailing predicate

    pl_u = [plaus[i] for i in range(len(rows)) if is_u[i]]
    pl_n = [plaus[i] for i in range(len(rows)) if not is_u[i]]
    te_u = [tens[i] for i in range(len(rows)) if is_u[i]]
    te_n = [tens[i] for i in range(len(rows)) if not is_u[i]]

    print(f"=== channel separation: {sum(is_u)} UDAAP vs {len(rows)-sum(is_u)} none ===")
    print(f"  PLAUSIBILITY (geodesic to {'cap' if cap_trained else 'operator'} center; lower=more plausible)")
    rho_note = (f"within rho: UDAAP {sum(d<=rho for d in pl_u)}/{len(pl_u)}  "
                f"none {sum(d<=rho for d in pl_n)}/{len(pl_n)}") if rho else "(no learned rho)"
    print(f"    UDAAP median={st.median(pl_u):.3f}  none median={st.median(pl_n):.3f}  {rho_note}")
    print(f"    AUC={_auc(pl_u, pl_n):.3f}")
    print(f"  TENSION (min over {preds}; 0=agree, ~1.41=irrelevant, 2=contradict)")
    print(f"    UDAAP median={st.median(te_u):.3f}  none median={st.median(te_n):.3f}")
    print(f"    AUC={_auc(te_u, te_n):.3f}")

    # ---- combined: z-normalized sum + conjunctive gate at a few thresholds ----
    def z(x, ref):
        mu, sd = st.mean(ref), (st.pstdev(ref) or 1.0)
        return [(v - mu) / sd for v in x]
    comb_u = [a + b for a, b in zip(z(pl_u, pl_u + pl_n), z(te_u, te_u + te_n))]
    comb_n = [a + b for a, b in zip(z(pl_n, pl_u + pl_n), z(te_n, te_u + te_n))]
    print(f"  COMBINED (z(plaus)+z(tension)): AUC={_auc(comb_u, comb_n):.3f}")
    pl_cut = rho if rho else st.median(pl_n)   # illustrative cut when not cap-trained
    print(f"\n  conjunctive gate  (plaus<={pl_cut:.3f} {'[rho]' if rho else '[none-median, illustrative]'} AND tension<=tau):")
    for tau in (0.70, 1.00, 1.41):
        fu = sum(p <= pl_cut and t <= tau for p, t in zip(pl_u, te_u))
        fn = sum(p <= pl_cut and t <= tau for p, t in zip(pl_n, te_n))
        print(f"    tau={tau:.2f}:  UDAAP fired {fu}/{len(pl_u)}   none false-fired {fn}/{len(pl_n)}")
    print("\n  (compare: discrete regex/LLM extraction gave UDAAP 5/17, none false 3/60)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
