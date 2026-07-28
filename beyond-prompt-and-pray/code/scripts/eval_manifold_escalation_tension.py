"""Phase 2: add the tension (u) channel and test whether plausibility AND tension
beats plausibility alone for UDAAP escalation.

Same A-frozen recipe (freeze v AND u embeddings -> both stay text-anchored), but
now also train P_u with the tension loss (agree-pairs + random irrelevance), so
the u-space projection becomes discriminative instead of a random map. Then score
each message on BOTH channels and test the conjunctive gate:

    fire UDAAP  iff  plaus <= tau_p  AND  tension <= tau_t

Hypothesis: tension lets us raise tau_p (more recall) while vetoing the `none`
cases plausibility alone would false-fire (better precision).

    python scripts/eval_manifold_escalation_tension.py
"""

from __future__ import annotations

import csv
import statistics as st
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from knowlytix.core.config import CapLossConfig, GeometryConfig
from knowlytix.core.graph.gkg import GeometricKnowledgeGraph
from knowlytix.core.losses.cap import cap_loss, sample_boundary_negatives
from knowlytix.core.data.prepare import prepare_training_data, sample_tension_batch
from knowlytix.core.train_finstructbench import GraphToGMS

from agentlab.capstone.regulatory_guard import get_default_guard

_V_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_U_MODEL = "sentence-transformers/nli-mpnet-base-v2"
_PREDS = ("unfair", "deceptive", "hidden_fee", "udaap")
EPOCHS, LR, D, LAMBDA_TE = 300, 5e-3, 256, 0.5


def encode_texts(texts, model_name, device):
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


def _match(v, d):
    return v if v.shape[1] == d else (v[:, :d] if v.shape[1] > d else F.pad(v, (0, d - v.shape[1])))


def _auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    w = t = 0.0
    for p in pos:
        for n in neg:
            t += 1
            w += 1.0 if p < n else (0.5 if p == n else 0.0)
    return w / t


def main() -> int:
    torch.manual_seed(42)
    guard = get_default_guard()
    dev = guard.store.device
    triples = [(h, r, t) for h, r, t in guard.store.triples]
    adapter = GraphToGMS(SimpleNamespace(triples=triples))
    N, R = adapter.num_entities, adapter.num_relations
    names = [adapter.idx_to_entity[i].replace("_", " ") for i in range(N)]

    print("warm-start (MiniLM v + nli-mpnet u) + train A-frozen with tension ...")
    v0 = _match(encode_texts(names, _V_MODEL, dev), D)
    u0 = _match(encode_texts(names, _U_MODEL, dev), D)
    geo = GeometryConfig(d_v=D, d_u=D, m=128, d=128)
    model = GeometricKnowledgeGraph(num_entities=N, num_relations=R, cfg=geo,
                                    cap_enabled=True, cap_use_diag=True).to(dev)
    model.dual_emb.init_from_pretrained(v0.to(dev), u0.to(dev))
    model.dual_emb.v_embed.weight.requires_grad_(False)   # freeze BOTH embeddings
    model.dual_emb.u_embed.weight.requires_grad_(False)   # -> stay text-anchored

    cap = CapLossConfig()
    td = prepare_training_data(adapter, num_neg=32)
    loader = DataLoader(td.dataset, batch_size=256, shuffle=True)
    pos_tails = td.dataset._positive_tails
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    tgt = torch.tensor([0.0, 2.0 ** 0.5, 2.0], device=dev)
    for epoch in range(EPOCHS):
        model.train()
        for h, r, t, neg_t in loader:
            h, r, t, neg_t = h.to(dev), r.to(dev), t.to(dev), neg_t.to(dev)
            if cap.n_boundary > 0 and epoch >= cap.boundary_start_epoch:
                bt = sample_boundary_negatives(model, h, r, adapter, cap.n_boundary, pos_tails)
                neg_t = torch.cat([neg_t, bt], dim=1)
            loss, _ = cap_loss(model, h, r, t, neg_t, cap)
            if td.agree_pairs or td.contra_pairs:
                a_t, b_t, lab = sample_tension_batch(td.agree_pairs, td.contra_pairs,
                                                     td.n_entities, batch_size=64)
                te = model.tension_energy_pairs(a_t.to(dev), b_t.to(dev))
                loss = loss + LAMBDA_TE * ((te - tgt[lab.to(dev)]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()

    e2i, r2i = adapter.entity_to_idx, adapter.relation_to_idx
    rev = r2i["has_evidence"]
    rho = float(model.cap_radii()[rev].item())
    ud = e2i["udaap"]

    rows = [r for r in csv.DictReader(Path("data/capstone_doe_results.csv").open())
            if r["regulatory"] in ("UDAAP", "none")]
    msgs = [r["message"] for r in rows]
    is_u = [r["regulatory"] == "UDAAP" for r in rows]
    P_v, P_u = model.dual_emb.P_v.detach(), model.dual_emb.P_u.detach()
    with torch.no_grad():
        mv = F.normalize(_match(encode_texts(msgs, _V_MODEL, dev), D).to(dev) @ P_v.T, dim=-1)
        mu = F.normalize(_match(encode_texts(msgs, _U_MODEL, dev), D).to(dev) @ P_u.T, dim=-1)
        c = F.normalize(model.cap_center(torch.tensor([ud], device=dev),
                                         torch.tensor([rev], device=dev)), dim=-1)
        plaus = torch.arccos((c * mv).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)).cpu().tolist()
        preds = [e2i[p] for p in _PREDS if p in e2i]
        tension = torch.stack([
            model.edge_computer.compute_tension_energy(
                mu, F.normalize(model.dual_emb.project_u(torch.tensor([pi], device=dev)), dim=-1).expand_as(mu))
            for pi in preds]).min(0).values.cpu().tolist()

    pl_u = [plaus[i] for i in range(len(rows)) if is_u[i]]
    pl_n = [plaus[i] for i in range(len(rows)) if not is_u[i]]
    te_u = [tension[i] for i in range(len(rows)) if is_u[i]]
    te_n = [tension[i] for i in range(len(rows)) if not is_u[i]]
    print(f"  trained. cap_radius={rho:.4f}")
    print(f"  plausibility AUC = {_auc(pl_u, pl_n):.3f}   tension AUC = {_auc(te_u, te_n):.3f} "
          f"(Phase-1 untrained tension was ~0.58)\n")

    print("=== UDAAP gate: plausibility-only vs plausibility AND tension ===")
    print(f"  {'tau_p':>10}  {'plaus-only (UDAAP/none-false)':>30}   {'+tension tau_t=1.2':>22}")
    for mp in (0.25, 0.30, 0.35, 0.40, 0.45):
        tp = rho + mp
        for tt in (1.2,):
            ru_p = sum(pl_u[i] <= tp for i in range(len(pl_u)))
            rn_p = sum(pl_n[i] <= tp for i in range(len(pl_n)))
            ru_c = sum(pl_u[i] <= tp and te_u[i] <= tt for i in range(len(pl_u)))
            rn_c = sum(pl_n[i] <= tp and te_n[i] <= tt for i in range(len(pl_n)))
        print(f"  rho+{mp:.2f}={tp:.2f}   {ru_p:>2}/17  false {rn_p:>2}/60          "
              f"{ru_c:>2}/17  false {rn_c:>2}/60")
    print("\nwin condition: +tension keeps UDAAP recall while cutting none-false at high tau_p")
    print("Phase-1 plausibility-only best 0-false point was 14/17 @ tau_p=rho+0.30.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
