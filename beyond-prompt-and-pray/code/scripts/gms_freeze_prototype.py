"""Prototype: rebuild the regulatory GMS with a MiniLM/MNLI warm-start and a
cap loss, three training strategies, and compare on BOTH axes that matter ---
relational fit and (text-anchored) message-readability.

The deployed store is xavier-init + point-loss, so the geometry can't read raw
messages. This rebuilds it so entity v-embeddings START at MiniLM(name) and asks:
how do we train the rotors + cap WITHOUT drifting the embeddings off encoder
space (which is what lets a raw message project onto the same manifold)?

Strategies
  A frozen        : warm-start, FREEZE v/u embeddings, train P_v/P_u + rotors + cap radius
  B freeze->thaw  : freeze E1 epochs (learn rotors), then unfreeze v_embed at low lr for E2
  C anchored      : train all jointly with an L2 pull  lambda * ||v_embed - v0||^2

Metrics per strategy
  drift_v         : mean cos(v_embed_trained, v0_warmstart)        (1.0 = no drift)
  graph_AUC       : cap_distance ranks UDAAP's TRUE evidence tails below others (relational fit)
  msg_plaus_AUC   : project DoE messages to v, cap_distance to UDAAP cap; UDAAP vs none
  msg_tension_AUC : project messages to u, tension to UDAAP predicate; UDAAP vs none
  msg_comb_AUC    : z(plaus)+z(tension)

    python scripts/gms_freeze_prototype.py
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
from knowlytix.core.data.prepare import prepare_training_data
from knowlytix.core.train_finstructbench import GraphToGMS

from agentlab.capstone.regulatory_guard import get_default_guard

_V_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_U_MODEL = "sentence-transformers/nli-mpnet-base-v2"
_PREDS = ("unfair", "deceptive", "hidden_fee", "udaap")
EPOCHS, E1, E2 = 300, 200, 120
LR, LR_THAW, ANCHOR_LAMBDA = 5e-3, 5e-4, 1.0


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


def _auc(pos, neg):  # P[pos has SMALLER score than neg]; 0.5 = chance
    if not pos or not neg:
        return float("nan")
    w = t = 0.0
    for p in pos:
        for n in neg:
            t += 1
            w += 1.0 if p < n else (0.5 if p == n else 0.0)
    return w / t


def build_model(N, R, d_v, v0, u0, device):
    geo = GeometryConfig(d_v=d_v, d_u=d_v, m=128, d=128)
    model = GeometricKnowledgeGraph(num_entities=N, num_relations=R, cfg=geo,
                                    cap_enabled=True, cap_use_diag=True).to(device)
    model.dual_emb.init_from_pretrained(v0.to(device), u0.to(device))
    return model


def train(model, adapter, cap, device, *, epochs, freeze_v, lr,
          anchor=None, v0=None):
    model.dual_emb.v_embed.weight.requires_grad_(not freeze_v)
    model.dual_emb.u_embed.weight.requires_grad_(not freeze_v)
    td = prepare_training_data(adapter, num_neg=32)
    loader = DataLoader(td.dataset, batch_size=256, shuffle=True, drop_last=False)
    pos_tails = td.dataset._positive_tails
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    for epoch in range(epochs):
        model.train()
        for h, r, t, neg_t in loader:
            h, r, t, neg_t = h.to(device), r.to(device), t.to(device), neg_t.to(device)
            if cap.n_boundary > 0 and epoch >= cap.boundary_start_epoch:
                bt = sample_boundary_negatives(model, h, r, adapter, cap.n_boundary, pos_tails)
                neg_t = torch.cat([neg_t, bt], dim=1)
            loss, _ = cap_loss(model, h, r, t, neg_t, cap)
            if anchor and model.dual_emb.v_embed.weight.requires_grad:
                loss = loss + anchor * ((model.dual_emb.v_embed.weight - v0.to(device)) ** 2).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def evaluate(model, adapter, v0, u0, msg_v_raw, msg_u_raw, is_udaap, device):
    dual = model.dual_emb
    e2i, r2i = adapter.entity_to_idx, adapter.relation_to_idx
    ud, rev = e2i["udaap"], r2i["has_evidence"]
    with torch.no_grad():
        drift = F.cosine_similarity(F.normalize(dual.v_embed.weight.cpu(), dim=-1),
                                    F.normalize(v0, dim=-1)).mean().item()
        # graph relational AUC: true udaap evidence tails should score lowest
        dists = model.score_all_tails_cap(torch.tensor([ud], device=device),
                                          torch.tensor([rev], device=device)).squeeze(0).cpu()
        true_t = {adapter.tails[i].item() for i in range(len(adapter.heads))
                  if adapter.heads[i].item() == ud and adapter.relations[i].item() == rev}
        pos = [float(dists[i]) for i in true_t]
        neg = [float(dists[i]) for i in range(len(dists)) if i not in true_t and i != ud]
        graph_auc = _auc(pos, neg)
        # message channels (project raw encoder vecs through the trained P_v/P_u)
        P_v, P_u = dual.P_v.detach(), dual.P_u.detach()
        c = F.normalize(model.cap_center(torch.tensor([ud], device=device),
                                         torch.tensor([rev], device=device)), dim=-1)
        mv = F.normalize(_match(msg_v_raw, P_v.shape[1]).to(device) @ P_v.T, dim=-1)
        plaus = torch.arccos((c * mv).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)).cpu().tolist()
        mu = F.normalize(_match(msg_u_raw, P_u.shape[1]).to(device) @ P_u.T, dim=-1)
        preds = [e2i[p] for p in _PREDS if p in e2i]
        tens = torch.stack([
            model.edge_computer.compute_tension_energy(
                mu, F.normalize(dual.project_u(torch.tensor([pi], device=device)), dim=-1).expand_as(mu))
            for pi in preds]).min(0).values.cpu().tolist()
    pl_u = [plaus[i] for i in range(len(plaus)) if is_udaap[i]]
    pl_n = [plaus[i] for i in range(len(plaus)) if not is_udaap[i]]
    te_u = [tens[i] for i in range(len(tens)) if is_udaap[i]]
    te_n = [tens[i] for i in range(len(tens)) if not is_udaap[i]]

    def z(x, ref):
        mu_, sd = st.mean(ref), (st.pstdev(ref) or 1.0)
        return [(v - mu_) / sd for v in x]
    cu = [a + b for a, b in zip(z(pl_u, pl_u + pl_n), z(te_u, te_u + te_n))]
    cn = [a + b for a, b in zip(z(pl_n, pl_u + pl_n), z(te_n, te_u + te_n))]
    return {"drift_v": drift, "graph_AUC": graph_auc,
            "msg_plaus_AUC": _auc(pl_u, pl_n), "msg_tension_AUC": _auc(te_u, te_n),
            "msg_comb_AUC": _auc(cu, cn)}


def main() -> int:
    guard = get_default_guard()
    dev = guard.store.device
    triples = [(h, r, t) for h, r, t in guard.store.triples]
    adapter = GraphToGMS(SimpleNamespace(triples=triples))
    N, R, d_v = adapter.num_entities, adapter.num_relations, 256
    names = [adapter.idx_to_entity[i].replace("_", " ") for i in range(N)]

    print("encoding warm-start (entity names) + DoE messages once...")
    v0 = _match(encode_texts(names, _V_MODEL, dev), d_v)
    u0 = _match(encode_texts(names, _U_MODEL, dev), d_v)
    rows = [r for r in csv.DictReader(Path("data/capstone_doe_results.csv").open())
            if r["regulatory"] in ("UDAAP", "none")]
    msgs = [r["message"] for r in rows]
    is_udaap = [r["regulatory"] == "UDAAP" for r in rows]
    msg_v_raw = _match(encode_texts(msgs, _V_MODEL, dev), d_v)
    msg_u_raw = _match(encode_texts(msgs, _U_MODEL, dev), d_v)
    print(f"entities={N} relations={R}  UDAAP={sum(is_udaap)} none={len(rows)-sum(is_udaap)}\n")

    cap = CapLossConfig()
    results = {}

    print("[A] frozen ...")
    mA = build_model(N, R, d_v, v0, u0, dev)
    train(mA, adapter, cap, dev, epochs=EPOCHS, freeze_v=True, lr=LR)
    results["A frozen"] = evaluate(mA, adapter, v0, u0, msg_v_raw, msg_u_raw, is_udaap, dev)

    print("[B] freeze -> thaw ...")
    mB = build_model(N, R, d_v, v0, u0, dev)
    train(mB, adapter, cap, dev, epochs=E1, freeze_v=True, lr=LR)
    train(mB, adapter, cap, dev, epochs=E2, freeze_v=False, lr=LR_THAW)
    results["B freeze->thaw"] = evaluate(mB, adapter, v0, u0, msg_v_raw, msg_u_raw, is_udaap, dev)

    print("[C] anchored ...")
    mC = build_model(N, R, d_v, v0, u0, dev)
    train(mC, adapter, cap, dev, epochs=EPOCHS, freeze_v=False, lr=LR,
          anchor=ANCHOR_LAMBDA, v0=v0)
    results["C anchored"] = evaluate(mC, adapter, v0, u0, msg_v_raw, msg_u_raw, is_udaap, dev)

    cols = ["drift_v", "graph_AUC", "msg_plaus_AUC", "msg_tension_AUC", "msg_comb_AUC"]
    print(f"\n{'strategy':16s} " + " ".join(f"{c:>15s}" for c in cols))
    for name, m in results.items():
        print(f"{name:16s} " + " ".join(f"{m[c]:>15.3f}" for c in cols))
    print("\ndrift_v: 1.0=embeddings stayed at warm-start (text-readable).  "
          "AUCs: 1.0=perfect separation, 0.5=chance.")
    print("baseline (deployed store, discrete extraction): UDAAP recall 5/17, +3 false on none.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
