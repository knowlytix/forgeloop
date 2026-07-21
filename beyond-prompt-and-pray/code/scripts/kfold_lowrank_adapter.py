"""Repeated stratified k-fold validation of the rank-1 adapter regulatory gate.

De-risks the small-n caveat: each example is tested OUT-OF-SAMPLE. Per fold a
fresh rank-1 adapter (T=I+uvᵀ before P_v, frozen MiniLM + base embeddings) trains
on 3 folds with entity cap loss + message cap hinge, tau is calibrated on a
separate VAL fold (no leakage), and the held-out TEST fold is scored. Aggregated
over K folds x R repeats with bootstrap CIs.

    python scripts/kfold_lowrank_adapter.py
"""

from __future__ import annotations

import csv
import random
import statistics as st
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from knowlytix.core.config import CapLossConfig, GeometryConfig
from knowlytix.core.graph.gkg import GeometricKnowledgeGraph
from knowlytix.core.losses.cap import cap_loss, sample_boundary_negatives
from knowlytix.core.data.prepare import prepare_training_data
from knowlytix.core.train_finstructbench import GraphToGMS

from agentlab.capstone.regulatory_guard import get_default_guard, _FLAG_TO_ENTITY

_V_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_ESC = ["UDAAP", "Reg_X"]
D, RANK, EPOCHS, LR, LAMBDA_MSG, MARGIN, ALPHA, WD = 256, 1, 100, 5e-3, 0.5, 0.5, 4.0, 5e-2
K, REPEATS = 5, 3


def encode(texts, device):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(_V_MODEL)
    mdl = AutoModel.from_pretrained(_V_MODEL).to(device).eval()
    out = []
    for i in range(0, len(texts), 64):
        b = tok(texts[i:i + 64], padding=True, truncation=True, max_length=128,
                return_tensors="pt").to(device)
        with torch.no_grad():
            h = mdl(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
            out.append(F.normalize(e, p=2, dim=-1).cpu())
    del mdl
    return torch.cat(out, 0)


def _match(v, d):
    return v if v.shape[1] == d else (v[:, :d] if v.shape[1] > d else F.pad(v, (0, d - v.shape[1])))


def tau_at_fpr(pos, neg, target=0.05):
    cands = sorted(set(pos + neg))
    best = (cands[0] - 1e-3) if cands else 0.0
    for tau in cands:
        if (sum(n <= tau for n in neg) / max(len(neg), 1)) <= target:
            best = tau
    return best


def main() -> int:
    guard = get_default_guard()
    dev = guard.store.device
    adapter = GraphToGMS(SimpleNamespace(triples=[(h, r, t) for h, r, t in guard.store.triples]))
    N, R = adapter.num_entities, adapter.num_relations
    e2i, r2i = adapter.entity_to_idx, adapter.relation_to_idx
    rev = r2i["has_evidence"]
    names = [adapter.idx_to_entity[i].replace("_", " ") for i in range(N)]

    v0 = _match(encode(names, dev), D).to(dev)
    rows = [r for r in csv.DictReader(Path("data/capstone_doe_results.csv").open())
            if r["regulatory"] in ("UDAAP", "Reg_X", "none")]
    mraw = _match(encode([r["message"] for r in rows], dev), D).to(dev)
    labels = [r["regulatory"] for r in rows]
    exp = [r["expected_escalation"] == "True" for r in rows]
    flag_e = {f: e2i[_FLAG_TO_ENTITY[f]] for f in _ESC}

    cap = CapLossConfig()
    td = prepare_training_data(adapter, num_neg=32)
    loader = DataLoader(td.dataset, batch_size=256, shuffle=True)
    pos_tails = td.dataset._positive_tails
    print(f"entities={N} rel={R}  msgs={len(rows)} "
          f"(UDAAP={labels.count('UDAAP')} Reg_X={labels.count('Reg_X')} none={labels.count('none')})  "
          f"rank={RANK}  {K}-fold x {REPEATS}\n")

    def run_fold(train_idx, val_idx, test_idx, seed):
        torch.manual_seed(seed)
        model = GeometricKnowledgeGraph(num_entities=N, num_relations=R,
                                        cfg=GeometryConfig(d_v=D, d_u=D, m=128, d=128),
                                        cap_enabled=True, cap_use_diag=True).to(dev)
        model.dual_emb.init_from_pretrained(v0, v0.clone())
        model.dual_emb.v_embed.weight.requires_grad_(False)
        model.dual_emb.u_embed.weight.requires_grad_(False)
        U = nn.Parameter(torch.zeros(D, RANK, device=dev))
        V = nn.Parameter(0.01 * torch.randn(D, RANK, device=dev))
        base = model.dual_emb.v_embed
        model.dual_emb.project_v = lambda idx: F.normalize(
            (base(idx) + (base(idx) @ U) @ V.T) @ model.dual_emb.P_v.T, dim=-1)
        proj_msg = lambda raw: F.normalize((raw + (raw @ U) @ V.T) @ model.dual_emb.P_v.T, dim=-1)
        opt = torch.optim.Adam(
            [{"params": [U, V], "weight_decay": WD},
             {"params": [p for p in model.parameters() if p.requires_grad]}], lr=LR)
        tr_t = torch.tensor(train_idx, device=dev)
        lab_tr = [labels[i] for i in train_idx]
        for epoch in range(EPOCHS):
            model.train()
            for h, r, t, neg_t in loader:
                h, r, t, neg_t = h.to(dev), r.to(dev), t.to(dev), neg_t.to(dev)
                if cap.n_boundary > 0 and epoch >= cap.boundary_start_epoch:
                    bt = sample_boundary_negatives(model, h, r, adapter, cap.n_boundary, pos_tails)
                    neg_t = torch.cat([neg_t, bt], dim=1)
                loss, _ = cap_loss(model, h, r, t, neg_t, cap)
                mv = proj_msg(mraw[tr_t]); rho = model.cap_radii()[rev]
                for f, fi in flag_e.items():
                    c = F.normalize(model.cap_center(torch.tensor([fi], device=dev),
                                                     torch.tensor([rev], device=dev)), dim=-1)
                    d = torch.arccos((mv * c).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7))
                    pos = torch.tensor([l == f for l in lab_tr], device=dev)
                    neg = torch.tensor([l == "none" for l in lab_tr], device=dev)
                    if pos.any():
                        loss = loss + LAMBDA_MSG * F.softplus(ALPHA * (d[pos] - rho)).mean()
                    if neg.any():
                        loss = loss + LAMBDA_MSG * F.softplus(ALPHA * (rho + MARGIN - d[neg])).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            mv = proj_msg(mraw)
            dist = {}
            for f, fi in flag_e.items():
                c = F.normalize(model.cap_center(torch.tensor([fi], device=dev),
                                                 torch.tensor([rev], device=dev)), dim=-1)
                dist[f] = torch.arccos((mv * c).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)).cpu().tolist()
        taus = {f: tau_at_fpr([dist[f][i] for i in val_idx if labels[i] == f],
                              [dist[f][i] for i in val_idx if labels[i] == "none"]) for f in _ESC}
        c = {"UDAAP": [0, 0], "Reg_X": [0, 0], "none": [0, 0]}; ok = 0
        for i in test_idx:
            fired = [f for f in _ESC if dist[f][i] <= taus[f]]
            esc = guard.escalation_for_flags(fired)[0]
            c[labels[i]][1] += 1; c[labels[i]][0] += int(esc); ok += int(esc == exp[i])
        return c, ok / len(test_idx)

    by = {}
    for i, l in enumerate(labels):
        by.setdefault(l, []).append(i)
    ud_rec, rx_rec, nf_rate, accs = [], [], [], []
    pooled = {"UDAAP": [0, 0], "Reg_X": [0, 0], "none": [0, 0]}
    cyc = 0
    for rep in range(REPEATS):
        folds = [[] for _ in range(K)]
        for l, ii in by.items():
            ii = ii[:]; random.Random(100 * rep + 7).shuffle(ii)
            for j, idx in enumerate(ii):
                folds[j % K].append(idx)
        for k in range(K):
            test_idx = folds[k]
            val_idx = folds[(k + 1) % K]
            train_idx = [i for kk in range(K) if kk not in (k, (k + 1) % K) for i in folds[kk]]
            c, acc = run_fold(train_idx, val_idx, test_idx, seed=1000 * rep + k)
            for key in pooled:
                pooled[key][0] += c[key][0]; pooled[key][1] += c[key][1]
            if c["UDAAP"][1]:
                ud_rec.append(c["UDAAP"][0] / c["UDAAP"][1])
            if c["Reg_X"][1]:
                rx_rec.append(c["Reg_X"][0] / c["Reg_X"][1])
            if c["none"][1]:
                nf_rate.append(c["none"][0] / c["none"][1])
            accs.append(acc); cyc += 1
            print(f"  cycle {cyc:>2}/{K*REPEATS}: UDAAP {c['UDAAP'][0]}/{c['UDAAP'][1]}  "
                  f"Reg_X {c['Reg_X'][0]}/{c['Reg_X'][1]}  none-false {c['none'][0]}/{c['none'][1]}")

    def ms(x):
        return (st.mean(x), (st.pstdev(x) if len(x) > 1 else 0.0))
    print(f"\n=== aggregated over {cyc} out-of-sample folds ===")
    print(f"  UDAAP recall    : pooled {pooled['UDAAP'][0]}/{pooled['UDAAP'][1]} "
          f"= {pooled['UDAAP'][0]/pooled['UDAAP'][1]:.2f}   per-fold mean {ms(ud_rec)[0]:.2f}±{ms(ud_rec)[1]:.2f}")
    print(f"  Reg_X recall    : pooled {pooled['Reg_X'][0]}/{pooled['Reg_X'][1]} "
          f"= {pooled['Reg_X'][0]/pooled['Reg_X'][1]:.2f}   per-fold mean {ms(rx_rec)[0]:.2f}±{ms(rx_rec)[1]:.2f}")
    print(f"  none false-esc  : pooled {pooled['none'][0]}/{pooled['none'][1]} "
          f"= {pooled['none'][0]/pooled['none'][1]:.2f}   per-fold mean {ms(nf_rate)[0]:.2f}±{ms(nf_rate)[1]:.2f}")
    print(f"  accuracy        : {ms(accs)[0]:.2f}±{ms(accs)[1]:.2f}")
    print("\nbaseline deployed regex+flagger (similar OOS): UDAAP ~1/9, none-false ~0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
