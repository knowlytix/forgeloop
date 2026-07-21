"""Low-rank embedding adapter + message-level cap loss for the regulatory guard.

Keeps MiniLM AND the GMS base embeddings frozen (text-readability + zero base
drift), and learns only a tiny low-rank map  T = I + U Vᵀ  (U,V in R^{d_v x r})
inserted BEFORE the Stiefel projection P_v --- where it can reshape the
post-projection geometry (a pure rotation there would be a no-op, since
cap_distance is rotation-invariant; pre-P_v + low-rank + non-orthogonal is what
lets it shear apart the `none`-near-UDAAP messages).

Trained with the entity cap loss PLUS a message-level cap hinge on labeled
messages (UDAAP/Reg_X pulled inside their flag's cap, `none` pushed outside), so
the adapter learns the message separation entity-only training never saw.
Thresholds bootstrap-calibrated; held-out recall / false-escalation with CIs.

    AGENTLAB_USE_LLM_FLAG=1 python scripts/train_lowrank_adapter_guard.py
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
from knowlytix.core.graph.embeddings import normalize as gms_norm  # noqa: F401
from knowlytix.core.losses.cap import cap_loss, sample_boundary_negatives
from knowlytix.core.data.prepare import prepare_training_data
from knowlytix.core.train_finstructbench import GraphToGMS

from agentlab.capstone.regulatory_guard import get_default_guard, _FLAG_TO_ENTITY

_V_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_ESC = ["UDAAP", "Reg_X"]
D, RANK, EPOCHS, LR, LAMBDA_MSG, MARGIN, ALPHA = 256, 1, 100, 5e-3, 0.5, 0.5, 4.0
WD = 5e-2   # rank-1 adapter + weight decay: minimal capacity to curb overfitting


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


def _auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    w = t = 0.0
    for p in pos:
        for n in neg:
            t += 1; w += 1.0 if p < n else (0.5 if p == n else 0.0)
    return w / t


def main() -> int:
    torch.manual_seed(42)
    guard = get_default_guard()
    dev = guard.store.device
    adapter = GraphToGMS(SimpleNamespace(triples=[(h, r, t) for h, r, t in guard.store.triples]))
    N, R = adapter.num_entities, adapter.num_relations
    names = [adapter.idx_to_entity[i].replace("_", " ") for i in range(N)]
    e2i, r2i = adapter.entity_to_idx, adapter.relation_to_idx
    rev = r2i["has_evidence"]

    v0 = _match(encode(names, dev), D).to(dev)
    geo = GeometryConfig(d_v=D, d_u=D, m=128, d=128)
    model = GeometricKnowledgeGraph(num_entities=N, num_relations=R, cfg=geo,
                                    cap_enabled=True, cap_use_diag=True).to(dev)
    model.dual_emb.init_from_pretrained(v0, v0.clone())
    model.dual_emb.v_embed.weight.requires_grad_(False)
    model.dual_emb.u_embed.weight.requires_grad_(False)

    # --- low-rank adapter T = I + U Vᵀ inserted before P_v ---
    U = nn.Parameter(torch.zeros(D, RANK, device=dev))           # init 0 -> T=I (frozen-A start)
    V = nn.Parameter(0.01 * torch.randn(D, RANK, device=dev))
    base_v_embed = model.dual_emb.v_embed

    def project_v_T(indices):
        v = base_v_embed(indices)
        v = v + (v @ U) @ V.T
        return F.normalize(v @ model.dual_emb.P_v.T, dim=-1)
    model.dual_emb.project_v = project_v_T                       # entities now see T

    def project_msg(raw):                                        # messages see the SAME T
        v = raw + (raw @ U) @ V.T
        return F.normalize(v @ model.dual_emb.P_v.T, dim=-1)

    # --- labeled messages, stratified train/test split ---
    rows = [r for r in csv.DictReader(Path("data/capstone_doe_results.csv").open())
            if r["regulatory"] in ("UDAAP", "Reg_X", "none")]
    mraw_all = _match(encode([r["message"] for r in rows], dev), D).to(dev)
    rng = random.Random(42)
    idx_by = {}
    for i, r in enumerate(rows):
        idx_by.setdefault(r["regulatory"], []).append(i)
    # 3-way: adapter-train / calibration-val (UNSEEN by adapter) / test
    tr_idx, val_idx, te_idx = [], [], []
    for reg, ii in idx_by.items():
        rng.shuffle(ii); a = len(ii) // 2; b = a + max(1, (len(ii) - a) // 2)
        tr_idx += ii[:a]; val_idx += ii[a:b]; te_idx += ii[b:]
    tr_idx_t = torch.tensor(tr_idx, device=dev)

    cap = CapLossConfig()
    td = prepare_training_data(adapter, num_neg=32)
    loader = DataLoader(td.dataset, batch_size=256, shuffle=True)
    pos_tails = td.dataset._positive_tails
    opt = torch.optim.Adam(
        [{"params": [U, V], "weight_decay": WD},
         {"params": [p for p in model.parameters() if p.requires_grad]}], lr=LR)
    flag_e = {f: e2i[_FLAG_TO_ENTITY[f]] for f in _ESC}
    lab_tr = [rows[i]["regulatory"] for i in tr_idx]

    print(f"entities={N} rel={R}  msgs train={len(tr_idx)} val={len(val_idx)} test={len(te_idx)}  rank={RANK}")
    for epoch in range(EPOCHS):
        model.train()
        for h, r, t, neg_t in loader:
            h, r, t, neg_t = h.to(dev), r.to(dev), t.to(dev), neg_t.to(dev)
            if cap.n_boundary > 0 and epoch >= cap.boundary_start_epoch:
                bt = sample_boundary_negatives(model, h, r, adapter, cap.n_boundary, pos_tails)
                neg_t = torch.cat([neg_t, bt], dim=1)
            loss, _ = cap_loss(model, h, r, t, neg_t, cap)
            # message-level cap hinge
            mv = project_msg(mraw_all[tr_idx_t])                 # (n_tr, m)
            rho = model.cap_radii()[rev]
            L_msg = 0.0
            for f, fi in flag_e.items():
                c = F.normalize(model.cap_center(torch.tensor([fi], device=dev),
                                                 torch.tensor([rev], device=dev)), dim=-1)
                d = torch.arccos((mv * c).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7))
                pos = torch.tensor([l == f for l in lab_tr], device=dev)
                neg = torch.tensor([l == "none" for l in lab_tr], device=dev)
                if pos.any():
                    L_msg = L_msg + F.softplus(ALPHA * (d[pos] - rho)).mean()
                if neg.any():
                    L_msg = L_msg + F.softplus(ALPHA * (rho + MARGIN - d[neg])).mean()
            loss = loss + LAMBDA_MSG * L_msg
            opt.zero_grad(); loss.backward(); opt.step()

    # --- distances on all messages ---
    model.eval()
    with torch.no_grad():
        mv = project_msg(mraw_all)
        rho = float(model.cap_radii()[rev].item())
        dist = {}
        for f, fi in flag_e.items():
            c = F.normalize(model.cap_center(torch.tensor([fi], device=dev),
                                             torch.tensor([rev], device=dev)), dim=-1)
            dist[f] = torch.arccos((mv * c).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)).cpu().tolist()

    # report train separation AUC
    for f in _ESC:
        pu = [dist[f][i] for i in tr_idx if rows[i]["regulatory"] == f]
        nu = [dist[f][i] for i in tr_idx if rows[i]["regulatory"] == "none"]
        print(f"  TRAIN {f} cap-dist AUC vs none = {_auc(pu, nu):.3f}")

    # --- bootstrap-calibrate tau per flag on TRAIN (target dev FPR <= 5%) ---
    def tau_at_fpr(pos, neg, target=0.05):
        cands = sorted(set(pos + neg))
        best = cands[0] - 1e-3 if cands else 0.0
        for tau in cands:
            if (sum(n <= tau for n in neg) / max(len(neg), 1)) <= target:
                best = tau
        return best
    B = 300
    taus = {}
    for f in _ESC:                       # calibrate on VAL (unseen by the adapter)
        pos = [dist[f][i] for i in val_idx if rows[i]["regulatory"] == f]
        neg = [dist[f][i] for i in val_idx if rows[i]["regulatory"] == "none"]
        boots = []
        for _ in range(B):
            bp = [pos[rng.randrange(len(pos))] for _ in pos] if pos else []
            bn = [neg[rng.randrange(len(neg))] for _ in neg] if neg else []
            boots.append(tau_at_fpr(bp, bn))
        taus[f] = st.median(boots)
    print(f"\nbootstrap tau (B={B}, VAL split, FPR<=5%): " + "  ".join(f"{f}={taus[f]:.3f}" for f in _ESC))

    # --- held-out TEST escalation, bootstrap CIs ---
    def escalate(i):
        fired = [f for f in _ESC if dist[f][i] <= taus[f]]
        return guard.escalation_for_flags(fired)[0]
    test = [(i, rows[i]["regulatory"], rows[i]["expected_escalation"] == "True") for i in te_idx]

    def metrics(sample):
        c = {"UDAAP": [0, 0], "Reg_X": [0, 0], "none": [0, 0]}
        ok = 0
        for i, reg, exp in sample:
            esc = escalate(i)
            c[reg][1] += 1; c[reg][0] += int(esc); ok += int(esc == exp)
        return c, ok / len(sample)
    c, acc = metrics(test)
    boot_ud, boot_nf = [], []
    for _ in range(B):
        s = [test[rng.randrange(len(test))] for _ in test]
        cc, _ = metrics(s)
        boot_ud.append(cc["UDAAP"][0] / max(cc["UDAAP"][1], 1))
        boot_nf.append(cc["none"][0] / max(cc["none"][1], 1))

    def ci(x):
        x = sorted(x); return x[int(0.05 * len(x))], x[int(0.95 * len(x))]
    print(f"\n=== held-out TEST (n={len(test)}: UDAAP={c['UDAAP'][1]} Reg_X={c['Reg_X'][1]} none={c['none'][1]}) ===")
    print(f"  UDAAP recall    : {c['UDAAP'][0]}/{c['UDAAP'][1]}   (boot 90% CI {ci(boot_ud)[0]:.2f}-{ci(boot_ud)[1]:.2f})")
    print(f"  Reg_X recall    : {c['Reg_X'][0]}/{c['Reg_X'][1]}")
    print(f"  none false-esc  : {c['none'][0]}/{c['none'][1]}   (boot 90% CI {ci(boot_nf)[0]:.2f}-{ci(boot_nf)[1]:.2f})")
    print(f"  accuracy        : {acc:.0%}")
    print("\nbaselines on similar split: deployed UDAAP 1/9 @0 false; frozen-manifold (Youden) 7/9 @7/30 false.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
