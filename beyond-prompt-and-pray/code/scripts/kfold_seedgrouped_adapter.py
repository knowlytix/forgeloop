"""SEED-GROUPED k-fold of the rank-1 adapter, trained on DoE synthetic paraphrases.

Honest generalization test: paraphrases of a seed go entirely to train OR test,
never both. Per fold the rank-1 adapter trains on the *synthetic paraphrases* of
the train seeds (entity cap loss + message cap hinge), tau is calibrated on the
real messages of val seeds, and the real messages of held-out TEST seeds are
scored. This measures whether synthetic augmentation lets the adapter generalize
to UNSEEN regulatory scenarios + phrasings (esp. misleading clarity).

    python scripts/kfold_seedgrouped_adapter.py
"""

from __future__ import annotations

import csv
import json
import random
import statistics as st
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from knowlytix.core.config import CapLossConfig, GeometryConfig, EmbeddingConfig
from knowlytix.core.graph.gkg import GeometricKnowledgeGraph
from knowlytix.core.graph.encoders import init_dual_embeddings
from knowlytix.core.losses.cap import cap_loss, sample_boundary_negatives
from knowlytix.core.data.prepare import prepare_training_data
from knowlytix.core.train_finstructbench import GraphToGMS
from agentlab.capstone.regulatory_guard import get_default_guard, _FLAG_TO_ENTITY

_V_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_AUG = Path("data/training/regulatory_doe_augmented.jsonl")
_CSV = Path("data/capstone_doe_results.csv")
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
    ent_names = [adapter.idx_to_entity[i] for i in range(N)]          # graph keys
    names = [e.replace("_", " ") for e in ent_names]                  # text to encode
    v0 = _match(encode(names, dev), D).to(dev)
    # Warm-start via the knowlytix EmbeddingConfig package API (Mode B,
    # name-keyed): persist v0 as a {entity_name: vector} file and let
    # init_dual_embeddings load + freeze it per fold. Same vectors for v and u
    # as the former init_from_pretrained(v0, v0.clone()); cap_distance is
    # v-only so the u channel is irrelevant beyond being frozen.
    vec_path = Path(tempfile.mkdtemp()) / "warmstart_v0.pt"
    torch.save({ent_names[i]: v0[i].cpu() for i in range(N)}, vec_path)
    emb_cfg = EmbeddingConfig(
        v_vectors_path=str(vec_path), u_vectors_path=str(vec_path),
        warm_start=False, freeze_base=True, dim_policy="truncate",
    )

    aug = [json.loads(l) for l in _AUG.read_text().splitlines() if l.strip()]
    real = [r for r in csv.DictReader(_CSV.open()) if r["regulatory"] in ("UDAAP", "Reg_X", "none")]
    araw = _match(encode([a["message"] for a in aug], dev), D).to(dev)
    rraw = _match(encode([r["message"] for r in real], dev), D).to(dev)
    real_seed = [r["seed_case"] for r in real]
    real_reg = [r["regulatory"] for r in real]
    real_exp = [r["expected_escalation"] == "True" for r in real]
    seeds = sorted(set(real_seed))
    seed_reg = {s: real_reg[real_seed.index(s)] for s in seeds}
    print(f"aug paraphrases={len(aug)}  real msgs={len(real)}  in-scope seeds={len(seeds)}  "
          f"(UDAAP={sum(v=='UDAAP' for v in seed_reg.values())} "
          f"Reg_X={sum(v=='Reg_X' for v in seed_reg.values())} "
          f"none={sum(v=='none' for v in seed_reg.values())})  rank={RANK}\n")

    cap = CapLossConfig()
    td = prepare_training_data(adapter, num_neg=32)
    loader = DataLoader(td.dataset, batch_size=256, shuffle=True)
    pos_tails = td.dataset._positive_tails
    flag_e = {f: e2i[_FLAG_TO_ENTITY[f]] for f in _ESC}

    def run_fold(train_seeds, val_seeds, test_seeds, seed):
        torch.manual_seed(seed)
        model = GeometricKnowledgeGraph(num_entities=N, num_relations=R,
                                        cfg=GeometryConfig(d_v=D, d_u=D, m=128, d=128),
                                        cap_enabled=True, cap_use_diag=True).to(dev)
        # Warm-start + freeze the dual embeddings through the package API
        # (replaces the former init_from_pretrained + manual requires_grad).
        init_dual_embeddings(model.dual_emb, ent_names, emb_cfg)
        U = nn.Parameter(torch.zeros(D, RANK, device=dev))
        V = nn.Parameter(0.01 * torch.randn(D, RANK, device=dev))
        base = model.dual_emb.v_embed
        model.dual_emb.project_v = lambda idx: F.normalize(
            (base(idx) + (base(idx) @ U) @ V.T) @ model.dual_emb.P_v.T, dim=-1)
        proj = lambda raw: F.normalize((raw + (raw @ U) @ V.T) @ model.dual_emb.P_v.T, dim=-1)
        opt = torch.optim.Adam([{"params": [U, V], "weight_decay": WD},
                                {"params": [p for p in model.parameters() if p.requires_grad]}], lr=LR)
        tr_mask = [i for i, a in enumerate(aug) if a["_seed"] in train_seeds]
        tr_t = torch.tensor(tr_mask, device=dev)
        lab_tr = [aug[i]["regulatory"] for i in tr_mask]
        for epoch in range(EPOCHS):
            model.train()
            for h, r, t, neg_t in loader:
                h, r, t, neg_t = h.to(dev), r.to(dev), t.to(dev), neg_t.to(dev)
                if cap.n_boundary > 0 and epoch >= cap.boundary_start_epoch:
                    bt = sample_boundary_negatives(model, h, r, adapter, cap.n_boundary, pos_tails)
                    neg_t = torch.cat([neg_t, bt], dim=1)
                loss, _ = cap_loss(model, h, r, t, neg_t, cap)
                mv = proj(araw[tr_t]); rho = model.cap_radii()[rev]
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
            mv = proj(rraw)
            dist = {}
            for f, fi in flag_e.items():
                c = F.normalize(model.cap_center(torch.tensor([fi], device=dev),
                                                 torch.tensor([rev], device=dev)), dim=-1)
                dist[f] = torch.arccos((mv * c).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)).cpu().tolist()
        taus = {f: tau_at_fpr([dist[f][i] for i in range(len(real)) if real_seed[i] in val_seeds and real_reg[i] == f],
                              [dist[f][i] for i in range(len(real)) if real_seed[i] in val_seeds and real_reg[i] == "none"])
                for f in _ESC}
        c = {"UDAAP": [0, 0], "Reg_X": [0, 0], "none": [0, 0]}; ok = tot = 0
        for i in range(len(real)):
            if real_seed[i] not in test_seeds:
                continue
            fired = [f for f in _ESC if dist[f][i] <= taus[f]]
            esc = guard.escalation_for_flags(fired)[0]
            c[real_reg[i]][1] += 1; c[real_reg[i]][0] += int(esc); ok += int(esc == real_exp[i]); tot += 1
        return c, ok / max(tot, 1)

    by = {}
    for s in seeds:
        by.setdefault(seed_reg[s], []).append(s)
    pooled = {"UDAAP": [0, 0], "Reg_X": [0, 0], "none": [0, 0]}; accs = []
    cyc = 0
    for rep in range(REPEATS):
        folds = [[] for _ in range(K)]
        for reg, ss in by.items():
            ss = ss[:]; random.Random(50 * rep + 3).shuffle(ss)
            for j, s in enumerate(ss):
                folds[j % K].append(s)
        for k in range(K):
            test_s = set(folds[k]); val_s = set(folds[(k + 1) % K])
            train_s = set(s for kk in range(K) if kk not in (k, (k + 1) % K) for s in folds[kk])
            if not test_s:
                continue
            c, acc = run_fold(train_s, val_s, test_s, seed=1000 * rep + k)
            for key in pooled:
                pooled[key][0] += c[key][0]; pooled[key][1] += c[key][1]
            accs.append(acc); cyc += 1
            print(f"  cycle {cyc}: UDAAP {c['UDAAP'][0]}/{c['UDAAP'][1]}  "
                  f"Reg_X {c['Reg_X'][0]}/{c['Reg_X'][1]}  none-false {c['none'][0]}/{c['none'][1]}")

    print(f"\n=== SEED-GROUPED, trained on DoE synthetic ({cyc} folds) ===")
    for key, tag in (("UDAAP", "recall"), ("Reg_X", "recall"), ("none", "false-esc")):
        p = pooled[key]
        print(f"  {key:6s} {tag:9s}: {p[0]}/{p[1]} = {p[0]/max(p[1],1):.2f}")
    print(f"  accuracy: {st.mean(accs):.2f}±{(st.pstdev(accs) if len(accs)>1 else 0):.2f}")
    print("\ncompare: row-level k-fold (possible seed leak) was UDAAP 0.75 @ 0.00 false.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
