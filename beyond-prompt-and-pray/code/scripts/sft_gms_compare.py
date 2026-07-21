"""Compare rank-1 SFT (full vs rotation) vs raw MiniLM, inserted into a cap-trained GMS.

Target classes = UDAAP / Reg_X / none ("the rest"), from the regulatory paraphrase
file. Seed-grouped k-fold (paraphrases of a test seed never appear in training).

Per (condition, fold):
  1. Embedding transform (frozen):
       - baseline : raw MiniLM, Stiefel/SVD-reduced 384 -> D
       - full/rot : rank-1 SFT (out_dim=D, learned Stiefel reduction) trained on the
                    train-seed paraphrases (message -> regulatory) with the chosen mode
  2. API-1 metric: nearest-prototype classification of held-out messages (the SFT's
     own classifier; for baseline, prototypes over the reduced raw embedding).
  3. Insert entity vectors into GMS via EmbeddingConfig Mode B (frozen, out_dim=D so
     no truncation), cap-train (cap loss on the entity graph + a message-cap hinge on
     the SFT-transformed train messages), then classify held-out messages by
     cap-distance to the UDAAP/Reg_X has_evidence caps -> guard escalation.

Reports per-class recall (UDAAP, Reg_X) and none false-escalation, pooled over folds,
for both the API-1 classifier and the GMS cap-distance classifier.

    python scripts/sft_gms_compare.py
"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from knowlytix.core.config import CapLossConfig, GeometryConfig, EmbeddingConfig
from knowlytix.core.graph.gkg import GeometricKnowledgeGraph
from knowlytix.core.graph.encoders import init_dual_embeddings, encode_texts
from knowlytix.core.geometry.stiefel import reduce_to_dim
from knowlytix.core.losses.cap import cap_loss, sample_boundary_negatives
from knowlytix.core.data.prepare import prepare_training_data
from knowlytix.core.train_finstructbench import GraphToGMS
from knowlytix.embedding import EmbeddingSFTConfig, finetune_embedding
from agentlab.capstone.regulatory_guard import get_default_guard, _FLAG_TO_ENTITY

D = 256
ESC = ["UDAAP", "Reg_X"]
AUG = Path("data/training/regulatory_doe_augmented.jsonl")
CSV = Path("data/capstone_doe_results.csv")
K = 5
CAP_EPOCHS, SFT_EPOCHS = 80, 60
LAMBDA_MSG, MARGIN, ALPHA = 0.5, 0.5, 4.0
V_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def tau_at_fpr(pos, neg, target=0.05):
    cands = sorted(set(pos + neg))
    best = (cands[0] - 1e-3) if cands else 0.0
    for tau in cands:
        if (sum(n <= tau for n in neg) / max(len(neg), 1)) <= target:
            best = tau
    return best


def prototype_classify(train_vecs, train_labels, test_vecs, classes):
    """Nearest-(class-mean) cosine classification. Returns predicted label per test."""
    tv = F.normalize(train_vecs, dim=-1)
    protos = torch.stack([
        F.normalize(tv[[i for i, l in enumerate(train_labels) if l == c]].mean(0), dim=-1)
        if any(l == c for l in train_labels) else torch.zeros(tv.shape[1], device=tv.device)
        for c in classes
    ])
    sims = F.normalize(test_vecs, dim=-1) @ protos.T
    return [classes[i] for i in sims.argmax(-1).tolist()]


def main() -> int:
    guard = get_default_guard()
    dev = guard.store.device
    adapter = GraphToGMS(type("S", (), {"triples": list(guard.store.triples)})())
    N, R = adapter.num_entities, adapter.num_relations
    e2i, r2i = adapter.entity_to_idx, adapter.relation_to_idx
    rev = r2i["has_evidence"]
    flag_e = {f: e2i[_FLAG_TO_ENTITY[f]] for f in ESC}
    ent_names = [adapter.idx_to_entity[i] for i in range(N)]
    ent_text = [n.replace("_", " ") for n in ent_names]

    aug = [json.loads(l) for l in AUG.read_text().splitlines() if l.strip()]
    real = [r for r in csv.DictReader(CSV.open()) if r["regulatory"] in ("UDAAP", "Reg_X", "none")]
    aug_seed = [a["_seed"] for a in aug]
    aug_lab = [a["regulatory"] for a in aug]
    real_seed = [r["seed_case"] for r in real]
    real_reg = [r["regulatory"] for r in real]
    real_exp = [r["expected_escalation"] == "True" for r in real]
    seeds = sorted(set(real_seed))
    seed_reg = {s: real_reg[real_seed.index(s)] for s in seeds}

    print(f"entities={N} relations={R} | aug={len(aug)} real={len(real)} seeds={len(seeds)} "
          f"(UDAAP={sum(v=='UDAAP' for v in seed_reg.values())} "
          f"Reg_X={sum(v=='Reg_X' for v in seed_reg.values())} "
          f"none={sum(v=='none' for v in seed_reg.values())})  D={D}\n")

    # Encode once (raw MiniLM, 384-d) -- reused by every condition/fold.
    print("encoding entities + messages with MiniLM (once)...")
    ENT = encode_texts(ent_text, V_MODEL, str(dev)).to(dev)
    AUGE = encode_texts([a["message"] for a in aug], V_MODEL, str(dev)).to(dev)
    REAL = encode_texts([r["message"] for r in real], V_MODEL, str(dev)).to(dev)

    cap = CapLossConfig()
    td = prepare_training_data(adapter, num_neg=32)
    loader = DataLoader(td.dataset, batch_size=256, shuffle=True)
    pos_tails = td.dataset._positive_tails

    # ---- seed-grouped folds (balanced by class), 1 repeat ----
    import random
    by = {}
    for s in seeds:
        by.setdefault(seed_reg[s], []).append(s)
    folds = [[] for _ in range(K)]
    for reg, ss in by.items():
        ss = ss[:]; random.Random(13).shuffle(ss)
        for j, s in enumerate(ss):
            folds[j % K].append(s)

    def transform(cond, train_idx):
        """Return (ent_v, aug_v, real_v) in D-dim for the condition, plus an SFT or None."""
        if cond == "baseline":
            return reduce_to_dim(ENT, D), reduce_to_dim(AUGE, D), reduce_to_dim(REAL, D), None
        # SFT: train on the train-seed paraphrases
        rows = [aug[i] for i in train_idx]
        tmp = Path(tempfile.mkdtemp()) / "train.jsonl"
        tmp.write_text("\n".join(json.dumps({"message": r["message"],
                                             "regulatory": r["regulatory"]}) for r in rows))
        cfg = EmbeddingSFTConfig(rank=1, mode=cond, base_dim=None, out_dim=D,
                                 epochs=SFT_EPOCHS, val_split=0.25, device=str(dev), seed=0)
        ft = finetune_embedding(tmp, cfg, text_col="message", label_col="regulatory")
        tr = lambda M: ft.transform(M.cpu()).to(dev)
        return tr(ENT), tr(AUGE), tr(REAL), ft

    def cap_gms(ent_v, aug_v, train_idx, aug_lab_tr):
        """Insert frozen entity vectors (Mode B) and cap-train (+ message hinge)."""
        model = GeometricKnowledgeGraph(num_entities=N, num_relations=R,
                                        cfg=GeometryConfig(d_v=D, d_u=D, m=128, d=128),
                                        cap_enabled=True, cap_use_diag=True).to(dev)
        # insert via EmbeddingConfig Mode B (name-keyed, frozen, out_dim==d_v so no resize)
        p = Path(tempfile.mkdtemp()) / "ent.pt"
        torch.save({ent_names[i]: ent_v[i].detach().cpu() for i in range(N)}, p)
        init_dual_embeddings(model.dual_emb, ent_names,
                             EmbeddingConfig(v_vectors_path=str(p), u_vectors_path=str(p),
                                             warm_start=False, freeze_base=True))
        opt = torch.optim.Adam([q for q in model.parameters() if q.requires_grad], lr=5e-3)
        proj = lambda mv: F.normalize(mv @ model.dual_emb.P_v.T, dim=-1)
        av = aug_v[train_idx]
        for epoch in range(CAP_EPOCHS):
            model.train()
            for h, r, t, neg_t in loader:
                h, r, t, neg_t = h.to(dev), r.to(dev), t.to(dev), neg_t.to(dev)
                if cap.n_boundary > 0 and epoch >= cap.boundary_start_epoch:
                    bt = sample_boundary_negatives(model, h, r, adapter, cap.n_boundary, pos_tails)
                    neg_t = torch.cat([neg_t, bt], dim=1)
                loss, _ = cap_loss(model, h, r, t, neg_t, cap)
                mv = proj(av); rho = model.cap_radii()[rev]
                for f, fi in flag_e.items():
                    c = F.normalize(model.cap_center(torch.tensor([fi], device=dev),
                                                     torch.tensor([rev], device=dev)), dim=-1)
                    d = torch.arccos((mv * c).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7))
                    pos = torch.tensor([l == f for l in aug_lab_tr], device=dev)
                    neg = torch.tensor([l == "none" for l in aug_lab_tr], device=dev)
                    if pos.any():
                        loss = loss + LAMBDA_MSG * F.softplus(ALPHA * (d[pos] - rho)).mean()
                    if neg.any():
                        loss = loss + LAMBDA_MSG * F.softplus(ALPHA * (rho + MARGIN - d[neg])).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        return model, proj

    def gms_distances(model, proj, real_v):
        out = {}
        model.eval()
        with torch.no_grad():
            mv = proj(real_v)
            for f, fi in flag_e.items():
                c = F.normalize(model.cap_center(torch.tensor([fi], device=dev),
                                                 torch.tensor([rev], device=dev)), dim=-1)
                out[f] = torch.arccos((mv * c).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)).cpu().tolist()
        return out

    conds = ["baseline", "full", "rotation"]
    api = {c: {"UDAAP": [0, 0], "Reg_X": [0, 0], "none": [0, 0]} for c in conds}
    gms = {c: {"UDAAP": [0, 0], "Reg_X": [0, 0], "none": [0, 0]} for c in conds}

    for k in range(K):
        test_s = set(folds[k]); val_s = set(folds[(k + 1) % K])
        train_s = set(s for kk in range(K) if kk not in (k, (k + 1) % K) for s in folds[kk])
        aug_tr = [i for i in range(len(aug)) if aug_seed[i] in train_s]
        real_te = [i for i in range(len(real)) if real_seed[i] in test_s]
        real_va = [i for i in range(len(real)) if real_seed[i] in val_s]
        if not real_te:
            continue
        aug_lab_tr = [aug_lab[i] for i in aug_tr]
        print(f"fold {k+1}/{K}: train_seeds={len(train_s)} test_msgs={len(real_te)}")

        for cond in conds:
            ent_v, aug_v, real_v, ft = transform(cond, aug_tr)

            # --- API-1: nearest-prototype classification (the embedding's own view) ---
            if ft is not None:
                preds, _ = ft.classify([real[i]["message"] for i in real_te])
            else:
                preds = prototype_classify(aug_v[aug_tr], aug_lab_tr,
                                           real_v[real_te], ["UDAAP", "Reg_X", "none"])
            for j, i in enumerate(real_te):
                esc = preds[j] in ESC
                api[cond][real_reg[i]][1] += 1
                api[cond][real_reg[i]][0] += int(esc if real_reg[i] in ESC else esc)

            # --- GMS: insert + cap-train + cap-distance classification ---
            model, proj = cap_gms(ent_v, aug_v, aug_tr, aug_lab_tr)
            dist = gms_distances(model, proj, real_v)
            taus = {f: tau_at_fpr([dist[f][i] for i in real_va if real_reg[i] == f],
                                  [dist[f][i] for i in real_va if real_reg[i] == "none"])
                    for f in ESC}
            for i in real_te:
                fired = [f for f in ESC if dist[f][i] <= taus[f]]
                esc = guard.escalation_for_flags(fired)[0]
                gms[cond][real_reg[i]][1] += 1
                gms[cond][real_reg[i]][0] += int(esc)

    def show(title, table):
        print(f"\n=== {title} (seed-grouped, {K} folds pooled) ===")
        print(f"  {'condition':10s} {'UDAAP rec':>10s} {'Reg_X rec':>10s} {'none false':>11s}")
        for c in conds:
            u, x, n = table[c]["UDAAP"], table[c]["Reg_X"], table[c]["none"]
            print(f"  {c:10s} {u[0]}/{u[1]}={u[0]/max(u[1],1):.2f}  "
                  f"{x[0]}/{x[1]}={x[0]/max(x[1],1):.2f}  "
                  f"{n[0]}/{n[1]}={n[0]/max(n[1],1):.2f}")

    show("API-1: SFT classifier (UDAAP/Reg_X = recall; none = false-escalation)", api)
    show("GMS: cap-distance after insertion (recall / false-escalation)", gms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
