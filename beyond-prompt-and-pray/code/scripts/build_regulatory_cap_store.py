"""Build the shipped regulatory cap artifact via the embedding-SFT + GMS pipeline.

Pipeline (the validated rotation arm of scripts/sft_gms_compare.py, trained on ALL
regulatory data for deployment):

  1. Fine-tune a rank-1 ROTATION embedding adapter (knowlytix.embedding) on the
     regulatory paraphrases (message -> regulatory), out_dim = d_v so the tuned
     embedding inserts into GMS with no truncation (Stiefel reduction, not a head cut).
  2. Insert the SFT-transformed entity vectors into a cap-enabled GMS over the
     regulatory graph via EmbeddingConfig Mode B (frozen), then cap-train (cap loss +
     a message-cap hinge on the SFT-transformed train messages).
  3. Calibrate a strict per-flag geodesic threshold tau (FPR <= target on `none`).
  4. Save data/gms_regulatory_cap/{model.pt, adapter.pt, meta.json, calibration.json}
     for ManifoldFlagScorer (agentlab/capstone/manifold_evidence.py).

    python scripts/build_regulatory_cap_store.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from knowlytix.core.config import CapLossConfig, GeometryConfig, EmbeddingConfig
from knowlytix.core.graph.gkg import GeometricKnowledgeGraph
from knowlytix.core.graph.encoders import init_dual_embeddings
from knowlytix.core.losses.cap import cap_loss, sample_boundary_negatives
from knowlytix.core.data.prepare import prepare_training_data
from knowlytix.core.train_finstructbench import GraphToGMS
from knowlytix.embedding import EmbeddingSFTConfig
from knowlytix.knowledge.geode.embed_loop import EmbedLoopConfig, GeodeEmbedLoop
from knowlytix.knowledge.geode.loop import make_default_trainer

from agentlab.capstone.regulatory_guard import get_default_guard, _FLAG_TO_ENTITY

_V_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_FLAGS = ["UDAAP", "Reg_X", "Reg_E", "Reg_Z", "FCRA"]
_HINGE_FLAGS = ["UDAAP", "Reg_X"]          # escalating flags present in the training data
D = 256
CAP_EPOCHS, SFT_EPOCHS = 80, 80
LAMBDA_MSG, MARGIN, ALPHA = 0.5, 0.5, 4.0
FPR_TARGET = 0.05
_AUG = Path("data/training/regulatory_doe_augmented.jsonl")
_DOC = Path("data/regulatory_guard.md")    # source doc the GEODE embed loop ingests
_ARTIFACT = Path("data/gms_regulatory_cap")


def tau_balanced(pos, neg, pos_pct=0.90):
    """Gap-midpoint threshold: place tau in the MIDDLE of the clean margin between
    the positive cluster (high percentile of positive distances) and the nearest
    negative above it. Centering tau in the gap -- rather than hugging the training
    positives (overfits, misses slightly looser OOS positives) or sitting at the
    negatives' FPR boundary (leaks onto OOS negatives) -- is robust on both sides.
    Falls back to a high positive percentile if positives and negatives overlap.
    """
    if not pos:
        return (min(neg) - 1e-3) if neg else 0.0
    sp = sorted(pos)
    pos_hi = sp[min(len(sp) - 1, int(pos_pct * len(sp)))]
    above = [n for n in neg if n > pos_hi]
    if above:
        return 0.5 * (pos_hi + min(above))       # midpoint of the separating gap
    return sp[min(len(sp) - 1, int(0.95 * len(sp)))]  # overlap: 95th-pct of positives


def main() -> int:
    torch.manual_seed(42)
    guard = get_default_guard()
    dev = guard.store.device
    triples = [(h, r, t) for h, r, t in guard.store.triples]
    adapter = GraphToGMS(SimpleNamespace(triples=triples))
    N, R = adapter.num_entities, adapter.num_relations
    e2i, r2i = adapter.entity_to_idx, adapter.relation_to_idx
    rev = r2i["has_evidence"]
    ent_names = [adapter.idx_to_entity[i] for i in range(N)]
    ent_text = [n.replace("_", " ") for n in ent_names]
    flag_e = {f: e2i[_FLAG_TO_ENTITY[f]] for f in _HINGE_FLAGS if _FLAG_TO_ENTITY[f] in e2i}

    # 1) GEODE embed loop (knowlytix.knowledge.geode.embed_loop): geometry-supervised
    # iterative encoder SFT over the regulatory doc's own alias/evidence structure,
    # with the customer-message paraphrases fed as the self-training surface `pool`
    # so downplayed/misleading phrasings get pseudo-labelled and folded into the
    # supervision. This is the canonical GEODE path (same loop build_geode_rag_store
    # uses), not a one-shot supervised SFT. The labels still drive the cap hinge below.
    print("[1/4] GEODE embed loop: geometry-supervised rotation SFT (out_dim=%d) ..." % D)
    aug = [json.loads(l) for l in _AUG.read_text().splitlines() if l.strip()]
    msgs = [a["message"] for a in aug]
    labs = [a["regulatory"] for a in aug]
    sft_cfg = EmbeddingSFTConfig(rank=1, mode="rotation", base_dim=None, out_dim=D,
                                 epochs=SFT_EPOCHS, device=str(dev), seed=42)
    eloop = GeodeEmbedLoop(make_default_trainer(dev, epochs=150),
                           EmbedLoopConfig(sft=sft_cfg, max_iters=2, use_geometry=True))
    eres = eloop.run(str(_DOC), pool=msgs)
    ft = eres.ft
    print(f"      embed loop: {eres.iterations} iter(s) converged={eres.converged} "
          f"labels={len(eres.labels)}")
    msg_v = ft.encode(msgs).to(dev)                  # SFT-transformed message vecs (N,256)
    ent_v = ft.encode(ent_text).to(dev)              # SFT-transformed entity vecs (N,256)

    # 2) insert entity vectors via EmbeddingConfig Mode B (frozen) + cap-train
    print("[2/4] insert (Mode B) + cap-train GMS ...")
    model = GeometricKnowledgeGraph(num_entities=N, num_relations=R,
                                    cfg=GeometryConfig(d_v=D, d_u=D, m=128, d=128),
                                    cap_enabled=True, cap_use_diag=True).to(dev)
    vec_path = _ARTIFACT / "_entity_vectors.pt"
    _ARTIFACT.mkdir(parents=True, exist_ok=True)
    torch.save({ent_names[i]: ent_v[i].detach().cpu() for i in range(N)}, vec_path)
    init_dual_embeddings(model.dual_emb, ent_names,
                         EmbeddingConfig(v_vectors_path=str(vec_path), u_vectors_path=str(vec_path),
                                         warm_start=False, freeze_base=True))
    cap = CapLossConfig()
    td = prepare_training_data(adapter, num_neg=32)
    loader = DataLoader(td.dataset, batch_size=256, shuffle=True)
    pos_tails = td.dataset._positive_tails
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=5e-3)
    proj = lambda mv: F.normalize(mv @ model.dual_emb.P_v.T, dim=-1)
    for epoch in range(CAP_EPOCHS):
        model.train()
        for h, r, t, neg_t in loader:
            h, r, t, neg_t = h.to(dev), r.to(dev), t.to(dev), neg_t.to(dev)
            if cap.n_boundary > 0 and epoch >= cap.boundary_start_epoch:
                bt = sample_boundary_negatives(model, h, r, adapter, cap.n_boundary, pos_tails)
                neg_t = torch.cat([neg_t, bt], dim=1)
            loss, _ = cap_loss(model, h, r, t, neg_t, cap)
            mv = proj(msg_v); rho = model.cap_radii()[rev]
            for f, fi in flag_e.items():
                c = F.normalize(model.cap_center(torch.tensor([fi], device=dev),
                                                 torch.tensor([rev], device=dev)), dim=-1)
                d = torch.arccos((mv * c).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7))
                pos = torch.tensor([l == f for l in labs], device=dev)
                neg = torch.tensor([l == "none" for l in labs], device=dev)
                if pos.any():
                    loss = loss + LAMBDA_MSG * F.softplus(ALPHA * (d[pos] - rho)).mean()
                if neg.any():
                    loss = loss + LAMBDA_MSG * F.softplus(ALPHA * (rho + MARGIN - d[neg])).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    rho = float(model.cap_radii()[rev].item())
    print(f"      cap_radius(has_evidence)={rho:.4f}")

    # 3) strict per-flag tau calibration (FPR<=target on `none`)
    print("[3/4] calibrate per-flag tau ...")
    model.eval()
    with torch.no_grad():
        mv = proj(msg_v)
        dist = {}
        for f, fi in flag_e.items():
            c = F.normalize(model.cap_center(torch.tensor([fi], device=dev),
                                             torch.tensor([rev], device=dev)), dim=-1)
            dist[f] = torch.arccos((mv * c).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)).cpu().tolist()
    # Anchor the operating point on the REAL labeled case cohort too: the original
    # seed messages sit farther from the cap than their (tighter) paraphrases, so
    # calibrating tau on paraphrases alone underestimates the positive spread. The
    # embedding is NOT trained on these -- this only sets the threshold, mirroring
    # the labeled-cohort calibration the alias-lexicon guard used for theta.
    cases = json.loads((Path("data") / "eval_cases" / "cases.json").read_text())
    inscope = [(c["message"], (c.get("factors") or {}).get("regulatory"))
               for c in cases if (c.get("factors") or {}).get("regulatory") in ("UDAAP", "Reg_X", "none")]
    cdist: dict[str, list[float]] = {f: [] for f in flag_e}
    clabs = [lab for _, lab in inscope]
    if inscope:
        with torch.no_grad():
            cv = proj(ft.encode([m for m, _ in inscope]).to(dev))
            for f, fi in flag_e.items():
                c = F.normalize(model.cap_center(torch.tensor([fi], device=dev),
                                                 torch.tensor([rev], device=dev)), dim=-1)
                cdist[f] = torch.arccos((cv * c).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)).cpu().tolist()

    flag_thresholds = {}
    for f in flag_e:
        pos = ([dist[f][i] for i in range(len(aug)) if labs[i] == f]
               + [cdist[f][j] for j in range(len(clabs)) if clabs[j] == f])
        neg = ([dist[f][i] for i in range(len(aug)) if labs[i] == "none"]
               + [cdist[f][j] for j in range(len(clabs)) if clabs[j] == "none"])
        flag_thresholds[f] = tau_balanced(pos, neg)
        rec = sum(p <= flag_thresholds[f] for p in pos) / max(len(pos), 1)
        fp = sum(n <= flag_thresholds[f] for n in neg) / max(len(neg), 1)
        print(f"      {f}: tau={flag_thresholds[f]:.4f}  train recall={rec:.2f} fpr={fp:.2f}")

    # 4) save artifact
    print("[4/4] save artifact ...")
    torch.save(model.state_dict(), _ARTIFACT / "model.pt")
    torch.save(ft.adapter.state_dict(), _ARTIFACT / "adapter.pt")
    (_ARTIFACT / "meta.json").write_text(json.dumps({
        "num_entities": N, "num_relations": R, "d_v": D, "d_u": D, "m": 128, "d": 128,
        "cap_use_diag": True, "cap_radius_has_evidence": rho,
        "entity_to_idx": e2i, "relation_to_idx": r2i,
        "flags": list(flag_e), "v_model": _V_MODEL,
        "default_margin": MARGIN,
        "adapter": {"d_in": ft.adapter.d_in, "out_dim": ft.adapter.out_dim,
                    "rank": ft.adapter.rank, "mode": ft.adapter.mode},
    }, indent=2))
    (_ARTIFACT / "calibration.json").write_text(json.dumps({"flag_thresholds": flag_thresholds}, indent=2))
    vec_path.unlink(missing_ok=True)
    print(f"saved -> {_ARTIFACT}/ (model.pt + adapter.pt + meta.json + calibration.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
