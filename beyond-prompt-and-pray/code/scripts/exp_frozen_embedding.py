"""Experiment: does freezing miniLM-seeded embeddings (train only the Cayley
rotors + Stiefel P) make the GMS manifold text-projectable?

Rebuilds the regulatory-guard store with v/u embeddings seeded from the
sentence encoders and FROZEN, so learning is concentrated on the relation
rotors. Then tests whether a raw message, encoded by the same miniLM and
projected through the trained P_v, lands near the transported flag center
R_has_evidence . v(flag) -- i.e. whether the geometry can read the message with
NO discrete extraction.

Success: cos(project_v(entity), project_text(name)) ~ 1 (seed preserved), and
the labelled flag is the nearest transported center (UDAAP closest for case-016,
not case-001/008/009).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import knowlytix.knowledge.ingest as ingest_mod
import knowlytix.core.train_finstructbench as T
from knowlytix.core.graph.encoders import encode_texts, init_dual_embeddings_from_text
from knowlytix.knowledge.query import DocGMSConfig, GMSExpertStore

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "data" / "regulatory_guard.md"
STORE = ROOT / "data" / "gms_regulatory_store_frozen"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _patched_train_gms(adapter, device, **kw):
    """Wrap train_gms: seed dual embeddings from text encoders and FREEZE them,
    so only the rotors + Stiefel projections train."""
    names = [adapter.idx_to_entity[i].replace("_", " ") for i in range(adapter.num_entities)]
    orig_gkg = T.GeometricKnowledgeGraph

    def seeded_frozen(*a, **k):
        model = orig_gkg(*a, **k)
        init_dual_embeddings_from_text(model.dual_emb, names, device=str(device))
        model.dual_emb.v_embed.weight.requires_grad_(False)
        model.dual_emb.u_embed.weight.requires_grad_(False)
        print(f"  [frozen-exp] seeded {len(names)} entities from miniLM/nli, embeddings FROZEN")
        return model

    T.GeometricKnowledgeGraph = seeded_frozen
    try:
        return _ORIG_TRAIN(adapter, device, **kw)
    finally:
        T.GeometricKnowledgeGraph = orig_gkg


_ORIG_TRAIN = ingest_mod.train_gms


def build_frozen():
    from knowlytix.harness.governance.memory import ingest_document as gov_ingest
    if STORE.exists():
        import shutil
        shutil.rmtree(STORE)
    cfg = DocGMSConfig(ingest_mode="regex", store_path=str(STORE))
    cfg.train.epochs = 600
    cfg.train.batch_size = 256
    cfg.train.lr = 5e-3
    store = GMSExpertStore(cfg, device=DEV)
    ingest_mod.train_gms = _patched_train_gms          # inject seed+freeze
    try:
        gov_ingest(store, str(DOC), llm=None, config=cfg, device=DEV)
    finally:
        ingest_mod.train_gms = _ORIG_TRAIN
    store.save()
    return store


def _match_dim(x, d):
    return x if x.shape[1] == d else (x[:, :d] if x.shape[1] > d else F.pad(x, (0, d - x.shape[1])))


def evaluate(store):
    m = store.model
    ad = store.adapter
    de = m.dual_emb
    P_v = de.P_v.detach().float()
    d_v = de.d_v
    e2i = ad.entity_to_idx
    i2e = ad.idx_to_entity

    def project_text(texts):
        raw = encode_texts(list(texts), "sentence-transformers/all-MiniLM-L6-v2", str(DEV))
        raw = _match_dim(torch.as_tensor(raw, device=DEV).float(), d_v)
        return F.normalize(raw @ P_v.T, dim=-1)

    # (1) DRIFT: project_v(trained, frozen seed) vs project_text(name)
    idxs = torch.arange(de.num_entities, device=DEV)
    trained = de.project_v(idxs)
    reenc = project_text([i2e[i].replace("_", " ") for i in range(de.num_entities)])
    cos = F.cosine_similarity(trained, reenc, dim=-1)
    print(f"\n[DRIFT] cos(project_v, project_text(name)): mean={cos.mean():.3f} "
          f"min={cos.min():.3f} med={cos.median():.3f}  (1=seed preserved)")

    # (2) trained relational geometry still meaningful?
    print("\n[score_triple] (udaap, has_evidence, ?) lower=plausible:")
    for t in ("unfair", "fee", "mortgage_servicing", "credit_report_dispute"):
        print(f"   {t:22} {store.score_triple('udaap','has_evidence',t)}")

    # (3) ADMISSION: transported flag center vs projected message
    r2i = ad.relation_to_idx
    he = r2i.get("has_evidence")
    flags = [f for f in ("udaap", "reg_e", "reg_z", "reg_x", "fcra") if f in e2i]
    fidx = torch.tensor([e2i[f] for f in flags], device=DEV)
    c = F.normalize(m.apply_relation(de.project_v(fidx), torch.tensor([he]*len(flags), device=DEV)), dim=-1)
    cases = json.load(open(ROOT / "data" / "eval_cases" / "cases.json"))
    cases = cases if isinstance(cases, list) else cases["cases"]
    print(f"\n[ADMISSION] geodesic(message -> R_has_evidence.flag); nearest flag per case:")
    print(f"{'case':9} {'reg':16} nearest(dist)            udaap_dist  udaap_rank")
    for c2 in cases:
        vmsg = project_text([c2["message"]])[0]
        d = torch.arccos((c * vmsg.unsqueeze(0)).sum(-1).clamp(-1+1e-6, 1-1e-6))
        order = torch.argsort(d).tolist()
        near = " ".join(f"{flags[i]}:{d[i]:.2f}" for i in order[:2])
        ui = flags.index("udaap")
        rank = order.index(ui) + 1
        print(f"{c2['id']:9} {str(c2['factors'].get('regulatory')):16} {near:24} "
              f"udaap={d[ui]:.2f}  rank={rank}/{len(flags)}")


def main():
    print(f"device={DEV}  doc={DOC.name}")
    store = build_frozen()
    evaluate(store)
    return 0


if __name__ == "__main__":
    sys.exit(main())
