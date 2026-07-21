"""Phase 1: end-to-end escalation before/after with the A-frozen manifold scorer.

Builds the rebuilt regulatory GMS (MiniLM/MNLI warm-start + cap loss + FROZEN
embeddings), then decides flags for each DoE message *geometrically* --- a flag
fires iff the message projected into v-space falls inside that flag's
``has_evidence`` cap (``cap_distance <= cap_radius``) --- with NO regex/LLM
extraction. Escalation is then the guard's own severity-path traversal, so the
only thing that changed is how flags are derived.

Reports the real metric: UDAAP / Reg_X recall and false-escalation on `none`,
versus the deployed regex+gate baseline (UDAAP 5/17, Reg_X 4/12, none 0/60).
Also saves the trained artifact to data/gms_regulatory_cap/ for guard wiring.

    python scripts/eval_manifold_escalation.py
"""

from __future__ import annotations

import csv
import json
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

from agentlab.capstone.regulatory_guard import get_default_guard, _FLAG_TO_ENTITY

_V_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_FLAGS = ["UDAAP", "Reg_X", "Reg_E", "Reg_Z", "FCRA"]
EPOCHS, LR, D = 300, 5e-3, 256
_ARTIFACT = Path("data/gms_regulatory_cap")


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

    print("warm-start encode + train A-frozen (cap) ...")
    v0 = _match(encode_texts(names, _V_MODEL, dev), D)
    u0 = v0.clone()  # u unused by cap/plausibility (v-only); placeholder warm-start
    geo = GeometryConfig(d_v=D, d_u=D, m=128, d=128)
    model = GeometricKnowledgeGraph(num_entities=N, num_relations=R, cfg=geo,
                                    cap_enabled=True, cap_use_diag=True).to(dev)
    model.dual_emb.init_from_pretrained(v0.to(dev), u0.to(dev))
    model.dual_emb.v_embed.weight.requires_grad_(False)   # FREEZE embeddings
    model.dual_emb.u_embed.weight.requires_grad_(False)

    cap = CapLossConfig()
    td = prepare_training_data(adapter, num_neg=32)
    loader = DataLoader(td.dataset, batch_size=256, shuffle=True)
    pos_tails = td.dataset._positive_tails
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    for epoch in range(EPOCHS):
        model.train()
        for h, r, t, neg_t in loader:
            h, r, t, neg_t = h.to(dev), r.to(dev), t.to(dev), neg_t.to(dev)
            if cap.n_boundary > 0 and epoch >= cap.boundary_start_epoch:
                bt = sample_boundary_negatives(model, h, r, adapter, cap.n_boundary, pos_tails)
                neg_t = torch.cat([neg_t, bt], dim=1)
            loss, _ = cap_loss(model, h, r, t, neg_t, cap)
            opt.zero_grad(); loss.backward(); opt.step()

    e2i, r2i = adapter.entity_to_idx, adapter.relation_to_idx
    rev = r2i["has_evidence"]
    rho = float(model.cap_radii()[rev].item())
    flag_idx = {f: e2i[_FLAG_TO_ENTITY[f]] for f in _FLAGS if _FLAG_TO_ENTITY[f] in e2i}
    print(f"  trained. cap_radius(has_evidence)={rho:.4f}  flags scored: {list(flag_idx)}\n")

    # ---- score DoE messages geometrically ----
    rows = [r for r in csv.DictReader(Path("data/capstone_doe_results.csv").open())
            if r["regulatory"] in ("UDAAP", "Reg_X", "none")]
    msgs = [r["message"] for r in rows]
    P_v = model.dual_emb.P_v.detach()
    with torch.no_grad():
        mv = F.normalize(_match(encode_texts(msgs, _V_MODEL, dev), D).to(dev) @ P_v.T, dim=-1)
        centers = {f: F.normalize(model.cap_center(torch.tensor([i], device=dev),
                                                   torch.tensor([rev], device=dev)), dim=-1)
                   for f, i in flag_idx.items()}
        # per-message cap distance to each flag
        dist = {f: torch.arccos((c * mv).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)).cpu().tolist()
                for f, c in centers.items()}

    # escalation via the guard's own severity traversal
    n = {"UDAAP": [0, 0], "Reg_X": [0, 0], "none": [0, 0]}   # [escalated, total]
    acc = [0, 0]
    udaap_d = {"UDAAP": [], "none": []}
    margin_curve = {round(mg, 2): {"UDAAP": 0, "Reg_X": 0, "none": 0}
                    for mg in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45)}
    for j, r in enumerate(rows):
        reg = r["regulatory"]
        exp = r["expected_escalation"] == "True"
        fired = [f for f in flag_idx if dist[f][j] <= rho]
        escalate = guard.escalation_for_flags(fired)[0]
        n[reg][1] += 1; n[reg][0] += int(escalate)
        acc[1] += 1; acc[0] += int(escalate == exp)
        if reg in ("UDAAP", "none"):
            udaap_d[reg].append(dist["UDAAP"][j])
        for mg in margin_curve:
            fired_m = [f for f in flag_idx if dist[f][j] <= rho + mg]
            if guard.escalation_for_flags(fired_m)[0]:
                margin_curve[mg][reg] += 1

    print("=== AFTER: manifold cap-distance scoring (frozen warm-start store) ===")
    print(f"  UDAAP recall : {n['UDAAP'][0]}/{n['UDAAP'][1]}   (baseline 5/17)")
    print(f"  Reg_X recall : {n['Reg_X'][0]}/{n['Reg_X'][1]}   (baseline 4/12)")
    print(f"  none FALSE   : {n['none'][0]}/{n['none'][1]}    (baseline 0/60)")
    print(f"  escalation accuracy: {acc[0]}/{acc[1]} ({acc[0]/acc[1]:.0%})")
    print(f"  UDAAP-cap-dist AUC (UDAAP vs none): {_auc(udaap_d['UDAAP'], udaap_d['none']):.3f}")
    print("\n  margin sweep (rho+margin):  UDAAP / Reg_X / none-false")
    for mg in margin_curve:
        c = margin_curve[mg]
        print(f"    margin={mg:+.2f}:  {c['UDAAP']:>2}/17  {c['Reg_X']:>2}/12  {c['none']:>2}/60")

    # ---- save artifact for guard wiring ----
    _ARTIFACT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), _ARTIFACT / "model.pt")
    (_ARTIFACT / "meta.json").write_text(json.dumps({
        "num_entities": N, "num_relations": R, "d_v": D, "d_u": D, "m": 128, "d": 128,
        "cap_use_diag": True, "cap_radius_has_evidence": rho,
        "entity_to_idx": e2i, "relation_to_idx": r2i,
        "flags": _FLAGS, "v_model": _V_MODEL,
    }, indent=2))
    print(f"\nsaved artifact -> {_ARTIFACT}/ (model.pt + meta.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
