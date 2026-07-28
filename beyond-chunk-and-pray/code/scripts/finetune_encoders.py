# SPDX-License-Identifier: Apache-2.0
"""Fine-tune the GMS encoders on the GENERATED enrichment data, then calibrate the
relevance gate (stage 3 of the build; see scripts/enrich_data.py for stage 2).

  * v-space (concepts-close): knowlytix.embedding.finetune_embedding over
    data/enrichment/embedding_sft.jsonl ({text, label}) -> tuned_encoder/, and
    name-keyed entity vectors -> v_emb.pt (GMS Mode-B). The DoE-enriched contextual
    questions teach the attribute IN CONTEXT, so a full query maps to it.

  * u-space (logical / contradiction): trained on GENUINE conflicting-value claim
    pairs mined from the store -- two claims that assert DIFFERENT values for the
    SAME (entity, attribute) contradict (high tension), paraphrases of the SAME
    value are consistent (low tension). This is a PAIR objective (the group-based
    contradiction_sft cannot express "same slot, different value"), so we drive the
    library's contradiction_loss over an `I+UVt` (mode="full") adapter directly --
    a rotation preserves angles and cannot move tension. -> contradiction_encoder/.

  * relevance gate operating point: calibrate_relevance_thresholds -> relevance_calibration.json.

Run (GPU):  python scripts/finetune_encoders.py
"""
from __future__ import annotations

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import REPO_ROOT, use_branch_library  # noqa: E402

use_branch_library()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from knowlytix.core.graph.encoders import encode_texts  # noqa: E402
from knowlytix.embedding import EmbeddingSFTConfig, finetune_embedding  # noqa: E402
from knowlytix.embedding.adapter import LowRankEmbeddingAdapter  # noqa: E402
from knowlytix.embedding.finetune import FineTunedEmbedding, _match_dim  # noqa: E402
from knowlytix.embedding.objectives import contradiction_loss, tension_energy  # noqa: E402
from knowlytix.knowledge.config import DocGMSConfig  # noqa: E402
from knowlytix.knowledge.rag.relevance import (  # noqa: E402
    calibrate_relevance_thresholds,
)
from knowlytix.knowledge.store import GMSExpertStore  # noqa: E402

STORE = os.environ.get("GMS_STORE",
                       os.path.join(REPO_ROOT, "data", "gms_annual_report_store"))
ENRICH = os.environ.get("GMS_ENRICH", os.path.join(REPO_ROOT, "data", "enrichment"))
EMB_SFT = os.path.join(ENRICH, "embedding_sft.jsonl")
U_GROUPS = os.path.join(ENRICH, "embedding_u_groups.json")
U_BASE = "sentence-transformers/nli-mpnet-base-v2"   # the logical (NLI) base encoder
_SKIP_REL = {"in_section", "has_alias", "has_division", "has_region", "has_head"}


def _is_num(t) -> bool:
    try:
        float(str(t)); return True
    except ValueError:
        return False


def _claim_templates(entity: str, rel: str, value: str) -> list[str]:
    r = (rel[4:] if rel.startswith("has_") else rel).replace("_", " ")
    e = entity.replace("_", " ")
    return [f"{e}'s {r} was {value}.",
            f"The {r} of {e} is {value}.",
            f"{e} reported {r} of {value}."]


def _contradiction_pairs(store, *, n_wrong: int = 3, seed: int = 0):
    """GENUINE contradictions from the store: same (entity, attribute) with a
    DIFFERENT value contradicts; same value, reworded, is consistent.

    Returns ``(pos, neg)`` lists of ``(claim_a, claim_b)``. ``pos`` = consistent
    (low tension), ``neg`` = contradictory (high tension). Wrong values are other
    REAL figures from the report, so the contradiction is a plausible confusion.
    """
    rng = random.Random(seed)
    facts = [(h, r, str(t)) for h, r, t in store.triples
             if r not in _SKIP_REL and _is_num(t)]
    all_vals = sorted({t for _, _, t in facts}, key=float)
    pos, neg = [], []
    for h, r, v in facts:
        trues = _claim_templates(h, r, v)
        others = [x for x in all_vals if abs(float(x) - float(v)) > 1e-9]
        wrongs = [_claim_templates(h, r, wv)[rng.randrange(3)]
                  for wv in rng.sample(others, min(n_wrong, len(others)))]
        for i in range(len(trues)):
            for j in range(i + 1, len(trues)):
                pos.append((trues[i], trues[j]))
        for tt in trues:
            for ww in wrongs:
                neg.append((tt, ww))
    return pos, neg


def _fit_contradiction_pairs(pos, neg, cfg) -> FineTunedEmbedding:
    """Train an I+UVt (full) adapter on explicit consistent/contradictory pairs via
    the library contradiction_loss. Returns a FineTunedEmbedding whose ``.encode``
    is the tuned u-encoder (the relevance/verification contradiction space)."""
    dev = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    texts = sorted({t for a, b in (pos + neg) for t in (a, b)})
    idx = {t: i for i, t in enumerate(texts)}
    base = _match_dim(encode_texts(texts, cfg.encoder, str(dev)).to(dev), cfg.base_dim)
    d = base.shape[1]
    ai = torch.tensor([idx[a] for a, _ in pos + neg], device=dev)
    bi = torch.tensor([idx[b] for _, b in pos + neg], device=dev)
    cons = torch.tensor([1.0] * len(pos) + [0.0] * len(neg), device=dev)
    adapter = LowRankEmbeddingAdapter(d, rank=cfg.rank, mode="full",
                                      out_dim=cfg.out_dim).to(dev)
    opt = torch.optim.Adam(adapter.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    for _ in range(cfg.epochs):
        adapter.train()
        perm = torch.randperm(len(ai), device=dev)
        for s in range(0, len(perm), cfg.batch_size):
            j = perm[s:s + cfg.batch_size]
            loss, _ = contradiction_loss(adapter, base[ai[j]], base[bi[j]], cons[j], cfg)
            opt.zero_grad(); loss.backward(); opt.step()
    adapter.eval()
    with torch.no_grad():
        proto = F.normalize(F.normalize(adapter(base), dim=-1).mean(0, keepdim=True), dim=-1)
    return FineTunedEmbedding(
        adapter=adapter, encoder=cfg.encoder, base_dim=cfg.base_dim,
        label_order=["claim"], prototypes=proto.cpu(),
        thresholds=torch.zeros(1), temperature=cfg.temperature)


def _uspace_tension(u_ft: FineTunedEmbedding, a: str, b: str) -> float:
    z = F.normalize(u_ft.encode([a, b]), dim=-1)
    return float(tension_energy(z[0:1], z[1:2]).item())


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from _bootstrap import load_store_geo
    store = load_store_geo(STORE, dev)
    d_v = int(store.model.cfg.d_v)
    assert os.path.isfile(EMB_SFT), f"missing {EMB_SFT}; run scripts/enrich_data.py first"

    # 1) v-space SFT directly on the generated {text,label} corpus.
    print("[1/3] v-space SFT (finetune_embedding) on generated contextual questions")
    v_ft = finetune_embedding(
        EMB_SFT,
        EmbeddingSFTConfig(rank=8, mode="full", out_dim=d_v,
                           objective="prototype", device=str(dev)),
        text_col="text", label_col="label")
    v_ft.save(os.path.join(STORE, "tuned_encoder"))
    ent_names = sorted({e for h, _r, t in store.triples for e in (h, t)})
    torch.save(v_ft.export_vectors(ent_names), os.path.join(STORE, "v_emb.pt"))
    print(f"  val_accuracy={v_ft.val_accuracy:.3f}; saved tuned_encoder/ + "
          f"{len(ent_names)} entity vectors -> v_emb.pt")

    # 2) u-space SFT on GENUINE conflicting-value claim pairs (full-mode, pair loss).
    print("[2/3] u-space SFT (conflicting-value claim pairs) -- the contradiction space")
    pos, neg = _contradiction_pairs(store)
    json.dump({"pos": pos, "neg": neg},
              open(os.path.join(ENRICH, "contradiction_pairs.json"), "w"), indent=2)
    ucfg = EmbeddingSFTConfig(rank=32, mode="full", objective="contradiction",
                              encoder=U_BASE, out_dim=d_v, epochs=400, margin=1.3,
                              drift_weight=0.05, weight_decay=1e-4, device=str(dev))
    u_ft = _fit_contradiction_pairs(pos, neg, ucfg)
    u_ft.save(os.path.join(STORE, "contradiction_encoder"))
    print(f"  trained on {len(pos)} consistent + {len(neg)} contradictory pairs")

    # validate the tension separation on held-out claims (the real test of u-space).
    cons_t = _uspace_tension(u_ft, "Retail's headcount was 520.",
                             "The headcount of Retail is 520.")     # same value
    contra_t = _uspace_tension(u_ft, "Retail's headcount was 520.",
                               "Retail's headcount was 210.")        # conflicting value
    print(f"  tension: consistent(same value)={cons_t:.3f}  "
          f"contradictory(wrong value)={contra_t:.3f}  "
          f"{'OK separates' if contra_t > cons_t + 0.3 else 'WEAK separation'}")

    # 3) calibrate the relevance gate (v-accept floor + per-relation u cut).
    print("[3/3] calibrate relevance gate")
    u_groups = json.load(open(U_GROUPS))
    relcal = calibrate_relevance_thresholds(u_groups, v_ft.encode, u_ft.encode)
    with open(os.path.join(STORE, "relevance_calibration.json"), "w") as fh:
        json.dump(relcal, fh, indent=2, sort_keys=True)
    print(f"  tau_accept={relcal.get('tau_accept')} "
          f"default_tau_contra={relcal.get('default_tau_contra')}")


if __name__ == "__main__":
    main()
