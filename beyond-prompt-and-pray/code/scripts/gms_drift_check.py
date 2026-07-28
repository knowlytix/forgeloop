"""Corrected drift check: is the stored entity embedding still the MiniLM/MNLI
warm-start, or did training move it?

The earlier probe compared THROUGH the trained Stiefel P_v and may have used the
wrong entity text. This isolates the question: compare the raw stored
``v_embed[idx]`` directly against ``MiniLM(name)`` in d_v space (no P_v), per
entity, trying a few name forms to rule out a text mismatch. High cos => embeddings
sit at the warm-start (light training); low cos => real drift.

    python scripts/gms_drift_check.py
"""

from __future__ import annotations

import statistics as st

import torch
import torch.nn.functional as F

from agentlab.capstone.regulatory_guard import get_default_guard

_V_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_U_MODEL = "sentence-transformers/nli-mpnet-base-v2"


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


def _cos_to_stored(stored_w, names, model_name, device):
    """For each name form, cos(stored_embed, encoder(name) truncated), in raw d_v space."""
    d = stored_w.shape[1]
    raw = _match(encode_texts(names, model_name, device), d)
    s = F.normalize(stored_w, p=2, dim=-1)
    r = F.normalize(raw.to(stored_w.device), p=2, dim=-1)
    return (s * r).sum(-1).cpu()


def main() -> int:
    guard = get_default_guard()
    dual = guard.store.model.dual_emb
    adapter = guard.store.adapter
    dev = guard.store.device

    idx_to_name = {i: n for n, i in adapter.entity_to_idx.items()}
    n = dual.v_embed.weight.shape[0]
    names_raw = [idx_to_name.get(i, f"__{i}__") for i in range(n)]
    names_spaced = [s.replace("_", " ") for s in names_raw]
    vw = dual.v_embed.weight.detach()
    uw = dual.u_embed.weight.detach()
    print(f"entities={n}  d_v={vw.shape[1]}  d_u={uw.shape[1]}")
    print(f"v_embed norms: mean={vw.norm(dim=-1).mean():.3f} "
          f"min={vw.norm(dim=-1).min():.3f} max={vw.norm(dim=-1).max():.3f}  "
          f"(MiniLM-384 truncated-to-256 unit vec ~ 0.8)\n")

    for label, model_name, w in (("v / MiniLM", _V_MODEL, vw), ("u / nli-mpnet", _U_MODEL, uw)):
        cos_raw = _cos_to_stored(w, names_raw, model_name, dev)
        cos_spc = _cos_to_stored(w, names_spaced, model_name, dev)
        # best of the two name forms per entity
        best = torch.maximum(cos_raw, cos_spc)
        print(f"=== {label}: cos(stored_embed, encoder(name)) in raw d-space ===")
        print(f"  name as-is   : mean={cos_raw.mean():.3f} median={cos_raw.median():.3f} min={cos_raw.min():.3f}")
        print(f"  name spaced  : mean={cos_spc.mean():.3f} median={cos_spc.median():.3f} min={cos_spc.min():.3f}")
        print(f"  best-of-two  : mean={best.mean():.3f} median={best.median():.3f} min={best.min():.3f}")
        # show a few named entities
        show = [e for e in ("udaap", "fee", "overdraft", "unfair", "deceptive",
                            "hidden_fee", "reg_e", "debit_fraud", "mortgage_servicing")
                if e in adapter.entity_to_idx]
        for e in show:
            i = adapter.entity_to_idx[e]
            print(f"    {e:20s} as-is={cos_raw[i]:+.3f}  spaced={cos_spc[i]:+.3f}")
        print()

    print("interpretation: cos near 1 => stored embedding IS the warm-start (light "
          "training, my earlier 0.01 was a probe artifact); cos near 0 => genuine drift.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
