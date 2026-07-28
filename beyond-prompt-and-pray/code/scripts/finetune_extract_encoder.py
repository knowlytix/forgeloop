#!/usr/bin/env python
"""Fine-tune low-rank task encoders for product and issue (knowlytix.embedding).

Prototype objective (cosine-softmax) over the generated {text,label} corpus, with
per-class abstain thresholds calibrated at fpr_target on the held-out split. The
`unknown`/`general` abstain classes are trained prototypes, so they are reachable
by construction -- the direct cure for the dominant extract trap. Saves two
FineTunedEmbedding artifacts under data/extract_encoder_{product,issue}/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from knowlytix.embedding import EmbeddingSFTConfig, finetune_embedding

_REPO = Path(__file__).resolve().parents[1]


def _fit(name: str, mode: str, rank: int, fpr: float):
    train = _REPO / "data" / "training" / f"extract_{name}_train.jsonl"
    cfg = EmbeddingSFTConfig(rank=rank, mode=mode, objective="prototype",
                             fpr_target=fpr, epochs=150, seed=42)
    ft = finetune_embedding(str(train), cfg)
    out = _REPO / "data" / f"extract_encoder_{name}"
    ft.save(out)
    thr = {l: round(float(t), 4) for l, t in zip(ft.label_order, ft.thresholds.tolist())}
    print(f"[{name}] mode={mode} rank={rank} val_acc={ft.val_accuracy:.3f} "
          f"classes={ft.label_order}")
    print(f"[{name}] per-class abstain thresholds={thr}")
    print(f"[{name}] saved -> {out}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["rotation", "full"], default="full")
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--fpr", type=float, default=0.10)
    args = ap.parse_args()
    _fit("product", args.mode, args.rank, args.fpr)
    _fit("issue", args.mode, args.rank, args.fpr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
