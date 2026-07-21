"""Generate label-preserving REGULATORY paraphrases via the DoE factor design, to
train the rank-1 adapter on more than the ~17 real UDAAP messages.

Each eval seed (cases.json) is mapped to its regulatory label (from
capstone_doe_results.csv), then rewritten across the DoE semantic factors --- with
the clarity axis swept hard (Clear/Ambiguous/Misleading), since misleading phrasing
is exactly where recall dies. Surface aliasing/noise stamped after. Dollar amounts
guarded. Output carries the seed id so training can split SEED-GROUPED (no leakage).

    python scripts/augment_regulatory_doe.py --per-seed 12
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import csv
import io
import json
import random
from pathlib import Path

# reuse the existing augmentation primitives
from augment_complaint_training_doe import (  # type: ignore
    _SEMANTIC, _apply_aliasing, _apply_noise, _amounts, _build_prompt,
)

_ROOT = Path(__file__).resolve().parents[1]
_CASES = _ROOT / "data" / "eval_cases" / "cases.json"
_CSV = _ROOT / "data" / "capstone_doe_results.csv"
_OUT = _ROOT / "data" / "training" / "regulatory_doe_augmented.jsonl"
_INSCOPE = {"UDAAP", "Reg_X", "none"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-seed", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    # seed id -> regulatory, from cases.json factors (the authoritative labels).
    # (Previously sourced from capstone_doe_results.csv, which can lag a relabel.)
    cases = json.loads(_CASES.read_text())
    seeds = [{"id": c["id"], "message": c["message"],
              "regulatory": (c.get("factors") or {}).get("regulatory")}
             for c in cases if (c.get("factors") or {}).get("regulatory") in _INSCOPE]
    dist = collections.Counter(s["regulatory"] for s in seeds)
    print(f"in-scope seeds: {len(seeds)}  by regulatory: {dict(dist)}")

    # clarity-heavy factor design: sweep clarity x style, moderate elsewhere
    clar = list(_SEMANTIC["clarity"].keys())
    styles = list(_SEMANTIC["style"].keys())
    rng = random.Random(args.seed)

    emo = list(_SEMANTIC["emotional_frame"].keys())
    exp = list(_SEMANTIC["expertise"].keys())

    def factor_row(i):
        return {"clarity": clar[i % len(clar)],
                "style": styles[(i // len(clar)) % len(styles)],
                "length": "Medium", "specificity": list(_SEMANTIC["specificity"])[i % 2],
                "paraphrase_depth": "Light",
                "expertise": exp[(i // len(clar)) % len(exp)],
                "emotional_frame": emo[i % len(emo)],
                "entity_aliasing": ["Canonical", "Synonym", "Abbreviated"][i % 3],
                "noise": "Clean" if i % 3 else "Typos"}

    plan = []
    for s in seeds:
        for i in range(args.per_seed):
            row = factor_row(i)
            plan.append((s, row, _build_prompt(s["message"], row)))

    import torch
    from agentlab.models.qwen_adapter import _DEFAULT_MODEL, _load
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, model = _load(_DEFAULT_MODEL, device)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    @torch.no_grad()
    def gen(prompts):
        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         tokenize=False, add_generation_prompt=True) for p in prompts]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        out = model.generate(**enc, max_new_tokens=128, do_sample=False, pad_token_id=tok.eos_token_id)
        return [tok.decode(g, skip_special_tokens=True).strip().strip('"')
                for g in out[:, enc["input_ids"].shape[1]:]]

    rewrites = []
    bs = max(1, args.batch_size)
    for b in range(0, len(plan), bs):
        with contextlib.redirect_stderr(io.StringIO()):
            rewrites.extend(gen([p for *_, p in plan[b:b + bs]]))
        print(f"  generated {min(b + bs, len(plan))}/{len(plan)}", flush=True)

    out_records, dropped = [], 0
    for (s, row, _), rw in zip(plan, rewrites):
        if not rw:
            rw = s["message"]
        rw = _apply_noise(_apply_aliasing(rw, row["entity_aliasing"]), row["noise"], rng)
        if set(_amounts(s["message"])) - set(_amounts(rw)):   # dollar amount must survive
            dropped += 1; continue
        out_records.append({"message": rw, "regulatory": s["regulatory"],
                            "_seed": s["id"], "_clarity": row["clarity"]})

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text("\n".join(json.dumps(r) for r in out_records) + "\n")
    print(f"\nwrote {len(out_records)} paraphrases ({dropped} dropped) -> {_OUT}")
    print("by regulatory:", dict(collections.Counter(r["regulatory"] for r in out_records)))
    print("by clarity   :", dict(collections.Counter(r["_clarity"] for r in out_records)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
