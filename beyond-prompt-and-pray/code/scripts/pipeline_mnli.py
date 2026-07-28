#!/usr/bin/env python
"""Local (Sonnet-free) policy mapping: complaint -> MNLI entailment -> policy atoms.

The applicability layer is a dedicated MNLI cross-encoder (DeBERTa-v3 trained on
MNLI/FEVER/ANLI), NOT an LLM-in-prompt. For each complaint (premise) and each
policy-atom trigger condition (hypothesis) it scores P(entailment); an atom's
score is the max over its triggers, a policy's score the max over its atoms. The
expected_policy should rank high; a vague complaint entails NO trigger, so a
calibrated accept threshold maps it to null (no fabricated policy). Runtime uses
ONLY local models (this NLI model; Qwen for facts in a later step) -- Sonnet is
reserved for offline training-data generation, never the pipeline.

Premise = raw complaint for v1; swap in Qwen-extracted structured facts later.
Scored on the 89 held-out rows: policy recall@3 + vague->null, vs search_policy ~0.80.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch
import yaml

_REPO = Path(__file__).resolve().parents[1]
_NLI = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
_TOPK = 3


def main() -> int:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(_NLI)
    model = AutoModelForSequenceClassification.from_pretrained(_NLI).to(device).eval()
    ent_idx = [i for i, l in model.config.id2label.items() if l.lower() == "entailment"][0]

    atoms = yaml.safe_load((_REPO / "data" / "policy_atoms.yaml").read_text())
    # flat (policy, trigger) hypotheses
    hyps = [(a["policy"], t) for a in atoms for t in a["trigger_conditions"]]

    diag = json.loads((_REPO / "data" / "diag_extract_facts.json").read_text())
    rows = diag["rows"]
    cases = json.loads((_REPO / "data" / "eval_cases" / "cases.json").read_text())
    by_id = {c["id"]: c for c in (cases if isinstance(cases, list) else cases["cases"])}

    @torch.no_grad()
    def entail(premise: str) -> dict[str, float]:
        pairs = [(premise, h) for _, h in hyps]
        scores = []
        for i in range(0, len(pairs), 32):
            chunk = pairs[i:i + 32]
            enc = tok([p for p, _ in chunk], [h for _, h in chunk],
                      truncation=True, max_length=256, padding=True,
                      return_tensors="pt").to(device)
            prob = model(**enc).logits.softmax(-1)[:, ent_idx]
            scores.extend(prob.tolist())
        per_policy: dict[str, float] = defaultdict(float)
        for (pol, _), s in zip(hyps, scores):
            per_policy[pol] = max(per_policy[pol], s)
        return dict(per_policy)

    # score every row
    row_scores = [entail(r["message"]) for r in rows]

    # calibrate accept threshold on vague cohort (expected_policy None) -> top policy
    # score should stay below threshold so the row maps to null. ceiling 0.10.
    vague_tops = [max(row_scores[i].values()) for i, r in enumerate(rows)
                  if by_id[r["seed_case"]].get("expected_policy") is None]
    ceiling = 0.10
    thr = 0.0
    for cut in [0.0] + sorted(set(vague_tops)):
        if sum(1 for t in vague_tops if t >= cut) / max(len(vague_tops), 1) <= ceiling:
            thr = float(cut); break
    else:
        thr = float(max(vague_tops)) + 1e-6 if vague_tops else 0.5

    pol_hit = pol_tot = vague_ok = vague_tot = 0
    detail = []
    for i, r in enumerate(rows):
        case = by_id[r["seed_case"]]
        exp = case.get("expected_policy")
        acc = set(case.get("acceptable_policies") or ([exp] if exp else []))
        ranked = sorted(row_scores[i].items(), key=lambda kv: kv[1], reverse=True)
        topk = [p for p, s in ranked if s >= thr][:_TOPK]
        if exp:
            pol_tot += 1
            pol_hit += int(bool(acc & set(topk)))
        else:
            vague_tot += 1
            vague_ok += int(len(topk) == 0)
        detail.append({"seed": case["id"], "exp": exp,
                       "top": [(p, round(s, 2)) for p, s in ranked[:3]], "mapped": topk})

    out = {
        "nli_model": _NLI, "premise": "raw_complaint", "accept_threshold": round(thr, 4),
        "policy_recall_at_%d" % _TOPK: [pol_hit, pol_tot,
                                        round(pol_hit / pol_tot, 3) if pol_tot else None],
        "vague_to_null": [vague_ok, vague_tot,
                          round(vague_ok / vague_tot, 3) if vague_tot else None],
    }
    print(json.dumps(out, indent=2))
    print("sample traces (seed | expected | top-3 entailment | mapped):")
    for d in detail[:10]:
        print(f"   {d['seed']} exp={d['exp']} top={d['top']} -> {d['mapped']}")
    (_REPO / "data" / "pipeline_mnli.json").write_text(
        json.dumps({"summary": out, "detail": detail}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
