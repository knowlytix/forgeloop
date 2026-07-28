#!/usr/bin/env python
"""Qwen-paraphrase arm: rewrite complaint -> policy language, then classify.

The intent->policy-language gap is why GMS routing failed on raw complaints. This
arm asks Qwen to restate the complaint in concise banking-policy terms (naming the
product and the policy issue), then classifies the PARAPHRASE with the same
DoE-trained embedding-SFT classifier (FineTunedEmbedding) used by geo_task. If
paraphrasing into policy language helps, issue accuracy should beat classifying
the raw complaint. Evaluated on the 89 held-out rows; product/issue/joint strict
+ accept-set. Compares to geo_task on raw text.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]

_SYS = ("Restate the customer's message as ONE concise banking-policy sentence that "
        "names the product (checking account, credit card, mortgage, loan, or none) "
        "and the policy issue (overdraft/NSF fee, account dispute or servicing, credit "
        "card billing, mortgage/escrow servicing, loan servicing, or a general/unclear "
        "concern). Do not resolve it. Reply with ONLY the sentence.")


def main() -> int:
    from agentlab.models.qwen_adapter import _DEFAULT_MODEL, _load
    from knowlytix.embedding import FineTunedEmbedding

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, model = _load(_DEFAULT_MODEL, device)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    fti = FineTunedEmbedding.load(_REPO / "data" / "extract_encoder_issue")
    ftp = FineTunedEmbedding.load(_REPO / "data" / "extract_encoder_product")

    diag = json.loads((_REPO / "data" / "diag_extract_facts.json").read_text())
    rows = diag["rows"]
    cases = json.loads((_REPO / "data" / "eval_cases" / "cases.json").read_text())
    by_id = {c["id"]: c for c in (cases if isinstance(cases, list) else cases["cases"])}
    msgs = [r["message"] for r in rows]

    @torch.no_grad()
    def paraphrase(batch):
        texts = [tok.apply_chat_template(
            [{"role": "system", "content": _SYS}, {"role": "user", "content": m}],
            tokenize=False, add_generation_prompt=True) for m in batch]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        out = model.generate(**enc, max_new_tokens=64, do_sample=False, pad_token_id=tok.eos_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        return [tok.decode(g, skip_special_tokens=True).strip().strip('"') for g in gen]

    paras = []
    for b in range(0, len(msgs), 16):
        paras.extend(paraphrase(msgs[b:b + 16]))
        print(f"  paraphrased {min(b+16,len(msgs))}/{len(msgs)}", flush=True)

    pi = [l or "general" for l in fti.classify(paras, use_threshold=False)[0]]
    pp = [l or "unknown" for l in ftp.classify(paras, use_threshold=False)[0]]

    acc = {k: [0, 0] for k in ("p_strict", "i_strict", "j_strict", "i_acc", "j_acc")}
    conf = Counter()
    for idx, r in enumerate(rows):
        ep, ei = r["expected_product"], r["expected_issue"]
        case = by_id[r["seed_case"]]
        acc_p = set(case.get("acceptable_products") or ([ep] if ep else []))
        acc_i = set(case.get("acceptable_issues") or ([ei] if ei else []))
        product, issue = pp[idx], pi[idx]
        sp = (product == ep) if ep else None
        si = (issue == ei) if ei else None
        ap = (product in acc_p) if ep else None
        ai = (issue in acc_i) if ei else None
        for key, val in (("p_strict", sp), ("i_strict", si), ("i_acc", ai)):
            if val is not None:
                acc[key][0] += int(val); acc[key][1] += 1
        sj = [v for v in (sp, si) if v is not None]
        aj = [v for v in (ap, ai) if v is not None]
        if sj:
            acc["j_strict"][0] += int(all(sj)); acc["j_strict"][1] += 1
        if aj:
            acc["j_acc"][0] += int(all(aj)); acc["j_acc"][1] += 1
        if ei is not None:
            conf[(ei, issue)] += 1
    rate = {k: (round(c / n, 3) if n else None, f"{c}/{n}") for k, (c, n) in acc.items()}
    print("\n=== qwen_paraphrase -> embedding-SFT classify, 89 rows ===")
    for k in ("p_strict", "i_strict", "j_strict", "i_acc", "j_acc"):
        print(f"  {k:10} {rate[k][0]}  ({rate[k][1]})")
    for ex in list(zip(msgs, paras))[:6]:
        print(f"   {ex[0][:42]!r:46} -> {ex[1][:60]!r}")
    (_REPO / "data" / "benchmark_paraphrase.json").write_text(json.dumps(
        {"rates": rate, "issue_confusion": {f"{e} -> {p}": n for (e, p), n in conf.most_common()},
         "samples": [{"msg": m, "paraphrase": p} for m, p in list(zip(msgs, paras))[:15]]},
        indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
