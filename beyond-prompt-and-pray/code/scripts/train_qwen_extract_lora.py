#!/usr/bin/env python
"""Fine-tune Qwen-3B LoRA classifiers for product + issue, and benchmark.

The geo_task arm tunes a low-rank adapter over the FROZEN MiniLM embedding
(prototype classifier, issue 0.62). This arm instead LoRA-fine-tunes the Qwen-3B
transformer itself (SEQ_CLS head), unfreezing the encoder to reshape features --
the same ladder rung classify_complaint climbed. Trains on the generated
synthetic corpus (data/training/extract_{issue,product}_train.jsonl), evaluates
on the same 89 held-out rephrased DoE rows, strict + accept-set, vs geo_task
(0.62) / hybrid (0.46). Saves adapters to data/extract_{issue,product}_lora/.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
_MAX_LEN = 64


def _read(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def _train_one(name: str, epochs: int, rank: int, device):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from peft import LoraConfig, TaskType, get_peft_model

    train = _read(_REPO / "data" / "training" / f"extract_{name}_train.jsonl")
    labels = sorted({r["label"] for r in train})
    lab2idx = {l: i for i, l in enumerate(labels)}
    tok = AutoTokenizer.from_pretrained(_MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        _MODEL_ID, num_labels=len(labels),
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map=device.type if device.type == "cuda" else None)
    if device.type != "cuda":
        model = model.to(device)
    model.config.pad_token_id = tok.pad_token_id
    lora = LoraConfig(task_type=TaskType.SEQ_CLS, r=rank, lora_alpha=2 * rank,
                      lora_dropout=0.05, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lora)

    def enc(msgs):
        return tok(msgs, truncation=True, max_length=_MAX_LEN, padding=True,
                   return_tensors="pt")

    ex = [(r["message"], lab2idx[r["label"]]) for r in train]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    model.train()
    n = len(ex)
    for epoch in range(epochs):
        perm = torch.randperm(n).tolist()
        tot = steps = 0
        for i in range(0, n, 16):
            b = [ex[j] for j in perm[i:i + 16]]
            e = enc([m for m, _ in b]).to(device)
            y = torch.tensor([c for _, c in b], device=device)
            out = model(**e, labels=y)
            out.loss.backward()
            opt.step(); opt.zero_grad()
            tot += float(out.loss); steps += 1
        if epoch % 3 == 0 or epoch == epochs - 1:
            print(f"  [{name}] epoch {epoch} loss {tot/steps:.4f}", flush=True)
    model.save_pretrained(_REPO / "data" / f"extract_{name}_lora")
    (_REPO / "data" / f"extract_{name}_lora" / "labels.json").write_text(json.dumps(labels))
    model.eval()

    @torch.no_grad()
    def predict(msgs):
        out = []
        for i in range(0, len(msgs), 32):
            e = enc(msgs[i:i + 32]).to(device)
            out.extend(labels[k] for k in model(**e).logits.argmax(-1).tolist())
        return out
    print(f"  [{name}] val_acc(train-set sanity) labels={labels}")
    return predict


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("training issue LoRA..."); pred_issue = _train_one("issue", 12, 16, device)
    print("training product LoRA..."); pred_prod = _train_one("product", 12, 16, device)

    diag = json.loads((_REPO / "data" / "diag_extract_facts.json").read_text())
    rows = diag["rows"]
    cases = json.loads((_REPO / "data" / "eval_cases" / "cases.json").read_text())
    by_id = {c["id"]: c for c in (cases if isinstance(cases, list) else cases["cases"])}
    msgs = [r["message"] for r in rows]
    pi = pred_issue(msgs)
    pp = pred_prod(msgs)

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
    print("\n=== qwen_lora (fine-tuned classifier) on 89 rows ===")
    for k in ("p_strict", "i_strict", "j_strict", "i_acc", "j_acc"):
        print(f"  {k:10} {rate[k][0]}  ({rate[k][1]})")
    print("issue confusion:")
    for (e, p), nn in conf.most_common():
        print(f"   {e:18} -> {str(p):18} {nn}{'' if e == p else '  <-- miss'}")
    (_REPO / "data" / "benchmark_qwen_lora.json").write_text(json.dumps(
        {"rates": rate, "issue_confusion": {f"{e} -> {p}": nn for (e, p), nn in conf.most_common()}},
        indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
