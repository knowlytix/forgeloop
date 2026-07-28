"""LoRA-plus-head rung of the classifier ladder: train + evaluate, for comparison.

The frozen-encoder logit head can only draw a linear boundary on fixed features.
The next rung unfreezes the encoder through PEFT LoRA adapters and trains a
classification head jointly, so the model can reshape the features. This script
trains Qwen2-for-sequence-classification + LoRA on the SAME train_doe split the
logit head used, and evaluates per-clarity on the SAME held-out test_doe and
external DoE sets, so the numbers drop straight into the head A/B table.

Self-contained: it does not touch the shipped ComplaintClassifier.

    python scripts/train_eval_classifier_lora.py --epochs 6
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
_DATA = _ROOT / "data" / "training" / "complaint_classification"
_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
_MAX_LEN = 128
_EVAL = {
    "test_doe.jsonl": _DATA / "test_doe.jsonl",
    "doe_clarity_eval.jsonl": _ROOT / "data" / "eval_cases" / "doe_clarity_eval.jsonl",
}


def _read(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def _clarity(d):
    return str(d.get("clarity") or d.get("_factors", {}).get("clarity", "clear")).lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-file", default="train_doe.jsonl")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-dir", default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train = _read(_DATA / args.train_file)
    labels = sorted({r["label"] for r in train})
    lab2idx = {l: i for i, l in enumerate(labels)}
    print(f"labels: {labels} | train={len(train)} ({args.train_file})")

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from peft import LoraConfig, TaskType, get_peft_model

    tok = AutoTokenizer.from_pretrained(_MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        _MODEL_ID, num_labels=len(labels),
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map=device.type if device.type == "cuda" else None,
    )
    if device.type != "cuda":
        model = model.to(device)
    model.config.pad_token_id = tok.pad_token_id

    lora = LoraConfig(task_type=TaskType.SEQ_CLS, r=args.rank, lora_alpha=args.alpha,
                      lora_dropout=0.05, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    def encode(msgs):
        return tok(msgs, truncation=True, max_length=_MAX_LEN, padding=True, return_tensors="pt")

    examples = [(r["message"], lab2idx[r["label"]]) for r in train]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    model.train()
    n = len(examples)
    for epoch in range(args.epochs):
        perm = torch.randperm(n).tolist()
        total, steps = 0.0, 0
        for i in range(0, n, args.batch_size):
            batch = [examples[j] for j in perm[i:i + args.batch_size]]
            enc = encode([m for m, _ in batch]).to(device)
            y = torch.tensor([c for _, c in batch], device=device)
            out = model(**enc, labels=y)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad()
            total += float(out.loss)
            steps += 1
        print(f"epoch {epoch}  loss {total / steps:.4f}", flush=True)

    # ---- evaluate per clarity ----
    model.eval()

    @torch.no_grad()
    def predict(msgs):
        preds = []
        for i in range(0, len(msgs), 32):
            enc = encode(msgs[i:i + 32]).to(device)
            logits = model(**enc).logits
            preds.extend(labels[k] for k in logits.argmax(-1).tolist())
        return preds

    print("\nclassifier: lora-head")
    for name, path in _EVAL.items():
        rows = _read(path)
        msgs = [r["message"] for r in rows]
        gold = [r["label"] for r in rows]
        clar = [_clarity(r) for r in rows]
        preds = predict(msgs)
        by = collections.defaultdict(lambda: [0, 0])
        c2o = ct = ok = 0
        for p, g, c in zip(preds, gold, clar):
            by[c][0] += int(p == g); by[c][1] += 1
            ok += int(p == g)
            if g == "complaint":
                ct += 1; c2o += int(p == "other")
        cl = sorted(by)
        print(f"=== {name} ({len(rows)} rows) ===")
        print("  " + " ".join(f"{c}={by[c][0]/by[c][1]:.2f}({by[c][1]})" for c in cl)
              + f"  overall={ok/len(rows):.2f}  comp->other={c2o}/{ct}")

    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.save_dir)
        (Path(args.save_dir) / "labels.json").write_text(json.dumps(labels))
        print(f"saved adapter -> {args.save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
