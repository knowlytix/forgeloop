"""Train + save the Qwen-3B LoRA injection classifier (the comparison winner) for
the Policy gate. Labels {none, prompt_injection, prohibited_advice} from
data/training/injection_doe.jsonl. Saves a PEFT adapter + labels.json that
LoraComplaintClassifier can load.

    python scripts/train_injection_classifier.py --epochs 6
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
_CORPUS = _ROOT / "data" / "training" / "injection_doe.jsonl"
_SAVE = _ROOT / "data" / "injection_classifier_lora"
_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from peft import LoraConfig, TaskType, get_peft_model

    rows = [json.loads(l) for l in _CORPUS.read_text().splitlines() if l.strip()]
    labels = sorted({r["label"] for r in rows})
    l2i = {l: i for i, l in enumerate(labels)}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"labels: {labels} | train={len(rows)}")

    tok = AutoTokenizer.from_pretrained(_MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        _MODEL_ID, num_labels=len(labels),
        dtype=torch.bfloat16 if dev.type == "cuda" else torch.float32)
    model.config.pad_token_id = tok.pad_token_id
    lora = LoraConfig(task_type=TaskType.SEQ_CLS, r=args.rank, lora_alpha=2 * args.rank,
                      lora_dropout=0.05, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lora).to(dev)

    def enc(msgs):
        return tok(msgs, return_tensors="pt", padding=True, truncation=True, max_length=128).to(dev)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    rng = random.Random(args.seed)
    model.train()
    for ep in range(args.epochs):
        order = list(range(len(rows))); rng.shuffle(order)
        tot = steps = 0
        for i in range(0, len(order), args.batch_size):
            batch = [rows[j] for j in order[i:i + args.batch_size]]
            out = model(**enc([b["message"] for b in batch]),
                        labels=torch.tensor([l2i[b["label"]] for b in batch], device=dev))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad()
            tot += float(out.loss.detach()); steps += 1
        print(f"  epoch {ep} loss {tot/steps:.4f}", flush=True)

    _SAVE.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(_SAVE)
    (_SAVE / "labels.json").write_text(json.dumps(labels))
    print(f"saved injection LoRA -> {_SAVE} (labels={labels})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
