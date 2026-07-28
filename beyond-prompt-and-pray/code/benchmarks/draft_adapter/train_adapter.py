"""Fine-tune a Qwen2.5-3B draft adapter on the MeMo v3 corpus: LoRA or RoRA.

Same masked-completion SFT objective for both methods (prompt tokens are
ignored; only the grounded reply is supervised), so the only variable is the
adapter family:

  - ``--method lora`` : additive low-rank update on the attention projections
    (PEFT). Retuned vs. the shipped adapter --- fewer epochs and a lower LR ---
    to stop the overfitting that dragged drafts below the prompted baseline.
  - ``--method rora`` : orthogonal input-rotation adapter (norm-preserving).

Trains on ``data/training/bank_policy/draft_response_memo_v3.jsonl`` and writes
to a NEW directory (never the shipped ``data/draft_response_lm_qwen``). Run from
the repo root with the venv active, e.g.:

    python -m benchmarks.draft_adapter.train_adapter --method lora \
        --out data/draft_response_lora_memo_v3 --epochs 5 --lr 1e-4
    python -m benchmarks.draft_adapter.train_adapter --method rora \
        --out data/draft_response_rora_memo_v3 --epochs 10 --lr 5e-3 --rank 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "training" / "bank_policy" / "draft_response_memo_v3.jsonl"
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MAX_LEN = 256


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def build_examples(pairs, tokenizer):
    eos = tokenizer.eos_token
    examples = []
    for ex in pairs:
        prompt = ex["user"] + " "             # trailing space matches inference
        completion = ex["assistant"].strip() + eos
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        input_ids = (p_ids + c_ids)[:MAX_LEN]
        labels = ([-100] * len(p_ids) + c_ids)[:MAX_LEN]
        examples.append((input_ids, labels))
    return examples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["lora", "rora"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--rank", type=int, default=None, help="LoRA r / RoRA rank")
    ap.add_argument("--alpha", type=int, default=32, help="LoRA alpha")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    pairs = read_jsonl(DATA)
    print(f"SFT pairs: {len(pairs)}  method={args.method}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"loading {MODEL_ID} on {device} ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map=device.type if device.type == "cuda" else None,
    )
    if device.type != "cuda":
        model = model.to(device)

    if args.method == "lora":
        from peft import LoraConfig, get_peft_model

        rank = args.rank or 16
        cfg = LoraConfig(
            r=rank, lora_alpha=args.alpha, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()
    else:
        from benchmarks.draft_adapter.rora_adapter import inject_rora, save_rora

        rank = args.rank or 8
        n = inject_rora(model, rank=rank)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"injected {n} RoRA adapters; trainable {trainable:,}/{total:,} "
              f"({100*trainable/total:.3f}%)")

    model.train()
    examples = build_examples(pairs, tokenizer)

    def collate(batch):
        maxlen = max(len(x[0]) for x in batch)
        pad = tokenizer.pad_token_id
        ids, labs, mask = [], [], []
        for input_ids, labels in batch:
            k = maxlen - len(input_ids)
            ids.append(input_ids + [pad] * k)
            labs.append(labels + [-100] * k)
            mask.append([1] * len(input_ids) + [0] * k)
        return (torch.tensor(ids, device=device),
                torch.tensor(labs, device=device),
                torch.tensor(mask, device=device))

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    n = len(examples)
    for epoch in range(args.epochs):
        perm = torch.randperm(n).tolist()
        total, steps = 0.0, 0
        for i in range(0, n, args.batch_size):
            batch = [examples[j] for j in perm[i:i + args.batch_size]]
            ids, labs, mask = collate(batch)
            out = model(input_ids=ids, attention_mask=mask, labels=labs)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad()
            total += float(out.loss.item())
            steps += 1
        print(f"epoch {epoch:2d}  loss {total/max(steps,1):.4f}")

    out_dir = Path(args.out)
    if args.method == "lora":
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
    else:
        from benchmarks.draft_adapter.rora_adapter import save_rora

        save_rora(model, out_dir, rank=rank, base_model=MODEL_ID)
        tokenizer.save_pretrained(out_dir)
    print(f"saved {args.method} adapter -> {out_dir}")


if __name__ == "__main__":
    main()
