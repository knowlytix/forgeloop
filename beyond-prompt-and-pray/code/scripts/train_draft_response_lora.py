"""LoRA fine-tune Qwen2.5-3B-Instruct for the draft_response tool.

Qwen port of the Chapter-31 draft-response LoRA (replaces the TinyGPT LoRA).
The base Qwen2.5-3B-Instruct is frozen; a small LoRA adapter on the attention
projections is trained by supervised fine-tuning on (complaint, issue, policy
summary) -> grounded reply pairs. Loss is masked to the completion only — the
prompt tokens are ignored so the adapter learns to *write the reply*, not to
reproduce the prompt.

Prompt format is kept identical to the training data so inference stays on
distribution:

    Complaint: <message>
    Issue: <issue>
    Policy: <one-line policy summary>
    Response: <reply>

The corpus is the MeMo v3 set (``draft_response_memo_v3.jsonl``): complaint-
shaped prompts paired with strict, graph-grounded replies, each validated
against the policy goldens (cite keyword + exact figure + no forbidden phrase).
It is built by ``benchmarks/draft_adapter/build_memo_data.py``. Training uses
fewer epochs and a lower LR than the original 121-pair run, which over-fit the
adapter below the base model's own prompted quality.

Artifacts: a PEFT adapter saved to ``data/draft_response_lm_qwen/`` via
``save_pretrained``. Greedy decoding at inference -> deterministic drafts.

Run:
    ~/cluster/spark-venv/bin/python scripts/train_draft_response_lora.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO_ROOT / "data" / "training" / "bank_policy"
_OUT_DIR = _REPO_ROOT / "data" / "draft_response_lm_qwen"
_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
_MAX_LEN = 256


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--model-id", default=_MODEL_ID)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    pairs = _read_jsonl(_DATA_DIR / "draft_response_memo_v3.jsonl")
    print(f"SFT pairs: {len(pairs)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"loading {args.model_id} on {device} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map=device.type if device.type == "cuda" else None,
    )
    if device.type != "cuda":
        model = model.to(device)

    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.train()

    # Build masked examples: prompt tokens -> label -100, completion -> ids.
    eos = tokenizer.eos_token
    examples = []
    for ex in pairs:
        prompt = ex["user"] + " "  # data's "Response:" line, trailing space
        completion = ex["assistant"].strip() + eos
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        input_ids = (p_ids + c_ids)[:_MAX_LEN]
        labels = ([-100] * len(p_ids) + c_ids)[:_MAX_LEN]
        examples.append((input_ids, labels))

    def collate(batch):
        maxlen = max(len(x[0]) for x in batch)
        pad = tokenizer.pad_token_id
        ids, labs, mask = [], [], []
        for input_ids, labels in batch:
            n = maxlen - len(input_ids)
            ids.append(input_ids + [pad] * n)
            labs.append(labels + [-100] * n)
            mask.append([1] * len(input_ids) + [0] * n)
        return (
            torch.tensor(ids, device=device),
            torch.tensor(labs, device=device),
            torch.tensor(mask, device=device),
        )

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    n = len(examples)
    for epoch in range(args.epochs):
        perm = torch.randperm(n).tolist()
        total = 0.0
        steps = 0
        for i in range(0, n, args.batch_size):
            batch = [examples[j] for j in perm[i : i + args.batch_size]]
            ids, labs, mask = collate(batch)
            out = model(input_ids=ids, attention_mask=mask, labels=labs)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            opt.step()
            opt.zero_grad()
            total += float(out.loss.item())
            steps += 1
        print(f"epoch {epoch:2d}  loss {total / max(steps,1):.4f}")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(_OUT_DIR)
    tokenizer.save_pretrained(_OUT_DIR)
    print(f"saved adapter -> {_OUT_DIR}")

    # Quick grounding eval on the held-out eval set.
    eval_path = _DATA_DIR / "draft_response_eval.jsonl"
    if eval_path.exists():
        model.eval()
        cases = _read_jsonl(eval_path)
        cite_ok = forb_ok = 0
        for c in cases:
            prompt = c["prompt"] + " "
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                gen = model.generate(
                    **enc, max_new_tokens=64, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            text = tokenizer.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            low = text.lower()
            if all(k.lower() in low for k in c.get("expected_citation_keywords", [])):
                cite_ok += 1
            if not any(re.search(re.escape(f.lower()), low) for f in c.get("forbidden_phrases", [])):
                forb_ok += 1
        print(f"eval: citation {cite_ok}/{len(cases)}  no-forbidden {forb_ok}/{len(cases)}")


if __name__ == "__main__":
    main()
