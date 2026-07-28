# SPDX-License-Identifier: Apache-2.0
"""LoRA SFT: fine-tune Qwen3-4B to map a question -> GMS query triples.

The hypothesis under test (Route B) is that fine-tuning on the DoE-inverted
(question -> triple) pairs lets a small model emit graph-vocabulary triples
directly, replacing the prompted schema-grounded extractor AND the v-space
binder. So training uses a MINIMAL system prompt -- task + output format only,
NO vocabulary list and NO few-shot examples. The graph's entities/relations must
be learned from the 292 training pairs, not injected at prompt time.

knowlytix has no LoRA utility (only memo/train.py, a full-FT trainer with a rigid
schema), so this is a self-contained peft+trl-free loop: manual prompt-masked
tokenization + HF Trainer. Promoting a general LoRA util to knowlytix is a
follow-up. Run on spark-ef84 (GPU), capped at 50% memory.
"""
from __future__ import annotations

import json
import os
import sys

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, Trainer, TrainingArguments)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.environ.get("NL2T_DATA", os.path.join(REPO_ROOT, "data", "nl2triple"))
ADAPTER_DIR = os.path.join(DATA, "qwen3_4b_triple_lora")
MODEL = "Qwen/Qwen3-4B-Instruct-2507"

SYSTEM = (
    "You translate a question about a company's annual report into knowledge-graph "
    "query triples. Output ONLY a JSON array of objects with keys head, relation, "
    'tail. Mark the single value the question asks for as the bare string "?". '
    "Relations are lowercase snake_case."
)


def triples_to_json(triples):
    return json.dumps([{"head": h, "relation": r, "tail": t} for h, r, t in triples],
                      separators=(",", ":"))


def build_dataset(tok, path):
    rows = [json.loads(l) for l in open(path)]
    feats = []
    for r in rows:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": r["question"]}]
        prompt = tok.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True)
        completion = triples_to_json(r["triples"])
        full = prompt + completion + tok.eos_token
        pids = tok(prompt, add_special_tokens=False)["input_ids"]
        fids = tok(full, add_special_tokens=False)["input_ids"]
        labels = [-100] * len(pids) + fids[len(pids):]
        feats.append({"input_ids": fids, "attention_mask": [1] * len(fids),
                      "labels": labels})
    return Dataset.from_list(feats)


def main():
    # Honor the shared-GPU 50% cap where the platform supports it.
    try:
        torch.cuda.set_per_process_memory_fraction(0.5, 0)
    except Exception as e:  # noqa: BLE001  (unified-memory GB10 may report N/A)
        print("note: could not set memory fraction:", e)

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ds = build_dataset(tok, os.path.join(DATA, "train.jsonl"))
    val_ds = build_dataset(tok, os.path.join(DATA, "val.jsonl"))
    print(f"train={len(train_ds)} val={len(val_ds)}")

    args = TrainingArguments(
        output_dir=os.path.join(DATA, "_train_ckpt"),
        num_train_epochs=8,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        report_to=[],
        seed=42,
    )
    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=val_ds, data_collator=collator)
    trainer.train()

    os.makedirs(ADAPTER_DIR, exist_ok=True)
    model.save_pretrained(ADAPTER_DIR)
    tok.save_pretrained(ADAPTER_DIR)
    print("saved adapter to", ADAPTER_DIR)

    # Sanity check: greedy-decode a few val questions.
    model.eval()
    rows = [json.loads(l) for l in open(os.path.join(DATA, "val.jsonl"))][:5]
    for r in rows:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": r["question"]}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=96, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\nQ: {r['question'][:70]}")
        print(f"  gold: {triples_to_json(r['triples'])}")
        print(f"  pred: {gen.strip()[:120]}")


if __name__ == "__main__":
    main()
