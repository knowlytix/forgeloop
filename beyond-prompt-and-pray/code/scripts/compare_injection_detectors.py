"""Compare two trainable prompt-injection detectors on the same data, picking the
better for the capstone Policy gate.

  A) Qwen3-4B + LoRA (SEQ_CLS head) -- the heavy approach (mirrors classify_complaint).
  B) knowlytix.embedding low-rank SFT (rotation) prototype classifier -- the light approach.

Both train on data/training/injection_doe.jsonl ({none, prompt_injection,
prohibited_advice}, clarity-augmented, with misleading-benign hard negatives).
Held-out eval = the DoE's OWN obfuscated rows from data/capstone_doe_results.csv
(prompt_injection = case-010/017, the ones today's zero-shot gate misses; none =
benign, incl. misleading). Gate semantics: a detector "fires" when it predicts a
label != none. We report injection RECALL (fire on injections), benign
FALSE-FIRE rate (fire on none), balanced accuracy, plus model size / train time /
inference latency so the pick weighs accuracy AND cost.

    python scripts/compare_injection_detectors.py
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from agentlab.models.constants import DEFAULT_QWEN_MODEL

_ROOT = Path(__file__).resolve().parents[1]
_CORPUS = _ROOT / "data" / "training" / "injection_doe.jsonl"
_DOE = _ROOT / "data" / "capstone_doe_results.csv"
_MODEL_ID = DEFAULT_QWEN_MODEL


def _load_corpus():
    return [json.loads(l) for l in _CORPUS.read_text().splitlines() if l.strip()]


def _eval_set():
    rows = list(csv.DictReader(_DOE.open()))
    inj = [r["message"] for r in rows if r["regulatory"] == "prompt_injection"]
    none = [r["message"] for r in rows if r["regulatory"] == "none"]
    return inj, none


def _metrics(pred_inj, pred_none):
    fire = lambda p: p not in (None, "none")
    rec = sum(fire(p) for p in pred_inj) / max(len(pred_inj), 1)
    fpr = sum(fire(p) for p in pred_none) / max(len(pred_none), 1)
    bal = 0.5 * (rec + (1 - fpr))
    return {"injection_recall": rec, "benign_false_fire": fpr, "balanced_acc": bal}


# --------------------------------------------------------------------------- #
# Approach B: embedding-SFT
# --------------------------------------------------------------------------- #
def run_embedding_sft(inj, none, device):
    from knowlytix.embedding import EmbeddingSFTConfig, finetune_embedding

    t0 = time.time()
    cfg = EmbeddingSFTConfig(rank=1, mode="rotation", epochs=120, val_split=0.2,
                             device=str(device), seed=0)
    ft = finetune_embedding(str(_CORPUS), cfg, text_col="message", label_col="label")
    train_s = time.time() - t0
    params = sum(p.numel() for p in ft.adapter.parameters())
    t1 = time.time()
    pred_inj, _ = ft.classify(inj)
    pred_none, _ = ft.classify(none)
    infer_ms = 1000 * (time.time() - t1) / max(len(inj) + len(none), 1)
    m = _metrics(pred_inj, pred_none)
    m.update(trainable_params=params, train_s=train_s, infer_ms_per_msg=infer_ms,
             classes=ft.label_order)
    return m


# --------------------------------------------------------------------------- #
# Approach A: Qwen 3B + LoRA (SEQ_CLS)
# --------------------------------------------------------------------------- #
def run_qwen_lora(inj, none, device, epochs=6, bs=8, rank=16):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from peft import LoraConfig, TaskType, get_peft_model

    rows = _load_corpus()
    labels = sorted({r["label"] for r in rows})
    l2i = {l: i for i, l in enumerate(labels)}
    tok = AutoTokenizer.from_pretrained(_MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    t0 = time.time()
    model = AutoModelForSequenceClassification.from_pretrained(
        _MODEL_ID, num_labels=len(labels),
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    model.config.pad_token_id = tok.pad_token_id
    lora = LoraConfig(task_type=TaskType.SEQ_CLS, r=rank, lora_alpha=2 * rank,
                      lora_dropout=0.05, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lora).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def enc(msgs):
        return tok(msgs, return_tensors="pt", padding=True, truncation=True,
                   max_length=128).to(device)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    import random as _r
    rng = _r.Random(0)
    model.train()
    for ep in range(epochs):
        order = list(range(len(rows))); rng.shuffle(order)
        tot = steps = 0
        for i in range(0, len(order), bs):
            batch = [rows[j] for j in order[i:i + bs]]
            e = enc([b["message"] for b in batch])
            y = torch.tensor([l2i[b["label"]] for b in batch], device=device)
            out = model(**e, labels=y)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad()
            tot += float(out.loss); steps += 1
        print(f"    [lora] epoch {ep} loss {tot/steps:.4f}", flush=True)
    train_s = time.time() - t0

    model.eval()

    @torch.no_grad()
    def predict(msgs):
        preds = []
        for i in range(0, len(msgs), 16):
            logits = model(**enc(msgs[i:i + 16])).logits
            preds.extend(labels[k] for k in logits.argmax(-1).tolist())
        return preds

    t1 = time.time()
    pred_inj = predict(inj); pred_none = predict(none)
    infer_ms = 1000 * (time.time() - t1) / max(len(inj) + len(none), 1)
    m = _metrics(pred_inj, pred_none)
    m.update(trainable_params=params, train_s=train_s, infer_ms_per_msg=infer_ms, classes=labels)
    return m


def main() -> int:
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inj, none = _eval_set()
    print(f"eval: {len(inj)} obfuscated injections (case-010/017) + {len(none)} benign (none)\n")

    print("[B] embedding-SFT (rotation, rank-1) ...")
    b = run_embedding_sft(inj, none, dev)
    print("[A] Qwen3-4B + LoRA (rank-16) ...")
    a = run_qwen_lora(inj, none, dev)

    def show(name, m):
        print(f"\n=== {name} ===")
        print(f"  injection recall : {m['injection_recall']:.2f}")
        print(f"  benign false-fire: {m['benign_false_fire']:.2f}")
        print(f"  balanced acc     : {m['balanced_acc']:.2f}")
        print(f"  trainable params : {m['trainable_params']:,}")
        print(f"  train time       : {m['train_s']:.1f}s")
        print(f"  infer / msg      : {m['infer_ms_per_msg']:.1f} ms")
    show("A) Qwen3-4B LoRA", a)
    show("B) embedding-SFT", b)

    print("\n=== verdict ===")
    print(f"  recall  A={a['injection_recall']:.2f}  B={b['injection_recall']:.2f}")
    print(f"  false-fire  A={a['benign_false_fire']:.2f}  B={b['benign_false_fire']:.2f}")
    print(f"  params  A={a['trainable_params']:,}  B={b['trainable_params']:,}  "
          f"(A/B ~ {a['trainable_params']/max(b['trainable_params'],1):.0f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
