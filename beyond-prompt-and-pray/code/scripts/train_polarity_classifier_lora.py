"""Train the Qwen3-4B LoRA polarity classifier (Gate B of the disclosure comparison).

Same LoRA-plus-head recipe as scripts/train_eval_classifier_lora.py (the capstone
complaint classifier), retargeted at the GMS+DoE polarity corpus
(data/training/polarity/). Input is the surface text; the label is the GMS/pole
ground truth (supported / contradicted / uncertain). Trained on the seed-grouped
train split, evaluated on the held-out test split with a per-surface-form
breakdown, so the phrasing-robustness story is visible. Saves a PEFT adapter that
LoraPolarityClassifier loads.

    python scripts/train_polarity_classifier_lora.py --epochs 6 \
        --save-dir data/polarity_classifier_qwen
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import torch
from agentlab.models.constants import DEFAULT_QWEN_MODEL

_ROOT = Path(__file__).resolve().parents[1]
_DATA = _ROOT / "data" / "training" / "polarity"
_MODEL_ID = DEFAULT_QWEN_MODEL
_MAX_LEN = 96

# The polarity decision is RELATIVE: an asserted stance is judged against the value
# the policy holds. The classifier must see both, so its input pairs the stored
# reference (premise) with the surface statement (hypothesis) -- the same
# information the geometric gate gets via (head, relation, asserted, stored).
_RELATION_PHRASE = {
    "has_unencrypted_channel_pii": "sending personal information over an unencrypted channel",
    "has_redaction": "redaction of personal information in tickets and logs",
    "has_identity_verification": "identity verification before an account is closed",
    "has_fraud_notice_exception": "the exception to the advance-notice rule when fraud is confirmed",
    "has_provisional_credit": "provisional credit to the customer while a dispute is investigated",
}


def _phrase(relation):
    return _RELATION_PHRASE.get(
        relation, (relation[4:] if relation.startswith("has_")
                   else relation).replace("_", " "))


def _pair_nl(r):
    """NL premise+claim: stored fact as a sentence, claim = DoE surface text."""
    return f"Policy: {_phrase(r['relation'])} is {r['stored']}. Claim: {r['message']}"


def _pair_tuple(r):
    """Structured (h,r,t) pair: canonical stored triple vs asserted triple
    (surface-invariant -- the apples-to-apples match to the geometric gate)."""
    return (f"stored: ({r['head']}, {r['relation']}, {r['stored']}) | "
            f"asserted: ({r['head']}, {r['relation']}, {r['asserted']})")


_BUILDERS = {"nl": _pair_nl, "tuple": _pair_tuple}


def _read(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def _surface(d):
    return str(d.get("_factors", {}).get("surface", "?"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--input", choices=["nl", "tuple"], default="nl",
                    help="classifier input representation")
    ap.add_argument("--save-dir", default="")
    args = ap.parse_args()
    build = _BUILDERS[args.input]
    save_dir = args.save_dir or f"data/polarity_classifier_qwen_{args.input}"

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train = _read(_DATA / "train_polarity.jsonl")
    test = _read(_DATA / "test_polarity.jsonl")
    labels = sorted({r["label"] for r in train})
    lab2idx = {l: i for i, l in enumerate(labels)}
    print(f"labels: {labels} | train={len(train)} test={len(test)} | input={args.input}")

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

    examples = [(build(r), lab2idx[r["label"]]) for r in train]
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
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad()
            total += float(out.loss)
            steps += 1
        print(f"epoch {epoch}  loss {total / steps:.4f}", flush=True)

    model.eval()

    @torch.no_grad()
    def predict(msgs):
        preds = []
        for i in range(0, len(msgs), 32):
            enc = encode(msgs[i:i + 32]).to(device)
            logits = model(**enc).logits
            preds.extend(labels[k] for k in logits.argmax(-1).tolist())
        return preds

    msgs = [build(r) for r in test]
    gold = [r["label"] for r in test]
    surf = [_surface(r) for r in test]
    preds = predict(msgs)
    ok = sum(int(p == g) for p, g in zip(preds, gold))
    by = collections.defaultdict(lambda: [0, 0])
    for p, g, s in zip(preds, gold, surf):
        by[s][0] += int(p == g)
        by[s][1] += 1
    print(f"\nGate B (Qwen LoRA) test accuracy: {ok/len(test):.3f} ({ok}/{len(test)})")
    print("  by surface: " + " ".join(
        f"{s}={by[s][0]/by[s][1]:.2f}({by[s][1]})" for s in sorted(by)))

    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    (out / "labels.json").write_text(json.dumps(labels))
    (out / "input_format.json").write_text(json.dumps({"input": args.input}))
    print(f"saved adapter -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
