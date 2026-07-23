"""Train the classify_complaint logit head on a frozen Qwen2.5-3B encoder.

This is the Qwen port of the Chapter-15 complaint classifier (it replaces the
vendored TinyGPT classifier). The recipe is deliberately the simplest thing
that can work — and the first rung of the escalation ladder:

    1. logit head only      <-- this script
    2. LoRA + logit head
    3. RoRA (Cayley) + logit head

The backbone (Qwen/Qwen2.5-3B-Instruct) is frozen. We pull the last non-pad
token's hidden state as a sentence feature, standardize, and fit a single
linear "logit head" (hidden -> 3 labels) on top. Because the encoder never
updates, features are extracted once and the head trains on cached vectors in
seconds.

Artifacts land in ``data/complaint_classifier_qwen/head.pt`` (head weights +
feature standardization + label list + the encoder id and max_length needed to
reproduce features at inference). Greedy / deterministic: no sampling anywhere.

Run:
    python scripts/train_complaint_classifier_qwen.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO_ROOT / "data" / "training" / "complaint_classification"
_EVAL_CASES = _REPO_ROOT / "data" / "eval_cases" / "cases.json"
_OUT_DIR = _REPO_ROOT / "data" / "complaint_classifier_qwen"
_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
_MAX_LENGTH = 128


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@torch.no_grad()
def extract_features(model, tokenizer, texts, device, batch_size=16, max_length=_MAX_LENGTH):
    """Last non-pad token hidden state from the frozen encoder, per text."""
    feats = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        out = model(**enc)
        hidden = out.last_hidden_state  # (b, t, h)
        last = enc["attention_mask"].sum(dim=1) - 1  # right-padded -> last real token
        pooled = hidden[torch.arange(hidden.size(0), device=device), last]
        feats.append(pooled.float().cpu())
    return torch.cat(feats, dim=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--model-id", default=_MODEL_ID)
    ap.add_argument("--train-file", default="train.jsonl",
                    help="training jsonl under data/training/complaint_classification/")
    ap.add_argument("--valid-file", default="valid.jsonl")
    ap.add_argument("--out-dir", default=str(_OUT_DIR),
                    help="where to write head.pt / labels.json")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    from transformers import AutoModel, AutoTokenizer

    print(f"loading encoder {args.model_id} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoder = AutoModel.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map=device.type if device.type == "cuda" else None,
    )
    if device.type != "cuda":
        encoder = encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    train = _read_jsonl(_DATA_DIR / args.train_file)
    valid = _read_jsonl(_DATA_DIR / args.valid_file)
    labels = sorted({r["label"] for r in train} | {r["label"] for r in valid})
    lab2idx = {lab: i for i, lab in enumerate(labels)}
    print(f"labels: {labels}  | train={len(train)} valid={len(valid)}")

    print("extracting features (frozen encoder, one pass) ...")
    Xtr = extract_features(encoder, tokenizer, [r["message"] for r in train], device)
    Xva = extract_features(encoder, tokenizer, [r["message"] for r in valid], device)
    ytr = torch.tensor([lab2idx[r["label"]] for r in train])
    yva = torch.tensor([lab2idx[r["label"]] for r in valid])

    # Standardize on train statistics; persisted so inference matches.
    mean = Xtr.mean(dim=0)
    std = Xtr.std(dim=0) + 1e-6
    Xtr_n = ((Xtr - mean) / std).to(device)
    Xva_n = ((Xva - mean) / std).to(device)
    ytr_d, yva_d = ytr.to(device), yva.to(device)

    head = nn.Linear(Xtr.shape[1], len(labels)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_acc, best_state, best_epoch = -1.0, None, -1
    for epoch in range(args.epochs):
        head.train()
        opt.zero_grad()
        loss = loss_fn(head(Xtr_n), ytr_d)
        loss.backward()
        opt.step()
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            head.eval()
            with torch.no_grad():
                acc = (head(Xva_n).argmax(dim=1) == yva_d).float().mean().item()
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            print(f"epoch {epoch:4d}  loss {loss.item():.4f}  valid_acc {acc:.3f}")

    print(f"best valid_acc {best_acc:.3f} @ epoch {best_epoch}")
    head.load_state_dict(best_state)

    # Eval on the 20 governance cases (expected_classification) for a second read.
    if _EVAL_CASES.exists():
        cases = json.loads(_EVAL_CASES.read_text())
        msgs = [c["message"] for c in cases]
        gold = [c["expected_classification"] for c in cases]
        Xc = ((extract_features(encoder, tokenizer, msgs, device) - mean) / std).to(device)
        head.eval()
        with torch.no_grad():
            pred = head(Xc).argmax(dim=1).cpu().tolist()
        correct = sum(labels[p] == g for p, g in zip(pred, gold))
        print(f"eval_cases classification: {correct}/{len(cases)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "head_state_dict": best_state,
            "mean": mean,
            "std": std,
            "labels": labels,
            "model_id": args.model_id,
            "max_length": _MAX_LENGTH,
            "hidden_size": Xtr.shape[1],
        },
        out_dir / "head.pt",
    )
    (out_dir / "labels.json").write_text(json.dumps(labels))
    print(f"saved -> {out_dir / 'head.pt'}")


if __name__ == "__main__":
    main()
