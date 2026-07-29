#!/usr/bin/env python
"""Generate a labeled training corpus for the task-tuned extract encoder.

For each product/issue class, sample diverse realistic customer messages from
Qwen conditioned on a class description (labels clean by construction). Sampling
(not greedy) for variety; deliberately generic scenarios so the corpus does NOT
reproduce the specific eval-case situations -- the benchmark tests on held-out
rephrased eval rows. Includes the `unknown`/`general` abstain classes so the SFT
learns a prototype for them (the unreachable-label trap, fixed by construction).

Writes data/training/extract_product_train.jsonl and extract_issue_train.jsonl.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from agentlab.models.constants import DEFAULT_QWEN_MODEL

_REPO = Path(__file__).resolve().parents[1]
_OUT = _REPO / "data" / "training"
_MODEL = DEFAULT_QWEN_MODEL

_PRODUCT = {
    "checking_account": "a checking or savings account, a debit card, or an overdraft on a bank account",
    "credit_card": "a credit card -- a card statement, APR, card payment, or a declined card",
    "mortgage": "a mortgage, a home loan, or an escrow account",
    "loan": "a personal loan or an auto/car loan (NOT a mortgage)",
    "unknown": "a general or procedural request that names NO specific product (updating an address, branch hours, how to contact support, a generic complaint)",
}
_ISSUE = {
    "overdraft_fee": "being charged a fee -- especially an overdraft fee -- and wanting it waived, reversed, or refunded",
    "account_issue": "a checking/savings account problem that is NOT about a fee: closing an account, disputing a charge they did not make, a duplicate charge, account access, or asking about account options",
    "credit_card_issue": "a credit card problem: a disputed card charge, a declined card, or a billing error on the card",
    "mortgage_issue": "a mortgage or escrow problem",
    "loan_issue": "a loan problem that is NOT a mortgage: a loan payment not credited, or a loan balance/interest question",
    "general": "a vague or procedural message where the specific problem is unclear, unstated, or just an emotional venting with no concrete issue",
}


def _gen(model, tok, device, desc: str, n: int, seed: int) -> list[str]:
    sys = ("You write short, realistic messages a retail-bank customer might send to "
           "support. One to three sentences. Vary tone (polite, confused, angry, "
           "terse), vocabulary, and detail. Also vary CLARITY across messages: some "
           "state the problem clearly and directly, some are vague and hedged as if "
           "unsure how to put it, some understate it as a casual aside. Each message "
           "must still be a DISTINCT situation, not a reword of another. Output ONLY "
           "the message text, nothing else.")
    user = f"Write a customer message about {desc}."
    chat = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
    inputs = tok.apply_chat_template(chat, add_generation_prompt=True,
                                     return_tensors="pt", return_dict=True).to(device)
    torch.manual_seed(seed)
    out = model.generate(**inputs, max_new_tokens=80, do_sample=True,
                         temperature=0.95, top_p=0.95, num_return_sequences=n,
                         pad_token_id=tok.eos_token_id)
    msgs = []
    for row in out:
        gen = row[inputs["input_ids"].shape[1]:]
        txt = tok.decode(gen, skip_special_tokens=True).strip().strip('"').strip()
        # collapse to a single message; drop empties / overlong boilerplate
        txt = " ".join(txt.split())
        if 8 <= len(txt) <= 400:
            msgs.append(txt)
    return msgs


def _build(model, tok, device, classes: dict[str, str], per_class: int, out_path: Path):
    rows, seen = [], set()
    for ci, (label, desc) in enumerate(classes.items()):
        got = []
        attempt = 0
        while len(got) < per_class and attempt < 4:
            batch = _gen(model, tok, device, desc, per_class, seed=1000 * ci + attempt)
            for m in batch:
                key = m.lower()
                if key not in seen:
                    seen.add(key)
                    got.append(m)
            attempt += 1
        got = got[:per_class]
        rows.extend({"text": m, "label": label} for m in got)
        print(f"  {label:18} {len(got)}")
    out_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"wrote {out_path} ({len(rows)} rows)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-class", type=int, default=30)
    args = ap.parse_args()
    _OUT.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(_MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        _MODEL, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map=device.type if device.type == "cuda" else None).eval()

    print("PRODUCT:")
    _build(model, tok, model.device, _PRODUCT, args.per_class,
           _OUT / "extract_product_train.jsonl")
    print("ISSUE:")
    _build(model, tok, model.device, _ISSUE, args.per_class,
           _OUT / "extract_issue_train.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
