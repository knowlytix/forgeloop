"""DoE augmentation of the extract-classifier training data (product + issue).

Mirrors scripts/augment_complaint_training_doe.py, but labels are the policy
complaint-type pair (product, issue) rather than the binary complaint/inquiry/
other. Takes hand-authored seeds (data/training/extract_seeds.jsonl, independent
of the capstone eval cases so paraphrases never leak into the held-out DoE),
paraphrases each across a Sobol design over presentation factors (clarity, style,
length, specificity, paraphrase_depth, expertise, emotional_frame + surface
entity_aliasing, noise), preserving (product, issue) and dollar amounts. Writes a
DoE-hardened {message, product, issue, _factors} corpus + seed-grouped split.

    python scripts/augment_extract_training_doe.py --sample 10   # quality check
    python scripts/augment_extract_training_doe.py --n 600       # full run
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import io
import json
import random
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_DATA = _ROOT / "data" / "training"
_SEEDS = _DATA / "extract_seeds.jsonl"
_OUT_AUG = _DATA / "extract_doe_augmented.jsonl"
_OUT_TRAIN = _DATA / "extract_doe_train.jsonl"
_OUT_TEST = _DATA / "extract_doe_test.jsonl"

_SEMANTIC = {
    "clarity": {
        "Clear": "state it clearly and directly",
        "Ambiguous": "phrase it vaguely and hedged, as if unsure how to put it, without stating the problem plainly",
        "Misleading": "downplay it as a casual aside that might be nothing, though the underlying issue is unchanged",
    },
    "style": {"Formal": "use formal, professional wording",
              "Casual": "use casual, conversational, everyday wording",
              "Technical": "use precise banking terminology"},
    "length": {"Short": "keep it very short, under ten words if you can",
               "Medium": "keep it to one or two sentences",
               "Long": "make it longer and more rambling, with extra throat-clearing context"},
    "specificity": {"General": "keep the wording broad and open-ended",
                    "Specific": "be precise and concrete"},
    "paraphrase_depth": {"None": "keep close to the original wording",
                         "Light": "reword lightly, keeping the same structure",
                         "Heavy": "rewrite deeply with new sentence structure but the same meaning"},
    "expertise": {"Novice": "write as a customer who does not know banking jargon",
                  "Intermediate": "write as a customer with some banking familiarity",
                  "Expert": "write as a customer fluent in banking terms"},
    "emotional_frame": {"Neutral": "keep a neutral tone",
                        "Urgent": "add urgency, as if it needs handling right away",
                        "Skeptical": "add a doubtful, skeptical tone"},
}
_SURFACE_ALIASING = ["Exact", "Synonym", "Abbreviated"]
_SYNONYMS = {"fee": "charge", "refund": "reversal", "overdraft": "NSF",
             "statement": "account summary", "transaction": "transfer"}
_ABBREV = {"account": "acct", "transaction": "txn", "minimum": "min",
           "overdraft": "OD", "statement": "stmt", "balance": "bal", "interest": "int."}
_NOISE = ["None", "Typos"]
_FACTORS = (
    [{"name": n, "type": "categorical", "categories": list(lv)} for n, lv in _SEMANTIC.items()]
    + [{"name": "entity_aliasing", "type": "categorical", "categories": _SURFACE_ALIASING}]
    + [{"name": "noise", "type": "categorical", "categories": _NOISE}])
_DOLLAR = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?")


def _prompt(message, row):
    directions = "; ".join(_SEMANTIC[f][row[f]] for f in _SEMANTIC)
    return ("You are simulating a bank customer writing one short message to their bank. "
            f"Rewrite the ORIGINAL message below, applying all of these style directions: {directions}. "
            "Keep the same underlying intent, product and problem type, and keep any dollar amounts "
            "exactly as written. Do not answer or resolve the issue --- only write the customer's "
            f"message. Reply with ONLY the rewritten message.\n\nORIGINAL: {message}\nMESSAGE:")


def _alias(msg, level):
    table = _SYNONYMS if level == "Synonym" else _ABBREV if level == "Abbreviated" else None
    if table is None:
        return msg
    for canon, variant in table.items():
        msg = re.sub(rf"\b{re.escape(canon)}\b", variant, msg, flags=re.IGNORECASE)
    return msg


def _noise(msg, level, rng):
    if level != "Typos":
        return msg
    words = msg.split()
    longish = [i for i, w in enumerate(words) if len(w.strip(".,!?$")) >= 5]
    for i in rng.sample(longish, min(2, len(longish))):
        w = list(words[i]); j = rng.randint(1, len(w) - 2)
        w[j], w[j + 1] = w[j + 1], w[j]; words[i] = "".join(w)
    return " ".join(words)


def _amounts(t):
    return [m.group(0).replace(" ", "") for m in _DOLLAR.finditer(t)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--test-frac", type=float, default=0.2)
    args = ap.parse_args()

    seeds = [json.loads(l) for l in _SEEDS.read_text().splitlines() if l.strip()]
    n = args.sample if args.sample else args.n
    from knowlytix.harness.graphdoe import DesignMatrix
    with contextlib.redirect_stderr(io.StringIO()):
        design = DesignMatrix(_FACTORS, method="sobol", n_runs=n, seed=args.seed).generate()
    rows = design.to_dict("records")
    order = list(range(len(seeds)))
    random.Random(args.seed).shuffle(order)

    import torch
    from agentlab.models.qwen_adapter import _DEFAULT_MODEL, _load
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, model = _load(_DEFAULT_MODEL, device)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    @torch.no_grad()
    def _gen(prompts):
        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                 tokenize=False, add_generation_prompt=True) for p in prompts]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        out = model.generate(**enc, max_new_tokens=128, do_sample=False, pad_token_id=tok.eos_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        return [tok.decode(g, skip_special_tokens=True).strip().strip('"') for g in gen]

    plan = []
    for i, row in enumerate(rows):
        seed = seeds[order[i % len(seeds)]]
        msg, product, issue = seed["message"], seed["product"], seed["issue"]
        erow = dict(row)
        # general has no concrete problem to carry through ambiguity -> a deep/long
        # rewrite fabricates a specific issue. Keep general seeds close to source.
        if issue == "general":
            erow["clarity"] = "Clear"
            if erow["length"] == "Long":
                erow["length"] = "Medium"
            if erow["paraphrase_depth"] == "Heavy":
                erow["paraphrase_depth"] = "Light"
        plan.append((msg, product, issue, erow, _prompt(msg, erow)))

    rewrites = []
    bs = max(1, args.batch_size)
    for b in range(0, len(plan), bs):
        rewrites.extend(_gen([p for *_, p in plan[b:b + bs]]))
        print(f"  generated {min(b + bs, len(plan))}/{len(plan)}", flush=True)

    out = []
    dropped = 0
    for i, ((msg, product, issue, erow, _), rw) in enumerate(zip(plan, rewrites)):
        if not rw:
            rw = msg
        rw = _alias(rw, erow["entity_aliasing"])
        rw = _noise(rw, erow["noise"], random.Random(args.seed + i))
        if set(_amounts(msg)) - set(_amounts(rw)):
            dropped += 1; continue
        if issue == "general" and len(rw) > 2.5 * len(msg) + 80:
            dropped += 1; continue
        rec = {"message": rw, "product": product, "issue": issue, "_seed": msg,
               "_factors": {k: erow[k] for k in design.columns}}
        out.append(rec)
        if args.sample:
            print(f"[{product}/{issue}] clarity={erow['clarity']} alias={erow['entity_aliasing']}")
            print(f"   seed: {msg}\n   ->  : {rw}\n")
    if args.sample:
        print(f"sample: {len(out)} kept, {dropped} dropped"); return 0

    _OUT_AUG.write_text("\n".join(json.dumps(r) for r in out) + "\n")
    by_key = collections.defaultdict(list)
    for s in seeds:
        by_key[(s["product"], s["issue"])].append(s["message"])
    rng = random.Random(args.seed + 999)
    test_seeds = set()
    for _k, msgs in by_key.items():
        uniq = sorted(set(msgs)); rng.shuffle(uniq)
        test_seeds.update(uniq[: max(1, int(round(len(uniq) * args.test_frac)))])
    orig = [{"message": s["message"], "product": s["product"], "issue": s["issue"],
             "_seed": s["message"], "_factors": {"clarity": "clear"}} for s in seeds]
    allrows = orig + out
    tr = [r for r in allrows if r["_seed"] not in test_seeds]
    te = [r for r in allrows if r["_seed"] in test_seeds]
    _OUT_TRAIN.write_text("\n".join(json.dumps(r) for r in tr) + "\n")
    _OUT_TEST.write_text("\n".join(json.dumps(r) for r in te) + "\n")
    print(f"augmented {len(out)} (dropped {dropped}); train {len(tr)} test {len(te)}")
    print("issue dist train:", dict(collections.Counter(r["issue"] for r in tr)))
    print("product dist train:", dict(collections.Counter(r["product"] for r in tr)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
