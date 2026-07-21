"""DoE-based augmentation of the complaint-classifier training data.

The classifier is trained on clean, canonical phrasings and collapses on hedged /
ambiguous phrasing (Chapter 16 DoE: clarity=clear 0.84 -> ambiguous 0.26, with the
dominant failure being complaint -> other). This script closes the train/test
distribution gap by bringing the same design-of-experiments variation into the
TRAINING set: it takes the training SEEDS and produces label-preserving
paraphrases across a balanced design over presentation factors.

Approach (mirrors the Ch16 capstone harness, but label-preserving and seeded from
the TRAINING split so there is no leakage into the held-out DoE test):

  - A space-filling design over presentation factors via knowlytix DesignMatrix.
  - Semantic factors (clarity, style, length, specificity, paraphrase_depth,
    expertise, emotional_frame) are realized by one greedy Qwen rewrite.
  - Surface factors (entity_aliasing, noise) are stamped on mechanically AFTER,
    so the model cannot normalize them away.
  - The label and any dollar amounts are preserved by construction.

Factor set: the "broader robustness" selection (see the GraphDOE factor catalog).

Usage (from repo root, with the venv active):
    python scripts/augment_complaint_training_doe.py --sample 12        # quality check
    python scripts/augment_complaint_training_doe.py --n 600            # full run
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_DATA = _ROOT / "data" / "training" / "complaint_classification"
_SEEDS = _DATA / "train.jsonl"
_OUT_AUG = _DATA / "train_doe_augmented.jsonl"   # all paraphrases + factor metadata
_OUT_TRAIN = _DATA / "train_doe.jsonl"           # train split (seed-grouped)
_OUT_TEST = _DATA / "test_doe.jsonl"             # held-out test split (seed-grouped)

# ── Semantic factors: one per-level instruction line, fed to Qwen ──────────────
_SEMANTIC = {
    "clarity": {
        "Clear": "state it clearly and directly",
        "Ambiguous": "phrase it vaguely and hedged, as if unsure how to put it, without stating the problem plainly",
        "Misleading": "understate it in a low-key, no-big-deal tone, as if it is probably nothing -- but still name the specific product or charge and state plainly what went wrong, including any claim that it was unfair, wrong, unauthorized or misapplied; soften the feeling, never drop the facts",
    },
    "style": {
        "Formal": "use formal, professional wording",
        "Casual": "use casual, conversational, everyday wording",
        "Technical": "use precise banking terminology",
    },
    "length": {
        "Short": "keep it very short, under ten words if you can",
        "Medium": "keep it to one or two sentences",
        "Long": "make it longer and more rambling, with extra throat-clearing context",
    },
    "specificity": {
        "General": "keep the wording broad and open-ended",
        "Specific": "be precise and concrete",
    },
    "paraphrase_depth": {
        "None": "keep close to the original wording",
        "Light": "reword lightly, keeping the same structure",
        "Heavy": "rewrite deeply with new sentence structure but the same meaning",
    },
    "expertise": {
        "Novice": "write as a customer who does not know banking jargon",
        "Intermediate": "write as a customer with some banking familiarity",
        "Expert": "write as a customer fluent in banking terms",
    },
    "emotional_frame": {
        "Neutral": "keep a neutral tone",
        "Urgent": "add urgency, as if it needs handling right away",
        "Skeptical": "add a doubtful, skeptical tone",
    },
}

# ── Surface factors: mechanical, applied after the rewrite ─────────────────────
_SURFACE_ALIASING = ["Exact", "Synonym", "Abbreviated"]
_SYNONYMS = {
    "fee": "charge", "refund": "reversal", "overdraft": "NSF",
    "statement": "account summary", "transaction": "transfer",
}
_ABBREV = {
    "account": "acct", "transaction": "txn", "minimum": "min",
    "overdraft": "OD", "statement": "stmt", "balance": "bal", "interest": "int.",
}
_NOISE = ["None", "Typos"]

_FACTORS = (
    [{"name": n, "type": "categorical", "categories": list(levels)} for n, levels in _SEMANTIC.items()]
    + [{"name": "entity_aliasing", "type": "categorical", "categories": _SURFACE_ALIASING}]
    + [{"name": "noise", "type": "categorical", "categories": _NOISE}]
)

_DOLLAR = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?")


def _build_prompt(message: str, row: dict) -> str:
    directions = "; ".join(_SEMANTIC[f][row[f]] for f in _SEMANTIC)
    return (
        "You are simulating a bank customer writing one short message to their bank. "
        "Rewrite the ORIGINAL message below, applying all of these style directions: "
        f"{directions}. "
        "Keep the same underlying intent and category, keep the specific product or "
        "charge and the stated problem (never generalize it to 'an issue with my "
        "account' or 'something seems off'), and keep any dollar amounts exactly as "
        "written. Do not answer or resolve the issue --- only write the "
        "customer's message. Reply with ONLY the rewritten message.\n\n"
        f"ORIGINAL: {message}\nMESSAGE:"
    )


def _apply_aliasing(msg: str, level: str) -> str:
    table = _SYNONYMS if level == "Synonym" else _ABBREV if level == "Abbreviated" else None
    if table is None:
        return msg
    out = msg
    for canon, variant in table.items():
        out = re.sub(rf"\b{re.escape(canon)}\b", variant, out, flags=re.IGNORECASE)
    return out


def _apply_noise(msg: str, level: str, rng: random.Random) -> str:
    if level != "Typos":
        return msg
    words = msg.split()
    longish = [i for i, w in enumerate(words) if len(w.strip(".,!?$")) >= 5]
    for i in rng.sample(longish, min(2, len(longish))):
        w = list(words[i])
        # transpose two adjacent interior characters (deterministic given rng)
        j = rng.randint(1, len(w) - 2)
        w[j], w[j + 1] = w[j + 1], w[j]
        words[i] = "".join(w)
    return " ".join(words)


def _amounts(text: str) -> list[str]:
    return [m.group(0).replace(" ", "") for m in _DOLLAR.finditer(text)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600, help="number of augmented examples")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sample", type=int, default=0,
                    help="if >0, only generate this many and print before/after (no files written)")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="prompts per Qwen generate() call (batched on GPU)")
    ap.add_argument("--test-frac", type=float, default=0.2,
                    help="fraction of SEEDS held out for test (seed-grouped, no leakage)")
    args = ap.parse_args()

    seeds = [json.loads(l) for l in _SEEDS.read_text().splitlines() if l.strip()]
    n = args.sample if args.sample else args.n

    from knowlytix.harness.graphdoe import DesignMatrix
    with contextlib.redirect_stderr(io.StringIO()):
        design = DesignMatrix(_FACTORS, method="sobol", n_runs=n, seed=args.seed).generate()
    rows = design.to_dict("records")

    # Balanced seed assignment: shuffle once (deterministic), cycle across rows so
    # every seed gets varied factor combinations and classes stay proportional.
    order = list(range(len(seeds)))
    random.Random(args.seed).shuffle(order)

    import torch
    from agentlab.models.qwen_adapter import _DEFAULT_MODEL
    from knowlytix.knowledge.llm_backend import LocalTransformersBackend
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Batched decoding via the knowlytix local backend: one padded forward pass per
    # batch (LocalTransformersBackend.call_batch), not one generate() per prompt.
    backend = LocalTransformersBackend(_DEFAULT_MODEL, device=str(device))

    def _gen_batch(prompts: list[str]) -> list[str]:
        return [t.strip().strip('"')
                for t in backend.call_batch(prompts, max_tokens=128)]

    # Plan every row first (seed + label-clamped factor levels + prompt), then
    # generate in GPU batches rather than one prompt at a time.
    plan = []
    for i, row in enumerate(rows):
        seed = seeds[order[i % len(seeds)]]
        msg, label = seed["message"], seed["label"]
        # Label-drift guard. 'other' (injections, chit-chat, off-topic) has no
        # legitimate customer-message paraphrase, so deep/long rewrites fabricate
        # a complaint. Keep 'other' close to its seed: Clear, not Long, not Heavy.
        # 'complaint' and 'inquiry' carry their intent through ambiguity, so they
        # get the full clarity sweep (the diagnostic axis).
        erow = dict(row)
        if label == "other":
            erow["clarity"] = "Clear"
            if erow["length"] == "Long":
                erow["length"] = "Medium"
            if erow["paraphrase_depth"] == "Heavy":
                erow["paraphrase_depth"] = "Light"
        plan.append((msg, label, erow, _build_prompt(msg, erow)))

    rewrites: list[str] = []
    bs = max(1, args.batch_size)
    for b in range(0, len(plan), bs):
        rewrites.extend(_gen_batch([p for *_, p in plan[b:b + bs]]))
        print(f"  generated {min(b + bs, len(plan))}/{len(plan)}", flush=True)

    out_records = []
    kept_label = dropped = 0
    for i, ((msg, label, erow, _), rewritten) in enumerate(zip(plan, rewrites)):
        if not rewritten:
            rewritten = msg
        # surface factors, stamped after the rewrite
        rewritten = _apply_aliasing(rewritten, erow["entity_aliasing"])
        rewritten = _apply_noise(rewritten, erow["noise"], random.Random(args.seed + i))

        # guard: any dollar amount in the seed must survive (label-critical signal)
        if set(_amounts(msg)) - set(_amounts(rewritten)):
            dropped += 1
            continue
        # guard: an 'other' rewrite that balloons in length has fabricated content
        # (an injection turned into a complaint); drop it.
        if label == "other" and len(rewritten) > 2.5 * len(msg) + 80:
            dropped += 1
            continue
        kept_label += 1
        rec = {"message": rewritten, "label": label,
               "_seed": msg, "_factors": {k: erow[k] for k in design.columns}}
        out_records.append(rec)
        if args.sample:
            f = rec["_factors"]
            print(f"[{label}] clarity={f['clarity']} style={f['style']} len={f['length']} "
                  f"depth={f['paraphrase_depth']} alias={f['entity_aliasing']} noise={f['noise']}")
            print(f"   seed : {msg}")
            print(f"   ->   : {rewritten}\n")

    if args.sample:
        print(f"sample: {kept_label} kept, {dropped} dropped (lost a $ amount)")
        return 0

    import collections
    _OUT_AUG.write_text("\n".join(json.dumps(r) for r in out_records) + "\n")

    # Seed-grouped split: assign each seed (label-stratified, deterministic) to
    # train or test, so a seed's original AND all its paraphrases stay on one
    # side --- no paraphrase of a test seed leaks into training.
    by_label = collections.defaultdict(list)
    for s in seeds:
        by_label[s["label"]].append(s["message"])
    rng = random.Random(args.seed + 999)
    test_seeds: set[str] = set()
    for lab, msgs in by_label.items():
        uniq = sorted(set(msgs))
        rng.shuffle(uniq)
        test_seeds.update(uniq[: int(round(len(uniq) * args.test_frac))])

    orig = [{"message": s["message"], "label": s["label"], "_seed": s["message"],
             "_factors": {"clarity": "clear"}} for s in seeds]
    allrows = orig + out_records
    train_rows = [r for r in allrows if r["_seed"] not in test_seeds]
    test_rows = [r for r in allrows if r["_seed"] in test_seeds]

    _OUT_TRAIN.write_text("\n".join(json.dumps(r) for r in train_rows) + "\n")
    _OUT_TEST.write_text("\n".join(json.dumps(r) for r in test_rows) + "\n")

    def _dist(rows):
        return dict(collections.Counter(r["label"] for r in rows))

    print(f"augmented kept: {len(out_records)}  (dropped {dropped}; original train.jsonl untouched)")
    print(f"seeds: {len(seeds)}  test seeds held out: {len(test_seeds)} ({args.test_frac:.0%})")
    print(f"train_doe : {len(train_rows)} rows {_dist(train_rows)} -> {_OUT_TRAIN.name}")
    print(f"test_doe  : {len(test_rows)} rows {_dist(test_rows)} -> {_OUT_TEST.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
