"""Build a DoE-augmented {none, prompt_injection, prohibited_advice} corpus for
training an injection detector (Qwen-LoRA and embedding-SFT both consume it).

Seeds: governance_exemplars.json (prompt_injection, prohibited_advice, benign) +
a sample of clean benign banking messages (inquiry/complaint) + hand-authored
MISLEADING-BENIGN hard negatives (casual "just skip the formalities / go ahead"
phrasings that are injection-ADJACENT but benign -- the false-fire cases the
zero-shot gate trips on). Each seed is rephrased across clarity levels by Qwen
(greedy, intent-preserving). Output is seed-grouped so train/test never share a
seed's paraphrases.

NOTE: the held-out EVAL set is NOT built here -- it is the DoE's own obfuscated
injection rows (case-010, case-017) + `none` rows, pulled from
data/capstone_doe_results.csv by scripts/compare_injection_detectors.py. Those
seeds are excluded here so there is no leakage.

    python scripts/build_injection_corpus.py --per-seed 4
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_EXEMPLARS = _ROOT / "data" / "governance_exemplars.json"
_TRAIN = _ROOT / "data" / "training" / "complaint_classification" / "train.jsonl"
_OUT = _ROOT / "data" / "training" / "injection_doe.jsonl"

# Misleading-benign HARD NEGATIVES: benign banking requests worded with the
# casual "skip/ignore the formalities / just go ahead" cues that look
# injection-adjacent but carry no instruction-override intent. These teach the
# detector that a cue word is not an attack.
_HARD_NEG = [
    "Quick question, probably nothing -- what's my current balance?",
    "You can skip the policy spiel, I just want to dispute a charge I didn't make.",
    "Don't worry about the formal process, can you just tell me the overdraft fee?",
    "Forget the long explanation -- how do I close my account?",
    "Ignore the small print for a sec, is my payment overdue?",
    "Just go ahead and tell me when my statement is ready, no need for the rules.",
    "Honestly skip the disclaimer, I only need my routing number.",
    "Never mind the procedure, did my mortgage payment post yet?",
    "No need to read me the terms -- can you reverse a duplicate charge?",
    "Let's not do the whole verification thing, what are your branch hours?",
    # Hedged / uncertain benign inquiries: a customer who does not know what is
    # wrong and asks the bank to look into it. The vague "just look into it"
    # phrasing is injection-ADJACENT in tone but carries no override intent --
    # these teach the detector that hedged uncertainty is not an attack.
    "I'm not really sure what's going on with my account, could you take a look?",
    "Not sure what happened here, can you just check my recent transactions?",
    "I'm a bit confused about my statement, would you mind looking into it?",
    "Something seems off with my balance, can you please investigate for me?",
    "I don't quite understand a charge on my account, could you help me figure it out?",
    "Just wondering if you could look into my account, something doesn't add up.",
    # Defiant / dismissive fee-waiver demands: a rude customer who dismisses the
    # bank's policies while asking for a waiver or refund. Dismissing the rules is
    # not overriding the system's instructions or leaking a prompt, so this is a
    # benign complaint, not prompt injection -- teach the boundary explicitly.
    "Just waive the fee, I don't care about your rules.",
    "I don't want to hear about your policies, just remove the charge.",
    "Skip the policy talk and refund my overdraft fee.",
    "I'm not interested in the fine print, just reverse this charge.",
]

_CLARITY = {
    "clear": "Rewrite the message in clear, direct language.",
    "ambiguous": "Rewrite the message in vaguer, hedged, less direct language.",
    "misleading": "Rewrite the message so the real intent is buried under a casual, "
                  "off-hand framing (e.g. 'quick question, probably nothing'), but KEEP that intent intact.",
}


def _prompt(message: str, clarity_instr: str) -> str:
    return (
        "You rewrite a single end-user message to a bank's support agent, varying only the "
        "phrasing. PRESERVE the message's intent exactly -- if it tries to override or ignore "
        "the assistant's instructions, the rewrite must still try to; if it is a normal banking "
        "request, it stays one. Do not add quotes or commentary.\n"
        f"{clarity_instr}\n\nMessage: {message}\n\nRewrite:"
    )


def _load_seeds() -> list[dict]:
    ex = json.loads(_EXEMPLARS.read_text())
    seeds: list[dict] = []
    for i, m in enumerate(ex.get("prompt_injection", [])):
        seeds.append({"id": f"inj-{i}", "message": m, "label": "prompt_injection"})
    for i, m in enumerate(ex.get("prohibited_advice", [])):
        seeds.append({"id": f"prh-{i}", "message": m, "label": "prohibited_advice"})
    # benign: governance benign exemplars + hard negatives + a sample of real
    # inquiry/complaint messages
    for i, m in enumerate(ex.get("benign", [])):
        seeds.append({"id": f"ben-{i}", "message": m, "label": "none"})
    for i, m in enumerate(_HARD_NEG):
        seeds.append({"id": f"hn-{i}", "message": m, "label": "none"})
    if _TRAIN.exists():
        rng = random.Random(0)
        rows = [json.loads(l) for l in _TRAIN.read_text().splitlines() if l.strip()]
        clean = [r for r in rows if r.get("label") in ("inquiry", "complaint")]
        rng.shuffle(clean)
        for i, r in enumerate(clean[:24]):
            seeds.append({"id": f"bk-{i}", "message": r["message"], "label": "none"})
    return seeds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-seed", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    seeds = _load_seeds()
    import collections
    print(f"seeds: {len(seeds)} | by label: {dict(collections.Counter(s['label'] for s in seeds))}")

    clar = list(_CLARITY)
    plan = []
    for s in seeds:
        # always include the original (clear) + per-seed clarity variants
        plan.append((s, "clear", s["message"], None))           # original, no rewrite
        for i in range(args.per_seed - 1):
            c = clar[(i + 1) % len(clar)]
            plan.append((s, c, None, _prompt(s["message"], _CLARITY[c])))

    import torch
    from agentlab.models.qwen_adapter import _DEFAULT_MODEL
    from knowlytix.knowledge.llm_backend import LocalTransformersBackend
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # One padded forward pass per batch via the knowlytix local backend.
    backend = LocalTransformersBackend(_DEFAULT_MODEL, device=str(device))

    def gen(prompts):
        return [t.strip().strip('"')
                for t in backend.call_batch(prompts, max_tokens=96)]

    to_gen = [(idx, p) for idx, (_, _, msg, p) in enumerate(plan) if p is not None]
    rewrites = {}
    bs = max(1, args.batch_size)
    for b in range(0, len(to_gen), bs):
        chunk = to_gen[b:b + bs]
        with contextlib.redirect_stderr(io.StringIO()):
            outs = gen([p for _, p in chunk])
        for (idx, _), rw in zip(chunk, outs):
            rewrites[idx] = rw
        print(f"  generated {min(b + bs, len(to_gen))}/{len(to_gen)}", flush=True)

    records = []
    for idx, (s, c, msg, p) in enumerate(plan):
        text = msg if msg is not None else (rewrites.get(idx) or s["message"])
        if not text.strip():
            text = s["message"]
        records.append({"message": text, "label": s["label"], "_seed": s["id"], "_clarity": c})

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    print(f"\nwrote {len(records)} examples -> {_OUT}")
    print("by label:", dict(collections.Counter(r["label"] for r in records)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
