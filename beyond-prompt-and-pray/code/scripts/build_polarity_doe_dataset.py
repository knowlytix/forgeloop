"""Generate a GMS-labeled, DoE-varied polarity dataset for the disclosure gate.

The comparison this dataset supports: the geometric value-polarity gate
(``ValuePolarityChecker``: cap + v-resolution + u-tension, pure geometry) versus a
Qwen3-4B LoRA classifier, on the SAME 3-class task -- does an asserted stance value
agree with, contradict, or fall unrelated to the value the policy actually holds.

Grounding and labels come from the system itself:

  * **GMS supplies the ground truth.** The store's stance facts -- a policy
    attribute whose value is a stance pole (forbidden/permitted, required/optional,
    issued/denied) -- are read straight from the graph. The stored value is the
    truth an asserted value is judged against.
  * **The pole taxonomy fixes the label** (not either gate, so the test is fair):
    a synonym of the stored pole -> ``supported``; the opposite pole -> ``contradicted``;
    a value from a different axis -> ``uncertain``.
  * **DoE supplies surface variation.** ``knowlytix.harness.graphdoe.DesignMatrix``
    designs a space-filling set over presentation factors (surface form, register,
    hedging); Qwen realizes each into natural language. The label is invariant across
    surface, so the design tests robustness to phrasing -- the axis on which a
    token-tuned geometric gate is expected to degrade.

Each row carries both gate inputs: ``head/relation/stored/asserted`` (for the
geometric gate, which scores the value token) and ``message`` (for the classifier,
which reads the surface text). Seed-grouped train/test split: every surface variant
of one (attribute, asserted-value) seed stays on one side, so no paraphrase leaks.

    python scripts/build_polarity_doe_dataset.py --n 480
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import io
import json
import random
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_STORE = _ROOT / "data" / "gms_governed_store"
_OUT = _ROOT / "data" / "training" / "polarity"

# Stance poles and same-stance synonyms (the validated axes from
# scripts/build_policy_value_polarity.py). Opposite poles pair within an axis;
# values across axes are unrelated.
AXES = [
    ("forbidden", ["prohibited", "banned", "disallowed", "outlawed"],
     "permitted", ["allowed", "authorized", "approved"]),
    ("required", ["mandatory", "compulsory", "obligatory", "enforced"],
     "optional", ["voluntary", "discretionary"]),
    ("issued", ["granted", "provided", "awarded", "conferred"],
     "denied", ["withheld", "refused", "rejected"]),
]
_SYN = {}
_OPP = {}
_POLES = set()
for _a, _sa, _b, _sb in AXES:
    _SYN[_a] = [_a] + _sa
    _SYN[_b] = [_b] + _sb
    _OPP[_a] = _b
    _OPP[_b] = _a
    _POLES |= {_a, _b}


def _cross_axis_values(pole: str) -> list[str]:
    """Values from the other two axes (unrelated to *pole*)."""
    out = []
    for a, sa, b, sb in AXES:
        if pole in (a, b):
            continue
        out += [a] + sa + [b] + sb
    return out


# Readable phrasings for the stance relations, so Qwen can render a natural
# sentence about the attribute. Keyed on relation; a relation not listed falls
# back to a de-underscored name.
_RELATION_PHRASE = {
    "has_unencrypted_channel_pii": "sending personal information over an unencrypted channel",
    "has_redaction": "redaction of personal information in tickets and logs",
    "has_identity_verification": "identity verification before an account is closed",
    "has_fraud_notice_exception": "the exception to the advance-notice rule when fraud is confirmed",
    "has_provisional_credit": "provisional credit to the customer while a dispute is investigated",
}


def _phrase(relation: str) -> str:
    return _RELATION_PHRASE.get(
        relation, (relation[4:] if relation.startswith("has_") else relation).replace("_", " "))


# DoE presentation factors. Surface form spans the token-to-prose spectrum on which
# the geometric gate is expected to degrade; register and hedging add orthogonal
# variation. The label is invariant across all of them.
_SURFACE = {
    "token": "just the single value word, nothing else",
    "phrase": "a short noun phrase, under six words",
    "sentence": "one complete, plain sentence",
    "verbose": "two sentences, with an extra clause of context",
}
_REGISTER = {
    "plain": "plain everyday wording",
    "formal": "formal, policy-manual wording",
    "casual": "casual, conversational wording",
    "technical": "precise banking/compliance terminology",
}
_HEDGING = {
    "direct": "state it directly",
    "hedged": "hedge it, using words like 'seems', 'I believe', or 'generally'",
}
_FACTORS = [
    {"name": "surface", "type": "categorical", "categories": list(_SURFACE)},
    {"name": "register", "type": "categorical", "categories": list(_REGISTER)},
    {"name": "hedging", "type": "categorical", "categories": list(_HEDGING)},
]


def _stance_facts(store) -> list[tuple[str, str, str]]:
    """Facts (head, relation, stored_pole) whose stored value is a stance pole."""
    facts = []
    for h, r, t in store.doc_graph.triples:
        tl = str(t).strip().lower()
        if tl in _POLES:
            facts.append((h, r, tl))
    return sorted(set(facts))


def _build_prompt(phrase: str, asserted: str, f: dict) -> str:
    return (
        "You write one short statement asserting a bank policy's stance on an "
        "attribute. Assert exactly this claim, even if it may be wrong; do NOT "
        "correct or qualify the stance itself.\n"
        f"Attribute: {phrase}.\n"
        f"Claimed stance: {asserted}.\n"
        f"Surface form: {_SURFACE[f['surface']]}. "
        f"Register: {_REGISTER[f['register']]}. "
        f"{_HEDGING[f['hedging']]}.\n"
        "Reply with ONLY the statement."
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=480, help="DoE runs (surface combos)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--per-class", type=int, default=4,
                    help="asserted values sampled per (fact,class)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--sample", type=int, default=0,
                    help="if >0, generate this many and print (no files written)")
    args = ap.parse_args()

    import torch
    from knowlytix.knowledge.geode import QWEN_4B
    from knowlytix.knowledge.llm_backend import LocalTransformersBackend
    from knowlytix.knowledge.query import DocGMSConfig, GMSExpertStore
    from knowlytix.harness.graphdoe import DesignMatrix

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = GMSExpertStore(DocGMSConfig(store_path=str(_STORE), ingest_mode="regex"),
                           device=device)
    assert store.load(), f"failed to load store {_STORE}"
    facts = _stance_facts(store)
    print(f"GMS stance facts ({len(facts)}):")
    for h, r, t in facts:
        print(f"  {h} / {r} / {t}")
    if not facts:
        print("no stance facts found; aborting", flush=True)
        return 1

    rng = random.Random(args.seed)

    # Semantic seeds: (head, relation, stored, asserted, label). Labels from the
    # pole taxonomy -- GMS gives the stored truth, the taxonomy gives the relation.
    seeds = []
    for h, r, stored in facts:
        supported = _SYN[stored]
        contradicted = _SYN[_OPP[stored]]
        uncertain = _cross_axis_values(stored)
        for label, pool in (("supported", supported),
                            ("contradicted", contradicted),
                            ("uncertain", uncertain)):
            picks = pool if len(pool) <= args.per_class else rng.sample(pool, args.per_class)
            for asserted in picks:
                seeds.append({"head": h, "relation": r, "stored": stored,
                              "asserted": asserted, "label": label})
    rng.shuffle(seeds)
    print(f"semantic seeds: {len(seeds)}  "
          f"{dict(collections.Counter(s['label'] for s in seeds))}")

    n = args.sample if args.sample else args.n
    with contextlib.redirect_stderr(io.StringIO()):
        design = DesignMatrix(_FACTORS, method="sobol", n_runs=n, seed=args.seed).generate()
    rows_doe = design.to_dict("records")

    backend = LocalTransformersBackend(QWEN_4B, device=str(device))

    # Plan every row: cycle seeds across the DoE surface combos so each seed gets
    # varied presentation and classes stay proportional.
    plan = []
    for i, f in enumerate(rows_doe):
        s = seeds[i % len(seeds)]
        plan.append((s, f))

    def _render(s, f) -> str:
        if f["surface"] == "token":
            return s["asserted"]  # the token spectrum's lower end: no LLM needed
        return None  # filled by batched generation

    # token rows need no LLM; the rest are batched through Qwen
    gen_idx = [i for i, (s, f) in enumerate(plan) if f["surface"] != "token"]
    prompts = [_build_prompt(_phrase(plan[i][0]["relation"]),
                             plan[i][0]["asserted"], plan[i][1]) for i in gen_idx]
    gen_out = {}
    bs = max(1, args.batch_size)
    for b in range(0, len(prompts), bs):
        outs = [t.strip().strip('"').replace("\n", " ")
                for t in backend.call_batch(prompts[b:b + bs], max_tokens=64)]
        for k, txt in zip(gen_idx[b:b + bs], outs):
            gen_out[k] = txt
        print(f"  materialized {min(b + bs, len(prompts))}/{len(prompts)}", flush=True)

    records = []
    for i, (s, f) in enumerate(plan):
        msg = _render(s, f) if f["surface"] == "token" else (gen_out.get(i) or s["asserted"])
        if not msg:
            continue
        rec = {"message": msg, "label": s["label"], "head": s["head"],
               "relation": s["relation"], "stored": s["stored"], "asserted": s["asserted"],
               "_seed": f"{s['head']}|{s['relation']}|{s['asserted']}",
               "_factors": {k: f[k] for k in ("surface", "register", "hedging")}}
        records.append(rec)
        if args.sample:
            print(f"[{rec['label']:12s}] {rec['_factors']}  {s['relation']}={s['asserted']} "
                  f"(stored={s['stored']})\n   -> {msg}")

    if args.sample:
        print(f"\nsample: {len(records)} rows "
              f"{dict(collections.Counter(r['label'] for r in records))}")
        return 0

    # Seed-grouped split: all surface variants of a (head,relation,asserted) seed
    # stay on one side.
    seed_ids = sorted({r["_seed"] for r in records})
    random.Random(args.seed + 999).shuffle(seed_ids)
    test_ids = set(seed_ids[: int(round(len(seed_ids) * args.test_frac))])
    train = [r for r in records if r["_seed"] not in test_ids]
    test = [r for r in records if r["_seed"] in test_ids]

    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / "train_polarity.jsonl").write_text("\n".join(json.dumps(r) for r in train) + "\n")
    (_OUT / "test_polarity.jsonl").write_text("\n".join(json.dumps(r) for r in test) + "\n")

    def _dist(rows):
        return dict(collections.Counter(r["label"] for r in rows))

    print(f"\nseeds: {len(seed_ids)}  test seeds: {len(test_ids)} ({args.test_frac:.0%})")
    print(f"train_polarity: {len(train)} {_dist(train)} -> {_OUT/'train_polarity.jsonl'}")
    print(f"test_polarity : {len(test)} {_dist(test)} -> {_OUT/'test_polarity.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
