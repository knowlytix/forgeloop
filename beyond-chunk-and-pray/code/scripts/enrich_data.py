# SPDX-License-Identifier: Apache-2.0
"""Data enrichment for the Chunk-and-Pray capstone (the Beyond Ship and Pray method).

One GMS store, generated three ways, all from library functionality:

  1. GENERATE base queries by mining the store with knowlytix.benchmark
     generators (the base types in the suite catalog: exact_recall, multi_hop,
     counting, cross_reference). Each carries graph-derived GROUND TRUTH.
  2. ENRICH by a DoE design over presentation factors (clarity, style, length,
     expertise, paraphrase_depth) -- knowlytix.harness.suite.compose +
     graphdoe_design -- then materialize each cell into CONTEXTUAL question text
     by conditioning Qwen on that cell's factor levels (ground truth invariant).
  3. EMIT three artifacts from the one enriched set:
       data/enrichment/rag_cohort.json       RAG test cohort (query + GT)
       data/enrichment/embedding_sft.jsonl    {text,label,_seed,_factors} for the
                                               v-space (concepts-close) SFT, grouped
                                               by target attribute
       data/enrichment/embedding_u_groups.json {attribute: [contextual questions]}
                                               for the u-space (contradiction) SFT
       data/enrichment/llm_draft_sft.jsonl     grounded (user, assistant) LLM pairs

The enriched questions are CONTEXTUAL on purpose: the encoders must learn the
attribute in context (a full question), not a bare keyword. The same enriched set
trains the embeddings + LLM (scripts/build_store.py) and tests the RAG (the
capstone). Generation uses Qwen at BUILD time only; the runtime gates are geometry.

Run (GPU, loads Qwen):  python scripts/enrich_data.py --n-runs 150
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import io
import json
import os
import re
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import REPO_ROOT, use_branch_library  # noqa: E402

use_branch_library()

import torch  # noqa: E402

import knowlytix.harness.suite as _suite  # noqa: E402
from knowlytix.harness.suite import (  # noqa: E402
    Catalog, CatalogBaseSource, compose, emit_classifier_sft, graphdoe_design,
    resolve, to_jsonl,
)
from knowlytix.knowledge.config import DocGMSConfig  # noqa: E402
from knowlytix.knowledge.geode import QWEN_3B  # noqa: E402
from knowlytix.knowledge.llm_backend import LocalTransformersBackend  # noqa: E402
from knowlytix.knowledge.store import GMSExpertStore  # noqa: E402

STORE = os.environ.get("GMS_STORE",
                       os.path.join(REPO_ROOT, "data", "gms_annual_report_store"))
OUTDIR = os.environ.get("GMS_ENRICH", os.path.join(REPO_ROOT, "data", "enrichment"))
# Resolve the DoE catalogs from the installed knowlytix package (works whether
# it is pip-installed or a source checkout) rather than a hardcoded branch path.
CATALOGS = os.path.join(os.path.dirname(_suite.__file__), "catalogs")

# Content-retrieval base types only: questions a reader actually asks of the
# report. (counting/cross_reference produce graph-STRUCTURE queries -- "how many
# has_revenue edges exist" -- which a content RAG is not built to answer.)
BASE_TYPES = ["exact_recall", "multi_hop"]
FACTORS = ["clarity", "style", "length", "expertise", "paraphrase_depth"]

# ENM section -> natural name (used to phrase the target, not the slug).
SECTION_NATURAL = {
    "segment_performance": "segment performance table",
    "income_statement": "income statement",
    "balance_sheet": "balance sheet",
}

# Factor level -> a natural-language style direction for the Qwen rewrite. The
# DoE design (graphdoe_design) chooses the level per scenario; this turns the
# chosen level into an instruction. Presentation only -- ground truth is invariant.
SEMANTIC = {
    "clarity": {
        "Clear": "state the question plainly and unambiguously",
        "Ambiguous": "phrase it a little vaguely, as a hurried reader might",
        "Misleading": "frame it hesitantly, opening with a brief aside like 'I might be misreading the table, but ...', yet still ask it as an OPEN question (never turn it into a yes/no and never guess a value)",
    },
    "style": {
        "Formal": "use formal, professional language",
        "Casual": "use a casual, conversational tone",
        "Technical": "use precise financial-analyst terminology",
    },
    "length": {
        "Short": "keep it to one short sentence",
        "Medium": "use one or two sentences",
        "Long": "add a little framing context, but still one question",
    },
    "expertise": {
        "Novice": "write as a non-expert who may not know the exact accounting term",
        "Intermediate": "write as someone reasonably familiar with financial reports",
        "Expert": "write as a financial analyst who knows the terminology",
    },
    "paraphrase_depth": {
        "None": "stay close to the original wording",
        "Light": "lightly reword it",
        "Heavy": "substantially reword it while keeping the exact same target",
    },
}


def _natural_target(item) -> tuple[str, str, list[str]]:
    """Turn a generator's schema-literal question into a NATURAL target phrase,
    its grouping attribute, and the anchor terms that must survive the rewrite.

    Returns ``(target_phrase, attribute, anchors)``. The schema slug
    ('Cloud Platform/Technology/Revenue', the relation name) never reaches Qwen --
    we hand it a plain-English target so it writes a question a person would ask.
    """
    m = item.metadata or {}
    if item.base == "exact_recall":
        parts = [p.strip() for p in str(m.get("enm_id", "")).split("/") if p.strip()]
        if m.get("enm_type") == "segment_performance" and len(parts) >= 3:
            entity, metric = parts[0], parts[-1]           # Seg / Division / Metric
            return (f"{metric.lower()} of {entity}", f"has_{metric.lower()}",
                    [entity.lower(), metric.lower()])
        # income_statement / balance_sheet: "Line Item" or "Line Item/FY2025"
        line = parts[0] if parts else "the figure"
        year = parts[1] if len(parts) > 1 else None
        if year:
            return (f"{line.lower()} in fiscal {year.replace('FY','')}",
                    f"has_{year.lower()}", [line.lower()])
        return (line.lower(), "has_amount", [line.lower()])
    if item.base == "multi_hop":
        which = m.get("which", "highest")
        section = SECTION_NATURAL.get(m.get("enm_type", ""), "report")
        kw = section.split()[0]                            # segment / income / balance
        return (f"the entry with the {which} reported value in the {section}",
                f"superlative:{m.get('enm_type','')}", [which, kw])
    return (item.query, item.base, [])


def _build_rewrite_prompt(target: str, levels: dict) -> str:
    directions = "; ".join(
        SEMANTIC[f][levels[f]] for f in SEMANTIC if f in levels and levels[f] in SEMANTIC[f])
    return (
        "You write ONE natural-language question that a person would ask a "
        "question-answering assistant about Northwind Industries' annual report. "
        f"The question must ask for: {target}. "
        f"Style: {directions}. "
        "Write it the way a real person speaks — NO database field names, NO "
        "relation names like 'has_revenue', NO quotes, slashes or category labels. "
        "Critically: do NOT invent or include ANY specific numbers, dollar amounts, "
        "percentages, years, or other company names — you are ASKING for the figure, "
        "never stating or guessing one. Keep the specific entity being asked about. "
        "Ask for exactly one thing and do NOT answer it. Reply with ONLY the "
        "question, nothing else.\n\n"
        "QUESTION:"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-runs", type=int, default=150,
                    help="size of the embedded DoE design (number of scenarios)")
    ap.add_argument("--max-per-category", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from _bootstrap import load_store_geo
    store = load_store_geo(STORE, dev)

    CAT = Catalog.load(CATALOGS)
    # Embedded design (Beyond Ship and Pray, ch. Enrichment Design Space): the
    # base question is itself a factor in the design, so #scenarios == n_runs --
    # a DoE of exactly the size we choose, not a full cross product. balance_base
    # blocks the base assignment so every base appears ~n_runs/k times (balanced
    # at small n); the presentation factors are space-filled by sobol+refine.
    # (mode="cross" would instead give full balance at #items x n_runs cost.)
    suite = resolve(CAT, BASE_TYPES, FACTORS, mode="embedded")
    items = CatalogBaseSource(store, max_per_category=args.max_per_category,
                              seed=args.seed).items(suite)
    design = partial(graphdoe_design, method="sobol+refine")
    scns = compose(suite, items, n_runs=args.n_runs, seed=args.seed,
                   design_fn=design, balance_base=True)
    print(f"mined {len(items)} base queries -> designed {len(scns)} DoE scenarios "
          f"(embedded, sobol+refine) over factors {suite.factor_names}")

    # Qwen rewrite, BATCHED through the backend's tokenizer/model (one
    # model.generate per batch with left padding) -- the same pattern the Ship
    # and Pray augmentation uses; far faster than one call at a time.
    backend = LocalTransformersBackend(QWEN_3B, device=str(dev))
    tok, model = backend._tokenizer, backend._model
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    @torch.no_grad()
    def rewrite_batch(prompts: list[str]) -> list[str]:
        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         tokenize=False, add_generation_prompt=True)
                 for p in prompts]
        enc = tok(texts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        with contextlib.redirect_stderr(io.StringIO()):
            out = model.generate(**enc, max_new_tokens=96, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        return [tok.decode(g, skip_special_tokens=True).strip().strip('"') for g in gen]

    # Derive a natural target + grouping attribute + anchor terms per scenario.
    targets = [_natural_target(s.base) for s in scns]
    prompts = [_build_rewrite_prompt(t, s.factor_levels)
               for s, (t, _a, _anc) in zip(scns, targets)]
    rewrites: list[str] = []
    for b in range(0, len(prompts), args.batch_size):
        rewrites.extend(rewrite_batch(prompts[b:b + args.batch_size]))
        print(f"  rewrote {min(b + args.batch_size, len(prompts))}/{len(prompts)}", flush=True)

    os.makedirs(OUTDIR, exist_ok=True)
    cohort, emb_rows, u_groups, draft = [], [], collections.defaultdict(list), []
    dropped = 0
    for j, (s, rw, (target, attr, anchors)) in enumerate(zip(scns, rewrites, targets)):
        norm = re.sub(r"[^a-z0-9 ]", " ", rw.lower())
        on_topic = any(re.sub(r"[^a-z0-9 ]", " ", a.lower()).strip() in norm for a in anchors)
        # Fabrication guard: a number the target did not ask for (a guessed value)
        # is not allowed -- the question must ask for the figure, never state one.
        allowed = set(re.findall(r"\d+", target))
        fabricated = any(n not in allowed for n in re.findall(r"\d+", rw))
        if not rw or not on_topic or fabricated:
            rw = f"What is {target}?"                       # clean natural fallback
            dropped += 1
        seed_id = s.base.qid
        ans = s.base.answer
        cohort.append({
            "id": f"{seed_id}-{j:03d}", "base": s.base.base, "attribute": attr,
            "question": rw, "expected_answer": _jsonable(ans),
            "answer_type": s.base.answer_type, "expect_decision": "accept",
            "_seed": seed_id, "_factors": dict(s.factor_levels),
        })
        emb_rows.append({"text": rw, "label": attr, "_seed": seed_id,
                         "_factors": dict(s.factor_levels)})
        u_groups[attr].append(rw)
        draft.append({"user": rw, "assistant": _golden_answer(s.base),
                      "attribute": attr, "_seed": seed_id})

    _write_json(os.path.join(OUTDIR, "rag_cohort.json"), cohort)
    _write_jsonl(os.path.join(OUTDIR, "embedding_sft.jsonl"), emb_rows)
    _write_json(os.path.join(OUTDIR, "embedding_u_groups.json"), dict(u_groups))
    _write_jsonl(os.path.join(OUTDIR, "llm_draft_sft.jsonl"), draft)

    print(f"\nwrote to {OUTDIR}/:")
    print(f"  rag_cohort.json        {len(cohort)} cases "
          f"({dropped} rewrites fell back to a plain question on the on-topic/fabrication guard)")
    print(f"  embedding_sft.jsonl    {len(emb_rows)} rows, "
          f"{len(u_groups)} attributes: {sorted(u_groups)}")
    print(f"  embedding_u_groups.json {sum(len(v) for v in u_groups.values())} questions")
    print(f"  llm_draft_sft.jsonl    {len(draft)} grounded pairs")
    by_attr = collections.Counter(r["label"] for r in emb_rows)
    print("  per-attribute counts:", dict(by_attr))
    return 0


def _jsonable(x):
    if isinstance(x, (set, tuple)):
        return sorted(x) if isinstance(x, set) else list(x)
    return x


def _golden_answer(item) -> str:
    """A short grounded answer string from the ground truth (LLM-SFT target)."""
    a = item.answer
    if isinstance(a, tuple) and len(a) == 2:
        return f"{a[0]} ({a[1]})"
    if isinstance(a, set):
        return ", ".join(sorted(a))
    return str(a)


def _write_json(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def _write_jsonl(path, rows):
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
