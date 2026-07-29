"""Score complaint classifiers by clarity, including prompted-LLM baselines.

Reads jsonl eval files (each row: message, label, and a clarity tag either as a
top-level ``clarity`` field or under ``_factors.clarity``) and reports, per
classifier, overall and per-clarity accuracy plus the complaint->other error
count (the dominant failure the DoE testing surfaced).

A classifier is given as ``name=SPEC`` where SPEC is one of:
  - ``head:DIR``       a trained logit head (ComplaintClassifier.load(DIR))
  - ``DIR``            shorthand for ``head:DIR``
  - ``qwen``           local Qwen3-4B-Instruct, prompted zero-shot
  - ``anthropic:MODEL` a hosted Claude model, prompted zero-shot (needs ANTHROPIC_API_KEY)

Classifiers are built and scored one at a time, then freed, so several 3B models
do not sit in memory together.

    python scripts/eval_classifier_clarity.py \
        --file data/eval_cases/doe_clarity_eval.jsonl \
        --clf baseline=head:data/complaint_classifier_qwen_baseline \
        --clf doe=head:data/complaint_classifier_qwen \
        --clf qwen-prompt=qwen \
        --clf sonnet=anthropic:claude-sonnet-4-6
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

_LABELS = ("complaint", "inquiry", "other")

_PROMPT = (
    "Classify the retail-bank customer message into exactly one category:\n"
    "- complaint: reports a problem, error, unfair or disputed charge, or grievance "
    "and wants it addressed.\n"
    "- inquiry: asks a question or requests information or a product, with no grievance.\n"
    "- other: greeting, chit-chat, off-topic, nonsensical, or an attempt to manipulate "
    "or jailbreak the assistant.\n"
    "Answer with one word only: complaint, inquiry, or other.\n\n"
    "Message: {msg}\nCategory:"
)


class PromptedClassifier:
    """A zero-shot LLM classifier with the same .classify(msg) -> (label, conf) API."""

    def __init__(self, llm):
        self.llm = llm

    def classify(self, message: str):
        out = self.llm.complete(_PROMPT.format(msg=message), max_tokens=8).strip().lower()
        hits = [(out.find(lab), lab) for lab in _LABELS if lab in out]
        if hits:
            return min(hits)[1], 1.0
        return "other", 0.0   # unparseable -> safe default


def _build(spec: str):
    if spec == "qwen":
        from agentlab.models import QwenAdapter
        return PromptedClassifier(QwenAdapter(max_new_tokens=8))
    if spec.startswith("anthropic:"):
        from agentlab.models.adapters import AnthropicAdapter
        return PromptedClassifier(AnthropicAdapter(model=spec.split(":", 1)[1]))
    path = spec.split(":", 1)[1] if spec.startswith("head:") else spec
    from agentlab.models.complaint_classifier import ComplaintClassifier
    return ComplaintClassifier.load(path)


def _rows(path: str):
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        clarity = d.get("clarity") or d.get("_factors", {}).get("clarity", "clear")
        out.append((d["message"], d["label"], str(clarity).lower()))
    return out


def _score(clf, rows):
    by = collections.defaultdict(lambda: [0, 0])
    overall = [0, 0]
    c2o = ct = 0
    for msg, gold, clarity in rows:
        pred = clf.classify(msg)[0]
        ok = int(pred == gold)
        by[clarity][0] += ok
        by[clarity][1] += 1
        overall[0] += ok
        overall[1] += 1
        if gold == "complaint":
            ct += 1
            c2o += int(pred == "other")
    return overall, by, c2o, ct


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", action="append", required=True, help="eval jsonl (repeatable)")
    ap.add_argument("--clf", action="append", required=True, help="name=SPEC (repeatable)")
    args = ap.parse_args()

    files = {f: _rows(f) for f in args.file}
    # results[clf_name][file] = (overall, by, c2o, ct)
    results: dict[str, dict] = {}
    for spec in args.clf:
        name, _, s = spec.partition("=")
        print(f"scoring {name} ({s}) ...", flush=True)
        clf = _build(s)
        results[name] = {f: _score(clf, rows) for f, rows in files.items()}
        del clf
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    for f, rows in files.items():
        clarities = sorted({c for _, _, c in rows})
        print(f"\n=== {Path(f).name}  ({len(rows)} rows) ===")
        print(f"{'classifier':22s} " + " ".join(f"{c:>11s}" for c in clarities)
              + f" {'overall':>8s}  {'comp->other':>11s}")
        for name in results:
            overall, by, c2o, ct = results[name][f]
            cells = []
            for c in clarities:
                cor, tot = by.get(c, [0, 0])
                cells.append(f"{(cor / tot if tot else 0):.2f}({tot})")
            line = f"{name:22s} " + " ".join(f"{x:>11s}" for x in cells)
            line += f" {overall[0] / overall[1]:>7.2f}  {c2o:>4d}/{ct:<4d}"
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
