"""Head-to-head: geometric polarity gate vs Qwen LoRA classifier.

Both gates decide the same 3-class question on the held-out GMS+DoE test split --
does an asserted stance value agree with, contradict, or fall unrelated to the
value the policy holds. Three columns are scored:

  * **Gate A (geometry, token)** -- ``ValuePolarityChecker.check`` on the asserted
    value token, the input the checker is designed for (cap + v-resolution + u-tension).
  * **Gate A (geometry, prose)** -- the same checker on the full surface sentence,
    the out-of-distribution input its own docstring warns about.
  * **Gate B (Qwen LoRA)** -- the fine-tuned classifier on the surface sentence.

Reported: 3-class accuracy, a per-surface-form breakdown (the phrasing-robustness
axis), and the operationally decisive binary -- contradiction detection
(precision/recall/F1 on the ``contradicted`` class, the dangerous case for
output disclosure). Writes data/polarity_gate_comparison.json.

    python scripts/compare_polarity_gates.py \
        --store data/gms_governed_store \
        --adapter data/polarity_classifier_qwen
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_TEST = _ROOT / "data" / "training" / "polarity" / "test_polarity.jsonl"

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
    return f"Policy: {_phrase(r['relation'])} is {r['stored']}. Claim: {r['message']}"


def _pair_tuple(r):
    return (f"stored: ({r['head']}, {r['relation']}, {r['stored']}) | "
            f"asserted: ({r['head']}, {r['relation']}, {r['asserted']})")


_BUILDERS = {"nl": _pair_nl, "tuple": _pair_tuple}


def _read(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def _acc(preds, gold):
    return sum(int(p == g) for p, g in zip(preds, gold)) / max(1, len(gold))


def _binary_contradiction(preds, gold):
    """Precision/recall/F1 treating 'contradicted' as the positive class."""
    tp = sum(int(p == "contradicted" and g == "contradicted") for p, g in zip(preds, gold))
    fp = sum(int(p == "contradicted" and g != "contradicted") for p, g in zip(preds, gold))
    fn = sum(int(p != "contradicted" and g == "contradicted") for p, g in zip(preds, gold))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn}


def _per_surface(rows, preds, gold):
    by = collections.defaultdict(lambda: [0, 0])
    for r, p, g in zip(rows, preds, gold):
        s = r.get("_factors", {}).get("surface", "?")
        by[s][0] += int(p == g)
        by[s][1] += 1
    return {s: round(by[s][0] / by[s][1], 3) for s in sorted(by)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="data/gms_governed_store")
    ap.add_argument("--adapters", nargs="+",
                    default=["data/polarity_classifier_qwen_tuple",
                             "data/polarity_classifier_qwen_nl"],
                    help="one or more trained LoRA adapter dirs (each carries "
                         "input_format.json declaring nl|tuple)")
    args = ap.parse_args()

    import torch
    from knowlytix.embedding import FineTunedEmbedding
    from knowlytix.knowledge.query import DocGMSConfig, GMSExpertStore
    from knowlytix.knowledge.rag import PolarityCuts, ValuePolarityChecker

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store_path = Path(args.store)
    rows = _read(_TEST)
    gold = [r["label"] for r in rows]
    print(f"test rows: {len(rows)}  {dict(collections.Counter(gold))}")

    # ---- Gate A: geometric ValuePolarityChecker ----
    store = GMSExpertStore(DocGMSConfig(store_path=str(store_path), ingest_mode="regex"),
                           device=device)
    assert store.load(), f"failed to load store {store_path}"
    v = FineTunedEmbedding.load(str(store_path / "tuned_encoder"))
    u = FineTunedEmbedding.load(str(store_path / "value_polarity_encoder"))
    cuts = PolarityCuts.load(str(store_path / "value_polarity_calibration.json"))
    checker = ValuePolarityChecker(store, v.encode, u.encode, cuts)

    def gate_a(asserted_field):
        preds = []
        for r in rows:
            try:
                verdict = checker.check(r["head"], r["relation"],
                                        r[asserted_field], r["stored"])
            except Exception:
                verdict = "uncertain"
            # checker returns supported/contradicted/uncertain; align to label space
            preds.append(verdict if verdict in ("supported", "contradicted") else "uncertain")
        return preds

    a_token = gate_a("asserted")   # designed input: the value token
    a_prose = gate_a("message")    # OOD input: the full sentence

    # ---- Gate B: Qwen LoRA classifier(s), one per input representation ----
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from peft import PeftModel

    def eval_adapter(adapter_dir):
        adapter = Path(adapter_dir)
        labels = json.loads((adapter / "labels.json").read_text())
        fmt = json.loads((adapter / "input_format.json").read_text())["input"]
        build = _BUILDERS[fmt]
        cfg = json.loads((adapter / "adapter_config.json").read_text())
        base_id = cfg["base_model_name_or_path"]
        tok = AutoTokenizer.from_pretrained(base_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        base = AutoModelForSequenceClassification.from_pretrained(
            base_id, num_labels=len(labels),
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            device_map=device.type if device.type == "cuda" else None)
        base.config.pad_token_id = tok.pad_token_id
        clf = PeftModel.from_pretrained(base, str(adapter))
        clf.eval()
        msgs = [build(r) for r in rows]
        preds = []
        with torch.no_grad():
            for i in range(0, len(msgs), 32):
                enc = tok(msgs[i:i + 32], truncation=True, max_length=96,
                          padding=True, return_tensors="pt").to(device)
                logits = clf(**enc).logits
                preds.extend(labels[k] for k in logits.argmax(-1).tolist())
        del clf, base
        torch.cuda.empty_cache()
        return fmt, preds

    # ---- report ----
    cols = {
        "gate_a_geometry_token": a_token,
        "gate_a_geometry_prose": a_prose,
    }
    for ad in args.adapters:
        fmt, preds = eval_adapter(ad)
        cols[f"gate_b_qwen_{fmt}"] = preds
    result = {"n_test": len(rows), "label_dist": dict(collections.Counter(gold)),
              "gates": {}}
    print("\n" + "=" * 68)
    print(f"{'gate':<28}{'3-class acc':>12}{'contra P':>10}{'contra R':>10}{'contra F1':>11}")
    print("-" * 68)
    for name, preds in cols.items():
        acc = _acc(preds, gold)
        binc = _binary_contradiction(preds, gold)
        per_s = _per_surface(rows, preds, gold)
        result["gates"][name] = {"accuracy": round(acc, 3),
                                 "contradiction": binc, "by_surface": per_s}
        print(f"{name:<28}{acc:>12.3f}{binc['precision']:>10.3f}"
              f"{binc['recall']:>10.3f}{binc['f1']:>11.3f}")
    print("-" * 68)
    print("\nper-surface 3-class accuracy:")
    for name in cols:
        print(f"  {name:<28} {result['gates'][name]['by_surface']}")

    out = _ROOT / "data" / "polarity_gate_comparison.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
