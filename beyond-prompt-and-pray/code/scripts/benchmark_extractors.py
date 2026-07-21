#!/usr/bin/env python
"""Compare extract_facts variants on the SAME held-out rephrased DoE rows.

Test set = the rows from data/diag_extract_facts.json (the materialized qwen DoE
messages + ground truth, 89 scored rows) -- reused so every variant is scored on
identical inputs with no re-run. Reference variants (rule-only, hybrid) come
straight from that file; the geometric variants are computed here:

  geo_minilm  : nearest hand-exemplar, stock MiniLM
  geo_geode   : nearest hand-exemplar, GEODE document-tuned encoder
  geo_task    : FineTunedEmbedding (task-tuned low-rank) nearest-prototype + its
                own per-class calibrated abstain (use_threshold=True)

Exemplar variants calibrate their abstain on the CLEAN seed cohort (train/test
split: clean->calibrate, rephrased->test). All embeddings are computed in batched
passes (a handful of encoder loads total). Each variant is scored per field and
joint, under BOTH the strict and the accept-set (layer 2) scorer. Writes
data/benchmark_extractors.json.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from agentlab.models.geometric_extractor import _ISSUE_EXEMPLARS, _PRODUCT_EXEMPLARS, _load_encoder

_REPO = Path(__file__).resolve().parents[1]
_GEODE = "tuned:" + str(_REPO / "data" / "gms_policy_store_geode")


def _norm(t):
    return F.normalize(torch.as_tensor(t, dtype=torch.float32), p=2, dim=-1)


class ExemplarScorer:
    """Batched nearest-exemplar label scorer for one encoder. Encodes every
    exemplar and every query in one pass each, then scores by max cosine per
    label. Abstain threshold (per classifier) calibrated by `fit_threshold`."""

    def __init__(self, encoder, exemplars, abstain_label):
        self.abstain = abstain_label
        self.labels = list(exemplars)
        owners, phrases = [], []
        for lab, exs in exemplars.items():
            owners += [lab] * len(exs); phrases += exs
        self.owners = owners
        self.emb = _norm(encoder(phrases))          # one load
        self.threshold = 0.0

    def label_scores(self, query_emb):
        sims = (query_emb @ self.emb.T)             # (N, n_exemplars)
        N = sims.shape[0]
        best = {lab: torch.full((N,), -1e9) for lab in self.labels}
        for j, owner in enumerate(self.owners):
            best[owner] = torch.maximum(best[owner], sims[:, j])
        return best  # {label: (N,)}

    def predict(self, query_emb):
        bs = self.label_scores(query_emb)
        labs = self.labels
        stack = torch.stack([bs[l] for l in labs], dim=1)  # (N, C)
        top = stack.argmax(1)
        out = []
        for i, c in enumerate(top.tolist()):
            out.append(self.abstain if stack[i, c] < self.threshold else labs[c])
        return out

    def fit_threshold(self, clean_emb, clean_labels, ceiling=0.10):
        bs = self.label_scores(clean_emb)
        labs = self.labels
        stack = torch.stack([bs[l] for l in labs], dim=1)
        top = stack.argmax(1)
        risky, n = [], 0
        for i, truth in enumerate(clean_labels):
            if truth != self.abstain:
                continue
            n += 1
            c = int(top[i])
            if labs[c] != self.abstain:
                risky.append(float(stack[i, c]))
        if n == 0:
            self.threshold = 0.0; return 0.0
        for cut in [0.0] + sorted(set(risky)):
            if sum(1 for t in risky if t >= cut) / n <= ceiling:
                self.threshold = float(cut); return cut
        self.threshold = float(max(risky)) + 1e-6
        return self.threshold


def _load_cases():
    cases = json.loads((_REPO / "data" / "eval_cases" / "cases.json").read_text())
    cases = cases if isinstance(cases, list) else cases.get("cases", cases)
    return {c["id"]: c for c in cases}, cases


def _score(rows, preds, by_id):
    """preds: {row_index: (product, issue)}."""
    out = {k: [0, 0] for k in ("p_strict", "i_strict", "j_strict", "i_acc", "j_acc")}
    conf = Counter()
    for idx, r in enumerate(rows):
        ep, ei = r["expected_product"], r["expected_issue"]
        case = by_id[r["seed_case"]]
        acc_p = set(case.get("acceptable_products") or ([ep] if ep else []))
        acc_i = set(case.get("acceptable_issues") or ([ei] if ei else []))
        pp, pi = preds[idx]
        sp = (pp == ep) if ep else None
        si = (pi == ei) if ei else None
        ap = (pp in acc_p) if ep else None
        ai = (pi in acc_i) if ei else None
        for key, val in (("p_strict", sp), ("i_strict", si), ("i_acc", ai)):
            if val is not None:
                out[key][0] += int(val); out[key][1] += 1
        sj = [v for v in (sp, si) if v is not None]
        aj = [v for v in (ap, ai) if v is not None]
        if sj:
            out["j_strict"][0] += int(all(sj)); out["j_strict"][1] += 1
        if aj:
            out["j_acc"][0] += int(all(aj)); out["j_acc"][1] += 1
        if ei is not None:
            conf[(ei, pi)] += 1
    rate = {k: (round(c / n, 3) if n else None, n) for k, (c, n) in out.items()}
    return rate, conf


def main() -> int:
    diag = json.loads((_REPO / "data" / "diag_extract_facts.json").read_text())
    rows = diag["rows"]
    by_id, cases = _load_cases()
    messages = [r["message"] for r in rows]
    clean = [c["message"] for c in cases]
    clean_prod = [c.get("expected_product") or "unknown" for c in cases]
    clean_issue = [c.get("expected_issue") or "general" for c in cases]

    variants = {}

    # reference variants from the diagnostic
    variants["rule_only"] = {i: (r["rule"].get("product"), r["rule"].get("issue"))
                             for i, r in enumerate(rows)}
    variants["hybrid_qwen"] = {i: (r["hybrid"].get("product"), r["hybrid"].get("issue"))
                               for i, r in enumerate(rows)}

    cal = {}
    # exemplar geometric variants
    for tag, spec in (("geo_minilm", "minilm"), ("geo_geode", _GEODE)):
        enc = _load_encoder(spec)
        encode = enc if enc is not None else None
        if encode is None:
            from knowlytix.core.graph.encoders import encode_texts
            encode = encode_texts
        ps = ExemplarScorer(encode, _PRODUCT_EXEMPLARS, "unknown")
        is_ = ExemplarScorer(encode, _ISSUE_EXEMPLARS, "general")
        q_emb = _norm(encode(messages))
        c_emb = _norm(encode(clean))
        pt = ps.fit_threshold(c_emb, clean_prod)
        it = is_.fit_threshold(c_emb, clean_issue)
        cal[tag] = {"product_threshold": round(pt, 4), "issue_threshold": round(it, 4)}
        pp, pi = ps.predict(q_emb), is_.predict(q_emb)
        variants[tag] = {i: (pp[i], pi[i]) for i in range(len(rows))}

    # task-tuned FineTunedEmbedding, with and without its per-class abstain.
    from knowlytix.embedding import FineTunedEmbedding
    ftp = FineTunedEmbedding.load(_REPO / "data" / "extract_encoder_product")
    fti = FineTunedEmbedding.load(_REPO / "data" / "extract_encoder_issue")
    tp_a = [l or "unknown" for l in ftp.classify(messages, use_threshold=True)[0]]
    ti_a = [l or "general" for l in fti.classify(messages, use_threshold=True)[0]]
    tp_n = ftp.classify(messages, use_threshold=False)[0]   # nearest prototype, no abstain
    ti_n = fti.classify(messages, use_threshold=False)[0]
    qprod = [rows[i]["hybrid"].get("product") for i in range(len(rows))]  # Qwen product
    variants["geo_task"] = {i: (tp_a[i], ti_a[i]) for i in range(len(rows))}
    variants["geo_task_noabs"] = {i: (tp_n[i], ti_n[i]) for i in range(len(rows))}
    variants["qwen_p+geo_i"] = {i: (qprod[i], ti_a[i]) for i in range(len(rows))}
    variants["qwen_p+geo_i_noabs"] = {i: (qprod[i], ti_n[i]) for i in range(len(rows))}

    results, confs = {}, {}
    for tag, preds in variants.items():
        results[tag], confs[tag] = _score(rows, preds, by_id)

    cols = ["p_strict", "i_strict", "j_strict", "i_acc", "j_acc"]
    hdr = f"{'variant':14}" + "".join(f"{c:>11}" for c in cols)
    print(hdr); print("-" * len(hdr))
    for tag in variants:
        row = results[tag]
        print(f"{tag:14}" + "".join(
            f"{(row[c][0] if row[c][0] is not None else 0):>11.3f}" for c in cols))
    print("\n(product strict | issue strict | joint strict | issue accept-set | joint accept-set)")
    print(f"calibrated exemplar thresholds: {json.dumps(cal)}")

    geo_tags = ["geo_minilm", "geo_geode", "geo_task", "geo_task_noabs",
                "qwen_p+geo_i", "qwen_p+geo_i_noabs"]
    best = max(geo_tags, key=lambda t: results[t]["j_strict"][0] or 0)
    print(f"\nissue confusion -- {best} (expected -> predicted):")
    for (e, p), n in confs[best].most_common():
        print(f"   {e:18} -> {str(p):18} {n}{'' if e == p else '  <-- miss'}")

    (_REPO / "data" / "benchmark_extractors.json").write_text(json.dumps({
        "n_rows": len(rows),
        "results": results,
        "exemplar_calibration": cal,
        "task_thresholds": {
            "product": {l: round(float(t), 4) for l, t in zip(ftp.label_order, ftp.thresholds.tolist())},
            "issue": {l: round(float(t), 4) for l, t in zip(fti.label_order, fti.thresholds.tolist())},
        },
        "best_geo": best,
        "issue_confusion_best_geo": {f"{e} -> {p}": n for (e, p), n in confs[best].most_common()},
    }, indent=2) + "\n")
    print(f"\nwrote {_REPO / 'data' / 'benchmark_extractors.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
