#!/usr/bin/env python
"""BM25 lexical baseline for the extractor (product + issue), 89 held-out rows.

Indexes the SAME DoE-augmented training corpus the embedding-SFT and Qwen-LoRA
arms use (extract_doe_train.jsonl), so this isolates lexical retrieval vs learned
representations. Classifies each test message by BM25 top-k nearest training
messages, majority-voting their (product, issue) labels -- a standard
BM25-as-classifier. `general`/`unknown` are ordinary labels in the corpus, so a
vague message that lexically matches general seeds is classified general (no
separate abstain). Self-contained BM25 (no rank_bm25 dependency).
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_TOK = re.compile(r"[a-z0-9$]+")


def _tok(s):
    return _TOK.findall(s.lower())


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avgdl = sum(self.len) / max(len(docs), 1)
        df = Counter()
        for d in docs:
            for t in set(d):
                df[t] += 1
        N = len(docs)
        self.idf = {t: math.log(1 + (N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def top_k(self, query, k):
        q = _tok(query)
        scores = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            dl = self.len[i]
            for t in q:
                if t not in tf:
                    continue
                idf = self.idf.get(t, 0.0)
                f = tf[t]
                s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            scores.append((s, i))
        scores.sort(reverse=True)
        return scores[:k]


def main() -> int:
    train = [json.loads(l) for l in (_REPO / "data" / "training" / "extract_doe_train.jsonl"
                                     ).read_text().splitlines() if l.strip()]
    docs = [_tok(r["message"]) for r in train]
    bm = BM25(docs)
    K = 5

    diag = json.loads((_REPO / "data" / "diag_extract_facts.json").read_text())
    rows = diag["rows"]
    cases = json.loads((_REPO / "data" / "eval_cases" / "cases.json").read_text())
    by_id = {c["id"]: c for c in (cases if isinstance(cases, list) else cases["cases"])}

    acc = {k: [0, 0] for k in ("p_strict", "i_strict", "j_strict", "i_acc", "j_acc")}
    conf = Counter()
    for r in rows:
        ep, ei = r["expected_product"], r["expected_issue"]
        case = by_id[r["seed_case"]]
        acc_p = set(case.get("acceptable_products") or ([ep] if ep else []))
        acc_i = set(case.get("acceptable_issues") or ([ei] if ei else []))
        hits = bm.top_k(r["message"], K)
        if hits and hits[0][0] > 0:
            pv = Counter(train[i]["product"] for _, i in hits)
            iv = Counter(train[i]["issue"] for _, i in hits)
            product = pv.most_common(1)[0][0]
            issue = iv.most_common(1)[0][0]
        else:
            product, issue = "unknown", "general"
        sp = (product == ep) if ep else None
        si = (issue == ei) if ei else None
        ap = (product in acc_p) if ep else None
        ai = (issue in acc_i) if ei else None
        for key, val in (("p_strict", sp), ("i_strict", si), ("i_acc", ai)):
            if val is not None:
                acc[key][0] += int(val); acc[key][1] += 1
        sj = [v for v in (sp, si) if v is not None]
        aj = [v for v in (ap, ai) if v is not None]
        if sj:
            acc["j_strict"][0] += int(all(sj)); acc["j_strict"][1] += 1
        if aj:
            acc["j_acc"][0] += int(all(aj)); acc["j_acc"][1] += 1
        if ei is not None:
            conf[(ei, issue)] += 1
    rate = {k: (round(c / n, 3) if n else None, f"{c}/{n}") for k, (c, n) in acc.items()}
    print(f"BM25 k={K} on {len(rows)} rows (corpus {len(train)})")
    for k in ("p_strict", "i_strict", "j_strict", "i_acc", "j_acc"):
        print(f"  {k:10} {rate[k][0]}  ({rate[k][1]})")
    print("issue confusion:")
    for (e, p), n in conf.most_common():
        print(f"   {e:18} -> {str(p):18} {n}{'' if e == p else '  <-- miss'}")
    (_REPO / "data" / "benchmark_bm25.json").write_text(json.dumps(
        {"k": K, "n_train": len(train), "rates": rate,
         "issue_confusion": {f"{e} -> {p}": n for (e, p), n in conf.most_common()}}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
