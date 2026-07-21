#!/usr/bin/env python
"""Strong-LLM ceiling test: how well can Sonnet map complaints -> policy labels?

Classifies the same 89 held-out capstone rows into (product, issue) with Claude
Sonnet via the authenticated `claude` CLI (headless -p), batched. This is an
upper-bound / oracle: if Sonnet is also limited on `issue`, the task is genuinely
hard (label ambiguity); if Sonnet is far above the small models (geo_task 0.63,
hybrid 0.46), the ceiling is higher and a bigger model / better data could close
the gap. Scored strict + accept-set, same as every other arm.
"""
from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

_SYS = (
    "You classify a retail-bank customer complaint into a fixed policy taxonomy. "
    "Return ONLY a JSON array, one object per input, in order: "
    '[{"i": <index>, "product": <p>, "issue": <iss>}].\n'
    'product is one of: "checking_account","credit_card","mortgage","loan","unknown".\n'
    'issue is one of: "overdraft_fee","account_issue","credit_card_issue",'
    '"mortgage_issue","loan_issue","general".\n'
    "Rules: overdraft_fee = a fee/overdraft/NSF charge the customer is charged or "
    "wants reversed/waived. account_issue = a checking/savings problem that is NOT "
    "a fee (closing, an unauthorized or duplicate charge dispute, access, account "
    "options). credit_card_issue / mortgage_issue / loan_issue = a problem with that "
    "product. general = vague/procedural message where the specific problem is "
    "unclear or unstated; product = unknown when no product is identifiable. "
    "issue and product MUST each be EXACTLY one of the listed strings -- never any "
    "other word (an account closure is account_issue; an unclear concern is "
    "general; no product is unknown). Do not add commentary."
)

_ISSUE_OK = {"overdraft_fee", "account_issue", "credit_card_issue",
             "mortgage_issue", "loan_issue", "general"}
_PROD_OK = {"checking_account", "credit_card", "mortgage", "loan", "unknown"}
_ISSUE_MAP = {"account_closure": "account_issue", "account_dispute": "account_issue",
              "dispute": "account_issue", "unauthorized_transaction": "account_issue",
              "unknown": "general", "none": "general", "": "general"}
_PROD_MAP = {"none": "unknown", "savings": "checking_account",
             "savings_account": "checking_account", "debit": "checking_account",
             "debit_card": "checking_account", "": "unknown"}


def _norm(product, issue):
    issue = str(issue or "").strip().lower()
    product = str(product or "").strip().lower()
    if issue not in _ISSUE_OK:
        issue = _ISSUE_MAP.get(issue, "general")
    if product not in _PROD_OK:
        product = _PROD_MAP.get(product, "unknown")
    return product, issue


def _classify_batch(items: list[tuple[int, str]]) -> dict[int, dict]:
    lines = "\n".join(f'{i}: {m}' for i, m in items)
    prompt = f"{_SYS}\n\nClassify these {len(items)} messages:\n{lines}"
    out = subprocess.run(
        ["claude", "-p", "--model", "sonnet"],
        input=prompt, capture_output=True, text=True, timeout=300).stdout
    m = re.search(r"\[.*\]", out, re.DOTALL)
    res: dict[int, dict] = {}
    if m:
        try:
            for o in json.loads(m.group(0)):
                p, iss = _norm(o.get("product"), o.get("issue"))
                res[int(o["i"])] = {"product": p, "issue": iss}
        except Exception:
            pass
    return res


def main() -> int:
    diag = json.loads((_REPO / "data" / "diag_extract_facts.json").read_text())
    rows = diag["rows"]
    cases = json.loads((_REPO / "data" / "eval_cases" / "cases.json").read_text())
    by_id = {c["id"]: c for c in (cases if isinstance(cases, list) else cases["cases"])}

    preds: dict[int, dict] = {}
    B = 15
    for b in range(0, len(rows), B):
        batch = [(i, rows[i]["message"]) for i in range(b, min(b + B, len(rows)))]
        preds.update(_classify_batch(batch))
        print(f"  classified {len(preds)}/{len(rows)}", flush=True)

    acc = {k: [0, 0] for k in ("p_strict", "i_strict", "j_strict", "i_acc", "j_acc")}
    conf = Counter()
    missing = 0
    for idx, r in enumerate(rows):
        ep, ei = r["expected_product"], r["expected_issue"]
        case = by_id[r["seed_case"]]
        acc_p = set(case.get("acceptable_products") or ([ep] if ep else []))
        acc_i = set(case.get("acceptable_issues") or ([ei] if ei else []))
        pr = preds.get(idx)
        if pr is None:
            missing += 1
            product, issue = "unknown", "general"
        else:
            product, issue = pr["product"], pr["issue"]
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
    print(f"\n=== Sonnet (ceiling) on {len(rows)} rows; unparsed {missing} ===")
    for k in ("p_strict", "i_strict", "j_strict", "i_acc", "j_acc"):
        print(f"  {k:10} {rate[k][0]}  ({rate[k][1]})")
    print("issue confusion:")
    for (e, p), n in conf.most_common():
        print(f"   {e:18} -> {str(p):18} {n}{'' if e == p else '  <-- miss'}")
    (_REPO / "data" / "benchmark_sonnet.json").write_text(json.dumps(
        {"rates": rate, "unparsed": missing,
         "issue_confusion": {f"{e} -> {p}": n for (e, p), n in conf.most_common()}}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
