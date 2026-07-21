#!/usr/bin/env python
"""Does a structured-fact query retrieve policy better than the raw complaint?

Local Qwen extracts structured facts (agentlab.models.structured_extractor) for
the 89 held-out rows; compose_query() turns them into a clean policy query. We
run the SAME GEODE retriever on (a) the raw message and (b) the structured query
and compare policy recall@3 vs expected_policy (accept-set; vague None-policy rows
should retrieve nothing). Also tabulates event/product -> the regulatory signal.
No Sonnet anywhere (book rule). Writes data/test_structured_query.json.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
_TOPK = 3
# event -> regulatory domain (the escalation signal extract should carry)
_EVENT_REG = {"unauthorized_transaction": "disputes/Reg_E", "fee_dispute": "overdraft/UDAAP",
              "account_closure": "account_closure", "credit_reporting": "Reg_V",
              "servicing_error": "Reg_X(mortgage)/servicing", "inquiry": "-", "none": "-"}


def main() -> int:
    from agentlab.models.qwen_adapter import _DEFAULT_MODEL, _load
    from agentlab.models.structured_extractor import parse_facts, compose_query, _SYS
    from agentlab.capstone.policy_rag import PolicyRagRetriever

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, model = _load(_DEFAULT_MODEL, device)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    diag = json.loads((_REPO / "data" / "diag_extract_facts.json").read_text())
    rows = diag["rows"]
    cases = json.loads((_REPO / "data" / "eval_cases" / "cases.json").read_text())
    by_id = {c["id"]: c for c in (cases if isinstance(cases, list) else cases["cases"])}

    @torch.no_grad()
    def extract_batch(msgs):
        texts = [tok.apply_chat_template(
            [{"role": "system", "content": _SYS}, {"role": "user", "content": m}],
            tokenize=False, add_generation_prompt=True) for m in msgs]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        out = model.generate(**enc, max_new_tokens=160, do_sample=False, pad_token_id=tok.eos_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        return [tok.decode(g, skip_special_tokens=True) for g in gen]

    facts = []
    for b in range(0, len(rows), 16):
        for txt in extract_batch([r["message"] for r in rows[b:b + 16]]):
            facts.append(parse_facts(txt) or {})
        print(f"  extracted {len(facts)}/{len(rows)}", flush=True)

    retr = PolicyRagRetriever()  # default deployed GEODE store search_policy uses

    def recall(route_fn):
        hit = tot = vague_ok = vague_tot = 0
        for i, r in enumerate(rows):
            case = by_id[r["seed_case"]]
            exp = case.get("expected_policy")
            acc = set(case.get("acceptable_policies") or ([exp] if exp else []))
            pols = route_fn(i)[:_TOPK]
            if exp:
                tot += 1; hit += int(bool(acc & set(pols)))
            else:
                vague_tot += 1; vague_ok += int(len(pols) == 0)
        return (round(hit / tot, 3), f"{hit}/{tot}"), (round(vague_ok / vague_tot, 3), f"{vague_ok}/{vague_tot}")

    raw_recall, raw_vague = recall(lambda i: retr.route(rows[i]["message"]))
    q_recall, q_vague = recall(lambda i: retr.route(compose_query(facts[i]) or rows[i]["message"]))

    ev = Counter(f.get("event") for f in facts)
    pr = Counter(f.get("product") for f in facts)
    print("\n=== RAG policy recall@3: raw message vs structured query ===")
    print(f"  raw message      recall {raw_recall}   vague->none {raw_vague}")
    print(f"  structured query recall {q_recall}   vague->none {q_vague}")
    print(f"\nextracted event dist:   {dict(ev)}")
    print(f"extracted product dist: {dict(pr)}")
    print("sample (msg -> query | event/product):")
    for i in range(0, 89, 12):
        print(f"   {rows[i]['message'][:40]!r} -> {compose_query(facts[i])[:55]!r} "
              f"| {facts[i].get('event')}/{facts[i].get('product')}")
    (_REPO / "data" / "test_structured_query.json").write_text(json.dumps({
        "raw_recall": raw_recall, "raw_vague": raw_vague,
        "structured_recall": q_recall, "structured_vague": q_vague,
        "event_dist": dict(ev), "product_dist": dict(pr),
        "facts": facts}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
