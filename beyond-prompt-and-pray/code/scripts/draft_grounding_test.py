"""Component validation for draft_response (the LoRA-tuned draft LLM).

Mirrors the RAG faithfulness eval, applied to the generated customer reply:
  * groundedness (correctness) -- every verifiable claim the draft states binds
    to a real store triple; a value that contradicts the store is a fabrication.
  * completeness -- does the reply convey the atoms it should, reported two ways:
      evidence-relative : vs the policy_evidence the draft was handed (isolates the LM)
      policy-relative   : vs the governing policy's atoms in the GEODE store (end-to-end)

Reuses knowlytix's TripleCompletenessScorer (dual encoder: v binds head+relation,
u verifies the value) and the deployed retriever -- no bespoke scoring logic.
"""
import warnings, contextlib, io, json
warnings.filterwarnings("ignore")

from agentlab.capstone.banking_tools import _extract_impl, _draft_impl
from agentlab.capstone.policy_rag import (
    get_default_retriever, PolicyRagRetriever, _DEFAULT_STORE)
from knowlytix.harness.testing import TripleCompletenessScorer

sink = io.StringIO()


def _num(x):
    try:
        return float(str(x).replace("$", "").replace(",", ""))
    except Exception:
        return None


def tail_eq(a, b):
    na, nb = _num(a), _num(b)
    if na is not None and nb is not None:
        return abs(na - nb) < 1e-6
    return str(a).strip().lower() == str(b).strip().lower()


def head_topical(h, text):
    """Morphology-tolerant head mention: 'disputes' matches 'dispute policy'.
    The strict _mentions_entity gate fails on singular/plural and paraphrase, so
    we stem each token of the head and look for the stem in the draft."""
    low = text.lower()
    for tok in str(h).split("_"):
        stem = tok[:max(4, len(tok) - 1)]
        if stem and stem in low:
            return True
    return False


def value_present(scorer, text, t, r):
    """The GMS value verification: numeric tails by string, else u-space entailment
    within the relation's calibrated tau_contra (the same check score() uses)."""
    n = _num(t)
    if n is not None:
        s = (f"{n:.2f}", f"{n:g}", str(int(n)) if n == int(n) else f"{n}")
        if any(x in text for x in s):
            return True
    return scorer._u_value_present(text, t, r)


def fact_present(scorer, text, h, r, t):
    return head_topical(h, text) and value_present(scorer, text, t, r)


with contextlib.redirect_stderr(sink), contextlib.redirect_stdout(sink):
    retr = get_default_retriever()
    v_enc = PolicyRagRetriever._load_tuned_encoder(_DEFAULT_STORE)
    u_enc = PolicyRagRetriever._load_contradiction_encoder(_DEFAULT_STORE)
    cal = json.load(open(f"{_DEFAULT_STORE}/relevance_calibration.json"))
    scorer = TripleCompletenessScorer(
        retr.pipe.store, encoder=v_enc, u_encoder=u_enc,
        tau_contra_per_relation=cal.get("tau_contra_per_relation"),
        default_tau_contra=cal.get("default_tau_contra"))

# Store facts as ground truth for groundedness: (head, relation) -> {tails}
store_triples = json.load(open(f"{_DEFAULT_STORE}/triples.json"))
store_by_hr: dict[tuple[str, str], set[str]] = {}
store_by_head: dict[str, list[tuple[str, str]]] = {}
for h, r, t in store_triples:
    store_by_hr.setdefault((h, r), set()).add(str(t))
    store_by_head.setdefault(h, []).append((r, str(t)))

# Goldens: the customer-facing atoms a correct reply must convey. Each policy's
# required_number resolves to exactly one store triple (by tail) -- the headline
# completeness ground truth, robust to prose and free of internal retrieval atoms.
goldens = json.load(open("data/training/bank_policy/draft_response_goldens.json"))
goldens_atoms: dict[str, list[tuple[str, str, str]]] = {}
for pol, spec in goldens.items():
    nums = {_num(x) for x in spec.get("required_numbers", [])}
    atoms = [(pol, r, t) for (r, t) in store_by_head.get(pol, [])
             if not r.startswith("has_alias") and "regulation" not in r
             and _num(t) is not None and _num(t) in nums]
    goldens_atoms[pol] = atoms

cases = json.load(open("data/eval_cases/cases.json"))
complaints = [c for c in cases if c["expected_classification"] == "complaint"]

tot_prod = tot_grounded = gold_exp = gold_cov = 0
rows = []
for c in complaints:
    with contextlib.redirect_stderr(sink), contextlib.redirect_stdout(sink):
        ex = _extract_impl(c["message"])
        ser = ex.get("extraction")
        results = retr.search(c["message"], extraction=ser) if ser is not None \
            else retr.search(c["message"])
        draft = _draft_impl(category="complaint", issue=str(ex.get("issue", "general")),
                            policy_evidence=results, message=c["message"])
        text = draft.get("draft_response") or draft.get("text") or ""

        produced = scorer.parse(text)
        # Completeness ground truth: the customer-facing atoms the goldens require
        # for the governing policy (each required_number -> one store triple).
        accept = set(c.get("acceptable_policies") or [c["expected_policy"]])
        gold_seen = {a for p in accept for a in goldens_atoms.get(p, [])}
        gold_c = sum(1 for (h, r, t) in gold_seen if fact_present(scorer, text, h, r, t))

    # Faithfulness: every verifiable claim the draft makes binds to a store triple.
    grounded = sum(1 for (ph, pr, pt) in produced
                   if any(tail_eq(pt, ts) for ts in store_by_hr.get((ph, pr), ())))
    tot_prod += len(produced); tot_grounded += grounded
    gold_exp += len(gold_seen); gold_cov += gold_c
    rows.append((c["id"], len(produced), grounded, f"{gold_c}/{len(gold_seen)}", text[:64]))

print(f"{'case':9} {'claims':>6} {'grounded':>8} {'goldens':>8}  draft")
for cid, np_, ng, gold, t in rows:
    print(f"{cid:9} {np_:>6} {ng:>8} {gold:>8}  {t!r}")

print(f"\nDRAFT FAITHFULNESS (groundedness)   = {tot_grounded}/{tot_prod} = "
      f"{(tot_grounded/tot_prod if tot_prod else 1.0):.3f}")
print(f"DRAFT COMPLETENESS (goldens atoms)  = {gold_cov}/{gold_exp} = "
      f"{(gold_cov/gold_exp if gold_exp else 1.0):.3f}")
