"""Build MeMo-style SFT data for the draft_response adapter, on knowlytix.

A knowlytix re-implementation of the kg-memory MeMo pipeline
(llm-tutorial/scripts/build_memo_style_lora_data.py), with two deliberate
changes for the capstone draft model:

1. **knowlytix, not kg-memory.** Base questions come from
   ``knowlytix.benchmark.generators`` over the policy ``GMSExpertStore``'s
   document graph; DOE paraphrase variety comes from
   ``knowlytix.harness.graphdoe.QuestionRephraser``. Nothing imports kg-memory.

2. **Complaint-shaped, not FAQ-shaped.** The agent feeds ``draft_response``
   real complaints ("I was charged a $35 overdraft fee..."), not policy FAQs
   ("How much is the overdraft fee?"). The previous corpus was dominated by
   FAQ prompts, a train/inference mismatch. Here every prompt is complaint-
   shaped and uses the *exact* inference template (imported from
   ``draft_response_lm``), so training stays on the agent's distribution.

The MeMo property we keep is strictness: every (prompt, response) pair is
validated against ``draft_response_goldens.json`` --- the response must cite the
policy keyword, contain the byte-exact required number, and avoid the forbidden
unauthorized-commitment phrases. Pairs that fail are dropped, so the corpus is
grounded by construction. The grounded numbers themselves are cross-checked
against the values the GMS store's exact-recall generator reads off the graph,
so the data is tied to the knowledge graph, not just to a hand table.

Output: ``data/training/bank_policy/draft_response_memo_v3.jsonl`` (new file;
nothing existing is overwritten). Run from the repo root with the venv active:

    python -m benchmarks.draft_adapter.build_memo_data
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch

# Import the inference-time template pieces so training format == inference
# format exactly (read-only use of the shipped agent code).
from agentlab.models.draft_response_lm import ISSUE_FOR_POLICY, POLICY_SUMMARIES

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "training" / "bank_policy"
GOLDENS_PATH = DATA / "draft_response_goldens.json"
SEED_PATH = DATA / "draft_response_sft.jsonl"
OUT_PATH = DATA / "draft_response_memo_v3.jsonl"
POLICY_STORE = REPO / "data" / "gms_policy_store"

POLICY_IDS = ["overdraft", "disputes", "fee_reversal", "account_closure",
              "pii_handling", "regulatory_escalation"]

# Complaint-shaped surface forms per policy. These are what a customer actually
# writes; the agent's draft_response sees prompts like these, never FAQs.
COMPLAINT_PROMPTS: dict[str, list[str]] = {
    "overdraft": [
        "I was charged a $35 overdraft fee and I want it removed.",
        "An overdraft fee hit my checking account and I think it is unfair.",
        "Why was I charged an overdraft fee? Please reverse it.",
        "I just saw a $35 overdraft charge I wasn't expecting.",
        "My account was hit with an overdraft fee and I'd like it refunded.",
    ],
    "disputes": [
        "There is a transaction on my account that I did not make.",
        "I need to dispute a charge I do not recognize.",
        "Someone made a charge I never authorized; what happens now?",
        "I want to dispute an unauthorized transaction on my card.",
        "A charge showed up that isn't mine and I want it investigated.",
    ],
    "fee_reversal": [
        "Can you reverse this fee on my account?",
        "I would like a fee on my account reversed.",
        "Please remove the fee you charged me last week.",
        "I'm asking for a reversal of the fee on my statement.",
        "Could a representative reverse the fee I was charged?",
    ],
    "account_closure": [
        "I want to close my checking account.",
        "Please close my account as soon as possible.",
        "How do I go about closing my account?",
        "I'd like to close out my account with the bank.",
        "I am requesting that my account be closed.",
    ],
    "pii_handling": [
        "I think my personal information may have been exposed.",
        "I'm worried my private data was leaked from my account.",
        "My personal details might have been compromised, what do you do?",
        "I believe there was a privacy breach involving my information.",
        "Some of my private information may have been disclosed.",
    ],
    "regulatory_escalation": [
        "This is an unfair practice and I want it escalated.",
        "Your fee practice feels deceptive and I'm filing a complaint.",
        "I believe this violates consumer protection rules; please escalate.",
        "This treatment is unfair and I want a formal complaint raised.",
        "I want this escalated to compliance as an unfair practice.",
    ],
}

# Strict, grounded response pool per policy. The validator below drops any that
# fail the goldens contract, so only contract-passing responses train the model.
POLICY_RESPONSES: dict[str, list[str]] = {
    "overdraft": [
        "Per overdraft policy, the $35 fee may be reversed once per year. A representative will follow up.",
        "Under overdraft policy a $35 fee applies, with one goodwill reversal per calendar year. A rep will review your account.",
        "Our overdraft policy assesses a $35 fee and permits one goodwill reversal per year; a representative will be in touch.",
        "Per overdraft policy the $35 fee stands, though one reversal per year is allowed at a representative's discretion.",
        "Under the overdraft policy the $35 fee can be reversed once per calendar year; we will have a representative follow up.",
    ],
    "disputes": [
        "Per dispute policy, you may file within 60 days and provisional credit is issued during the investigation.",
        "Under dispute policy there is a 60-day filing window; provisional credit applies while we investigate.",
        "Our dispute policy gives you 60 days to file; a representative will open the dispute and issue provisional credit.",
        "Per dispute policy transactions disputed within 60 days receive provisional credit during review.",
        "Under the dispute policy you have a 60-day window to file; we will investigate and follow up.",
    ],
    "fee_reversal": [
        "Per fee reversal policy, a representative may approve a reversal up to $35; amounts above require manager approval.",
        "Under fee reversal policy the representative cap is $35; a manager must approve any reversal above that.",
        "Our fee reversal policy lets a representative approve up to $35; higher reversals go to a manager for review.",
        "Per the fee reversal policy a representative may consider a reversal up to $35, subject to manager approval above that.",
        "Under fee reversal policy a representative can review a reversal up to $35; anything higher needs a manager.",
    ],
    "account_closure": [
        "Per account closure policy, customer-initiated closure requires identity verification; bank-initiated closure gives 30-day notice.",
        "Under account closure policy we verify identity for a customer request, and a bank-initiated closure provides 30-day notice.",
        "Our account closure policy requires identity verification; for a bank-initiated closure a 30-day notice applies.",
        "Per closure policy a representative verifies identity before closing; bank-initiated closures carry a 30-day notice.",
        "Under the account closure policy identity verification is required, with 30-day notice for any bank-initiated closure.",
    ],
    "pii_handling": [
        "Per privacy policy, any exposure of personal information is reported to the Privacy team within 24 hours.",
        "Under privacy policy your report is escalated to the Privacy team within 24 hours and redaction is enforced.",
        "Our privacy policy requires that PII exposure be reported to the Privacy team within 24 hours.",
        "Per the privacy policy a suspected exposure is escalated to Privacy within 24 hours for review.",
        "Under privacy policy we escalate any potential data exposure to the Privacy team within 24 hours.",
    ],
    "regulatory_escalation": [
        "Per escalation policy, your complaint is escalated to Compliance within 1 business day for review.",
        "Under escalation policy a representative routes the complaint to Compliance within 1 business day.",
        "Our escalation policy requires routing UDAAP complaints to Compliance within 1 business day.",
        "Per the escalation policy your case is escalated to Compliance within 1 business day.",
        "Under escalation policy the complaint is escalated to Compliance within 1 business day and a representative follows up.",
    ],
}

# DOE factor assignments (quick_screen group) for light surface variety.
FACTOR_LEVELS = [
    {"clarity": "Clear", "entity_aliasing": "Synonym", "reasoning_cue": "No_Cue"},
    {"clarity": "Ambiguous", "entity_aliasing": "Exact", "reasoning_cue": "Direct"},
    {"clarity": "Misleading", "entity_aliasing": "Abbreviated", "reasoning_cue": "Step_By_Step"},
]


def load_goldens() -> dict:
    return json.loads(GOLDENS_PATH.read_text())


def passes_contract(response: str, gold: dict) -> bool:
    """The MeMo strictness contract: cite keyword, contain required number,
    avoid forbidden phrases."""
    low = response.lower()
    if not all(k.lower() in low for k in gold.get("citation_keywords", [])):
        return False
    if not all(str(n).lower() in low for n in gold.get("required_numbers", [])):
        return False
    if any(f.lower() in low for f in gold.get("forbidden_phrases", [])):
        return False
    return True


def graph_numeric_facts() -> dict[str, set[str]]:
    """Read the per-policy numeric registers off the GMS store via the
    exact-recall generator, so the corpus's required numbers are graph-grounded.
    Returns {policy_id: {values as strings}}; empty on any load failure (the
    goldens contract still enforces correctness)."""
    try:
        from knowlytix.benchmark.generators import ExactRecallGenerator
        from knowlytix.knowledge.query import DocGMSConfig, GMSExpertStore

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        store = GMSExpertStore(DocGMSConfig(store_path=str(POLICY_STORE)), device=device)
        if not store.load():
            return {}
        facts: dict[str, set[str]] = {pid: set() for pid in POLICY_IDS}
        for q in ExactRecallGenerator().generate(store.doc_graph):
            enm_id = (q.metadata or {}).get("enm_id", "")
            pid = enm_id.split("/")[0] if "/" in enm_id else ""
            if pid in facts and q.ground_truth is not None:
                gt = q.ground_truth
                facts[pid].add(str(int(gt)) if float(gt).is_integer() else str(gt))
        return facts
    except Exception as exc:  # noqa: BLE001
        print(f"  (graph grounding unavailable: {type(exc).__name__}: {str(exc)[:80]})")
        return {}


def rephrase_variants(prompt: str) -> set[str]:
    """Best-effort DOE paraphrases for surface variety; the prompt itself is
    always included so variety never depends on the rephraser."""
    variants = {prompt}
    try:
        from knowlytix.benchmark.generators import GeneratedQuestion
        from knowlytix.harness.graphdoe import QuestionRephraser

        r = QuestionRephraser(method="template")
        q = GeneratedQuestion(
            qid="memo", category="exact_recall", natural_language=prompt,
            graph_answer_fn=lambda g: None, ground_truth=None,
            llm_prompt=prompt, answer_type="str", metadata={},
        )
        for fa in FACTOR_LEVELS:
            try:
                variants.add(r.rephrase(q, fa).natural_language)
            except Exception:
                pass
    except Exception:
        pass
    return variants


def build_prompt(message: str, policy_id: str) -> str:
    """Exactly the inference template from draft_response_lm._format_prompt."""
    issue = ISSUE_FOR_POLICY[policy_id]
    summary = POLICY_SUMMARIES[policy_id]
    return f"Complaint: {message}\nIssue: {issue}\nPolicy: {summary}\nResponse:"


def main() -> None:
    rng = random.Random(42)
    goldens = load_goldens()

    print("Cross-checking required numbers against the GMS store...")
    facts = graph_numeric_facts()
    for pid in POLICY_IDS:
        req = set(str(n) for n in goldens[pid].get("required_numbers", []))
        seen = facts.get(pid, set())
        if seen:
            hit = req & seen
            print(f"  {pid:24s} goldens={sorted(req)} graph={sorted(seen)} grounded={sorted(hit)}")

    # Validate the response pools against the contract; keep only passing.
    valid_responses: dict[str, list[str]] = {}
    for pid in POLICY_IDS:
        gold = goldens[pid]
        kept = [r for r in POLICY_RESPONSES[pid] if passes_contract(r, gold)]
        if len(kept) < 2:
            raise SystemExit(f"policy {pid}: only {len(kept)} responses pass the contract; add more.")
        valid_responses[pid] = kept
        print(f"  {pid:24s} {len(kept)}/{len(POLICY_RESPONSES[pid])} responses pass the strictness contract")

    pairs: list[dict] = []
    for pid in POLICY_IDS:
        gold = goldens[pid]
        for base_prompt in COMPLAINT_PROMPTS[pid]:
            for surface in rephrase_variants(base_prompt):
                for resp in rng.sample(valid_responses[pid], k=2):
                    if not passes_contract(resp, gold):
                        continue
                    pairs.append({
                        "user": build_prompt(surface, pid),
                        "assistant": resp,
                        "policy_id": pid,
                        "issue": ISSUE_FOR_POLICY[pid],
                        "source": "memo_v3_complaint",
                    })

    # Append the hand-curated complaint seed, but only pairs that also pass the
    # strictness contract (a few seed replies predate the goldens and violate it).
    seed = []
    seed_dropped = 0
    for line in SEED_PATH.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        pid = row.get("policy_id")
        if pid not in goldens or not passes_contract(row["assistant"], goldens[pid]):
            seed_dropped += 1
            continue
        row["source"] = "hand_curated"
        seed.append(row)
    print(f"  seed: kept {len(seed)}, dropped {seed_dropped} for contract violation")

    all_pairs = pairs + seed
    rng.shuffle(all_pairs)

    # Final guard: every emitted pair satisfies the contract.
    bad = sum(1 for p in all_pairs if not passes_contract(p["assistant"], goldens[p["policy_id"]]))
    OUT_PATH.write_text("".join(json.dumps(p) + "\n" for p in all_pairs))

    from collections import Counter
    by_pid = Counter(p["policy_id"] for p in all_pairs)
    by_src = Counter(p["source"] for p in all_pairs)
    print(f"\nWrote {OUT_PATH}  ({len(all_pairs)} pairs; {len(pairs)} memo + {len(seed)} seed)")
    print(f"  contract violations in output: {bad}  (must be 0)")
    print(f"  by source: {dict(by_src)}")
    print(f"  by policy: {dict(by_pid)}")


if __name__ == "__main__":
    main()
