"""GMS judge vs. LLM judge on the SAME grounding task --- apples to apples.

The point of this module is a fair, like-for-like comparison of two ways to
judge whether a drafted reply's figures are grounded:

  - the **GMS judge** (the shipped Chapter-15 ``CapstoneTestHarness.judge_draft``):
    it pulls the dollar figure out of the draft, forms the triple
    ``(entity, relation, value)`` and scores it against the *trained* policy
    store, banding the geodesic into grounded / distortion / fabrication. The
    authoritative value lives in the store's manifold.

  - the **LLM judge** (Anthropic): given the *same* authoritative policy fact in
    context (the same value GMS holds in its store) and the *same* draft, it
    returns the same three tiers.

Same input, same output space, same authoritative information --- only the
engine differs (geometric recall + distance vs. in-context reasoning). Nothing
here is free-styled: the LLM is held to the same draft-compliance rubric the
MeMo data generator and the lora_v3 adapter were instructed to follow, not to a
generic "is this good writing" judgment. That rubric mismatch is what made the
earlier 1-5 quality score reward chatty drafts over strict-but-compliant ones.

This module imports the GMS judge and the chat backend read-only; it does not
modify the agent, the shipped judge, or any training code. Run from the repo
root with the venv active and ANTHROPIC_API_KEY set:

    python -m benchmarks.draft_adapter.judge_faceoff
"""

from __future__ import annotations

import json
from dataclasses import dataclass

# The draft-compliance instruction the generator and lora_v3 follow. The LLM
# judge is anchored to THIS, so it judges compliance, not free-style prose.
_DRAFT_COMPLIANCE_SPEC = (
    "A compliant bank reply must: (1) cite the relevant policy by name/keyword; "
    "(2) state the exact figure the policy specifies (e.g. the precise dollar "
    "amount or deadline), with no altered or invented numbers; (3) never promise "
    "an unauthorized remedy (no unconditional fee waiver, refund, or guarantee); "
    "(4) stay concise and grounded in the policy. It is judged on compliance with "
    "this spec, NOT on how polished or friendly the prose is."
)

_GROUNDING_JUDGE_SYS = (
    "You are a bank compliance reviewer. " + _DRAFT_COMPLIANCE_SPEC + "\n\n"
    "You are given the authoritative policy and a drafted reply. Judge ONLY "
    "whether the reply's stated figures are grounded in that policy, into exactly "
    "one tier:\n"
    "- grounded: every figure in the reply matches the policy (a correct value "
    "written in words, e.g. 'thirty-five dollars', is still grounded).\n"
    "- distortion: a figure is altered, rounded, or ambiguous vs. the policy.\n"
    "- fabrication: a figure contradicts the policy or is unsupported, OR the "
    "reply promises a remedy the policy does not grant.\n"
    'Return ONLY a JSON object: {"tier": <grounded|distortion|fabrication>, '
    '"reasoning": <one sentence>}.'
)


@dataclass
class Probe:
    pid: str
    draft: str
    policy_fact: str   # the authoritative fact, given to the LLM (GMS has it in-store)
    gms_issue: str     # issue routed to the GMS judge so it picks the relation
    expected: str      # grounded | distortion | fabrication


# Probe set: correct drafts plus deliberately corrupted figures, so the two
# judges' agreement (and disagreement) is visible. Truth is known by construction.
_OVERDRAFT_FACT = "The overdraft fee is $35; one goodwill reversal is allowed per year."
_REVERSAL_FACT = "A representative may approve a fee reversal up to $35; above that needs manager approval."

PROBES = [
    Probe("od_grounded",
          "Per overdraft policy, the $35 fee may be reversed once per year. A representative will follow up.",
          _OVERDRAFT_FACT, "overdraft_fee", "grounded"),
    Probe("od_distortion",
          "Per overdraft policy, the $40 fee may be reversed once per year.",
          _OVERDRAFT_FACT, "overdraft_fee", "distortion"),
    Probe("od_fabrication",
          "Good news -- we will refund your $500 overdraft fee in full today.",
          _OVERDRAFT_FACT, "overdraft_fee", "fabrication"),
    Probe("od_wordform",
          "Under overdraft policy a thirty-five dollar fee applies, reversible once a year.",
          _OVERDRAFT_FACT, "overdraft_fee", "grounded"),
    Probe("rev_grounded",
          "Per fee reversal policy, a representative may approve up to $35; above that needs manager approval.",
          _REVERSAL_FACT, "fee_reversal_request", "grounded"),
    Probe("rev_distortion",
          "Per fee reversal policy, a representative may approve a reversal up to $100.",
          _REVERSAL_FACT, "fee_reversal_request", "distortion"),
]


def gms_tier(harness, draft: str, issue: str) -> tuple[str, float | None]:
    triple, geo, tier = harness.judge_draft(draft, issue)
    return tier, geo


def llm_tier(backend, draft: str, policy_fact: str) -> str:
    user = f"Authoritative policy: {policy_fact}\nDrafted reply to check:\n{draft}"
    obj = backend.chat_json(_GROUNDING_JUDGE_SYS, user, max_new_tokens=120) or {}
    tier = str(obj.get("tier", "")).strip().lower()
    return tier if tier in ("grounded", "distortion", "fabrication") else "fabrication"


def main() -> int:
    from agentlab.testing.capstone_harness import CapstoneTestHarness
    from benchmarks.prompted_baseline.chat_backends import AnthropicChatBackend

    backend = AnthropicChatBackend.from_env()
    if backend is None:
        print("ANTHROPIC_API_KEY required for the LLM judge."); return 1
    harness = CapstoneTestHarness()
    harness.calibrate_groundedness()

    print(f"Judge face-off on {len(PROBES)} fee-claim probes "
          f"(LLM = {backend.name}; same task, same authoritative fact)\n")
    def is_grounded(tier: str) -> bool:
        return tier == "grounded"  # distortion / fabrication / n/a all = not grounded

    hdr = f"{'probe':16s} {'expected':>12s} {'GMS':>16s} {'LLM':>12s}  {'match':>8s}"
    print(hdr); print("-" * 72)
    gms_tier3 = llm_tier3 = agree3 = 0      # exact 3-tier
    gms_bin = llm_bin = agree_bin = 0       # grounded vs not-grounded
    for p in PROBES:
        g_tier, g_geo = gms_tier(harness, p.draft, p.gms_issue)
        l_tier = llm_tier(backend, p.draft, p.policy_fact)
        exp_g = is_grounded(p.expected)
        gms_tier3 += int(g_tier == p.expected)
        llm_tier3 += int(l_tier == p.expected)
        agree3 += int(g_tier == l_tier)
        gms_bin += int(is_grounded(g_tier) == exp_g)
        llm_bin += int(is_grounded(l_tier) == exp_g)
        agree_bin += int(is_grounded(g_tier) == is_grounded(l_tier))
        g_show = f"{g_tier}" + (f"({g_geo:.2f})" if g_geo is not None else "")
        match = "agree" if is_grounded(g_tier) == is_grounded(l_tier) else "DIFFER"
        print(f"{p.pid:16s} {p.expected:>12s} {g_show:>16s} {l_tier:>12s}  {match:>8s}")

    n = len(PROBES)
    print("-" * 72)
    print("  grounded vs not-grounded (the decision that matters):")
    print(f"     vs truth:  GMS {gms_bin}/{n}   LLM {llm_bin}/{n}    agreement {agree_bin}/{n}")
    print("  exact 3-tier (grounded/distortion/fabrication; band-sensitive):")
    print(f"     vs truth:  GMS {gms_tier3}/{n}   LLM {llm_tier3}/{n}    agreement {agree3}/{n}")
    print("\n  GMS tier shows the geodesic in parentheses; 'n/a' = no $-figure the")
    print("  regex could extract (e.g. a number written in words) -> GMS abstains,")
    print("  a structural limit; the LLM reads the word-form. Conversely the LLM is")
    print("  non-deterministic and needs the fact in context, while GMS recalls it")
    print("  from the trained store with no per-call reference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
