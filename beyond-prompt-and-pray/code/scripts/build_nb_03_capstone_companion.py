#!/usr/bin/env python
"""Builder for notebooks/03_capstone_companion.ipynb.

Capstone companion to Chapter 3 (Reasoning Traces Are Not Evidence): the concept
read on the running banking complaint agent. Teaching notebook in the style of
the main chapter notebooks -- real capstone imports, no pre-embedded outputs
(the reader runs it). Store/GPU-backed verification is shown as reader-runnable
code and is not executed during validation.
"""
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = Path(__file__).resolve().parents[2] / "notebooks" / "03_capstone_companion.ipynb"

cells = [
    new_markdown_cell(
        "# Capstone companion --- Chapter 3: Reasoning Traces Are Not Evidence\n"
        "\n"
        "Chapter~3 establishes a distinction that governs the rest of the book: a "
        "model's stated reasoning is an artifact to be grounded, never trusted on its "
        "own credit. A reasoning trace records what the model *said* it did; it does not "
        "certify that what it said is correct. Correctness is established only by "
        "checking a claim against the substrate --- the authoritative store of facts --- "
        "not against the fluency or plausibility of the explanation that accompanies it.\n"
        "\n"
        "This companion reads that principle on the capstone banking complaint agent. "
        "The final tool in its pipeline, `draft_response`, is the one place where a "
        "generative model writes prose rather than transcribing a lookup. The draft "
        "carries a paraphrase of policy and a stated fee amount. The paraphrase is the "
        "reasoning; the fee amount is a factual claim. The companion shows how the "
        "capstone separates the two and grounds the claim against the store."
    ),
    new_markdown_cell(
        "## The draft is prose plus a checkable claim\n"
        "\n"
        "The drafting tool returns a `DraftOutput`. Its `text` field is what the model "
        "wrote: an explanation of the disposition, phrased in the register of a customer "
        "reply. Reading the output schema shows that the tool's contract does not, on its "
        "own, distinguish the persuasive prose from the facts embedded in it --- that "
        "separation is the verifier's work, performed after generation."
    ),
    new_code_cell(
        "import json\n"
        "from agentlab.capstone.banking_tools import DraftInput, DraftOutput\n"
        "\n"
        "print('DraftInput  :', json.dumps(DraftInput.model_json_schema()['properties'], indent=2))\n"
        "print('DraftOutput :', json.dumps(DraftOutput.model_json_schema()['properties'], indent=2))"
    ),
    new_markdown_cell(
        "The output carries `text` alongside `corrected`, `corrections`, "
        "`requires_escalation` and `reason`. Those four fields are not written by the "
        "generative model. They are set by the verifier, and they record the outcome of "
        "grounding the draft's claims against the substrate. A draft that reads perfectly "
        "and one that has been silently corrected are distinguished only here, because "
        "the prose alone cannot report on its own accuracy."
    ),
    new_markdown_cell(
        "## The verifier grounds the claim, not the reasoning\n"
        "\n"
        "`DraftVerifier` implements the Chapter~3 principle directly. It does not read the "
        "draft's explanation and judge whether the explanation is convincing. It parses "
        "the factual claim the draft asserts --- a dollar amount --- looks up the "
        "authoritative value in the store's Exact Numerical Memory, and compares. The "
        "reasoning text is never the object of the check; the substrate is."
    ),
    new_code_cell(
        "import inspect\n"
        "from agentlab.capstone import draft_verifier\n"
        "from agentlab.capstone.draft_verifier import DraftVerifier\n"
        "\n"
        "# The class docstring states the two failure modes the verifier grounds against.\n"
        "print(inspect.getdoc(DraftVerifier))"
    ),
    new_markdown_cell(
        "The verifier's two responsibilities correspond to the two ways a draft can be "
        "wrong while remaining fluent. A drifted digit in a fee amount is a numeric error "
        "the model cannot rule out on its own; it is caught by an Exact Numerical Memory "
        "lookup and deterministically corrected. An unconditional promise to waive a fee "
        "is a policy contradiction rather than a typo; it is escalated to a human, never "
        "silently rewritten. Both dispositions are decided against the store, not against "
        "the draft's account of itself."
    ),
    new_code_cell(
        "# The static routing table: which policy's stated fee is checkable, and where\n"
        "# the authoritative figure lives in the store. This is inspected, not executed --\n"
        "# reading it shows what the verifier treats as a groundable claim.\n"
        "print('policy -> ENM (register, key):')\n"
        "for pid, enm in draft_verifier._POLICY_FEE_ENM.items():\n"
        "    print(f'  {pid:14s} -> {enm}')\n"
        "print()\n"
        "print('dollar-amount pattern:', draft_verifier._DOLLAR.pattern)\n"
        "print()\n"
        "print(inspect.getsource(DraftVerifier.verify))"
    ),
    new_markdown_cell(
        "Reading `verify` makes the separation explicit. The method parses the dollar "
        "amounts the draft states with a regular expression, and only when a single "
        "amount is present does it compare that amount to the value returned by "
        "`store.lookup_enm`. A mismatch produces a `corrections` entry recording the "
        "draft value and the authoritative value side by side. The escalation branch "
        "calls the semantic intent guard on the same text; a `fee_waiver` intent sets "
        "`escalate`. Nowhere does the method decide anything from the persuasiveness of "
        "the prose."
    ),
    new_markdown_cell(
        "## Running the verifier against the store\n"
        "\n"
        "The check above is cheap because it reads structure. The grounding itself "
        "requires the deployed GMS store and, on this hardware, a GPU. The following "
        "cell is reader-runnable: it loads the store, drafts one claim with a drifted "
        "fee amount, and shows the verifier substituting the authoritative value. It is "
        "not executed here, because it loads the substrate."
    ),
    new_code_cell(
        "# Reader-runnable. Requires the built GMS banking store (data/gms_banking_store)\n"
        "# and loads it onto the GPU when available.\n"
        "#\n"
        "#   from agentlab.capstone.draft_verifier import get_default_verifier\n"
        "#\n"
        "#   verifier = get_default_verifier()\n"
        "#\n"
        "#   # A draft that paraphrases the overdraft policy correctly but drifts the fee\n"
        "#   # figure from the authoritative amount. The prose is fluent; the claim is wrong.\n"
        "#   draft = (\n"
        "#       'We reviewed the $30 overdraft fee applied to your account. Under the '\n"
        "#       'overdraft policy this fee is charged per occurrence.'\n"
        "#   )\n"
        "#   result = verifier.verify(draft, policy_id='overdraft')\n"
        "#\n"
        "#   print('corrected  :', result['corrected'])\n"
        "#   print('corrections:', result['corrections'])   # draft_value vs authoritative\n"
        "#   print('text       :', result['text'])          # the grounded, substituted draft\n"
        "#   print('escalate   :', result['escalate'])\n"
        "#\n"
        "# The verdict does not depend on how convincing the draft reads. The digit is\n"
        "# corrected because the store, not the sentence, holds the authoritative value."
    ),
    new_markdown_cell(
        "## The step record: what was proposed, not what was thought\n"
        "\n"
        "The same distinction shapes what the harness seals for audit. A reasoning trace "
        "is not evidence, so it is not what the audit chain commits to. What is sealed is "
        "the proposed action --- a typed record of *what the agent moved to do* --- and a "
        "hash of the state before the step. `replay.py` reconstructs a run from that "
        "trail and checks it against the primitives, so a claim about the run is verified "
        "against the sealed record rather than trusted from a narrative of it."
    ),
    new_code_cell(
        "from agentlab.audit.event import AuditEvent, SealedEvent\n"
        "\n"
        "# The audit event is the sealed step record. Its fields are the checkable facts\n"
        "# of a step; note there is no free-text 'reasoning' field to trust.\n"
        "print('AuditEvent fields :', list(AuditEvent.__dataclass_fields__))\n"
        "print('SealedEvent fields:', list(SealedEvent.__dataclass_fields__))"
    ),
    new_markdown_cell(
        "`AuditEvent` records the `run_id`, `step`, `state_hash`, the `proposed_action` "
        "as a JSON dict and the `observation`. There is no field for the model's stated "
        "rationale, because a rationale is not what gets verified. The proposed action is "
        "a typed `Action` --- a `ToolCall`, `AskUser`, `Finish` or `Escalate` --- "
        "discriminated on its `kind`, and it is this typed object that the replay path "
        "reconstructs and re-checks."
    ),
    new_code_cell(
        "import inspect\n"
        "from agentlab.capstone.replay import ReplayStep, ReplayReport, replay_run\n"
        "\n"
        "# The replay step's booleans are the two grounding checks: the action rebuilt\n"
        "# from sealed JSON reproduces the typed action, and the reconstructed pre-step\n"
        "# state re-hashes to the sealed hash.\n"
        "print('ReplayStep fields:', list(ReplayStep.__dataclass_fields__))\n"
        "print()\n"
        "print(inspect.getsource(replay_run))"
    ),
    new_markdown_cell(
        "Each `ReplayStep` reports `action_roundtrips` and `state_hash_matches`. The "
        "first confirms that `parse_action` applied to the sealed JSON reproduces the "
        "typed action recorded in the trajectory; the second confirms that "
        "`AgentState.from_dict` reconstructs the pre-step state and that re-hashing it "
        "reproduces the hash the chain sealed. A run's integrity is thus established by "
        "reconstructing and re-checking the sealed facts, not by reading and believing a "
        "trace of what the agent claimed to have done."
    ),
    new_markdown_cell(
        "## Connecting back to Chapter 3, and forward to Chapter 15\n"
        "\n"
        "Chapter~3 draws the line between a reasoning trace and evidence: the trace "
        "explains, the substrate certifies. The capstone realizes that line in two "
        "places. `DraftVerifier` grounds the one claim a generative model writes --- a "
        "fee amount --- against the store's Exact Numerical Memory, correcting a drifted "
        "digit deterministically and escalating an unauthorized promise, and it does so "
        "without ever judging the draft's prose. `replay.py` seals the proposed action "
        "and the pre-step state hash, not a narrative, so a run is verified by "
        "reconstruction rather than trusted from its own account.\n"
        "\n"
        "Chapter~15 assembles the five tools into the governed workflow, where the "
        "drafting tool's output is verified before it becomes the agent's externally "
        "visible commitment and every step is sealed into the audit chain that "
        "`replay.py` reconstructs. The principle read here in isolation is the discipline "
        "the full capstone applies at every step: ground the claim, do not trust the "
        "trace."
    ),
]

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    nbformat.write(nb, f)
print("wrote", OUT, "cells", len(nb.cells))
