"""Build the Chapter 13 companion notebook (Governed Retrieval).

Teaching-style: runnable code cells against the real knowlytix governed-retrieval
API + the real gms_governed_store, with the actual results captured from the
spark-ef84 run embedded as markdown (data/governed_scenarios.json,
data/polarity_gate_comparison.json). Mirrors the chapter's structure.

    python scripts/build_nb_13_governed_retrieval.py
"""
from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "notebooks" / "13_governed_retrieval.ipynb"


def md(t):
    return nbf.v4.new_markdown_cell(t)


def code(t):
    return nbf.v4.new_code_cell(t)


def main() -> int:
    scen = json.loads((_ROOT / "data" / "governed_scenarios.json").read_text())
    cmp = json.loads((_ROOT / "data" / "polarity_gate_comparison.json").read_text())
    cells = []

    cells.append(md(
        "# Chapter 13 --- Governed Retrieval\n\n"
        "Retrieval is the highest-risk read channel an agent has. This notebook builds a "
        "**governed retriever** over the real `gms_governed_store`: the agent never searches "
        "the store directly, it submits an intent-scoped request to a layer that enforces a "
        "per-workflow contract before, during and after retrieval. Access control stays "
        "geometric and calibrated; the output-disclosure scanner is a Qwen classifier trained "
        "on GMS+DoE-generated data.\n\n"
        "The results shown as output blocks are the actual results captured on spark-ef84 "
        "(Qwen3-4B-Instruct-2507 on the GB10); the code cells are runnable against the store."))

    cells.append(md("## 1. Build the governed retriever\n\n"
        "`GovernedRetriever` composes an existing `RagPipeline`, a `RetrievalContract` and a "
        "`SensitivityMap`. The store carries no sensitivity field, so the map supplies the "
        "zone labels (fail-closed) and the contract states the per-workflow envelope."))
    cells.append(code(
        "import torch\n"
        "from knowlytix.knowledge.rag.governed import (\n"
        "    GovernedRetriever, RetrievalContract, SensitivityMap,\n"
        "    ClassifierDisclosureGuard, ProtectedProbe)\n"
        "from agentlab.capstone.policy_rag import PolicyRagRetriever\n"
        "from agentlab.governance.polarity_classifier import (\n"
        "    LoraPolarityClassifier, RELATION_PHRASE)\n\n"
        "STORE = 'data/gms_governed_store'\n"
        "retr = PolicyRagRetriever(store_path=STORE)          # real pipeline over the store\n"
        "smap = SensitivityMap.load(f'{STORE}/sensitivity_map.json')\n"
        "contract = RetrievalContract.load('data/governed_contracts/complaint_policy_mapping.json')\n"))

    cells.append(md("The contract is an allowlist of zones, a least-context allowlist of the "
        "relations the purpose needs, a table of field redactors and a sensitivity ceiling. "
        "Purpose-based, not role-based: the complaint-mapping purpose needs only the last four "
        "digits of the account number, and none of the SSN, date of birth, address or balance."))
    cells.append(code(
        "print('allowed zones :', sorted(contract.allowed_zones))\n"
        "print('redactors     :', contract.redact_relations)\n"
        "print('max sensitivity:', contract.max_sensitivity)\n"
        "print('zone of customer_bob :', smap.zone_of('customer_bob'),\n"
        "      smap.sensitivity_of('customer_bob'))\n"
        "print('unclassified (fail-closed):', smap.sensitivity_of('mystery_entity'))"))

    cells.append(md("## 2. The disclosure gate: geometry vs a trained classifier\n\n"
        "The one control geometry does not serve well is the output-disclosure scan over prose. "
        "We generate a held-out test set whose **labels come from GMS** (the store's stance facts) "
        "and whose **surface variation comes from DoE** (a `DesignMatrix` over presentation "
        "factors, realized by Qwen), then score the geometric `ValuePolarityChecker` against a "
        "Qwen LoRA classifier on the same split.\n\n"
        "Generation and training (run on spark-ef84):\n"
        "```\n"
        "python scripts/build_polarity_doe_dataset.py --n 480\n"
        "python scripts/train_polarity_classifier_lora.py --input nl\n"
        "python scripts/compare_polarity_gates.py\n"
        "```"))

    g = cmp["gates"]
    def row(name, label):
        d = g[name]; c = d["contradiction"]
        return (f"| {label} | {d['accuracy']:.3f} | {c['precision']:.3f} | "
                f"{c['recall']:.3f} | {c['f1']:.3f} |")
    table = (
        "**Head-to-head on the held-out DoE test set "
        f"({cmp['n_test']} rows, {cmp['label_dist']}):**\n\n"
        "| gate | 3-class acc | contradiction P | contradiction R | contradiction F1 |\n"
        "|---|---|---|---|---|\n"
        + row("gate_a_geometry_token", "geometry, value token") + "\n"
        + row("gate_a_geometry_prose", "geometry, prose sentence") + "\n"
        + row("gate_b_qwen_tuple", "**Qwen classifier, triple pair**") + "\n"
        + row("gate_b_qwen_nl", "**Qwen classifier, prose pair**") + "\n\n"
        "The classifier catches every contradiction (recall 1.0) at F1 0.83, against the "
        "geometric gate's 0.62 on the token; geometry collapses on prose "
        f"({g['gate_a_geometry_prose']['accuracy']:.2f} 3-class) while the classifier holds "
        "~0.70 across the phrasing spectrum. Access control stays geometric; the disclosure "
        "scanner is the classifier.")
    cells.append(md(table))

    cells.append(code(
        "clf = LoraPolarityClassifier.load()      # data/polarity_classifier_qwen_nl\n"
        "probes = [ProtectedProbe('financial_distress', 'customer_alice',\n"
        "                         'has_days_balance_negative',\n"
        "                         'the customer is in financial distress')]\n"
        "guard = ClassifierDisclosureGuard(clf.classify, RELATION_PHRASE, tau=0.0, probes=probes)\n\n"
        "admitted = [('pii_handling', 'has_unencrypted_channel_pii', 'forbidden')]\n"
        "print('contradicting:', guard.scan('unencrypted PII transmission is permitted', admitted))\n"
        "print('consistent   :', guard.scan('unencrypted PII transmission is forbidden', admitted))"))
    dg = scen["disclosure_gate"]
    cells.append(md(
        "Captured output --- the contradicting answer is flagged, the consistent one is clean:\n\n"
        "```\n"
        f"contradicting: {dg['contradicting_answer']['findings']}\n"
        f"consistent   : {dg['consistent_answer']['findings']}\n"
        "```"))

    cells.append(md("## 3. The governed retriever in action\n\n"
        "A grounded synthesizer answers from the admitted, redacted facts alone --- the model "
        "never sees an ungoverned fact --- so least-context and redaction are real, not cosmetic."))
    cells.append(code(
        "def synth(query, facts):\n"
        "    if not facts: return ''\n"
        "    bullets = '\\n'.join(f'- {f.head} {f.relation[4:]}: {f.tail}' for f in facts)\n"
        "    sys = ('Answer using ONLY these facts; if a detail is absent say it is not '\n"
        "           'available for this workflow; reply CANNOT_ANSWER if none is relevant.')\n"
        "    out = (retr.llm.call(system=sys, user=f'Facts:\\n{bullets}\\n\\nQ: {query}\\nA:',\n"
        "                         max_tokens=80) or '').strip()\n"
        "    return '' if 'CANNOT_ANSWER' in out else out\n\n"
        "gov = GovernedRetriever(retr.pipe, contract, smap,\n"
        "                        disclosure_guard=guard, synthesizer=synth)\n"
        "r = gov.retrieve('what is the overdraft fee and the dispute filing window?')\n"
        "print(r.decision, '::', r.answer)"))

    def block(tag):
        s = scen[tag]
        lines = [f"[{tag}]  {s['query']}",
                 f"  decision = {s['decision']}   ({s['reason']})"]
        if s.get("answer"):
            lines.append(f"  answer   = {s['answer'][:150]}")
        if s.get("denied"):
            d0 = s["denied"][0]
            lines.append(f"  withheld = {len(s['denied'])} fact(s), e.g. "
                         f"{d0['r']} ({d0['zone']})")
        return "\n".join(lines)

    cells.append(md(
        "### Captured decisions across the scenario suite\n\n"
        "One allowed path, cross-customer and blocked-source denials, least-context + "
        "redaction, prompt injection in the query channel, purpose-based access, and a search "
        "failure distinguished from a policy denial:\n\n"
        "```\n"
        + "\n\n".join(block(t) for t in [
            "allowed_policy_mapping", "cross_customer_case", "blocked_hr",
            "least_context_pii", "prompt_injection", "aml_denied_to_complaint",
            "aml_allowed_to_review", "search_failure"])
        + "\n```"))

    cells.append(md("## 4. Purpose-based access and the audit record\n\n"
        "The same SAR query is denied to the complaint workflow and admitted to an AML-review "
        "workflow that holds the `aml_authorized` flag. Every retrieval emits an immutable, "
        "provenance-anchored audit record; admitted values are already redacted, so it is safe "
        "to retain."))
    cells.append(code(
        "aml = RetrievalContract.load('data/governed_contracts/aml_review.json')\n"
        "gov_aml = GovernedRetriever(retr.pipe, aml, smap,\n"
        "                            granted_flags=frozenset({'aml_authorized'}),\n"
        "                            disclosure_guard=guard, synthesizer=synth)\n"
        "q = 'what is the rationale class of the SAR filing sr21'\n"
        "print('complaint workflow:', gov.retrieve(q).decision)\n"
        "print('aml review        :', gov_aml.retrieve(q).decision, '::',\n"
        "      gov_aml.retrieve(q).answer)"))

    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {"language_info": {"name": "python"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}}
    _OUT.write_text(nbf.writes(nb))
    print(f"wrote {_OUT} ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
