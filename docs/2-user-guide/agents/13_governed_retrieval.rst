Chapter 13 — Governed Retrieval
===============================

A single search tool that returns the fee schedule can also return a second
customer's account, an employee file or the rationale behind a suspicious-activity
report, and nothing in the query text separates the authorized retrieval from the
one that should be refused. This page shows how to place a store behind a
retrieval layer that binds the query geometrically, admits only the facts a
workflow is entitled to and scans the synthesized answer for a reversed policy
stance. The governed primitives live in
:mod:`knowlytix.knowledge.rag.governed`; the retriever they wrap is
:class:`~forgeloop.agents.capstone.policy_rag.PolicyRagRetriever`.

The pieces
----------

Governed retrieval composes a plain triple-mediated retriever with three
policy objects and one trained gate.

.. list-table::
   :header-rows: 1

   * - Object
     - Role
   * - :class:`~forgeloop.agents.capstone.policy_rag.PolicyRagRetriever`
     - Binds the query, retrieves grounded facts through the GMS operators.
   * - :class:`~knowlytix.knowledge.rag.governed.SensitivityMap`
     - Assigns each entity a zone and level; fails closed on the unclassified.
   * - :class:`~knowlytix.knowledge.rag.governed.RetrievalContract`
     - Per-workflow allowlist of zones and relations, field redactors and a sensitivity ceiling.
   * - :class:`~knowlytix.knowledge.rag.governed.ClassifierDisclosureGuard`
     - Scans the answer prose for a stance that contradicts an admitted fact.
   * - :class:`~knowlytix.knowledge.rag.governed.GovernedRetriever`
     - The driver: bind, admit, redact, synthesize, scan, audit.

Load the store, the sensitivity map and the contract
-----------------------------------------------------

:class:`~knowlytix.knowledge.rag.governed.SensitivityMap` and
:class:`~knowlytix.knowledge.rag.governed.RetrievalContract` are loaded from JSON
that ships beside the store, so access is data, not code. The retriever is the
same GEODE pipeline the capstone's ``search_policy`` tool uses.

.. code-block:: python

   import torch
   from knowlytix.knowledge.rag.governed import (
       GovernedRetriever, RetrievalContract, SensitivityMap,
       ClassifierDisclosureGuard, ProtectedProbe)
   from forgeloop.agents.capstone.policy_rag import PolicyRagRetriever
   from forgeloop.agents.governance.polarity_classifier import (
       LoraPolarityClassifier, RELATION_PHRASE)

   STORE = "data/gms_governed_store"
   retr = PolicyRagRetriever(store_path=STORE)
   smap = SensitivityMap.load(f"{STORE}/sensitivity_map.json")
   contract = RetrievalContract.load("data/governed_contracts/complaint_policy_mapping.json")

The contract states what a purpose may see. ``allowed_zones`` and
``redact_relations`` express purpose-based access, and ``sensitivity_of`` returns
a ceiling for an entity the map has never classified, so an unlisted entity is
treated as maximally sensitive rather than open.

.. code-block:: python

   print("allowed zones :", sorted(contract.allowed_zones))
   print("redactors     :", contract.redact_relations)
   print("max sensitivity:", contract.max_sensitivity)
   print("unclassified   :", smap.sensitivity_of("mystery_entity"))  # fail-closed

The disclosure gate: geometry against a trained classifier
----------------------------------------------------------

Most of the pipeline is geometric, so an injection in a retrieved document
cannot move what binds. Reading a synthesized answer for a reversed policy
stance is the exception the geometry does not serve well. On short value tokens
the geometric value-polarity check is accurate, but on the full prose an agent
produces its u-space tension is out of distribution and it degrades (held-out
DoE: contradiction F1 0.62). A Qwen3-4B ``SEQ_CLS`` head fine-tuned with LoRA
reads the prose directly and is stronger (contradiction F1 0.83, recall 1.0).
The classifier's input pairs the admitted policy stance with the claim, using the
``RELATION_PHRASE`` reference table, so the judgment has both sides.

.. code-block:: python

   clf = LoraPolarityClassifier.load()      # data/polarity_classifier_qwen_nl
   probes = [ProtectedProbe("financial_distress", "customer_alice",
                            "has_days_balance_negative",
                            "the customer is in financial distress")]
   guard = ClassifierDisclosureGuard(clf.classify, RELATION_PHRASE, tau=0.0, probes=probes)

   admitted = [("pii_handling", "has_unencrypted_channel_pii", "forbidden")]
   guard.scan("unencrypted PII transmission is permitted", admitted)  # flagged: reversed
   guard.scan("unencrypted PII transmission is forbidden", admitted)  # clean: consistent

Retrieve under the contract
---------------------------

:class:`~knowlytix.knowledge.rag.governed.GovernedRetriever` takes the retriever's
pipeline, the contract and the map, plus the disclosure guard and a synthesizer
that reads only the admitted, redacted facts. Synthesis therefore cannot restate
a field it was never given, and a denial by policy is returned as a decision
distinct from an empty search result.

.. code-block:: python

   def synth(query, facts):
       if not facts:
           return ""
       bullets = "\n".join(f"- {f.head} {f.relation[4:]}: {f.tail}" for f in facts)
       sys = ("Answer using ONLY these facts; if a detail is absent say it is not "
              "available for this workflow; reply CANNOT_ANSWER if none is relevant.")
       out = (retr.llm.call(system=sys, user=f"Facts:\n{bullets}\n\nQ: {query}\nA:",
                            max_tokens=80) or "").strip()
       return "" if "CANNOT_ANSWER" in out else out

   gov = GovernedRetriever(retr.pipe, contract, smap,
                           disclosure_guard=guard, synthesizer=synth)
   r = gov.retrieve("what is the overdraft fee and the dispute filing window?")
   print(r.decision, "::", r.answer)

Purpose, not role
-----------------

The same query resolves differently under two contracts, because access follows
the workflow rather than the identity of the caller. A complaint workflow is
denied the rationale class of a suspicious-activity report; an AML review holding
the ``aml_authorized`` flag is admitted to it. ``granted_flags`` carries the
purpose-scoped authorization the contract requires.

.. code-block:: python

   aml = RetrievalContract.load("data/governed_contracts/aml_review.json")
   gov_aml = GovernedRetriever(retr.pipe, aml, smap,
                               granted_flags=frozenset({"aml_authorized"}),
                               disclosure_guard=guard, synthesizer=synth)
   q = "what is the rationale class of the SAR filing sr21"
   print("complaint workflow:", gov.retrieve(q).decision)   # denied
   print("aml review        :", gov_aml.retrieve(q).decision)

What the layer enforces
-----------------------

Retrieval returns typed facts rather than document spans, so field-level masking
names the relation whose tail is sensitive instead of locating a substring, and
least-context drops a fact whose only role would be to enable an unwanted
inference. Binding is geometric, so injected text in a retrieved document cannot
change which entity a query resolves to, and an injection in the query is handled
by starvation: it binds nothing and the pipeline abstains. Every retrieval emits
a provenance-anchored audit record, so a denial and its cause are reconstructable
after the fact.

See also
--------

- :doc:`/3-api-reference/modules/knowlytix/rag/governed` — the governed
  primitives and their signatures.
- :doc:`/3-api-reference/modules/capstone_internals/policy_rag` — the
  :class:`~forgeloop.agents.capstone.policy_rag.PolicyRagRetriever` this layer wraps.
- :doc:`/3-api-reference/modules/agents_backends/polarity_classifier` — the LoRA
  disclosure classifier.
- :doc:`12_runtime_governance` — the writing-channel counterpart of this gate.
- :doc:`/4-notebook-examples/agents/index` — the governed retriever run end to end.
