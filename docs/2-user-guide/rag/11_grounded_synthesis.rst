Chapter 11 — Grounded Synthesis
===============================

Retrieval already decided the answer values through the graph: bound query
triples resolved to asserted facts, each carrying its source span
(:doc:`10_answering_through_the_gms`). What remains is to turn those facts into
prose without letting the model reach into parametric memory. This page shows
how :class:`~knowlytix.knowledge.rag.assemble.Assembler` formats the retrieved
facts into an evidence block and calls the synthesis model under a prompt that
permits answering only from that block, and how the synthesizer refuses when the
evidence is empty.

Format the evidence block
--------------------------

The synthesizer sees the retrieved facts as
:class:`~knowlytix.knowledge.rag.retrieve.RetrievedFact` values, each with its
verbatim source excerpt and location. ``Assembler._context`` renders the
``<evidence>`` block that is handed to the model.

.. code-block:: python

   from knowlytix.knowledge.rag.assemble import Assembler
   from knowlytix.knowledge.rag.retrieve import RetrievedFact

   facts = [
       RetrievedFact(
           head="cloud platform", relation="has_revenue", tail="120.0",
           score=0.0, confidence=1.0, source="triple",
           location=":15:553-558", raw="Cloud Platform | Technology | 120.0 | 340",
       ),
   ]
   print(Assembler._context("What was Cloud Platform revenue?", facts))

There is no document body, no neighboring paragraph and no top-k chunk in the
block; the only figure present is the one retrieval verified.

Synthesize from the facts
-------------------------

:class:`~knowlytix.knowledge.rag.assemble.Assembler` takes any
:class:`~knowlytix.knowledge.llm_backend.LLMBackend`. In deployment the backend
is a local model. :class:`~knowlytix.knowledge.llm_backend.LocalTransformersBackend`
loads the tutorial's synthesis model, :data:`~knowlytix.knowledge.geode.QWEN_3B`
(``Qwen/Qwen3-4B-Instruct-2507``), and needs no API key.

.. code-block:: python

   from knowlytix.knowledge.llm_backend import LocalTransformersBackend
   from knowlytix.knowledge.geode import QWEN_3B

   qwen = LocalTransformersBackend(QWEN_3B)
   answer = Assembler(qwen).assemble("What was Cloud Platform revenue?", facts)
   print(answer)   # prose stating revenue of 120.0, no other figure

Because the ``120.0`` is the only number in the evidence block, the model cannot
drift onto a plausible-sounding figure: no other figure is present to drift
onto.

Refuse when the evidence is empty
---------------------------------

The other half of grounded synthesis is refusal. When retrieval returns no
facts, as it does for a prose section that carries no triples, the evidence
block is empty and the synthesizer declines rather than forecasting.

.. code-block:: python

   no_facts: list[RetrievedFact] = []
   refusal = Assembler(qwen).assemble(
       "What does the Outlook section forecast for FY2026?", no_facts)
   print(refusal)   # a decline, not an invented forecast

Keep the LLM roles separate
---------------------------

:class:`~knowlytix.knowledge.rag.config.RagConfig` holds three LLM roles: ``llm``
for synthesis, ``llm_extract`` for turning a question into query triples, and
``llm_verify`` for decomposing an answer into claim triples. Unset roles fall
back through ``extract_llm()`` and ``verify_llm()``, so swapping the synthesizer
is a single-field change.

.. code-block:: python

   from knowlytix.knowledge.rag.config import RagConfig

   cfg = RagConfig(llm=synth_backend, llm_extract=extract_backend)
   assert cfg.llm.model_name == synth_backend.model_name       # synthesis: swapped
   assert cfg.extract_llm() is extract_backend                 # extract: set
   assert cfg.verify_llm() is extract_backend                  # verify -> extract

Here ``synth_backend`` and ``extract_backend`` are any two
:class:`~knowlytix.knowledge.llm_backend.LLMBackend` instances; the verify role
was left unset and falls back to the extract role.

Mechanism
---------

Grounded synthesis constrains where the answer comes from: the values were
decided by the geometry, and synthesis is confined to phrasing them. It does not
prove that the prose faithfully restates the facts, so a wayward model could
still mis-phrase a value, which is why answer self-verification is a separate
stage (:doc:`12_self_verification`). Synthesis also inherits the coverage of
retrieval, so a blind-spot question yields a refusal
(:doc:`13_abstention_and_coverage`).

See also
--------

- :doc:`/3-api-reference/modules/knowlytix/rag/index` — signatures for
  :class:`~knowlytix.knowledge.rag.assemble.Assembler`,
  :class:`~knowlytix.knowledge.rag.config.RagConfig` and
  :class:`~knowlytix.knowledge.rag.retrieve.RetrievedFact`.
- :doc:`/3-api-reference/modules/knowlytix/llm/index` — the
  :class:`~knowlytix.knowledge.llm_backend.LLMBackend` interface and
  :class:`~knowlytix.knowledge.llm_backend.LocalTransformersBackend`.
- :doc:`10_answering_through_the_gms` — where the retrieved facts come from.
- :doc:`/4-notebook-examples/rag/index` — synthesis run end to end.
