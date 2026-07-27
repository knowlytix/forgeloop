Pluggable LLMs and the Distrusted Dense Fallback
================================================

A deployment has three LLM-shaped jobs: extract the query triples from the
question, synthesize the grounded answer from retrieved facts, and optionally
verify the answer's own claims against the store. This page shows how
:class:`~knowlytix.knowledge.rag.config.RagConfig` assigns a backend to each
role, and how the opt-in dense vector fallback is wired so that an answer it
produces is quarantined rather than trusted.

The three roles
---------------

The config carries three backend slots, each an
:class:`~knowlytix.knowledge.llm_backend.LLMBackend`. Only the synthesis slot is
required; the other two resolve through a fallback chain.

.. list-table::
   :header-rows: 1

   * - Role
     - Slot
     - Resolution when unset
   * - synthesize the answer
     - ``llm``
     - required
   * - extract query triples
     - ``llm_extract``
     - falls back to ``llm``
   * - verify the answer
     - ``llm_verify``
     - falls back to ``llm_extract()``

A single-model deployment therefore needs one argument, and every role can be
pinned to one local model.

Assign a backend per role
--------------------------

The library accepts any :class:`~knowlytix.knowledge.llm_backend.LLMBackend`
subclass, so a scripted backend keeps the contract testable without a GPU. The
resolution helpers ``extract_llm()`` and ``verify_llm()`` implement the fallback
chain.

.. code-block:: python

   from knowlytix.knowledge.rag import RagConfig
   from knowlytix.knowledge.llm_backend import LLMBackend

   class ScriptedBackend(LLMBackend):
       def __init__(self, name: str, reply: str = ""):
           self._name = name
           self._reply = reply

       def call(self, system: str, user: str, max_tokens: int = 2048) -> str:
           return self._reply

       @property
       def model_name(self) -> str:
           return self._name

   cfg = RagConfig(
       llm=ScriptedBackend("synth"),
       llm_extract=ScriptedBackend("extract"),
       llm_verify=ScriptedBackend("verify"),
   )
   cfg.llm.model_name           # "synth"
   cfg.extract_llm().model_name # "extract"
   cfg.verify_llm().model_name  # "verify"

Dropping a slot lets it inherit from the one above it. One backend then serves
all three roles.

.. code-block:: python

   shared = ScriptedBackend("qwen-shared")
   cfg2 = RagConfig(llm=shared)
   assert cfg2.extract_llm() is shared   # llm_extract defaults to llm
   assert cfg2.verify_llm() is shared    # llm_verify defaults to extract_llm()

For a real deployment each role is a local model loaded through
:class:`~knowlytix.knowledge.llm_backend.LocalTransformersBackend`. The
following requires a GPU.

.. code-block:: python

   from knowlytix.knowledge.llm_backend import LocalTransformersBackend
   from knowlytix.knowledge.geode import QWEN_3B

   qwen = LocalTransformersBackend(QWEN_3B, device="cuda")
   qwen_cfg = RagConfig(llm=qwen)   # all three roles -> local Qwen
   assert qwen_cfg.extract_llm().model_name == QWEN_3B

Load the store and abstain on prose
-----------------------------------

The pipeline runs over a trained store loaded with
:func:`forgeloop.rag.load_store`. A question that binds to no triple has nothing
to retrieve, so the default triple-mediated route abstains rather than
fabricating an answer.

.. code-block:: python

   from forgeloop.rag import load_store
   from knowlytix.knowledge.rag import RagPipeline, RagConfig

   store = load_store()

   default_cfg = RagConfig(llm=ScriptedBackend("synth"))
   assert default_cfg.dense_fallback is False   # bank-grade default
   assert default_cfg.strict_mode is False

   ans = RagPipeline(store, default_cfg).query(
       "What does management say about customer concentration risk?")

   assert ans.decision == "abstain"
   assert ans.route == "triple"
   assert ans.verified is True   # an abstention is still verified-true

Enable the dense fallback
-------------------------

Setting ``dense_fallback=True`` indexes the document's paragraph spans into an
:class:`~knowlytix.knowledge.rag.dense.InMemoryVectorBackend`. When nothing
binds, the pipeline searches that index and synthesizes from the raw passages.
The resulting :class:`~knowlytix.knowledge.rag.pipeline.RagAnswer` is routed as
``dense_fallback``, flagged ``verified=False`` and stamped with a notice.

.. code-block:: python

   dense_cfg = RagConfig(
       llm=ScriptedBackend("synth", reply="Management states ..."),
       dense_fallback=True,   # opt in to the distrusted path
       strict_mode=False,     # return the flagged dense answer
   )
   assert dense_cfg.vector_backend is None   # defaults to InMemoryVectorBackend

   ans = RagPipeline(store, dense_cfg).query(
       "What does management say about customer concentration risk?")

   assert ans.route == "dense_fallback"
   assert ans.verified is False   # quarantined
   assert ans.notice == "Answer from dense fallback; NOT GMS-verified."
   assert ans.dense_sources       # the spans it used, with line-range provenance

Refuse the dense answer with strict mode
----------------------------------------

A deployment may treat an unverifiable answer as worse than none.
``strict_mode=True`` keeps the dense index wired but downgrades the dense route
back to an abstention, retaining the coverage signal without returning an
unverified figure.

.. code-block:: python

   strict_cfg = RagConfig(
       llm=ScriptedBackend("synth"),
       dense_fallback=True,
       strict_mode=True,   # dense route -> abstain
   )
   ans = RagPipeline(store, strict_cfg).query(
       "What does management say about customer concentration risk?")

   assert ans.decision == "abstain"
   assert ans.route == "triple"   # never reaches the dense route
   assert ans.verified is True

Point the fallback at a different backend
-----------------------------------------

The dense index is pluggable through the
:class:`~knowlytix.knowledge.rag.dense.VectorBackend` contract of two methods,
``index`` and ``search``. The default is
:class:`~knowlytix.knowledge.rag.dense.InMemoryVectorBackend` with an injectable
encoder, so the fallback is offline-testable.
:func:`~knowlytix.knowledge.rag.dense.build_spans` chunks markdown into
:class:`~knowlytix.knowledge.rag.dense.DenseSpan` objects, each carrying a
line-range provenance location.

.. code-block:: python

   from knowlytix.knowledge.rag import InMemoryVectorBackend, build_spans
   from knowlytix.knowledge.rag.dense import VectorBackend

   report_md = "## 6. Management Discussion and Analysis\n\nManagement ...\n"
   spans = build_spans(report_md, source_path="data/annual_report.md")
   assert spans[0].location.startswith("data/annual_report.md:")
   assert spans[0].line_start >= 1   # header-only blocks are skipped

   # Any VectorBackend subclass can replace the default -- e.g. an external DB.
   assert issubclass(InMemoryVectorBackend, VectorBackend)

Mechanism
---------

Per-role selection changes which model extracts, synthesizes and verifies; it
does not change what is verifiable. The dense fallback carries the same failure
modes as a chunk-and-pray baseline, and the ``verified=False`` flag labels that
limitation rather than removing it. A dense hit is a lead. The response to a
recurring dense hit is to extract the missing triples and close the coverage
blind spot, not to trust the fallback; ``strict_mode`` exists so a deployment
can forbid the unverified route outright.

See also
--------

- :doc:`/3-api-reference/modules/knowlytix/llm/index` — ``LLMBackend`` and
  ``LocalTransformersBackend``.
- :doc:`/3-api-reference/modules/knowlytix/rag/index` — the dense fallback, pipeline
  and config types.
- :doc:`13_abstention_and_coverage` — why the triple route abstains on prose.
- :doc:`/4-notebook-examples/rag/index` — the same configuration run end to end.
