Chapter 12 — Self-Verification: The GMS as a Hallucination Detector
===================================================================

Grounded synthesis constrains the model; it does not prove the model obeyed
(:doc:`11_grounded_synthesis`). A small local model can still drop a digit, swap
a segment or paste a half-remembered figure. This page shows how
:class:`~knowlytix.knowledge.rag.verify.AnswerVerifier` decomposes an answer
back into claim triples and checks each against the trained store, and how
:class:`~knowlytix.knowledge.rag.pipeline.RagPipeline` abstains when a claim is
contradicted. The check is calibration-free: a numeric-aware contradiction test
where the answer says one value and the graph asserts another, plus, for
cap-trained stores, an admissibility test against the relation's learned cap
radius. A language model is used only to decompose the draft; the verdict comes
from the store.

The store and the decomposition backend
----------------------------------------

The verifier reads the trained store loaded by
:func:`forgeloop.rag.load_store` and calls an
:class:`~knowlytix.knowledge.llm_backend.LLMBackend` to turn the draft answer
into claim triples. The backend's only job is decomposition; the geometry
decides supported against contradicted, so a deterministic backend that returns
the claim triples a competent extractor would produce is enough to exercise the
adjudication offline.

.. code-block:: python

   import json
   from forgeloop.rag import load_store
   from knowlytix.knowledge.llm_backend import LLMBackend

   store = load_store()

   class ScriptedBackend(LLMBackend):
       """Maps a draft (by substring) to the JSON claim-triple array a real
       extractor would emit. The geometry, not this stub, adjudicates."""
       def __init__(self, script):
           self._script = script
       def call(self, system, user, max_tokens=2048):
           for needle, claims in self._script.items():
               if needle in user:
                   return json.dumps(claims)
           return "[]"
       @property
       def model_name(self):
           return "scripted-verify"

In deployment the same local model that synthesizes serves as the verify LLM;
only its role differs. Because a 3B extractor is nondeterministic, pinning the
decomposition with a scripted backend keeps the demonstration reproducible while
leaving the adjudication identical to the live path.

A faithful draft verifies as supported
---------------------------------------

The first draft states the correct figure.
:meth:`~knowlytix.knowledge.rag.verify.AnswerVerifier.verify` decomposes it to
the claim ``(total, has_revenue, 355.0)``, binds it to the graph, finds the
asserted edge and numeric-aware equality holds, so the claim is ``supported``
and ``report.ok`` is ``True``.

.. code-block:: python

   from knowlytix.knowledge.rag.verify import AnswerVerifier

   good_draft = "Total revenue for FY2025 was 355.0 million."
   good_script = {good_draft: [{"head": "total", "relation": "has_revenue",
                                "tail": "355.0"}]}
   verifier = AnswerVerifier(store, ScriptedBackend(good_script))

   report = verifier.verify(good_draft)
   print("ok       :", report.ok)
   print("verdicts :", [(v.triple.as_tuple(), v.status) for v in report.verdicts])
   print("failures :", report.failures)

The claim matched an asserted edge, so no calibration and no cap radius were
needed; the contradiction test alone settled it.

A hallucinated figure verifies as contradicted
-----------------------------------------------

A draft that reports ``455.0`` states a wrong total. The graph asserts
``355.0``, numeric-aware equality fails, the verdict is ``contradicted`` and
``report.ok`` is ``False``. The verdict's ``detail`` names what the graph
actually holds.

.. code-block:: python

   bad_draft = "Total revenue for FY2025 was 455.0 million."
   bad_script = {bad_draft: [{"head": "total", "relation": "has_revenue",
                              "tail": "455.0"}]}
   bad_report = AnswerVerifier(store, ScriptedBackend(bad_script)).verify(bad_draft)

   print("ok      :", bad_report.ok)
   for v in bad_report.verdicts:
       print(v.triple.as_tuple(), "->", v.status, "|", v.detail)

The verifier did not judge whether ``455.0`` is reasonable; it found a directly
asserted fact that disagrees. That is the difference between a faithfulness
check and an opinion.

The pipeline abstains on a failed verification
-----------------------------------------------

Verification is a pipeline stage.
:class:`~knowlytix.knowledge.rag.config.RagConfig` exposes ``verify_llm_output``
and ``on_verify_fail``. With ``verify_llm_output=True`` and
``on_verify_fail="abstain"``, a contradicted claim forces
:class:`~knowlytix.knowledge.rag.pipeline.RagPipeline` down the abstain branch:
``decision == "abstain"``, the answer becomes the standard refusal, and the
``verification`` audit record carries ``ok=False``.

.. code-block:: python

   from knowlytix.knowledge.rag.config import RagConfig
   from knowlytix.knowledge.rag.pipeline import RagPipeline

   question = "What was total revenue in FY2025?"
   extract_script = {question: [{"head": "total", "relation": "has_revenue",
                                 "tail": "?"}]}

   class SynthBackend(LLMBackend):
       def __init__(self, text): self._text = text
       def call(self, system, user, max_tokens=2048): return self._text
       @property
       def model_name(self): return "scripted-synth"

   cfg = RagConfig(
       llm=SynthBackend(bad_draft),               # emits the tampered draft
       llm_extract=ScriptedBackend(extract_script),
       llm_verify=ScriptedBackend(bad_script),
       verify_llm_output=True,
       on_verify_fail="abstain",
       relevance_gate=False,      # offline: skip the extra relevance LLM call
       ground_extraction=False,   # offline: scripted extractor needs no schema
   )
   ans = RagPipeline.from_store(store, cfg).query(question)

   print("decision     :", ans.decision)        # abstain
   print("answer       :", ans.answer)           # the standard refusal
   print("verified ok? :", ans.verification.get("ok"))   # False
   print("notice       :", ans.notice)

``ans`` is a :class:`~knowlytix.knowledge.rag.pipeline.RagAnswer`. The tampered
figure never reaches the user. Setting ``on_verify_fail="regenerate"`` instead
re-synthesizes once before giving up; when the second draft still fails,
``RagPipeline`` annotates rather than abstains, driving ``ans.confidence`` to
``0.0`` and attaching a notice that names the unsupported claim.

Mechanism
---------

The adjudicator is the geometry, not a language model judging a language model.
The contradiction test compares the claim's value against the value the graph
asserts through
:meth:`~knowlytix.knowledge.store.GMSExpertStore.query_triples`; for cap-trained
stores the admissibility test uses the relation's learned cap radius through
:meth:`~knowlytix.knowledge.store.GMSExpertStore.score_triple`. No threshold is
set by hand. Verification is only as good as claim decomposition: a claim the
model fails to extract is never checked, a claim that touches nothing in the
graph is unverifiable rather than safe, and the check confirms agreement with
the store, not truth, so a wrong fact in the store is faithfully echoed. It is
the last gate, not the first, and it feeds the abstention branch of
:doc:`13_abstention_and_coverage`.

See also
--------

- :doc:`/3-api-reference/modules/knowlytix/rag/index` — signatures for
  :class:`~knowlytix.knowledge.rag.verify.AnswerVerifier`,
  :class:`~knowlytix.knowledge.rag.pipeline.RagPipeline`,
  :class:`~knowlytix.knowledge.rag.pipeline.RagAnswer` and
  :class:`~knowlytix.knowledge.rag.config.RagConfig`.
- :doc:`/3-api-reference/modules/knowlytix/llm/index` — the
  :class:`~knowlytix.knowledge.llm_backend.LLMBackend` interface.
- :doc:`11_grounded_synthesis` and :doc:`13_abstention_and_coverage` — the stage
  before and the branch a failed verification takes.
- :doc:`/4-notebook-examples/rag/index` — verification run end to end.
