Generating Training Data
========================

This page shows how to consume composed scenarios as a supervised corpus. The
same scenarios that drive evaluation drive training; only the consumer differs.
Two emitters ship, one for a classifier and one for a generative model, and each
enforces a different invariant so the corpus is correct by construction.

Two emitters
------------

.. list-table::
   :header-rows: 1

   * - Emitter
     - Invariant enforced
   * - :func:`~forgeloop.testing.emit.emit_classifier_sft`
     - The label is preserved while presentation varies.
   * - :func:`~forgeloop.testing.emit.emit_draft_sft`
     - Each output satisfies a grounding contract or is dropped.

Label-preserving classifier data
---------------------------------

:func:`~forgeloop.testing.emit.emit_classifier_sft` turns each scenario into a
``(message, label)`` record. Presentation factors vary the surface; the label is
fixed by the base, so the enrichment produces harder variants without corrupting
the target.

.. code-block:: python

   import forgeloop.testing as gt

   cat = gt.Catalog.load()
   items = gt.UserBaseSource([
       {'query': 'I was overcharged on my overdraft', 'answer': 'complaint'},
       {'query': 'How do I close my account?',        'answer': 'inquiry'},
   ]).items()
   scns = gt.compose(gt.resolve(cat, ['exact_recall'], ['clarity', 'noise'], mode='cross'),
                     items, n_runs=4, seed=1)

   recs = gt.emit_classifier_sft(scns)
   print(len(recs), 'records; labels', {r['label'] for r in recs})   # {'complaint', 'inquiry'}

Grounded generative data under a contract
-----------------------------------------

:func:`~forgeloop.testing.emit.emit_draft_sft` builds ``(user, assistant)`` pairs
whose ``user`` field is the exact inference-time template from
:func:`~forgeloop.testing.emit.draft_prompt`. A per-policy golden states the
contract: the required citation keywords, the byte-exact numbers the answer must
contain and the phrases it must not. Each candidate is checked by
:func:`~forgeloop.testing.emit.passes_contract`; a failing pair is returned in
the dropped list rather than kept.

.. code-block:: python

   goldens = {'overdraft': {'citation_keywords': ['overdraft'],
                            'required_numbers': ['35'],
                            'forbidden_phrases': ['we will waive']}}
   pol = [gt.QAItem(qid='p0', query='charged $35 overdraft', answer=None,
                    metadata={'policy_id': 'overdraft'})]
   psc = gt.compose(gt.resolve(cat, ['exact_recall'], ['clarity'], mode='cross'), pol, n_runs=2)

   good = lambda s: 'Your overdraft fee of $35 may be reversed once per year.'
   bad  = lambda s: 'We will waive your fee.'
   kept_g, drop_g = gt.emit_draft_sft(psc, good, goldens=goldens)
   kept_b, drop_b = gt.emit_draft_sft(psc, bad,  goldens=goldens)
   print('good kept/dropped:', len(kept_g), len(drop_g))   # 2 0
   print('bad  kept/dropped:', len(kept_b), len(drop_b))   # 0 2

The good response cites the keyword and holds the number, so both scenarios are
kept. The bad response uses a forbidden phrase, so both are dropped. Text
generation is pluggable through the ``response_fn`` argument, so the format and
contract logic stay dependency-light and testable while a model supplies the
surface text in production. :func:`~forgeloop.testing.emit.to_jsonl` writes the
kept records.

The functionality check on a mined base
----------------------------------------

When the base is mined from the graph rather than user-supplied, a question is
admitted only after the store recovers its stated answer. The check runs two
ways: a symbolic recompute re-applies the generator's answer function to the
recorded facts, and a geometric check confirms the trained manifold supports the
answer's triple under the calibrated operating point. A question survives only
when both agree, so a question the store cannot answer is discarded before it can
poison the corpus.

See also
--------

- :doc:`/3-api-reference/modules/testing/index` — the emit module and the
  contract check.
- :doc:`ch05_base_taxonomy` — the base sources whose scenarios feed the emitters.
- :doc:`ch09_agentic` — the evaluation consumer of the same scenarios.
- :doc:`/4-notebook-examples/testing/index` — both emitters worked through.
