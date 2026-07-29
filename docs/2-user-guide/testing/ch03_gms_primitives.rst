Chapter 3 — Geometric Memory Systems and Their Primitives
=========================================================

This page shows the primitives a trained ``GMSExpertStore`` exposes, how a raw
distance becomes a calibrated decision read from the store, and how the two
embedding channels are fine-tuned. Each primitive computes the ground truth for
one class of question, which is why the base taxonomy in :doc:`ch05_base_taxonomy`
can name a primitive for every category.

The primitives
--------------

Two primitives are exact reads of the asserted graph. The rest are geometric
reads of the trained manifold, and each returns a continuous quantity rather than
a Boolean.

.. list-table::
   :header-rows: 1

   * - Primitive
     - What it computes
   * - ``lookup_enm``
     - Byte-exact numeric value from the exact register.
   * - ``query_triples``
     - Asserted edges matching a pattern (exact).
   * - ``score_triple``
     - Plausibility of a triple as a distance.
   * - ``link_predict``
     - Distinguishes an asserted tail from a predicted one.
   * - ``tension_energy``
     - Contradiction between two entities, as an angle in u-space.

Each entity is embedded twice because "about the same topic" and "logically
compatible" are different questions. The semantic channel, the v-space, places an
entity by what it is about, and the relation-conditioned cap that ``score_triple``
reads lives there. The logical channel, the u-space, carries relational and
contradictory structure, and the angle ``tension_energy`` reads lives there.

Reading the primitives
-----------------------

.. code-block:: python

   import torch
   from knowlytix.knowledge.query import DocGMSConfig, GMSExpertStore

   store = GMSExpertStore(
       DocGMSConfig(store_path=str(STORE), ingest_mode="regex"),
       device=torch.device("cpu"),
   )
   assert store.load()

   d35 = store.score_triple('overdraft', 'has_fee_amount', '35.0')
   d50 = store.score_triple('overdraft', 'has_fee_amount', '50.0')
   print(f'score_triple: 35.0={d35:.3f}  50.0={d50:.3f}')
   print('asserted    :', store.query_triples(head='overdraft')[:2])
   print('tension     :', round(store.tension_energy('overdraft', 'udaap'), 3))

The decision is read from a calibrated operating point
------------------------------------------------------

A raw distance is not a verdict. The decision boundary is a per-relation
operating point fit by ``GMSJudge.calibrate`` and persisted in the store's
``calibration.json`` under a known false-accept rate, so the threshold is a
property of the trained store rather than a hand-set cutoff.

.. code-block:: python

   import json
   from knowlytix.harness.testing.judge import GMSJudge

   tau_d = json.loads(
       (STORE / GMSJudge.CALIBRATION_JSON).read_text()
   )['thresholds']['geodesic']

   def decide(h, r, t):
       d = store.score_triple(h, r, t)
       return round(d, 3), ('grounded' if d <= tau_d else 'fabricated')

   print(decide('overdraft', 'has_fee_amount', '35.0'))   # ('...', 'grounded')
   print(decide('overdraft', 'has_fee_amount', '50.0'))   # ('...', 'fabricated')

Fine-tuning the two channels
----------------------------

A stock encoder has never seen the document's vocabulary, so a customer's
phrasing may miss the entity it names. The knowlytix embedding SFT tunes each
channel separately: ``finetune_embedding`` fits the v-space to text-to-entity
labels, and ``finetune_contradiction`` fits the u-space so confusable attributes
repel. Both export vectors that ``init_dual_embeddings`` inserts into a
``DualEmbedding``.

.. code-block:: python

   from knowlytix.embedding import (EmbeddingSFTConfig, finetune_embedding,
                                    finetune_contradiction)

   v_ft = finetune_embedding(
       str(labels_jsonl),
       EmbeddingSFTConfig(mode='rotation', rank=4, epochs=20, out_dim=64, device='cpu'),
       text_col='text', label_col='label',
   )
   u_ft = finetune_contradiction(
       {'has_fee_amount': ['overdraft fee', 'nsf charge'],
        'has_interest_rate': ['apr', 'interest rate'],
        'has_window_days': ['filing window', 'days to file']},
       EmbeddingSFTConfig(mode='full', objective='contradiction',
                          rank=4, epochs=20, out_dim=64, device='cpu'),
   )
   print(v_ft.adapter.mode, u_ft.adapter.mode)     # rotation full

The tuned encoder is carried into the production build, so the agent under test
and its grader read the same encoder at query time.

See also
--------

- :doc:`/3-api-reference/modules/knowlytix/store/index` — the store and its
  primitive methods.
- :doc:`/3-api-reference/modules/knowlytix/embedding/index` — ``finetune_embedding``,
  ``finetune_contradiction`` and ``EmbeddingSFTConfig``.
- :doc:`/3-api-reference/modules/gms_backend/index` — the forgeloop gate that
  reads the calibrated operating point.
- :doc:`/4-notebook-examples/testing/index` — the primitives, calibration and
  SFT run end to end.
