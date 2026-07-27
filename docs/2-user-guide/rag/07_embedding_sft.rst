Tuning the Encoders: Embedding SFT
==================================

Retrieval and the relevance gate both rest on an encoder that maps text to a
vector. A frozen general-purpose encoder treats "topline" and "revenue" as
similar strings and has no notion of two claims contradicting each other. This
page tunes two encoders on the data produced by the designed experiment
(:doc:`06_doe_enrichment`), so the geometry carries two relationships: semantic
proximity in the v-space, where a paraphrased question lands near the attribute
it asks about, and logical contradiction in the u-space, where a claim asserting
the wrong value sits far from the truth. Both are supervised fine-tunes of a
low-rank adapter over a frozen base encoder; the base never moves. The v-space
tune drives binding (:doc:`09_binding`) and the u-space tune drives
self-verification (:doc:`12_self_verification`).

The two encoders
----------------

.. list-table::
   :header-rows: 1

   * - Encoder
     - What it decides
   * - v-space
     - Which attribute a question is about (semantic proximity).
   * - u-space
     - Whether two statements agree or conflict (tension).

Both are configured through :class:`~knowlytix.embedding.config.EmbeddingSFTConfig`
and fit by :func:`~knowlytix.embedding.finetune.finetune_embedding`, which returns
a :class:`~knowlytix.embedding.finetune.FineTunedEmbedding` wrapping a
:class:`~knowlytix.embedding.adapter.LowRankEmbeddingAdapter`.

Load the store and read its v-dimension
---------------------------------------

The adapter's output dimension must match the store's v-space, so the store is
loaded first and ``d_v`` is read from its model config. In
:mod:`forgeloop.rag`, :func:`forgeloop.rag.load_store` loads the trained store.

.. code-block:: python

   from forgeloop.rag import load_store

   store = load_store("data/gms_annual_report_store")
   d_v = store.model.cfg.d_v

Tune the v-space encoder
------------------------

The v-space encoder is tuned on the contextual questions, each labeled with its
attribute, under the ``"prototype"`` objective that pulls same-attribute
questions together and pushes attributes apart. The supervision is the
contextual question rather than a bare keyword, so the encoder learns the
attribute in context. The training data is the ``embedding_sft.jsonl`` file
written by the enrichment step, read through ``text_col`` and ``label_col``.

.. code-block:: python

   from knowlytix.embedding import EmbeddingSFTConfig, finetune_embedding

   v_ft = finetune_embedding(
       "data/enrichment/embedding_sft.jsonl",
       EmbeddingSFTConfig(rank=8, mode="full", out_dim=d_v, objective="prototype"),
       text_col="text",
       label_col="label",
   )
   print("v-space val_accuracy =", round(v_ft.val_accuracy, 3))

:class:`~knowlytix.embedding.finetune.FineTunedEmbedding` reports a held-out
``val_accuracy`` on the generated questions.

Tune the u-space encoder
-------------------------

The u-space encoder is the logical half: it must give low tension to two
statements that agree and high tension to two that conflict. The supervision
must be genuine contradiction, the same fact asserted with different values, not
merely different topics. The contradiction objective requires ``mode="full"``: a
rotation preserves all angles and so cannot change tension, which would leave a
contradiction looking as consistent as the truth. The notebook builds the
conflicting-value pairs and fits them through script helpers over the same
configuration object.

.. code-block:: python

   from finetune_encoders import (
       _contradiction_pairs, _fit_contradiction_pairs, _uspace_tension,
   )

   pos, neg = _contradiction_pairs(store)
   u_ft = _fit_contradiction_pairs(
       pos, neg,
       EmbeddingSFTConfig(
           rank=32, mode="full", objective="contradiction",
           encoder="sentence-transformers/nli-mpnet-base-v2",
           out_dim=d_v, epochs=400,
       ),
   )

   print("consistent:", round(_uspace_tension(
       u_ft, "Retail’s headcount was 520.",
       "The headcount of Retail is 520."), 3))
   print("contradictory:", round(_uspace_tension(
       u_ft, "Retail’s headcount was 520.",
       "Retail’s headcount was 210."), 3))

``_contradiction_pairs``, ``_fit_contradiction_pairs`` and ``_uspace_tension``
are helpers in the book's ``finetune_encoders`` script; they wrap the knowlytix
contradiction objective. The underlying calls are
:func:`~knowlytix.embedding.finetune.finetune_contradiction`, which fits the
adapter under :func:`~knowlytix.embedding.objectives.contradiction_loss`, and
:func:`~knowlytix.embedding.objectives.tension_energy`, which scores a pair.

Why full mode
-------------

The v-space objective compares questions by direction, so an angle-preserving
rotation adapter suffices. The u-space objective compares two claims by tension,
and :func:`~knowlytix.embedding.objectives.tension_energy` is invariant under a
shared rotation. A rotation-only adapter therefore cannot separate a
conflicting-value pair from an agreeing one, which is why the contradiction tune
sets ``mode="full"``.

Run it end to end
-----------------

The book's ``scripts/finetune_encoders.py`` runs both fine-tunes and the
relevance-gate calibration in one pass, writing ``tuned_encoder/``,
``contradiction_encoder/`` and ``relevance_calibration.json`` beside the store.
The persisted operating point, the v-accept floor and the per-attribute u-veto
cut, is fit from the same data and exercised under calibration
(:doc:`14_calibration`). The encoders are only as good as the generated data
behind them, and a high held-out accuracy on generated questions is not a
guarantee on phrasings far outside the factor design.

See also
--------

- :doc:`/3-api-reference/modules/knowlytix/embedding/index` — full signatures for
  :class:`~knowlytix.embedding.config.EmbeddingSFTConfig`,
  :func:`~knowlytix.embedding.finetune.finetune_embedding` and the objectives.
- :doc:`08_triple_mediated_retrieval` and :doc:`09_binding` — where the tuned
  v-encoder is consumed.
- :doc:`14_calibration` — fitting the relevance gate's operating point.
- :doc:`/4-notebook-examples/rag/index` — the same fine-tunes run end to end.
