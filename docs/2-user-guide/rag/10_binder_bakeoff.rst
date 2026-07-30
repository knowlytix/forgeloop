Chapter 10 — Choosing a Binder: a Bake-Off
==========================================

Binding resolves each query term to a term the store contains, and encoder
tuning is one way to make that resolution reliable (:doc:`09_binding`). It is not
the only way, and its alternatives are not obviously worse. The step that turns a
question into a bound query triple can instead be a small language model
finetuned to compile the question directly, or a frozen model shown a few worked
examples at inference time, or a trained model that supplies the relations while
a lookup table supplies the entities. Each of these has a claim to robustness,
and on a first reading each is plausible.

Plausibility does not settle a deployment decision, because two binders that look
equally reasonable can differ sharply in the ways that matter: in how often they
resolve to the wrong graph term, in whether they fail loudly by abstaining or
silently by binding to a plausible-but-wrong relation, and in what it costs to
correct a mistake once one is found. A binder that resolves confidently to the
wrong relation returns a well-provenanced wrong answer, which is the failure mode
:doc:`09_binding` set out to remove, so an ill-chosen binder reintroduces the
risk the graph was meant to eliminate. The choice among binders is therefore
settled by measurement rather than by argument.

How the comparison is run
-------------------------

A fair comparison requires that the binder be the only thing that varies, since
otherwise a difference in accuracy could be traced to a different retriever or
relevance gate rather than to the binder. The retriever, the relevance gate, the
verifier, the assembler, the numeric route and the synthesis model are fixed as a
single configuration, and each binder is inserted into that configuration at one
seam. Each binder is then run over two cohorts: a synthetic design cohort whose
presentation is varied along measured factors, and a separate real-language
holdout written so that no binder can have seen its phrasing. Every answer is
scored by a catalog that separates a correct bind from a silent mis-bind and from
a safe abstention, because the character of a failure decides whether the
pipeline can catch it. Every metric is then sliced by the presentation factor the
question occupies, so the report shows where each binder leads rather than a
single pooled average. A decision rule finally admits only binders that clear
stated ceilings on the mis-bind rate, the calibrated coverage and the
synthetic-to-real gap, and recuses when none does.

The four binders
----------------

The four configurations differ only in how a question becomes a bound query
triple, which is what makes the seam a fair place to intervene, since a binder is
any object that maps a question to query triples. Arm A is the shipped encoder
binder of :doc:`09_binding`, in which a prompted extractor parses the question and
an embedding binder resolves each slot against the tuned encoder. Arm B is the
finetuned compiler, a small language model trained on the store to emit the named
entity's ``(head, relation)`` hop chain, which is then walked against the graph by
exact lookup through
:class:`~knowlytix.knowledge.rag.compiler.walk.StoreChainWalker`. Arm C keeps that
emission-and-walk contract but removes the training: a frozen model receives, in
its prompt, the handful of training examples nearest the question in the tuned
encoder's space, and is asked to produce the same hop chain. Arm D is a hybrid,
in which the trained compiler supplies the relations while the head entity is
resolved through a patchable alias table that maps surface forms to canonical
identifiers, falling back to the nearest store entity in the encoder's space only
when its similarity clears a calibrated margin.

The two new binders are built into the library's ``bakeoff`` package and injected
as the pipeline's parser, so they reuse the models the first two already load ---
the frozen model of arm C is the base model arm A prompts, and the compiler of
arm D is the one arm B trained --- and the four-arm comparison costs no more
memory than the two-arm one.

.. code-block:: python

   from knowlytix.knowledge.rag.bakeoff import (
       ExemplarIndex, ExemplarFewShotParser,
       AliasTableResolver, AliasHybridParser,
       calibrate_alias_resolver, decide)

   # Arm C: a frozen model prompted with the k nearest training exemplars.
   index = ExemplarIndex.from_rows(exemplar_rows, v_encoder.encode)
   pipe_c.extractor = ExemplarFewShotParser(
       store, base_lm, index, v_encoder.encode, k=6)

   # Arm D: compiler relations, alias-table head resolution with a calibrated
   # fallback.
   tau = calibrate_alias_resolver(
       entities, v_encoder.encode, far_ceiling=0.05)["tau"]
   resolver = AliasTableResolver.from_store(store, encoder=v_encoder.encode, tau=tau)
   pipe_d.extractor = AliasHybridParser(store, compiler, resolver)

Generating the samples and fine-tuning the compiler
---------------------------------------------------

The code above takes the ``compiler`` and its ``exemplar_rows`` as given. Arm A's
encoder is tuned as in :doc:`07_embedding_sft`, but arm B's compiler is built in two
steps --- generating the supervised samples from the store, then fine-tuning a small
language model on them --- and because those steps are what a practitioner carries
out to obtain the binder, they are set out here. Sample generation enumerates the
store's asserted facts and, for each, produces a natural-language question paired
with the gold ``(head, relation)`` hop chain that answers it, varying the question's
presentation across the designed factors of :doc:`06_doe_enrichment` so the compiler
sees many surface forms of the same fact rather than a single canonical phrasing. A
presentation level and a fraction of the facts are withheld, which yields the
held-out-level and novel-fact splits the bake-off scores on later, so the model is
never trained on the questions it is tested against.

.. code-block:: python

   from knowlytix.knowledge.rag.compiler import (
       build_compiler_dataset, train_compiler, CompilerSFTConfig, QWEN_4B)

   # 1) Generate (question -> hop-chain) samples from the store over the DOE
   #    factors, holding out a presentation level and a fraction of facts.
   splits = build_compiler_dataset(
       store, llm, group="comprehensive", variants_per_base=12,
       heldout_levels={"clarity": "Misleading"}, heldout_fact_frac=0.15, seed=42)
   # -> {"train": [...], "test_heldout_level": [...], "test_heldout_fact": [...]};
   #    each row pairs a question with its gold hop chain and the realized factors.

   # 2) Fine-tune a small language model on the samples with a low-rank adapter.
   compiler_dir = train_compiler(
       splits["train"], out_dir=f"{store_dir}/query_compiler",
       config=CompilerSFTConfig(base_model=QWEN_4B, lora_r=16, lora_alpha=32, epochs=3))

The fine-tune fits a low-rank adapter on the base model so that, given a question,
the model emits the hop chain for the entity the question names in the store's
schema rather than free text, and it writes the adapter and tokenizer to the store's
``query_compiler/`` directory, from which
:class:`~knowlytix.knowledge.rag.compiler.model.CompiledQueryParser` loads them. The
single call ``build_query_compiler(store_dir, llm)`` runs both steps and the in-run
evaluation together when the two need not be separated. The remaining binders reuse
these products rather than adding a training run: arm C indexes the same generated
samples in the encoder's space, so ``exemplar_rows`` is ``splits["train"]`` and no
gradient step is taken, and arm D loads arm B's adapter for its relations while its
alias-table fallback margin ``tau`` is calibrated once from the store's entities and
persisted beside it.

The substrate and the two cohorts
----------------------------------

The substrate is the annual-report store used throughout the tutorial, a small
graph of forty-four entities and thirty-four content facts, so the comparison
measures the binders on the same modest body of facts rather than on a large
corpus. The first cohort is the design-of-experiments cohort of the enrichment
chapter, three hundred and twenty-nine questions whose presentation is varied
along measured factors --- clarity, style, length, expertise and paraphrase
depth --- and which includes both questions at held-out presentation levels and
questions about held-out facts, the latter labeled novel. The second cohort is a
separate real-language holdout of forty questions, disjoint from the first on both
axes that could leak an advantage: it varies a different group of presentation
factors, and it is written by a different language model than the one that
generated the training data, so a binder that has merely memorized the design's
phrasing cannot score on it. Every question is answered by executing the bound
query against the graph and comparing to the graph's own answer, which means a
query that differs from the reference but is equivalent under the graph still
counts as correct.

What is measured, and why mis-binding is the load-bearing column
----------------------------------------------------------------

Accuracy alone does not characterize a binder, because two binders with the same
accuracy can fail in opposite ways. A binder that abstains when it cannot resolve
a term is safe, since it surfaces its own uncertainty; a binder that resolves to a
schema-valid but wrong relation is confidently wrong and expensive to catch
downstream. The metric catalog therefore partitions each answerable case into a
correct bind, a mis-bind that is accepted but executes to the wrong answer, or a
fail-bind that abstains, and scores relation-absent questions by whether the
binder correctly abstained or falsely accepted. The mis-bind rate is the column
that carries the governance decision, because it counts the errors the pipeline
will not catch on its own. Alongside it the catalog reports the novel-fact score,
the false-accept rate on relation-absent questions, the accuracy gap between the
synthetic cohort and the real-language holdout, the conformal coverage of a
calibrated abstention set on the holdout, and the patch cost, the number of steps
needed to correct a discovered mis-binding, which is one for an alias-table edit,
two for retuning an adapter, and three for a full retrain.

The crossover
-------------

On the synthetic cohort the three arms that either train on the store or are shown
its examples occupy a single accuracy band: arm B reaches ``0.967``, arm C
``0.954`` and arm D ``0.970``, and since the spread among them is a few cases on a
cohort of this size, they are comparable rather than ranked. All three commit no
mis-binds and answer the novel-fact questions at the same ``0.95`` rate. The
encoder baseline is the exception, at ``0.748`` accuracy with a mis-bind rate of
``0.178`` and a false-accept rate of ``0.211`` on relation-absent questions, which
is the peripheral-head drift that the exact walk was designed to remove. The
separation on the synthetic cohort is therefore between the walk-and-abstain
binders and the encoder binder, not among the first three.

The real-language holdout tells a different story, because every arm loses most of
its synthetic accuracy on it: arm A falls to ``0.500``, arms B and C to ``0.375``,
and arm D to ``0.400``. The synthetic-to-real gap is accordingly large for the
three walk-and-abstain binders, near ``0.57`` to ``0.59``, and smaller for the
encoder binder at ``0.248``, though the encoder binder's smaller gap follows from
its lower synthetic accuracy rather than from better generalization. The gap
measures how much of each binder's synthetic performance was tied to the design's
phrasing, and on this substrate that dependence is substantial for the binders
that learned or were shown the design.

The decision rule recuses
-------------------------

The decision rule turns the report into a governed recommendation by admitting
only binders that clear every ceiling: a mis-bind rate at or below ``0.05``, a
conformal coverage at or above ``0.90``, and a synthetic-to-real gap at or below
``0.10``. These ceilings are placeholders until model risk management ratifies
them, and the audit record marks them as un-ratified. Applied to this report, no
binder is admissible, because the conformal coverage of every arm sits well below
``0.90`` and the gap of every arm exceeds ``0.10``, and arm A additionally fails
the mis-bind ceiling. The rule therefore recuses from a single pick rather than
shipping a binder that violates a bound, and records the nearest binder, arm D,
which fails only the coverage and gap ceilings, together with the reason.

.. code-block:: python

   import json
   from knowlytix.knowledge.rag.bakeoff import decide

   report = json.load(open("data/enrichment/bakeoff_ABCD.json"))
   out = decide(report, ratified=False, timestamp="2026-07-30")

   assert out["recommended"] is None and out["recused"] is True
   assert out["audit"]["per_arm"]["A_encoder"]["mis_bind_rate"] > 0.05
   assert all(out["arms"][a]["metrics"]["patch_cost"] == 1
              for a in ("C_fewshot", "D_hybrid"))
   print("recused; nearest =", out["audit"]["recommended"] or "none")

Recusal is the correct outcome under the stated conditions, since a coverage of
``0.90`` is a strong requirement and the holdout has forty cells. Within that
recusal the per-regime crossover still identifies a leading binder at each factor
level, and no single binder leads everywhere: the compiler tends to lead on
shorter and more formal questions, while the hybrid tends to lead on longer and
more ambiguous ones. A deployment that must choose can route by regime rather than
commit to one binder globally.

Reading the result: cost is the separating axis
------------------------------------------------

Because the three walk-and-abstain binders are comparable on synthetic accuracy
and share the same holdout collapse, accuracy does not separate them, and the axis
that does is the cost of correction. Arm D reproduces the compiler's
in-distribution profile --- no mis-binds, no false accepts on absent relations, the
same novel-fact score --- while resolving entities through a table that is edited
rather than trained, so a discovered mis-binding is fixed by adding one alias row
at a patch cost of one rather than by a full retrain at a patch cost of three.
Arm C reaches the same accuracy band with no training at all, at the price of a
single false accept on the relation-absent questions, which suggests that dynamic
exemplars recover most of what the finetuned compiler learns without the training
run. The finetuned compiler remains the most conservative on the absent questions,
but it is also the most expensive to correct, and on this substrate it buys no
synthetic accuracy over the hybrid.

The scope of these conclusions is bounded by the substrate. The graph is small, so
the rates carry wide confidence intervals and a larger graph is the natural next
test. The novel questions withhold facts rather than entities, so they do not
exercise the alias table's intended advantage, resolving an entity the binder has
never seen, which remains to be measured on a substrate whose novelty is in the
entities. The ceilings are un-ratified, so the recusal is a statement about
placeholder bounds and not a ratified verdict.

With the binder chosen or, as here, deliberately left open per regime, the next
page turns the bound query triples into answers with provenance attached
(:doc:`11_answering_through_the_gms`).

See also
--------

- :doc:`/3-api-reference/modules/knowlytix/rag/index` — signatures for
  :class:`~knowlytix.knowledge.rag.bakeoff.ExemplarFewShotParser`,
  :class:`~knowlytix.knowledge.rag.bakeoff.AliasHybridParser`,
  :class:`~knowlytix.knowledge.rag.bakeoff.AliasTableResolver` and
  :func:`~knowlytix.knowledge.rag.bakeoff.decide`.
- :doc:`09_binding` — the encoder binder that arm A ships and the failure mode the
  bake-off measures.
- :doc:`11_answering_through_the_gms` — resolving the bound query triples into
  facts once a binder is chosen.
