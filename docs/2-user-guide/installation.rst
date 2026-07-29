Installation
============

forgeloop requires Python 3.12 or later. The core install is light; heavier and
licensed pieces are optional extras.

.. code-block:: bash

    pip install forgeloop                 # core: loop, typed actions, tools, gates, audit, planning, memory, evaluation, catalogs
    pip install "forgeloop[ml]"           # + open-weight model tools (torch, transformers, peft, accelerate)
    pip install "forgeloop[artifacts]"    # + huggingface_hub, to fetch the trained-artifact bundle
    pip install "forgeloop[anthropic]"    # + the optional hosted-model adapter
    pip install "forgeloop[notebooks]"    # + JupyterLab, matplotlib, pandas
    pip install "forgeloop[all]"          # everything above (public deps only)
    pip install "forgeloop[dev]"          # + pytest, ruff, black

Extras
------

.. list-table::
   :header-rows: 1
   :widths: 15 40 45

   * - Extra
     - Pulls in
     - Needed for
   * - *(core)*
     - ``pydantic``, ``numpy``, ``pyyaml``
     - the loop, types, tools, gates, audit, planning, memory, evaluation, and the bundled catalog/app YAML
   * - ``ml``
     - ``torch``, ``transformers``, ``peft``, ``accelerate``
     - the classifier head, the Qwen adapters, the draft LoRA
   * - ``artifacts``
     - ``huggingface_hub``
     - fetching the trained-artifact bundle
   * - ``gms``
     - ``knowlytix`` (licensed, separate), ``pandas``
     - the GMS substrate — see below
   * - ``anthropic``
     - ``anthropic``
     - the hosted-model adapter
   * - ``notebooks``
     - ``jupyterlab``, ``matplotlib``, ``pandas``
     - running the chapter notebooks
   * - ``dev``
     - ``pytest``, ``pytest-cov``, ``ruff``, ``black``
     - tests and linting

From a checkout, use ``-e`` for an editable install, for example
``pip install -e ".[ml,notebooks,dev]"``.

The GMS backend (licensed)
--------------------------

The GMS-backed features — the geometric plausibility gate, the regulatory
guard, the policy Graph-RAG store, exact numerical memory, the design-of-
experiments test harness and the governed-retrieval store — run on the
``knowlytix`` package. ``knowlytix`` is licensed and distributed separately; it
is not on public PyPI, and a license is required.

.. code-block:: bash

    pip install knowlytix --index-url <KNOWLYTIX_INDEX_URL>   # license required; index from Knowlytix
    pip install "forgeloop[gms]"                              # records the dependency

forgeloop imports ``knowlytix`` lazily, so everything outside the GMS features
works without it.

Data and trained artifacts
--------------------------

The examples read from a book's ``data/`` directory, resolved by
:func:`forgeloop.data_root` and :func:`forgeloop.data_path` (override with the
``FORGELOOP_DATA_DIR`` environment variable). Authored inputs — policy corpora,
evaluation cases, governance exemplars — ship with the books. Trained artifacts
— classifier and drafter adapters, calibrated GMS stores, fine-tuned extractor
encoders — are large and are not committed or bundled in the wheel; they are
reproducible from each book's ``scripts/``, or fetched from a published bundle
with :func:`forgeloop.ensure_artifacts` (which uses the ``artifacts`` extra).
