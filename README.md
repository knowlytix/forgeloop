# ForgeLoop

Companion library for the *Beyond Prompt / Ship / Chunk and Pray* trilogy — one
installable package spanning the three books' companion code.

| Subpackage | Book | What it covers |
| --- | --- | --- |
| `forgeloop.agents`  | *Beyond Prompt and Pray* | building governed agents: the loop, typed tools, gates, audit, planning, memory, the capstone |
| `forgeloop.testing` | *Beyond Ship and Pray*   | the GMS test framework (base/factor/profile catalogs, DOE composition, evaluation) |
| `forgeloop.rag`     | *Beyond Chunk and Pray*  | governed retrieval over a trained GMS store |
| `forgeloop.apps`, `forgeloop.catalogs` | — | the capstone system-under-test and the test catalogs |

## Install

```bash
pip install forgeloop-0.2.0-py3-none-any.whl          # from a Release asset
pip install "forgeloop[ml]"                            # + torch/transformers/peft for the model tools
```

The GMS substrate (geometry, stores, DOE harness) runs on the licensed
`knowlytix` package, installed separately.

## Documentation

The User Guide, API Reference and Gallery of Examples build as one Sphinx site
under `docs/`, organized like the modeva package. The Sphinx source directory is
`docs/`; the config directory is `docs/docs/`:

| Path | Contents |
| --- | --- |
| `docs/2-user-guide/` | User Guide (`.rst`), three parts: agents, testing, rag |
| `docs/3-api-reference/` | API reference (`.rst`, autodoc), one page per subpackage |
| `docs/docs/` | Sphinx config (`conf.py`, `Makefile`) and gallery source under `auto_galleries/` |
| `docs/auto_examples/` | sphinx-gallery output (generated, gitignored) |

Build:

```bash
pip install -e .                        # forgeloop, so autodoc can import it
pip install -r docs/docs/requirements.txt   # sphinx, sphinx-gallery, sphinx-rtd-theme
cd docs/docs && make html               # output in docs/docs/_build/html
```

The API reference imports `forgeloop` with `torch`, `knowlytix`, `transformers`,
`peft`, `anthropic`, `huggingface_hub` and `pandas` mocked (see
`autodoc_mock_imports` in `conf.py`), so it builds without a licensed GMS
environment. Gallery scripts prefixed `plot_` are executed at build time
(only the core-install quickstart); those prefixed `noplot_` need the models and
the licensed backend and are rendered with their code but not executed.

## Data and trained artifacts

The books' notebooks read from each book's `code/data/`. Authored inputs are
committed with the book; the large trained artifacts (adapters, GMS stores,
pinned result files) are not, and are fetched on demand:

```python
import forgeloop
forgeloop.data_root()          # resolves the current book's data directory
forgeloop.ensure_artifacts()   # fetches the trained artifacts that are missing
```

`data_root()` honors `FORGELOOP_DATA_DIR`; `ensure_artifacts()` reads a local
`FORGELOOP_ARTIFACTS_DIR` or the Hugging Face Hub dataset named by
`FORGELOOP_ARTIFACTS_REPO`.
