# forgeloop

**Beyond Prompt-Ship-Chunk and Pray** — open-source companion code and runnable
notebooks for building, testing, and grounding trustworthy agentic AI systems on
a geometric memory substrate.

| Directory | Package | Focus |
|---|---|---|
| [`beyond-prompt-and-pray/`](beyond-prompt-and-pray/) | `agentlab` | Building agentic AI systems: the agent loop, typed tools, governance, memory, evaluation, and a capstone banking agent. |
| [`beyond-ship-and-pray/`](beyond-ship-and-pray/) | `gmstest` | Testing agentic systems: DoE-driven test design, component and system attribution, resilience. |
| [`beyond-chunk-and-pray/`](beyond-chunk-and-pray/) | `book_kit` | Trustworthy RAG with geometric memory: triple-mediated retrieval, grounded synthesis, self-verification, calibrated abstention. |
| [`beyond-vibe-and-pray/`](beyond-vibe-and-pray/) | `reasonloop` | A semantic execution substrate: the LLM as compiler and grounded synthesizer over a deterministic reasoning engine. *(Work in progress.)* |

## Source of truth, and what is generated

**The three book directories are the source.** Everything else that contains the
same material is a build artifact assembled from them:

```
beyond-*/                    ← edit here
   │  scripts/build_forgeloop.py
   ├─► forgeloop/                   the installable package (committed)
   └─► docs/4-notebook-examples/    the docs gallery (generated, git-ignored)
```

Each book directory holds:

- **`code/`** — the companion Python package plus supporting scripts, data and
  tests. `make install` installs all of them; see [Install](#install).

- **`notebooks/`** — the runnable notebooks, one per chapter, importing from that
  book's package and the stdlib.

Assembling rewrites import statements book-local → package (`agentlab` →
`forgeloop.agents`, `gmstest` → `forgeloop.testing`, `book_kit` →
`forgeloop.rag`), so the books stay runnable from a checkout while the package
ships a single `forgeloop.*` namespace.

```bash
python scripts/build_forgeloop.py             # build both distributions
python scripts/build_forgeloop.py --package   # forgeloop/ only
python scripts/build_forgeloop.py --docs      # docs gallery only
python scripts/build_forgeloop.py --check     # fail if either has drifted
```

`forgeloop/` is committed so that `pip install` from a checkout and the autodoc
docs build both work without a build step. **Do not edit it by hand** — the next
assembly overwrites it, and CI runs `--check` to catch exactly that. Change the
book, then re-assemble and commit the result.

The handful of package-level files with no book source — the package `__init__`
and `forgeloop.rag` — live in [`packaging/`](packaging/) and are copied in
verbatim.

## Install

One command installs every book's package plus `forgeloop`, in the right order:

```bash
git clone https://github.com/knowlytix/forgeloop.git && cd forgeloop
make install
```

After that you can start in **any** book and run its notebooks — `agentlab`,
`gmstest`, `book_kit`, `apps.complaint_sut` and `forgeloop` all import with no
`PYTHONPATH` set. Order matters and `make` handles it: Ship-and-Pray's system
under test imports `agentlab`, which is not declared as a dependency because it
is not published to PyPI.

`make install-dev` adds the test and lint extras. To consume the assembled
package on its own, without the book sources:

```bash
pip install forgeloop-<version>-py3-none-any.whl   # from a Release asset
pip install "forgeloop[ml]"                        # + torch/transformers/peft for the model tools
```

## The `knowlytix` substrate (licensed)

The GMS-backed features — geometric plausibility gates, the regulatory guard, the
policy Graph-RAG store, Exact Numerical Memory, and the design-of-experiments test
harness — run on the **`knowlytix`** package. Code paths that don't require it run
without it, so a fresh clone lints, tests, and runs the non-GMS material out of
the box.

`knowlytix` is published on PyPI but **gated by a license key** you validate at
runtime — the package is free to install; using it requires a license. To enable
the GMS features locally:

1. **Get a developer license.** Sign up at <https://knowlytix.ai/signup/> and
   follow the steps to download your license key. The flow writes it to
   `~/.knowlytix/license.key` and records EULA acceptance in
   `~/.knowlytix/eula-accepted`. Keep the key private — never commit it.
2. **Install the package:** `pip install knowlytix`.
   - *If the install fails building a `litellm` wheel* (recent `litellm` needs a
     Rust toolchain and has no prebuilt wheel on some platforms), pre-install a
     wheel-available version first — knowlytix only requires `litellm>=1.40`:
     `pip install "litellm==1.74.9"` then `pip install knowlytix`.
3. **Verify:** `python -c "import knowlytix"` prints a one-line banner naming the
   licensed customer and tier if the key is valid.

## Trained artifacts are not committed

Trained-model weights (`*.safetensors`, `*.pt`, large `tokenizer.json`) and the
generated GMS stores are **git-ignored** to keep clones lean. A fresh checkout has
none of them, so most chapters fail with *"failed to load GMS … store"* until they
are built.

Rebuild them with each book's setup notebook, which drives that book's
`code/scripts/build_*.py` and `train_*.py` and is idempotent — every stage is
skipped when its output already exists:

```
beyond-prompt-and-pray/notebooks/00_setup.ipynb
beyond-ship-and-pray/notebooks/00_setup.ipynb
beyond-chunk-and-pray/notebooks/00_setup.ipynb
```

Tier 1 is CPU-only and covers what most chapters need; Tier 2 downloads
Qwen2.5-3B-Instruct and expects a CUDA GPU for the retrieval chapters. Both
require the licensed `knowlytix` substrate.

## Reading the documentation

The User Guide, API Reference and notebook gallery form one Sphinx site. It is
**not hosted and not committed** — the built site is ~130 MB against a ~70 MB
repository, and HTML diffs poorly, so every rebuild would add another near-complete
copy to git history. Build it locally instead; it is a two-command job.

**1. Build it from a clone** (a few minutes)

```bash
git clone https://github.com/knowlytix/forgeloop.git && cd forgeloop

python -m pip install -e .                        # so autodoc can import forgeloop
python -m pip install -r docs/docs/requirements.txt

make -C docs/docs html                            # assembles the gallery, then builds

open docs/docs/_build/html/index.html             # macOS
xdg-open docs/docs/_build/html/index.html         # Linux
start docs\docs\_build\html\index.html            # Windows
```

That produces the whole site — User Guide, API Reference and all 114 chapter
notebooks — as static HTML you can read offline.

Two things worth knowing:

- `make` assembles the notebook gallery from the books before running Sphinx, so
  there is no separate step and `docs/4-notebook-examples/` does not need to exist
  beforehand.
- Install `forgeloop` **before** building. The site stamps `forgeloop.__version__`
  on every page, read from installed metadata; without an install it reads
  `0.0.0+unknown` and the build still succeeds, so it fails quietly rather than
  loudly.

The API Reference is generated by autodoc, so docstrings written in the books
render straight through — nothing to regenerate by hand. It covers the `knowlytix`
substrate as well: 87 of its ~194 pages are knowlytix, which is why the
requirements file installs it. **No license key is needed to build those pages** —
the key gates runtime use, and autodoc only imports and reads signatures. Without
the package installed the site still builds, but those pages render as empty stubs.

**2. Read the sources directly**

No build needed — GitHub renders both:

- User Guide — [`docs/2-user-guide/`](docs/2-user-guide/) (`.rst`, one file per chapter)
- Chapter notebooks — `beyond-*/notebooks/`

Docs are not built in CI either: the site takes several minutes, most of it
rendering the `knowlytix` API reference, and nothing about it needs to run on
every push.

## Development

```bash
make install-dev    # every package, plus pytest and ruff
make check          # exactly what CI runs
make assemble       # rebuild forgeloop/ and the docs gallery from the books
make docs           # build the documentation site
```

`make check` runs the same four gates as CI: `ruff`, no committed notebook output
(`scripts/notebook_outputs.py --check`), `forgeloop/` matching the books
(`scripts/build_forgeloop.py --package --check`), and the test suites. A green
`make check` locally means a green CI run.

Tests that require the licensed `knowlytix` substrate are skipped automatically
when it is not installed.

## License

[Apache-2.0](LICENSE). The separately-licensed `knowlytix` substrate is **not**
covered by this license — see above.
