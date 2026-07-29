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

## Setup

```bash
git clone https://github.com/knowlytix/forgeloop.git && cd forgeloop
make install
```

Installs everything the notebooks need.

---

## 1. Learn with the notebooks

```bash
jupyter lab
```

Open any notebook under `beyond-*/notebooks/`. Chapters are numbered in reading
order, each pairing a concept notebook with an `NN_capstone` that adds that
chapter's layer to a running banking agent.

Plain-Python notebooks run now. The rest need the GMS substrate — two one-time
steps:

**Step 1 — licence key.** `knowlytix` is installed but needs a key to run. Sign up
at <https://knowlytix.ai/signup/>; the flow writes `~/.knowlytix/license.key`.
Never commit it. Verify:

```bash
python -c "import knowlytix"    # prints a banner naming the licensed customer
```

**Step 2 — build the stores.** Run the setup notebook for the book you are
reading:

```text
beyond-prompt-and-pray/notebooks/00_setup.ipynb    building governed agents
beyond-ship-and-pray/notebooks/00_setup.ipynb      testing them
beyond-chunk-and-pray/notebooks/00_setup.ipynb     grounded RAG
```

Idempotent, and it reports what it will build before starting. Tier 1 is CPU-only
and covers most chapters; Tier 2 downloads `Qwen3-4B-Instruct` (~8 GB) and wants a
CUDA GPU.

Weights and stores are not committed; `00_setup` builds them from the corpora in
the repo.

> *`litellm` wheel fails to build?* Recent versions need a Rust toolchain and have
> no prebuilt wheel on some platforms. `knowlytix` only requires `litellm>=1.40`,
> so run `pip install "litellm==1.74.9"` first.

---

## 2. Read the User Guide

User Guide, API Reference and all 114 chapter notebooks, as one offline site:

```bash
pip install -r docs/docs/requirements.txt
make docs
open docs/docs/_build/html/index.html      # xdg-open on Linux, start on Windows
```

Takes a few minutes. Or read the sources on GitHub:

- User Guide — [`docs/2-user-guide/`](docs/2-user-guide/), one file per chapter
- Chapter notebooks — `beyond-*/notebooks/`

The API Reference comes from the docstrings via autodoc, and covers `knowlytix`
too. Building it needs no licence key.

---

## Contributing

**The book directories are the source.** `forgeloop/` and the docs gallery are
assembled from them:

```text
beyond-*/                    ← edit here
   │  scripts/build_forgeloop.py
   ├─► forgeloop/                   the installable package (committed)
   └─► docs/4-notebook-examples/    the docs gallery (generated, git-ignored)
```

Assembly rewrites imports book-local → package (`agentlab` → `forgeloop.agents`,
and so on). `forgeloop/` is committed, so **do not edit it by hand** — CI checks
it against the books. Files with no book source live in
[`packaging/`](packaging/).

```bash
make install-dev    # everything, plus pytest and ruff
make check          # exactly what CI runs
make assemble       # rebuild forgeloop/ and the docs gallery
```

`make check` runs the four CI gates: `ruff`, no committed notebook output,
`forgeloop/` matching the books, and the tests. Green locally means green in CI.

## License

[Apache-2.0](LICENSE). The separately-licensed `knowlytix` substrate is **not**
covered by this license — see above.
