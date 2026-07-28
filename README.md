# forgeloop

**Beyond Prompt-Ship-Chunk and Pray** — open-source companion code and runnable
notebooks for building, testing, and grounding trustworthy agentic AI systems on
a geometric memory substrate.

Each topic lives in its own directory with a consistent `code/` · `notebooks/`
layout. The library in `code/` is the source of truth; the notebooks are the
runnable, chapter-by-chapter walkthroughs.

| Directory | Package | Focus |
|---|---|---|
| [`beyond-prompt-and-pray/`](beyond-prompt-and-pray/) | `agentlab` | Building agentic AI systems: the agent loop, typed tools, governance, memory, evaluation, and a capstone banking agent. |
| [`beyond-ship-and-pray/`](beyond-ship-and-pray/) | `gmstest` | Testing agentic systems: DoE-driven test design, component and system attribution, resilience. |
| [`beyond-chunk-and-pray/`](beyond-chunk-and-pray/) | `book_kit` | Trustworthy RAG with geometric memory: triple-mediated retrieval, grounded synthesis, self-verification, calibrated abstention. |
| [`beyond-vibe-and-pray/`](beyond-vibe-and-pray/) | `reasonloop` | A semantic execution substrate: the LLM as compiler and grounded synthesizer over a deterministic reasoning engine. *(Work in progress.)* |

## Layout

Each topic directory contains:

- **`code/`** — the companion Python package plus supporting scripts, data, and
  tests. Install a topic editable, e.g.:

  ```bash
  python -m pip install -e beyond-prompt-and-pray/code
  ```

- **`notebooks/`** — the runnable notebooks. Each imports only from its topic
  package (and the stdlib) and ends with a self-check assertion.

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

The store data the notebooks and GMS tests use is *not* shipped — regenerate it
per topic (see below and each `code/README.md`; bcap has a `notebooks/00_setup.ipynb`).

## Model weights are not committed

Trained-model weights (`*.safetensors`, `*.pt`, large `tokenizer.json`) and
generated GMS/lab stores are **git-ignored** to keep clones lean. Regenerate them
with each topic's `code/scripts/train_*.py` / `build_*.py`, or fetch the base
models from their upstream source. See each topic's `code/README.md` for specifics.

## Development

```bash
# from a topic's code/ directory
python -m pip install -e ".[dev]"   # where a dev extra is defined
ruff check .
pytest -q
```

Tests that require the licensed `knowlytix` substrate are skipped automatically
when it is not installed.

## License

[Apache-2.0](LICENSE). The separately-licensed `knowlytix` substrate is **not**
covered by this license — see above.
