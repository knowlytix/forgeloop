# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

**forgeloop** — open-source companion code and runnable notebooks for building,
testing, and grounding trustworthy agentic AI systems on the Geometric Memory
Systems (GMS) substrate. It is a monorepo of independent topic directories, each
with its own `code/` package and `notebooks/`. Ported from the private
`GMS-Agents` repo; the book manuscripts are intentionally excluded.

> **This repo is intended to become open source.** Never commit secrets,
> credentials, or large trained-model weights (see the Hard rules below).

## Repository map

| Directory | Package | Focus |
|---|---|---|
| `beyond-prompt-and-pray/` | `agentlab` | Agent loop, typed tools, governance, memory, evaluation, capstone banking agent. |
| `beyond-ship-and-pray/` | `gmstest` | DoE-driven test design, component/system attribution, resilience. |
| `beyond-chunk-and-pray/` | `book_kit` (+ `knowlytix`) | Trustworthy RAG with geometric memory. |
| `beyond-vibe-and-pray/` | `reasonloop` | Semantic execution substrate — LLM as compiler + grounded synthesizer. WIP. |

Each topic: `code/` (the package, `scripts/`, `data/`, `tests/`) is the source of
truth; `notebooks/` are thin runnable walkthroughs that import only from the topic
package and the stdlib.

## Reach for `knowlytix` / the topic package first

Before writing new code, search the existing `knowlytix` APIs and the topic's own
package layer (e.g. `agentlab`, which wraps `knowlytix`) for an equivalent — for
retrieval, graph queries and traversal, calibration, evaluation, and
comparison/benchmark utilities. A notebook or `scripts/` file should be a *thin
caller* of a package API, not a reimplementation of its logic. Only implement
something new when the package does not already provide it; when it's genuinely
needed, add it to the package rather than leaving a one-off in a script.

## The `knowlytix` substrate (licensed, separate)

The GMS-backed features run on the **`knowlytix`** package, which is **licensed
and distributed separately** — not on public PyPI. Do not vendor it, commit it, or
hardcode license keys. Code that needs it should degrade or skip cleanly when it
is absent (tests guard the import — see below).

## Development workflow

Per topic, from its `code/` directory:

```bash
python -m pip install -e ".[dev]"   # where a dev extra exists; else -e .
ruff check .
pytest -q
```

- **Lint:** `ruff check .` should pass across the repo.
- **Types:** `mypy` is advisory (non-blocking) for now.
- **Tests:** `pytest`. Tests that require `knowlytix` must be skipped when it is
  not importable (e.g. `pytest.importorskip("knowlytix")`) so the suite is green
  without the proprietary substrate.

CI (`.github/workflows/ci.yml`) runs ruff across the repo and mypy/pytest per
topic package with CPU-only torch.

## Hard rules

- **Never commit trained-model weights** (`*.safetensors`, `*.pt`, `*.pth`, large
  `tokenizer.json`) or generated GMS/lab/agent stores. They are git-ignored;
  regenerate via `scripts/train_*.py` / `build_*.py` or fetch base models from
  upstream.
- **Never commit secrets.** Git-ignored: `.env*` (except `.env.example`),
  `*.pem`, `*.key`, `*.lic`, `credentials.json`, `service-account-*.json`,
  `.aws/`, `.gcp/`, `.ssh/`, `id_rsa*`, `id_ed25519*`. Use `.env.example` with
  `YOUR_KEY_HERE` placeholders.
- **No book material.** The `book/` trees (LaTeX/`.docx`/slides/substack) and
  book-authoring tooling from the source repo are intentionally excluded. Don't
  re-introduce manuscript content here.
- **Keep notebooks thin.** A notebook imports from its topic package; it does not
  reimplement package logic.
