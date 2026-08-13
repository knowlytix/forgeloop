# Contributing

Thanks for taking a look. This repo is the companion code for the *Beyond
Prompt / Ship / Chunk and Pray* trilogy, so the bar for a change is "does it
make the code a reader runs clearer or more correct".

## Before you start

Read the [README](README.md) setup section first — `make install` in a Python
3.12 venv. Most of the repo runs without the licensed `knowlytix` substrate;
the parts that need it skip cleanly rather than failing.

## The one rule that surprises people

**The `beyond-*/` book directories are the source. `forgeloop/` is generated.**

```text
beyond-*/                    ← edit here
   │  scripts/build_forgeloop.py
   ├─► forgeloop/                   the installable package (committed)
   └─► docs/4-notebook-examples/    the docs gallery (generated, git-ignored)
```

Edit `forgeloop/` by hand and the next `make assemble` silently reverts you. CI
fails the PR before that happens. Change the book file, then run
`make assemble`, then commit both.

Files with no book source (the package `__init__`) live in
[`packaging/`](packaging/).

## Checks

```bash
make install-dev    # everything, plus pytest and ruff
make check          # exactly what CI runs
```

`make check` runs the five gates: `ruff`, no committed secrets, no committed
notebook output, `forgeloop/` matching the books, and the tests. Green locally
means green in CI.

Three of those catch mistakes that are easy to make:

- **Secrets.** `python scripts/secret_scan.py` looks for API tokens, licence
  keys and absolute paths off your machine. If it fires on a real credential,
  **rotate it** — dropping the commit does not un-leak it.
- **Notebook hygiene.** Never commit executed cells — it bloats diffs and
  re-renders stale results on GitHub. The same gate rejects a notebook saved
  against a kernel only your machine has (readers get "kernel not found" before
  they read a line). Fix both with
  `python scripts/notebook_outputs.py --strip`.
- **Assembly drift.** Fix with `python scripts/build_forgeloop.py`.

## What not to commit

The [`.gitignore`](.gitignore) covers these, but they are worth stating, because
this repo is public:

- **No secrets.** No license keys, `.env` files, API tokens, or anything from
  `~/.knowlytix/`. See [SECURITY.md](SECURITY.md).
- **No trained weights or generated stores.** `*.safetensors`, `*.pt`, adapters,
  GMS stores. They are rebuilt from committed corpora by `00_setup.ipynb` and
  the `scripts/train_*.py`.
- **No book manuscript material.** LaTeX, `.docx` and slides from the source
  repo are deliberately not here.
- **No real data.** Every corpus in `data/` is synthetic and stays that way.

## Style

- Notebooks stay thin: a notebook imports from its book's package, it does not
  reimplement package logic. If a notebook needs new logic, add it to the
  package.
- Reach for the existing `knowlytix` / book-package API before writing a new
  helper — retrieval, calibration, evaluation and comparison utilities mostly
  exist already.
- `ruff check .` must pass. `mypy` is advisory for now.

## Pull requests

Describe what a reader gains from the change. If it touches a notebook, say
which chapter and confirm you stripped the output. Small, focused PRs get read
faster than large ones.

By contributing you agree your contribution is licensed under
[Apache-2.0](LICENSE).
