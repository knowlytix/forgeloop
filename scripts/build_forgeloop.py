"""Assemble the `forgeloop` package and the docs notebook gallery from the books.

The three book directories under ``beyond-*/`` are the source of truth. This
script produces two distributions of that same material:

  forgeloop/                  the pip package  (``--package``)
  docs/4-notebook-examples/   the Sphinx gallery (``--docs``)

Both are committed build artifacts. ``--check`` re-assembles into a temporary
directory and diffs against what is committed, so CI fails if either drifts
from the books. Run with no flags to build everything.

Import statements are rewritten book-local -> package (``agentlab`` ->
``forgeloop.agents``). Prose is deliberately left alone: a docstring that
mentions "the agentlab library" reads the same in both distributions, and
rewriting prose would churn the diff for no benefit.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- source -> package ------------------------------------------------------
# (book path, path under forgeloop/). Order matters only for readability.
PACKAGE_TREES = [
    ("beyond-prompt-and-pray/code/agentlab", "agents"),
    ("beyond-ship-and-pray/code/gmstest", "testing"),
    ("beyond-ship-and-pray/code/apps", "apps"),
    ("beyond-ship-and-pray/code/catalogs", "catalogs"),
]

# Scaffold that lives inside agentlab for the book's sake but belongs at the
# package root. Moving these up is why the package needs an assembler at all.
PACKAGE_MOVES = [
    ("beyond-prompt-and-pray/code/agentlab/_env.py", "_env.py"),
    ("beyond-prompt-and-pray/code/agentlab/_datapaths.py", "_paths.py"),
    ("beyond-prompt-and-pray/code/agentlab/artifacts.py", "artifacts.py"),
]

# Files with no book source at all; copied verbatim from packaging/.
SCAFFOLD = "packaging/forgeloop"

# Authored corpora and fixtures the package bundles. Enumerated via git, so the
# gitignored store build outputs sitting in the same directories cannot leak in.
DATA_DIRS = [
    "beyond-prompt-and-pray/code/data",
    "beyond-chunk-and-pray/code/data",
    # Ship-and-Pray owns the pinned capstone campaign results, which the
    # analysis notebooks in both books read.
    "beyond-ship-and-pray/code/data",
]

# Book notebooks -> package notebooks and docs gallery sections.
BOOKS = [
    ("beyond-prompt-and-pray", "agents", "Agent Builder (Beyond Prompt and Pray)"),
    ("beyond-ship-and-pray", "testing", "Testing (Beyond Ship and Pray)"),
    ("beyond-chunk-and-pray", "rag", "Governed RAG (Beyond Chunk and Pray)"),
]

# Longest first so agentlab._paths is not shadowed by a shorter prefix.
IMPORT_SUBS = [
    ("agentlab._datapaths", "forgeloop._paths"),
    ("agentlab.artifacts", "forgeloop.artifacts"),
    ("agentlab._env", "forgeloop._env"),
    ("agentlab", "forgeloop.agents"),
    ("gmstest", "forgeloop.testing"),
    ("book_kit", "forgeloop.rag"),
    ("apps", "forgeloop.apps"),
    ("catalogs", "forgeloop.catalogs"),
]

SKIP_PARTS = {"__pycache__", ".ipynb_checkpoints"}


def rewrite_imports(text: str) -> str:
    """Rewrite import statements only, leaving prose untouched.

    No ``\\b`` before ``from``/``import``: inside a single-string .ipynb cell the
    statement is preceded by a literal backslash-n, whose "n" is a word
    character, so a word-boundary assertion would silently skip it.
    """
    for old, new in IMPORT_SUBS:
        o = re.escape(old)
        text = re.sub(rf"(?<=from ){o}(?=[\s.,)]|\\n|$)", new, text)
        text = re.sub(rf"(?<=import ){o}(?=[\s.,)]|\\n|$)", new, text)
    return text


def _iter(src: Path):
    for p in sorted(src.rglob("*")):
        if p.is_dir() or SKIP_PARTS & set(p.parts):
            continue
        yield p


def _emit(src: Path, dst: Path, rewrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if rewrite and src.suffix in (".py", ".ipynb"):
        dst.write_text(rewrite_imports(src.read_text(encoding="utf-8")), encoding="utf-8")
    else:
        shutil.copy2(src, dst)


def build_package(out: Path) -> int:
    n = 0
    # Scaffold that PACKAGE_MOVES lifts to the package root must not also be
    # copied in place as part of the agentlab tree, or it ships twice.
    moved = {Path(src).name for src, _ in PACKAGE_MOVES}
    for book_rel, pkg_rel in PACKAGE_TREES:
        src = ROOT / book_rel
        for p in _iter(src):
            if p.parent == src and p.name in moved:
                continue
            _emit(p, out / pkg_rel / p.relative_to(src), rewrite=True)
            n += 1
    for src_rel, dst_rel in PACKAGE_MOVES:
        _emit(ROOT / src_rel, out / dst_rel, rewrite=True)
        n += 1
    # agentlab/_paths.py is the book-anchored wrapper -> forgeloop/agents/_paths.py
    # (already carried by the agents tree). Scaffold with no book source:
    scaffold = ROOT / SCAFFOLD
    for p in _iter(scaffold):
        _emit(p, out / p.relative_to(scaffold), rewrite=False)
        n += 1
    # Packages that need an __init__ the book does not carry.
    for pkg in ("apps", "catalogs"):
        init = out / pkg / "__init__.py"
        if not init.exists():
            init.write_text(f'"""Bundled {pkg} shipped with forgeloop."""\n', encoding="utf-8")
            n += 1
    # Authored data, tracked files only.
    for d in DATA_DIRS:
        src = ROOT / d
        if not src.exists():
            continue
        listed = subprocess.run(
            ["git", "ls-files", d], cwd=ROOT, capture_output=True, text=True
        ).stdout.split()
        for rel in listed:
            p = ROOT / rel
            if p.is_file():
                _emit(p, out / "data" / p.relative_to(src), rewrite=False)
                n += 1
    # Marker naming the book the bundled data belongs to; read by artifacts.py
    # to anchor data_root() when the package is installed rather than checked out.
    marker = out / "data" / ".forgeloop_book"
    if marker.parent.exists():
        marker.write_text("beyond-prompt-and-pray\n", encoding="utf-8")
        n += 1
    # Notebooks, one directory per book.
    for book, _section, _title in BOOKS:
        src = ROOT / book / "notebooks"
        if not src.exists():
            continue
        for p in _iter(src):
            _emit(p, out / "notebooks" / book / p.relative_to(src), rewrite=True)
            n += 1
    return n


def _ensure_title(nb_path: Path, fallback: str) -> None:
    """Sphinx needs an H1; give a notebook one if its first cell lacks it."""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])
    first = "".join(cells[0].get("source", [])) if cells else ""
    if first.lstrip().startswith("# "):
        return
    cells.insert(0, {"cell_type": "markdown", "metadata": {}, "source": f"# {fallback}\n"})
    nb["cells"] = cells
    nb_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")


def build_docs(out: Path) -> int:
    n = 0
    for book, section, title in BOOKS:
        src = ROOT / book / "notebooks"
        if not src.exists():
            continue
        dst = out / section
        names = []
        for p in sorted(src.glob("*.ipynb")):
            _emit(p, dst / p.name, rewrite=True)
            pretty = p.stem.replace("_", " ").title()
            _ensure_title(dst / p.name, pretty)
            names.append(p.stem)
            n += 1
        toc = "\n".join(f"   {x}" for x in sorted(names))
        (dst / "index.rst").write_text(
            f"{title}\n{'=' * len(title)}\n\n"
            "Generated from the book notebooks by ``scripts/build_forgeloop.py``.\n"
            "Edit the notebooks under the book directory, not these copies.\n\n"
            ".. toctree::\n   :maxdepth: 1\n\n" + toc + "\n",
            encoding="utf-8",
        )
        n += 1
    return n


def _diff(a: Path, b: Path) -> list[str]:
    """Report paths that differ between a freshly built tree and the committed one."""
    out: list[str] = []
    built = {p.relative_to(a) for p in a.rglob("*") if p.is_file()}
    have = {
        p.relative_to(b)
        for p in b.rglob("*")
        if p.is_file() and not SKIP_PARTS & set(p.parts)
    }
    for rel in sorted(built - have):
        out.append(f"  missing from repo: {rel}")
    for rel in sorted(have - built):
        out.append(f"  stale in repo:     {rel}")
    for rel in sorted(built & have):
        if not filecmp.cmp(a / rel, b / rel, shallow=False):
            out.append(f"  differs:           {rel}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", action="store_true", help="build forgeloop/ only")
    ap.add_argument("--docs", action="store_true", help="build the docs gallery only")
    ap.add_argument("--check", action="store_true", help="verify committed output matches the books")
    args = ap.parse_args()
    do_pkg = args.package or not args.docs
    do_docs = args.docs or not args.package

    if args.check:
        import tempfile

        problems = []
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            if do_pkg:
                build_package(tmp / "forgeloop")
                problems += [f"forgeloop/: {p}" for p in _diff(tmp / "forgeloop", ROOT / "forgeloop")]
            if do_docs:
                build_docs(tmp / "docs")
                problems += [
                    f"docs/4-notebook-examples/: {p}"
                    for p in _diff(tmp / "docs", ROOT / "docs/4-notebook-examples")
                ]
        if problems:
            print("DRIFT: committed output does not match the books.\n")
            for p in problems[:40]:
                print(p)
            if len(problems) > 40:
                print(f"  ... and {len(problems) - 40} more")
            print("\nRun `python scripts/build_forgeloop.py` and commit the result.")
            return 1
        print("ok: committed output matches the books")
        return 0

    if do_pkg:
        target = ROOT / "forgeloop"
        if target.exists():
            shutil.rmtree(target)
        print(f"forgeloop/: {build_package(target)} files")
    if do_docs:
        target = ROOT / "docs/4-notebook-examples"
        if target.exists():
            shutil.rmtree(target)
        print(f"docs/4-notebook-examples/: {build_docs(target)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
