"""Resolution of a book's ``data/`` directory (shared across the trilogy).

The three books each keep their own ``code/data/`` directory. A notebook runs
with its working directory inside one book's ``notebooks/``, so the data it
needs is the one belonging to that book. ``data_root()`` finds it the same way
for every book:

1. an explicit override via ``FORGELOOP_DATA_DIR`` (or legacy ``AGENTLAB_DATA_DIR``),
   or a prior call to :func:`set_data_dir`;
2. discovery from the working directory, walking upward and checking ``data/``
   and ``code/data/`` at each level — so a notebook run from anywhere inside a
   book checkout finds that book's data with no configuration;
3. the editable-install fallback (``<install parent>/data``).

A candidate only counts if it contains one of the marker entries in
:data:`_MARKERS`, so an unrelated ``data/`` on the path is not chosen.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from forgeloop._env import getenv

_PKG_ROOT = Path(__file__).resolve().parent          # .../forgeloop
_INSTALL_PARENT = _PKG_ROOT.parent
# Small data shipped inside the wheel (GMS stores, calibrations, pinned results
# and the authored text inputs incl. banking_policy.md). The large model
# adapters are not here; they are fetched on demand (see forgeloop.artifacts).
_BUNDLED = _PKG_ROOT / "data"

# Entries that identify a book's data directory (union across the trilogy).
_MARKERS = (
    "policies", "eval_cases", "banking_policy_full.md", "policy_atoms.yaml",
    "annual_report.md", "gms_annual_report_store", "eval_cohort.json",
    "capstone_run.json", "capstone_companions.json",
)

_DATA_ROOT: Path | None = None


def _looks_like_data(d: Path) -> bool:
    return d.is_dir() and any((d / m).exists() for m in _MARKERS)


def _from_env() -> Path | None:
    v = getenv("DATA_DIR")
    return Path(v).expanduser().resolve() if v else None


def _discover() -> Path | None:
    if _looks_like_data(_INSTALL_PARENT / "data"):
        return _INSTALL_PARENT / "data"
    here = Path.cwd().resolve()
    for base in (here, *here.parents):
        for rel in ("data", "code/data"):
            cand = base / rel
            if _looks_like_data(cand):
                return cand
    return None


def _materialize_bundled() -> Path | None:
    """Copy the wheel-bundled data to a writable cache once, and return it.

    A pip install with no repository checkout still has the small data shipped
    in the wheel (`forgeloop/data`), but that lives under ``site-packages`` and
    is treated as read-only. Copy it to ``~/.forgeloop/data`` (override with
    ``FORGELOOP_CACHE_DIR``) so a notebook can build a store or fetch the large
    adapters into the same directory.
    """
    if not _looks_like_data(_BUNDLED):
        return None
    cache = getenv("CACHE_DIR")
    dest = (Path(cache).expanduser() if cache else Path.home() / ".forgeloop") / "data"
    if not (dest / ".forgeloop_bundled").exists():
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(_BUNDLED, dest, dirs_exist_ok=True)
        (dest / ".forgeloop_bundled").write_text("materialized from the forgeloop wheel\n")
    return dest


def data_root() -> Path:
    """Return the resolved ``data/`` directory for the current book.

    Order: explicit override, then discovery from the working directory (a repo
    checkout), then the small data bundled in the wheel (materialized to a
    writable cache), then the install-parent fallback.
    """
    global _DATA_ROOT
    if _DATA_ROOT is None:
        _DATA_ROOT = (_from_env() or _discover() or _materialize_bundled()
                      or (_INSTALL_PARENT / "data"))
    return _DATA_ROOT


def data_path(*parts: str) -> Path:
    """Join ``parts`` onto :func:`data_root`."""
    return data_root().joinpath(*parts)


def set_data_dir(path: str | os.PathLike) -> Path:
    """Override the data directory for the current process. Returns the path set."""
    global _DATA_ROOT
    _DATA_ROOT = Path(path).expanduser().resolve()
    return _DATA_ROOT


def book_data_root(book: str) -> Path:
    """Return the ``code/data`` directory of a specific book by name.

    A facade from one book (e.g. ``forgeloop.agents``) may be called from another
    book's notebook (e.g. a Ship-and-Pray capstone test that drives the Prompt
    agent). Its artifacts live with *its* book, not the caller's, so it anchors to
    that book's data regardless of the working directory. The book directory is
    located as a sibling by walking up from the working directory, then from the
    resolved :func:`data_root`; falls back to :func:`data_root` if not found.
    """
    def _hit(base: Path) -> Path | None:
        cand = base / book / "code" / "data"
        return cand if _looks_like_data(cand) else None

    here = Path.cwd().resolve()
    for base in (here, *here.parents):
        h = _hit(base)
        if h is not None:
            return h
    for anc in data_root().parents:
        h = _hit(anc)
        if h is not None:
            return h
    return data_root()
