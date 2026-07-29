"""Access to the optional, licensed GMS backend (``knowlytix``).

The GMS-backed features of this library --- the geometric plausibility gate, the
regulatory guard, the policy Graph-RAG store, Exact Numerical Memory and the DOE
test harness --- run on the ``knowlytix`` package. ``knowlytix`` is licensed and
distributed separately (it is not on public PyPI); a license is required.

Knowlytix: https://knowlytix.ai/

Install it from the Knowlytix index, then ``agentlab`` picks it up automatically
(the GMS imports throughout the library are lazy, so everything outside those
features works without it)::

    pip install knowlytix --index-url <KNOWLYTIX_INDEX_URL>   # license required
    pip install "agentlab[gms]"                               # records the dependency

Use :func:`available` to branch on whether the backend is installed, or
:func:`require` to get a clear, actionable error when it is missing.
"""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType

#: Canonical reference for obtaining the licensed backend.
KNOWLYTIX_URL = "https://knowlytix.ai/"

#: Human-readable install pointer, reused in error messages and the docs.
INSTALL_HINT = (
    "GMS features require the licensed 'knowlytix' package, which is installed "
    "separately and is not on public PyPI.\n"
    f"  Obtain a license and the package index from Knowlytix: {KNOWLYTIX_URL}\n"
    "  pip install knowlytix --index-url <KNOWLYTIX_INDEX_URL>   # license required\n"
    "  pip install 'agentlab[gms]'\n"
    "See the README section 'GMS / Knowlytix (licensed)'."
)


def available() -> bool:
    """Return ``True`` if the licensed ``knowlytix`` backend is importable."""
    return importlib.util.find_spec("knowlytix") is not None


def require() -> ModuleType:
    """Import and return ``knowlytix``, or raise a clear error pointing to it.

    Raises:
        ImportError: with :data:`INSTALL_HINT` when ``knowlytix`` is not installed.
    """
    try:
        return importlib.import_module("knowlytix")
    except ImportError as exc:  # pragma: no cover - exercised only without the backend
        raise ImportError(INSTALL_HINT) from exc
