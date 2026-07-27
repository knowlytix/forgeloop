"""Data resolution anchored to the agents book (*Beyond Prompt and Pray*).

The ``forgeloop.agents`` facades ship the Prompt-and-Pray artifacts (classifier
and drafter adapters, the banking/regulatory/policy stores). Those live with the
agents book, so the facades resolve to it even when called from another book's
notebook — e.g. a Ship-and-Pray capstone test that drives the agent. This wraps
the shared resolver with that book fixed, and mirrors the ``data_path`` /
``ensure_default`` names the facades already import.
"""
from __future__ import annotations

from forgeloop._paths import book_data_root

_BOOK = "beyond-prompt-and-pray"


def data_root():
    return book_data_root(_BOOK)


def data_path(*parts: str):
    return data_root().joinpath(*parts)


def ensure_default(entry: str):
    from forgeloop.artifacts import ensure_default as _ensure_default
    return _ensure_default(entry, root=data_root())
