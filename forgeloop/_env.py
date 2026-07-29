"""Environment-variable overrides, with legacy-name fallback.

The trilogy's companion code was unified into ``forgeloop``; its runtime knobs
are spelled ``FORGELOOP_*``. The pre-unification names — ``AGENTLAB_*`` (agents),
and the book bootstraps ``KNOWLYTIX_SRC`` / ``GMS_RAG_TUTORIAL`` — are still
honored as fallbacks so existing setups keep working.
"""
from __future__ import annotations

import os

_PREFIX = "FORGELOOP_"
_LEGACY_PREFIX = "AGENTLAB_"


def getenv(name: str, default: str | None = None) -> str | None:
    """Return ``FORGELOOP_<name>``, then ``AGENTLAB_<name>``, then ``default``."""
    return os.environ.get(_PREFIX + name, os.environ.get(_LEGACY_PREFIX + name, default))
