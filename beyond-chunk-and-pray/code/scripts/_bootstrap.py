# SPDX-License-Identifier: Apache-2.0
"""Make `knowlytix` resolve to the GMS-knowlytix *branch* source (latest version).

The tutorial is developed against the unpublished branch of the library, not a
released wheel. Every script and notebook prepends ``KNOWLYTIX_SRC`` to
``sys.path`` so ``import knowlytix.knowledge.rag`` picks up that working tree.

The path is **configurable**: override with the ``KNOWLYTIX_SRC`` environment
variable, or edit ``_DEFAULT`` below, when the library moves (e.g. once it ships
as a wheel, drop this bootstrap entirely).
"""

from __future__ import annotations

import os
import sys

_DEFAULT = "/home/user/jupyterlab/GMS-knowlytix"

KNOWLYTIX_SRC = os.environ.get("KNOWLYTIX_SRC", _DEFAULT)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def use_branch_library() -> str:
    """Prepend the branch library to sys.path; return the path used."""
    if KNOWLYTIX_SRC and KNOWLYTIX_SRC not in sys.path:
        sys.path.insert(0, KNOWLYTIX_SRC)
    return KNOWLYTIX_SRC


def load_store_geo(store_path, dev):
    """Load a GMSExpertStore honoring the geometry saved in model_dims.json.

    store.load() rebuilds the model from ``config.geometry`` (only the entity/
    relation counts come from model_dims.json), so a bare DocGMSConfig loads a
    store only while the library default geometry matches the stored one. Reading
    the stored geometry keeps every foundation script loadable across library
    versions (the shipped store is d=32; the current default drifted to d=128).
    """
    import json as _json
    import os as _os
    from knowlytix.knowledge.config import DocGMSConfig
    from knowlytix.knowledge.store import GMSExpertStore
    dims = _os.path.join(store_path, "model_dims.json")
    geo = None
    if _os.path.isfile(dims):
        g = _json.load(open(dims)).get("geometry")
        if g:
            from knowlytix.core.config import GeometryConfig
            geo = GeometryConfig(d_v=g["d_v"], d_u=g["d_u"], m=g["m"], d=g["d"])
    cfg = (DocGMSConfig(store_path=store_path, geometry=geo) if geo is not None
           else DocGMSConfig(store_path=store_path))
    store = GMSExpertStore(cfg, device=dev)
    assert store.load(), f"store not found at {store_path}"
    return store


# Notebooks copy these three lines into their first cell instead of importing
# this module (so a notebook is self-contained):
#
#     import os, sys
#     KNOWLYTIX_SRC = os.environ.get("KNOWLYTIX_SRC", "/home/user/jupyterlab/GMS-knowlytix")
#     sys.path.insert(0, KNOWLYTIX_SRC)
