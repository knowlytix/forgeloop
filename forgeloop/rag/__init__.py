# SPDX-License-Identifier: Apache-2.0
"""Shared notebook helper: repo-root paths and a store loader that uses the
*exact* build-time config. knowlytix is used from the installed wheel.

Importing this gives notebooks:
  * data paths resolved through ``forgeloop.data_root`` (not the notebook's cwd), and
  * a store loader that rebuilds the model with the geometry/cap it was built with
    (GMSExpertStore.load rebuilds the model from config, so the config must
    match scripts/build_store.py or the state_dict will not load).

Optional environment overrides (developer use only):
  KNOWLYTIX_SRC       -> a local knowlytix checkout to prepend to sys.path
  GMS_RAG_TUTORIAL    -> a specific repo data root (defaults to data_root())
"""

from __future__ import annotations

import os
import sys

from forgeloop._paths import data_root

# Optional: a local knowlytix source checkout on sys.path (dev only). The
# installed knowlytix wheel is used when this is unset.
KNOWLYTIX_SRC = os.environ.get("KNOWLYTIX_SRC")
if KNOWLYTIX_SRC and KNOWLYTIX_SRC not in sys.path:
    sys.path.insert(0, KNOWLYTIX_SRC)

# Data resolves through the shared resolver (this book's code/data), overridable
# with FORGELOOP_DATA_DIR; GMS_RAG_TUTORIAL still forces a specific repo data dir.
_REPO_OVERRIDE = os.environ.get("GMS_RAG_TUTORIAL")
_DATA = os.path.join(_REPO_OVERRIDE, "data") if _REPO_OVERRIDE else str(data_root())
CORPUS = os.path.join(_DATA, "annual_report.md")
STORE = os.path.join(_DATA, "gms_annual_report_store")
EVAL_COHORT = os.path.join(_DATA, "eval_cohort.json")

import torch  # noqa: E402
from knowlytix.core.config import GeometryConfig, TrainConfig  # noqa: E402
from knowlytix.knowledge.config import DocGMSConfig  # noqa: E402
from knowlytix.knowledge.store import GMSExpertStore  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def store_config() -> DocGMSConfig:
    """The exact config scripts/build_store.py trained the store with."""
    return DocGMSConfig(
        store_path=STORE,
        ingest_mode="regex",
        loss_mode="cap",
        geometry=GeometryConfig(d_v=64, d_u=64, m=32, d=32),
        train=TrainConfig(epochs=150, batch_size=64, neg_samples=16,
                          lr=5e-3, lr_riemannian=2e-3),
    )


def load_store(device=None) -> GMSExpertStore:
    """Load the trained annual-report store (correct shape, ready to query)."""
    store = GMSExpertStore(store_config(), device or DEVICE)
    if not store.load():
        raise FileNotFoundError(
            f"no trained store at {STORE}; run `python scripts/build_store.py`")
    return store
