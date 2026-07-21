# SPDX-License-Identifier: Apache-2.0
"""Shared notebook helper: branch-library bootstrap, repo-root paths and a
store loader that uses the *exact* build-time config.

Notebooks import this after the KNOWLYTIX_SRC bootstrap so that:
  * data paths resolve from the repo root (not the notebook's cwd), and
  * the trained store loads with the geometry/cap it was built with
    (GMSExpertStore.load rebuilds the model from config, so the config must
    match scripts/build_store.py or the state_dict will not load).

Both locations are configurable via environment variables:
  KNOWLYTIX_SRC       -> the GMS-knowlytix branch library (default below)
  GMS_RAG_TUTORIAL    -> this repo (default below)
"""

from __future__ import annotations

import os
import sys

KNOWLYTIX_SRC = os.environ.get(
    "KNOWLYTIX_SRC", "/home/user/jupyterlab/GMS-knowlytix")
REPO = os.environ.get(
    "GMS_RAG_TUTORIAL", os.path.dirname(os.path.abspath(__file__)))
for _p in (KNOWLYTIX_SRC, REPO):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

CORPUS = os.path.join(REPO, "data", "annual_report.md")
STORE = os.path.join(REPO, "data", "gms_annual_report_store")
EVAL_COHORT = os.path.join(REPO, "data", "eval_cohort.json")

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
