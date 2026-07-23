# SPDX-License-Identifier: Apache-2.0
"""Shared notebook helper: knowlytix bootstrap, repo-root paths and a store
loader that uses the *exact* build-time config.

Notebooks import this so that:
  * `import knowlytix` resolves (the licensed substrate; pip-installed or a checkout),
  * data paths resolve from the repo root (not the notebook's cwd), and
  * the trained store loads with the geometry/cap it was built with
    (GMSExpertStore.load rebuilds the model from config, so the config must
    match scripts/build_store.py or the state_dict will not load).

The store data itself is never committed or packaged — regenerate it with
`notebooks/00_setup.ipynb` (which drives scripts/build_store.py and the rest of
the pipeline). load_store() raises with that pointer when the store is absent.

Locations are configurable via environment variables:
  KNOWLYTIX_SRC     -> the GMS-knowlytix source checkout (auto-detected if unset)
  GMS_RAG_TUTORIAL  -> this repo's code/ dir (defaults to this file's dir)
"""

from __future__ import annotations

import importlib.util
import os
import sys

REPO = os.environ.get(
    "GMS_RAG_TUTORIAL", os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

CORPUS = os.path.join(REPO, "data", "annual_report.md")
STORE = os.path.join(REPO, "data", "gms_annual_report_store")
EVAL_COHORT = os.path.join(REPO, "data", "eval_cohort.json")


def resolve_knowlytix() -> str | None:
    """Make ``import knowlytix`` work; return the path added to ``sys.path`` (or
    None if it was already importable).

    The GMS substrate installs from PyPI (``pip install knowlytix``) but may also
    live in a separate source checkout. If it is already importable we do nothing;
    otherwise we honor ``$KNOWLYTIX_SRC`` first, then a few common checkout
    locations. When found, we also export ``KNOWLYTIX_SRC`` so child build scripts
    (scripts/_bootstrap.py) inherit the same path. Raises with guidance when it
    cannot be located.
    """
    if importlib.util.find_spec("knowlytix") is not None:
        return None
    candidates = [
        os.environ.get("KNOWLYTIX_SRC"),
        os.path.expanduser("~/source/GMS-knowlytix"),
        os.path.expanduser("~/GMS-knowlytix"),
        os.path.normpath(os.path.join(REPO, "..", "..", "GMS-knowlytix")),
    ]
    for c in candidates:
        if c and os.path.isdir(os.path.join(c, "knowlytix")):
            if c not in sys.path:
                sys.path.insert(0, c)
            os.environ["KNOWLYTIX_SRC"] = c
            return c
    raise ModuleNotFoundError(
        "The licensed `knowlytix` substrate was not found. Install it with "
        "`pip install knowlytix` (it needs a license key at ~/.knowlytix/"
        "license.key; sign up at https://knowlytix.ai/signup/), or point "
        "KNOWLYTIX_SRC at a source checkout and restart the kernel:\n"
        "    import os; os.environ['KNOWLYTIX_SRC'] = '/path/to/GMS-knowlytix'\n"
        "See the repo README (\"The knowlytix substrate\")."
    )


resolve_knowlytix()

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
            f"no trained store at {STORE}. The store data is not committed or "
            "packaged — build it by running notebooks/00_setup.ipynb (or "
            "`python scripts/build_store.py`), then re-run this cell."
        )
    return store
