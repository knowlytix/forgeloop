"""Shared pytest configuration and fixtures.

The `gms_store` fixture loads the trained GMS store from `data/gms_demo_store`
(built once from `data/model_risk_assessment.md`). Tests that exercise the
GMS-backed adapters use this fixture instead of mocks.

If the GMSH/docgms libraries are not installed, GMS-backed tests are skipped.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
STORE_PATH = REPO_ROOT / "data" / "gms_demo_store"


@pytest.fixture(scope="session")
def gms_store():
    """Load the persisted demo GMS store. Session-scoped so we load once."""
    try:
        import torch
        from docgms.config import DocGMSConfig
        from docgms.store import GMSExpertStore
    except ImportError:
        pytest.skip("docgms not installed")
    if not STORE_PATH.exists():
        pytest.skip(f"GMS demo store not found at {STORE_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = DocGMSConfig(store_path=str(STORE_PATH))
    store = GMSExpertStore(config, device=device)
    store.load()
    return store
