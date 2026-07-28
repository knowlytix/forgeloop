# SPDX-License-Identifier: Apache-2.0
"""Shared capstone pipeline wiring (Chunk and Pray, Ch16).

One source of truth so the agent under test and the gate calibration use the
SAME pipeline (oracle == agent). Everything here is assembled from library
primitives and the artifacts the GEODE build wrote beside the store:

  tuned_encoder/             GMS-tuned v-encoder  (concepts-close SFT)
  contradiction_encoder/     GMS-tuned u-encoder  (contradiction SFT)
  relevance_calibration.json relevance gate operating point (v floor, u veto)
  rag_gate_calibration.json  accept/abstain operating point (written by
                             scripts/calibrate_accept_gate.py)

The store is self-describing (model_dims.json carries its geometry), so it loads
with a bare DocGMSConfig — no caller-supplied dimensions.
"""
from __future__ import annotations

import json
import os

import torch

from knowlytix.embedding import FineTunedEmbedding
from knowlytix.knowledge.config import DocGMSConfig
from knowlytix.knowledge.geode import QWEN_3B
from knowlytix.knowledge.llm_backend import LocalTransformersBackend
from knowlytix.knowledge.rag import RagConfig, RagPipeline
from knowlytix.knowledge.store import GMSExpertStore


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_store(store_path: str, dev: torch.device | None = None) -> GMSExpertStore:
    """Load the GEODE store, honoring the geometry saved in model_dims.json.

    ``store.load()`` recreates the model from ``self.config.geometry`` (only the
    entity/relation counts come from model_dims.json), so a bare DocGMSConfig
    loads correctly only while the library default geometry matches the stored
    one. When they drift (a store built at d=32 vs a later default d=128), the
    weight load fails. Read the stored geometry when present so the store stays
    loadable across library versions; fall back to the bare config otherwise.
    """
    dev = dev or device()
    dims_path = os.path.join(store_path, "model_dims.json")
    geo = None
    if os.path.isfile(dims_path):
        g = json.load(open(dims_path)).get("geometry")
        if g:
            from knowlytix.core.config import GeometryConfig
            geo = GeometryConfig(d_v=g["d_v"], d_u=g["d_u"], m=g["m"], d=g["d"])
    cfg = (DocGMSConfig(store_path=store_path, geometry=geo) if geo is not None
           else DocGMSConfig(store_path=store_path))
    store = GMSExpertStore(cfg, device=dev)
    assert store.load(), f"store not found at {store_path}"
    return store


def make_qwen(dev: torch.device | None = None) -> LocalTransformersBackend:
    """Local Qwen2.5-3B-Instruct — the default synthesis/verify/extract backend."""
    dev = dev or device()
    return LocalTransformersBackend(QWEN_3B, device=str(dev))


def _load_json(path: str, default: dict) -> dict:
    return json.load(open(path)) if os.path.isfile(path) else default


def build_rag(store: GMSExpertStore, llm, *, llm_extract=None,
              accept_threshold: float | None = None) -> RagConfig:
    """Assemble the bank-grade RagConfig from the store's tuned encoders and
    calibrated gates.

    - query parsing and the relevance v-accept run on the GMS-tuned v-encoder;
      the relevance u-veto runs on the GMS-tuned contradiction (u) encoder.
    - relevance and accept operating points are read from the persisted
      calibration files (no hand-set thresholds).
    - dense fallback off, answers self-verified, contradicted claims abstain.

    Pass ``accept_threshold=0.0`` to disable the accept gate for the calibration
    pass; otherwise the calibrated value from rag_gate_calibration.json is used.
    """
    sp = store.store_path
    v_ft = FineTunedEmbedding.load(os.path.join(sp, "tuned_encoder"))
    u_ft = FineTunedEmbedding.load(os.path.join(sp, "contradiction_encoder"))
    relcal = _load_json(os.path.join(sp, "relevance_calibration.json"), {})
    gatecal = _load_json(os.path.join(sp, "rag_gate_calibration.json"), {})

    if accept_threshold is None:
        if "accept_threshold" not in gatecal:
            raise FileNotFoundError(
                "rag_gate_calibration.json missing accept_threshold; run "
                "scripts/calibrate_accept_gate.py first (no default permitted).")
        accept_threshold = gatecal["accept_threshold"]

    return RagConfig(
        llm=llm,
        llm_extract=llm_extract or llm,
        encoder=v_ft.encode,                 # GMS-tuned v-encoder
        binding="fuzzy",
        ground_extraction=True,
        relevance_gate=True,
        relevance_mode="geometric",          # LLM-free: v accepts, u vetoes
        relevance_u_encoder=u_ft.encode,     # GMS-tuned contradiction encoder
        relevance_tau_accept=relcal.get("tau_accept", 0.30),
        relevance_tau_contra=relcal.get("default_tau_contra", 0.75),
        relevance_tau_contra_per_relation=relcal.get("tau_contra_per_relation", {}),
        verify_llm_output=True,
        on_verify_fail="abstain",
        dense_fallback=False,
        accept_threshold=accept_threshold,
    )


def build_pipeline(store: GMSExpertStore, llm, **kw) -> RagPipeline:
    return RagPipeline.from_store(store, build_rag(store, llm, **kw))
