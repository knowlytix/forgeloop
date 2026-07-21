"""Manifold flag scorer: derive regulatory flags by reading the message
geometrically, instead of regex/LLM evidence extraction.

A flag fires when the customer message, projected into the GMS semantic (v)
space, falls inside that flag's ``has_evidence`` spherical cap --- i.e. the
geodesic distance from the message to the operator-transported flag center is
within a calibrated threshold. This replaces the brittle text->entity projection
(``evidence_in_message``) with the manifold itself, which is why it recovers the
UDAAP cases the alias lexicon missed without over-firing.

Requires the rebuilt cap store artifact (``data/gms_regulatory_cap/``: MiniLM/MNLI
warm-start + cap loss + frozen embeddings), built by
``scripts/eval_manifold_escalation.py``. Degrades safely: if the artifact or its
encoder is unavailable, :meth:`load` raises and the guard falls back to its
regex/hybrid evidence path.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ARTIFACT = _REPO_ROOT / "data" / "gms_regulatory_cap"
_FLAG_TO_ENTITY = {"UDAAP": "udaap", "Reg_E": "reg_e", "Reg_Z": "reg_z", "Reg_X": "reg_x", "FCRA": "fcra"}


def _match(v: torch.Tensor, d: int) -> torch.Tensor:
    return v if v.shape[1] == d else (v[:, :d] if v.shape[1] > d else F.pad(v, (0, d - v.shape[1])))


@dataclass
class ManifoldFlagScorer:
    model: Any
    device: Any
    d_v: int
    v_model: str
    centers: dict[str, torch.Tensor]              # flag -> normalized cap center (m,)
    thresholds: dict[str, float]                  # flag -> calibrated geodesic tau
    adapter: Any = None                           # low-rank SFT embedding adapter (or None)
    _enc_cache: dict[str, torch.Tensor] = field(default_factory=dict, repr=False)
    _encoder: Any = field(default=None, repr=False)   # cached (tokenizer, model)

    @classmethod
    def load(cls, artifact_dir: Path | None = None, device=None) -> "ManifoldFlagScorer":
        from knowlytix.core.config import GeometryConfig
        from knowlytix.core.graph.gkg import GeometricKnowledgeGraph

        art = Path(artifact_dir) if artifact_dir is not None else _DEFAULT_ARTIFACT
        meta = json.loads((art / "meta.json").read_text())
        # The flag geometry (a MiniLM encode, the low-rank adapter's Cayley solve,
        # and a Stiefel projection) is small and numerically delicate. On a GPU
        # whose compute capability exceeds the installed PyTorch build's support
        # (e.g. a GB10, cuda 12.1 > torch max 12.0) it can silently miscompute
        # per-process, flipping a flag's geodesic and firing a spurious escalation.
        # Default this path to CPU --- it is cheap (one short message at a time)
        # and deterministic; ``AGENTLAB_MANIFOLD_DEVICE=cuda`` opts back in.
        if device is None:
            env = os.environ.get("AGENTLAB_MANIFOLD_DEVICE", "cpu").lower()
            device = torch.device("cuda" if env == "cuda" and torch.cuda.is_available() else "cpu")

        cfg = GeometryConfig(d_v=meta["d_v"], d_u=meta["d_u"], m=meta["m"], d=meta["d"])
        model = GeometricKnowledgeGraph(
            num_entities=meta["num_entities"], num_relations=meta["num_relations"],
            cfg=cfg, cap_enabled=True, cap_use_diag=meta.get("cap_use_diag", True),
        ).to(device)
        model.load_state_dict(torch.load(art / "model.pt", map_location=device))
        model.eval()

        e2i, r2i = meta["entity_to_idx"], meta["relation_to_idx"]
        rev = r2i["has_evidence"]
        centers: dict[str, torch.Tensor] = {}
        for flag in meta["flags"]:
            ent = _FLAG_TO_ENTITY.get(flag)
            if ent in e2i:
                with torch.no_grad():
                    c = model.cap_center(torch.tensor([e2i[ent]], device=device),
                                         torch.tensor([rev], device=device))
                centers[flag] = F.normalize(c, dim=-1).squeeze(0)

        # Calibrated per-flag thresholds (calibration.json) override the default.
        calib = {}
        cpath = art / "calibration.json"
        if cpath.exists():
            calib = json.loads(cpath.read_text()).get("flag_thresholds", {})
        default_tau = float(meta.get("cap_radius_has_evidence", 0.0)) + meta.get("default_margin", 0.30)
        thresholds = {f: float(calib.get(f, default_tau)) for f in centers}

        # Optional low-rank SFT embedding adapter: messages traverse the same
        # fine-tuned geometry as the inserted entities. Absent in legacy artifacts.
        adapter = None
        aspec = meta.get("adapter")
        if aspec and (art / "adapter.pt").exists():
            from knowlytix.embedding.adapter import LowRankEmbeddingAdapter
            adapter = LowRankEmbeddingAdapter(
                aspec["d_in"], rank=aspec["rank"], mode=aspec["mode"],
                out_dim=aspec["out_dim"],
            ).to(device)
            adapter.load_state_dict(torch.load(art / "adapter.pt", map_location=device))
            adapter.eval()

        return cls(model=model, device=device, d_v=meta["d_v"], v_model=meta["v_model"],
                   centers=centers, thresholds=thresholds, adapter=adapter)

    def _embed(self, texts: list[str]) -> torch.Tensor:
        """Project texts into the trained v-sphere (cached encoder)."""
        if self._encoder is None:
            from transformers import AutoModel, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(self.v_model)
            mdl = AutoModel.from_pretrained(self.v_model).to(self.device).eval()
            self._encoder = (tok, mdl)
        tok, mdl = self._encoder
        embs = []
        for i in range(0, len(texts), 64):
            b = tok(texts[i:i + 64], padding=True, truncation=True, max_length=128,
                    return_tensors="pt").to(self.device)
            with torch.no_grad():
                h = mdl(**b).last_hidden_state
                m = b["attention_mask"].unsqueeze(-1).float()
                e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
                embs.append(F.normalize(e, p=2, dim=-1))
        raw = torch.cat(embs, 0)                      # (n, encoder_dim)
        if self.adapter is not None:                  # SFT: encoder -> tuned out_dim
            with torch.no_grad():
                raw = self.adapter(raw.to(self.device))
        raw = _match(raw, self.d_v)                   # no-op when out_dim == d_v
        return F.normalize(raw @ self.model.dual_emb.P_v.detach().T, dim=-1)   # (n, m)

    def _distances(self, mv: torch.Tensor) -> dict[str, list[float]]:
        out = {}
        for flag, c in self.centers.items():
            dot = (c.unsqueeze(0) * mv).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)
            out[flag] = torch.arccos(dot).cpu().tolist()
        return out

    def flag_distances(self, message: str) -> dict[str, float]:
        """Geodesic distance from the message to each flag's has_evidence cap center."""
        if message not in self._enc_cache:
            self._enc_cache[message] = self._embed([message]).squeeze(0)
        mv = self._enc_cache[message].unsqueeze(0)
        return {f: d[0] for f, d in self._distances(mv).items()}

    def flag_distances_batch(self, messages: list[str]) -> dict[str, list[float]]:
        """Per-flag geodesic distances for many messages (one encoder pass)."""
        return self._distances(self._embed(messages))

    def flags_for_message(self, message: str) -> set[str]:
        """Flags whose cap admits the message (distance within calibrated threshold)."""
        d = self.flag_distances(message)
        return {f for f, dist in d.items() if dist <= self.thresholds[f]}


_DEFAULT_SCORER: ManifoldFlagScorer | None = None
_LOAD_FAILED = False


def get_default_scorer() -> ManifoldFlagScorer | None:
    """Cached scorer, or None if the cap artifact is unavailable (degrade-safe)."""
    global _DEFAULT_SCORER, _LOAD_FAILED
    if _DEFAULT_SCORER is None and not _LOAD_FAILED:
        try:
            _DEFAULT_SCORER = ManifoldFlagScorer.load()
        except Exception:
            _LOAD_FAILED = True
            return None
    return _DEFAULT_SCORER
