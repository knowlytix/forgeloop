"""Windowed manifold entity extractor: a deterministic, context-sensitive third
path alongside regex and the LLM.

Where :class:`~agentlab.capstone.manifold_evidence.ManifoldFlagScorer` asks
"does the *whole message* fall inside a *flag's* has_evidence cap", this asks the
finer question "which *evidence entities* does a *local span* of the message
carry". It slides a window over the text, projects each window through the same
SFT-tuned embedding into the GMS semantic (v) sphere, and reports an entity when
*any* window lands within a calibrated geodesic radius of that entity's own
learned v-point (``dual_emb.project_v``). Max-pooling over windows is what makes
it context-sensitive: a bare "$35" only fires ``hidden_fee`` when the surrounding
window also carries the undisclosed-charge meaning, not when it reads "$35 to the
merchant".

It is the manifold analogue of regex/LLM extraction and exposes the same
``extract(text) -> set[str]`` contract as
:class:`~agentlab.extraction.hybrid_entities.HybridEntityExtractor`, so it drops
into the hybrid as a deterministic alternative to (or backstop for) the LLM half.

Reuses the loaded model, adapter and encoder of ``ManifoldFlagScorer`` verbatim
for embedding --- only the *targets* differ (entity v-points here, transported
flag caps there). Degrades safely: if the cap artifact or its encoder is
unavailable, :meth:`load` raises and the caller falls back to regex/LLM.

Per-entity thresholds should be *calibrated* against labeled data
(:meth:`calibrate`); the un-calibrated default is for smoke tests only and will
emit a warning.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from agentlab.capstone.manifold_evidence import _DEFAULT_ARTIFACT, ManifoldFlagScorer

# Structural / taxonomy nodes in the store that are not extractable evidence
# entities (catalog roots, severity levels, action verbs, relation-ish helpers).
_NON_EVIDENCE = {
    "flag_catalog", "flag_evidence", "flag_severity", "evidence_aliases",
    "severity_action", "high", "standard", "escalate",
}
_DEFAULT_TAU = 1.0  # geodesic radians; smoke-test only --- calibrate per entity


@dataclass
class ManifoldEntityExtractor:
    """Extract canonical evidence entities a text carries, geometrically.

    Args:
        embedder: a loaded :class:`ManifoldFlagScorer`, reused purely for its
            ``_embed`` (encoder + SFT adapter + v-projection). Not consulted for
            its flag caps.
        centers: entity name -> unit v-point on the m-sphere (``project_v``).
        thresholds: entity name -> calibrated geodesic radius (max distance to fire).
        window_size: window length in whitespace tokens.
        window_stride: step between window starts in tokens.
    """

    embedder: ManifoldFlagScorer
    centers: dict[str, torch.Tensor]
    thresholds: dict[str, float]
    window_size: int = 12
    window_stride: int = 6
    _matrix: torch.Tensor = field(default=None, repr=False)   # (E, m) stacked centers
    _names: list[str] = field(default=None, repr=False)        # row order of _matrix

    def __post_init__(self) -> None:
        self._names = list(self.centers)
        self._matrix = torch.stack([self.centers[n] for n in self._names], dim=0)

    @classmethod
    def load(
        cls,
        entities: Iterable[str] | None = None,
        artifact_dir: Path | None = None,
        device=None,
        *,
        default_tau: float = _DEFAULT_TAU,
        window_size: int = 12,
        window_stride: int = 6,
    ) -> ManifoldEntityExtractor:
        """Build from the cap artifact, reusing ``ManifoldFlagScorer`` for embedding.

        ``entities`` selects the target vocabulary (default: every store entity
        except the structural taxonomy nodes). Per-entity thresholds come from
        ``entity_calibration.json`` in the artifact when present, else ``default_tau``.
        """
        art = Path(artifact_dir) if artifact_dir is not None else _DEFAULT_ARTIFACT
        embedder = ManifoldFlagScorer.load(art, device=device)
        meta = json.loads((art / "meta.json").read_text())
        e2i = meta["entity_to_idx"]

        vocab = list(entities) if entities is not None else [
            e for e in e2i if e not in _NON_EVIDENCE
        ]
        vocab = [e for e in vocab if e in e2i]

        idx = torch.tensor([e2i[e] for e in vocab], device=embedder.device)
        with torch.no_grad():
            v = embedder.model.dual_emb.project_v(idx)        # (E, m)
        v = F.normalize(v, dim=-1)
        centers = {name: v[i] for i, name in enumerate(vocab)}

        calib = {}
        cpath = art / "entity_calibration.json"
        if cpath.exists():
            calib = json.loads(cpath.read_text()).get("entity_thresholds", {})
        else:
            warnings.warn(
                "ManifoldEntityExtractor: no entity_calibration.json --- using "
                f"un-calibrated default tau={default_tau}. Call calibrate() before "
                "relying on precision.",
                stacklevel=2,
            )
        thresholds = {e: float(calib.get(e, default_tau)) for e in vocab}

        return cls(embedder=embedder, centers=centers, thresholds=thresholds,
                   window_size=window_size, window_stride=window_stride)

    def _windows(self, text: str) -> list[str]:
        """Overlapping token windows plus the full text, deduped, order-preserving.

        The full text is always included so the extractor never does *worse* than
        a whole-message scorer; the windows add the local context resolution.
        """
        toks = text.split()
        out = [text]
        w, s = self.window_size, self.window_stride
        if len(toks) > w:
            for i in range(0, len(toks) - w + 1, s):
                out.append(" ".join(toks[i:i + w]))
            if (len(toks) - w) % s:
                out.append(" ".join(toks[-w:]))           # ensure the tail is covered
        return list(dict.fromkeys(out))

    def _window_distances(self, text: str) -> tuple[torch.Tensor, list[str]]:
        """(E, nw) geodesic distances from every entity center to every window."""
        wins = self._windows(text)
        wv = self.embedder._embed(wins)                   # (nw, m), same projection
        dot = (self._matrix @ wv.T).clamp(-1 + 1e-7, 1 - 1e-7)   # (E, nw)
        return torch.arccos(dot), wins

    def entity_distances(self, text: str) -> dict[str, float]:
        """Per-entity *minimum* geodesic distance over all windows (max-pool)."""
        d, _ = self._window_distances(text)
        mins = d.min(dim=1).values.cpu().tolist()
        return dict(zip(self._names, mins))

    def extract(self, text: str) -> set[str]:
        """Entities whose closest window falls within the calibrated radius.

        Same contract as ``HybridEntityExtractor.extract`` --- a drop-in
        deterministic third path.
        """
        return {n for n, dist in self.entity_distances(text).items()
                if dist <= self.thresholds[n]}

    def explain(self, text: str) -> dict[str, dict[str, object]]:
        """Debug view: for each fired entity, its distance and the winning window."""
        d, wins = self._window_distances(text)
        mins, argmins = d.min(dim=1)
        out: dict[str, dict[str, object]] = {}
        for i, name in enumerate(self._names):
            dist = float(mins[i])
            if dist <= self.thresholds[name]:
                out[name] = {"distance": dist, "tau": self.thresholds[name],
                             "window": wins[int(argmins[i])]}
        return out

    def calibrate(
        self,
        labeled: list[tuple[str, set[str]]],
        grid: Iterable[float] | None = None,
    ) -> dict[str, float]:
        """Fit per-entity tau by maximizing F1 over a geodesic-radius grid.

        ``labeled`` is ``[(text, {gold entity names}), ...]``. Mutates and returns
        ``self.thresholds``; persist it to ``entity_calibration.json`` to make the
        choice durable. Entities never seen positive keep their current threshold.
        """
        grid = list(grid) if grid is not None else [round(0.1 * k, 2) for k in range(1, 26)]
        # distance[name] -> list of (min_dist, is_gold) over the corpus
        dists: dict[str, list[tuple[float, bool]]] = {n: [] for n in self._names}
        for text, gold in labeled:
            ed = self.entity_distances(text)
            for name, dist in ed.items():
                dists[name].append((dist, name in gold))

        for name, rows in dists.items():
            pos = sum(1 for _, g in rows if g)
            if pos == 0:
                continue
            best_tau, best_f1 = self.thresholds[name], -1.0
            for tau in grid:
                tp = sum(1 for d, g in rows if g and d <= tau)
                fp = sum(1 for d, g in rows if not g and d <= tau)
                fn = pos - tp
                f1 = 0.0 if tp == 0 else tp / (tp + 0.5 * (fp + fn))
                if f1 > best_f1:
                    best_tau, best_f1 = tau, f1
            self.thresholds[name] = best_tau
        return self.thresholds


_DEFAULT_EXTRACTOR: ManifoldEntityExtractor | None = None
_LOAD_FAILED = False


def get_default_extractor() -> ManifoldEntityExtractor | None:
    """Cached extractor, or None if the cap artifact is unavailable (degrade-safe)."""
    global _DEFAULT_EXTRACTOR, _LOAD_FAILED
    if _DEFAULT_EXTRACTOR is None and not _LOAD_FAILED:
        try:
            _DEFAULT_EXTRACTOR = ManifoldEntityExtractor.load()
        except Exception:
            _LOAD_FAILED = True
            return None
    return _DEFAULT_EXTRACTOR
