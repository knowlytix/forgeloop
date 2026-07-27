"""Compatibility shim for the GMS knowledge-graph sphere visualizer.

The visualizer is general and reusable, so it lives in knowlytix as
``knowlytix.knowledge.viz`` (single source of truth). This module re-exports it
under the names the book notebooks use, and keeps the ``load_gms_store`` /
``visualize`` / ``visualize_store`` aliases and the command-line entry point so
existing notebooks and scripts keep working.

    from knowlytix.knowledge.viz import KGData, sphere_figure, from_store_dir

Command line:

    python kg_sphere.py <store_dir> [--out graph.html] [--source v|u]
                        [--title "..."] [--relations is_type,enables]
"""

from __future__ import annotations

import argparse

from knowlytix.knowledge.viz import (
    KGData,
    build_figure,
    from_store,
    from_store_dir,
    project_to_sphere,
    slerp_arc,
    sphere_figure,
)

__all__ = [
    "KGData", "build_figure", "from_store", "from_store_dir",
    "project_to_sphere", "slerp_arc", "sphere_figure",
    "load_gms_store", "visualize", "visualize_store",
]


# ── Legacy aliases used by the book notebooks ──────────────────────────────

def load_gms_store(store_dir: str, source: str = "v") -> KGData:
    """Load a saved knowlytix GMS store directory into :class:`KGData`."""
    return from_store_dir(store_dir, source=source)


def visualize(kg: KGData, out_html: str | None = None, show: bool = False,
              standardize: bool = False, **fig_kwargs):
    """Project ``kg`` to the sphere, build the figure, optionally write/show."""
    return sphere_figure(kg, out_html=out_html, show=show,
                         standardize=standardize, **fig_kwargs)


def visualize_store(store_dir: str, out_html: str | None = None,
                    source: str = "v", relations: list[str] | None = None,
                    **kwargs):
    """Load a knowlytix GMS store and visualize it, optionally keeping only the
    given ``relations`` (drops all other triples before rendering)."""
    kg = from_store_dir(store_dir, source=source)
    if relations:
        kg.triples = [(h, r, t) for (h, r, t) in kg.triples if r in relations]
    return sphere_figure(kg, out_html=out_html, relations=relations, **kwargs)


# ── CLI ────────────────────────────────────────────────────────────────────

def _main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("store_dir", help="path to a KG store directory")
    p.add_argument("--out", default="kg_sphere.html", help="output HTML path")
    p.add_argument("--source", default="v", choices=["v", "u"],
                   help="which dual entity embedding to use (default v)")
    p.add_argument("--title", default="Knowledge-Graph Embedding")
    p.add_argument("--relations", default=None,
                   help="comma-separated relations to keep (default all)")
    p.add_argument("--light", action="store_true", help="light theme")
    p.add_argument("--standardize", action="store_true",
                   help="z-score embedding dims before PCA")
    a = p.parse_args()
    rels = a.relations.split(",") if a.relations else None
    fig = visualize_store(a.store_dir, out_html=a.out, source=a.source,
                          relations=rels, title=a.title, dark=not a.light,
                          standardize=a.standardize)
    print(f"wrote {a.out}")
    return 0 if fig is not None else 1


if __name__ == "__main__":
    raise SystemExit(_main())
