"""Build the Chapter 13 governed-retrieval store (``data/gms_governed_store``).

The governed-retrieval example reads a store that no single script produced: the
metadata, value-polarity and calibration scripts all *consume* the store (they
load it and write artifacts *into* it), but nothing ingested the governed corpus
in the first place. On a clean checkout, where the store is git-ignored, Chapter
13 could not be built even with a GPU. This script is the missing producer: it
chains the existing builders in dependency order at the governed path.

The store is the GEODE policy store built from ``data/governed_banking_corpus.md``
(the same ``loss_mode="cap"`` geometry as ``gms_policy_store_cap``), with the
governance layer added on top: the sensitivity map + retrieval contracts, the
value-polarity verifier, and the calibrated RAG and disclosure gates.

Stages (each skipped if its output already exists):

  1. GEODE geometry ...... build_geode_rag_store.py   (GPU + Qwen)
  2. Governance metadata . build_governed_metadata.py (CPU)
  3. Value-polarity ...... build_policy_value_polarity.py (GPU + Qwen)
  4. RAG accept gate ..... calibrate_policy_rag_gate.py   (GPU)

With ``--scenarios`` it also builds the narrative artifacts the Chapter 13
notebook embeds (all GPU + Qwen):

  5. Polarity DoE split .. build_polarity_doe_dataset.py
  6. Governed scenarios .. run_governed_scenarios.py   (writes disclosure_gate_calibration.json)
  7. Gate comparison ..... compare_polarity_gates.py

Run::

    HF_HUB_OFFLINE=1 python scripts/build_governed_store.py            # store only
    HF_HUB_OFFLINE=1 python scripts/build_governed_store.py --scenarios  # + notebook artifacts
    python scripts/build_governed_store.py --force                     # rebuild, ignore existing
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
_STORE = "data/gms_governed_store"


def _run(script: str, args=(), *, produces=(), force=False) -> None:
    """Run scripts/<script> unless every path in *produces* already exists."""
    produces = list(produces)
    if not force and produces and all((_REPO / p).exists() for p in produces):
        print(f"✓ {script}: outputs present — skipping")
        return
    cmd = [sys.executable, str(_SCRIPTS / script), *map(str, args)]
    print(f"▶ {script} " + " ".join(map(str, args)), flush=True)
    r = subprocess.run(cmd, cwd=_REPO)
    if r.returncode:
        raise SystemExit(f"{script} failed (exit {r.returncode})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenarios", action="store_true",
                    help="also build the notebook narrative artifacts (stages 5-7)")
    ap.add_argument("--force", action="store_true",
                    help="rebuild every stage even if its outputs exist")
    args = ap.parse_args()

    corpus = _REPO / "data" / "governed_banking_corpus.md"
    if not corpus.exists():
        print(f"FAIL: missing corpus {corpus}", file=sys.stderr)
        return 1

    # 1. GEODE geometry from the governed corpus (cap loss, binding + polarity encoders).
    _run("build_geode_rag_store.py",
         ["--doc", "data/governed_banking_corpus.md", "--store-path", _STORE],
         produces=[f"{_STORE}/model.pt", f"{_STORE}/tuned_encoder"], force=args.force)

    # 2. Sensitivity map + retrieval contracts (CPU; hard-coded to the governed path).
    _run("build_governed_metadata.py",
         produces=[f"{_STORE}/sensitivity_map.json"], force=args.force)

    # 3. Value-polarity verifier (needs the store's stance facts + tuned_encoder).
    _run("build_policy_value_polarity.py", [_STORE],
         produces=[f"{_STORE}/value_polarity_encoder",
                   f"{_STORE}/value_polarity_calibration.json"], force=args.force)

    # 4. Calibrate the RAG accept gate on the governed store.
    _run("calibrate_policy_rag_gate.py", [_STORE],
         produces=[f"{_STORE}/rag_gate_calibration.json"], force=args.force)

    if args.scenarios:
        # 5. Polarity DoE split consumed by the disclosure calibration + gate comparison.
        _run("build_polarity_doe_dataset.py",
             produces=["data/training/polarity/test_polarity.jsonl"], force=args.force)
        # 6. Run the governed scenarios; calibrates + writes disclosure_gate_calibration.json.
        _run("run_governed_scenarios.py",
             produces=[f"{_STORE}/disclosure_gate_calibration.json",
                       "data/governed_scenarios.json"], force=args.force)
        # 7. Geometric-vs-classifier polarity gate comparison (embedded in the notebook).
        _run("compare_polarity_gates.py", ["--store", _STORE],
             produces=["data/polarity_gate_comparison.json"], force=args.force)

    print(f"\n✓ governed store ready at {_STORE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
