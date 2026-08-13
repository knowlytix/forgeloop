"""Build the MeMo SFT corpus for the draft_response LoRA adapter.

Thin caller so 00_setup can invoke this as a scripts/ entry alongside the
other stage() calls. The real implementation lives in
``benchmarks/draft_adapter/build_memo_data.py``; this script delegates to it
so the module is resolved from the repo root (cwd=REPO in stage()).

    python scripts/build_draft_response_corpus.py
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
runpy.run_module("benchmarks.draft_adapter.build_memo_data", run_name="__main__")
