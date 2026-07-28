#!/bin/bash
# rag_test (ship verdict) on spark-ef84 with the geometric self-verifier.
cd /path/to/forgeloop || exit 1
source /path/to/venv/bin/activate
export PYTHONPATH=/path/to/forgeloop:$PYTHONPATH
export KNOWLYTIX_EULA_ACCEPTED=1 TOKENIZERS_PARALLELISM=false AGENTLAB_USE_LLM_RAG=1
python scripts/run_rag_test_ch12.py
