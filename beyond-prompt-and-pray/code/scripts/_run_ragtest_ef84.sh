#!/bin/bash
# rag_test (ship verdict) on spark-ef84 with the geometric self-verifier.
cd /home/user/jupyterlab/forgeloop || exit 1
source /home/user/cluster/spark-venv/bin/activate
export PYTHONPATH=/home/user/jupyterlab/forgeloop:$PYTHONPATH
export KNOWLYTIX_EULA_ACCEPTED=1 TOKENIZERS_PARALLELISM=false AGENTLAB_USE_LLM_RAG=1
python scripts/run_rag_test_ch12.py
