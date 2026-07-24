#!/bin/bash
# Launcher for the final (b) faithful-qwen capstone DOE on spark-ef84.
cd /path/to/forgeloop || exit 1
source /path/to/venv/bin/activate
# ef84's venv lacks fe12's editable install of agentlab; put the repo on the path.
export PYTHONPATH=/path/to/forgeloop:$PYTHONPATH
export KNOWLYTIX_EULA_ACCEPTED=1 TOKENIZERS_PARALLELISM=false AGENTLAB_USE_LLM_RAG=1
python scripts/run_capstone_doe.py --n-runs 120 --seed 42 --rephrase qwen \
  --batch-materialize --out data/doe_ch12_final_b.csv
