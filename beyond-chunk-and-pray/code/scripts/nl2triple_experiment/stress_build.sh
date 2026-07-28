#!/bin/bash
# Stress test: full foundation rebuild on the larger corpus (separate paths, no
# clobber of shipped artifacts), then retrain the Route B parser and re-run the
# 3-route comparison at ~3x facts / +1 relation.
set -e
ROOT=/path/to/forgeloop/beyond-chunk-and-pray/code
PY=/path/to/venv/bin/python
SCR=$ROOT/scripts
EXP=$SCR/nl2triple_experiment

export GMS_MD=$ROOT/data/annual_report_large.md
export GMS_STORE=$ROOT/data/gms_annual_report_store_large
export GMS_FACTS=$ROOT/data/corpus_facts_large.md
export GMS_ENRICH=$ROOT/data/enrichment_large
export NL2T_DATA=$ROOT/data/nl2triple_large

echo "===== [0] generate large corpus ====="
$PY $EXP/make_large_corpus.py
echo "===== [1] build large store (GEODE + train) ====="
$PY $SCR/build_store.py
echo "===== [2] enrich (DoE cohort) ====="
$PY $SCR/enrich_data.py --n-runs 300 --max-per-category 50
echo "===== [3] finetune encoders (v + u + relevance cal) ====="
$PY $SCR/finetune_encoders.py
echo "===== [4] make NL->triple data ====="
$PY $EXP/make_data.py
echo "===== [5] train Route B LoRA (large) ====="
$PY $EXP/train_lora.py
echo "===== [6] novel-fact set ====="
$PY $EXP/make_novelfact.py
echo "===== [7] compare 3 routes (main sets) ====="
$PY $EXP/compare_parsers.py
echo "===== [8] compare 3 routes (novel facts) ====="
$PY $EXP/compare_parsers.py --novel-only
echo "===== STRESS DONE ====="
