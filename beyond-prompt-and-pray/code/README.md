# Beyond Prompt and Pray

*Building Agentic AI Systems from Scratch.* Two artifacts, one source of truth.

| Artifact | Path | What it is |
| --- | --- | --- |
| Python library | `agentlab/` | Source of truth. Types, protocols, the loop, gates, audit. |
| Notebooks | `notebooks/` | One JupyterLab notebook per chapter, importing only from `agentlab` and the stdlib. Every notebook ends with a self-check assertion. |

## Install

`agentlab` is pip-installable. The core install is light; heavier and licensed pieces are optional extras.

```bash
pip install agentlab                 # core: loop, typed actions, tools, gates, audit, planning, memory, evaluation
pip install "agentlab[ml]"           # + open-weight model tools (torch, transformers, peft)
pip install "agentlab[notebooks]"    # + JupyterLab, matplotlib, pandas to run the chapter notebooks
pip install "agentlab[anthropic]"    # + the optional hosted-model adapter (notebooks default to local Qwen / MockLM)
pip install "agentlab[all]"          # everything above (public deps only)
pip install "agentlab[dev]"          # + pytest, ruff, black
```

| Extra | Pulls in | Needed for |
| --- | --- | --- |
| *(core)* | `pydantic`, `numpy` | the loop, types, tools, gates, audit, planning, memory, evaluation |
| `ml` | `torch`, `transformers`, `peft`, `accelerate` | the classifier head, Qwen adapters, the draft LoRA |
| `gms` | `knowlytix` (licensed, separate) | the GMS substrate — see below |
| `anthropic` | `anthropic` | the hosted-model adapter |
| `notebooks` | `jupyterlab`, `matplotlib`, `pandas` | running the chapter notebooks |
| `dev` | `pytest`, `ruff`, `black` | tests and linting |

From a checkout, use `-e` for an editable install, e.g. `pip install -e ".[ml,notebooks,dev]"`.

### GMS / Knowlytix (licensed)

The GMS-backed features — the geometric plausibility gate, the regulatory guard, the policy Graph-RAG store, Exact Numerical Memory and the design-of-experiments test harness — run on the **`knowlytix`** package. `knowlytix` is **licensed and distributed separately**; it is not on public PyPI, and a license is required.

Get it from Knowlytix: **https://knowlytix.ai/**

```bash
pip install knowlytix --index-url <KNOWLYTIX_INDEX_URL>   # license required; index from Knowlytix
pip install "agentlab[gms]"                               # records the dependency
```

`agentlab` imports `knowlytix` lazily, so everything outside the GMS features works without it. To check at runtime:

```python
import agentlab.gms as gms
gms.available()   # True if the licensed backend is installed
gms.require()     # returns the knowlytix module, or raises with install instructions
```

The `data/*` model artifacts (classifier head, LoRA adapter, GMS stores) are **not committed** — the large weight files (`*.safetensors`, `*.pt`, tokenizers) are git-ignored to keep the repo lean. Reproduce them from the scripts in `scripts/` (`train_*.py`, `build_*.py`); the small config/metadata files in each `data/*` dir are kept.

## Run the tests

```bash
pytest
```

200 tests, run in under half a second.

## Run the capstone

```bash
python scripts/run_complaint_agent.py --verbose
```

Evaluates the governed complaint agent against 20 synthetic cases. Escalation accuracy is the headline; classification accuracy is honest about where keyword classifiers fall short.

## Build the notebooks

The notebooks are generated from `scripts/build_notebooks.py`. Edit the spec functions and rerun:

```bash
python scripts/build_notebooks.py
```

Then execute one to verify:

```bash
python -m nbconvert --to notebook --execute notebooks/15_capstone.ipynb --output /tmp/out.ipynb
```

## Repo layout

```
beyond-prompt-and-pray/code/
├── agentlab/                  # the library
│   ├── core/                  # Chapter 2, 4, 7
│   ├── reasoning/             # Chapter 3
│   ├── tools/                 # Chapter 5, 6
│   ├── memory/                # Chapter 9
│   ├── planning/              # Chapter 8
│   ├── models/                # Chapter 2, 7
│   ├── evaluation/            # Chapter 10, 11
│   ├── governance/            # Chapter 12, 13
│   ├── audit/                 # Chapter 12
│   ├── multiagent/            # Chapter 14
│   └── capstone/              # Chapter 15
├── notebooks/                 # one per chapter (in the topic root)
├── data/policies/             # synthetic banking policy corpus
├── data/eval_cases/           # 20 synthetic complaint cases
├── configs/                   # capstone wiring config
├── scripts/                   # build, eval, consistency check
└── tests/                     # unit tests
```
