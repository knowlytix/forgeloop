"""The open-weight model the books run on.

One constant, one download. Before this, the id was written out as a literal in
roughly twenty places across three books, and had already drifted into two
different models: the runtime loaded Qwen3-4B while
``scripts/train_qwen_extract_lora.py`` trained its adapter against Qwen2.5-3B,
so the adapter it produced could never load onto the backbone that reads it.
A reader working through both books also downloaded ~14 GB for what should be
a single model.

Import this rather than restating the id::

    from agentlab.models.constants import DEFAULT_QWEN_MODEL

Anything that trains an adapter must use the same backbone the runtime loads,
so training scripts read this constant too.
"""

from __future__ import annotations

#: Hugging Face id of the instruct model used for generation, extraction,
#: classification heads and every LoRA backbone in the trilogy.
DEFAULT_QWEN_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

#: Sentence-transformers model used for the dense-vector baselines the books
#: compare against the geometric store.
DEFAULT_DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

__all__ = ["DEFAULT_DENSE_MODEL", "DEFAULT_QWEN_MODEL"]
