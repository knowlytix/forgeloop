"""TinyGPT classifier vendored from the llm-tutorial repo.

Source: /path/to/llm-tutorial/lm_from_scratch/ at the
commit current as of agent-tutorial's switch from a keyword classifier to
the fine-tuned TinyGPT classifier. Files are byte-identical to upstream
except for import rewrites (lm_from_scratch.X -> agentlab.models.tinygpt.X)
so this package can run without the llm-tutorial source on sys.path.

The intended public surface is narrow — only what the agent's
classify_complaint tool needs:

    from agentlab.models.tinygpt import TinyGPTClassifier, BPETokenizer
"""

from agentlab.models.tinygpt.bpe_tokenizer import BPETokenizer
from agentlab.models.tinygpt.classifier import (
    ClassifierConfig,
    TinyGPTClassifier,
)
from agentlab.models.tinygpt.configs import GPTConfig
from agentlab.models.tinygpt.gpt import TinyGPT

__all__ = [
    "BPETokenizer",
    "ClassifierConfig",
    "GPTConfig",
    "TinyGPT",
    "TinyGPTClassifier",
]
