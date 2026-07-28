# SPDX-License-Identifier: Apache-2.0
"""Route B parser: a fine-tuned Qwen3-4B that emits GMS query triples directly.

Drop-in for GeometricQueryParser / QueryTripleExtractor: exposes ``.extract`` and
returns ``list[QueryTriple]``. The emitted triples flow through the pipeline's
existing TripleBinder (mode="fuzzy"), so a term the graph does not hold returns
None -> unbound -> the pipeline abstains (pipeline.query line ~246). That binder
step is the residual vocabulary-membership check; abstention needs no new wiring.
"""
from __future__ import annotations

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from knowlytix.knowledge.rag.query_triples import QueryTripleExtractor

# Same minimal system prompt used at training time (no vocab, no few-shot):
# the fine-tune must supply the graph vocabulary itself.
SYSTEM = (
    "You translate a question about a company's annual report into knowledge-graph "
    "query triples. Output ONLY a JSON array of objects with keys head, relation, "
    'tail. Mark the single value the question asks for as the bare string "?". '
    "Relations are lowercase snake_case."
)


class FineTunedTripleParser:
    def __init__(self, adapter_dir, base_model="Qwen/Qwen3-4B-Instruct-2507",
                 device="cuda", system=SYSTEM, max_new_tokens=96):
        self.tok = AutoTokenizer.from_pretrained(adapter_dir)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, device_map=device)
        self.model = PeftModel.from_pretrained(base, adapter_dir).eval()
        self.system = system
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def extract(self, question: str, *, hint: str | None = None):
        user = question if hint is None else f"{question}\n\nNote: {hint}"
        msgs = [{"role": "system", "content": self.system},
                {"role": "user", "content": user}]
        text = self.tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt",
                       add_special_tokens=False).to(self.model.device)
        out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                  do_sample=False, pad_token_id=self.tok.eos_token_id)
        raw = self.tok.decode(out[0, enc["input_ids"].shape[1]:],
                              skip_special_tokens=True)
        # Reuse the shipped parser's JSON/fallback parsing + asked-slot guard.
        return QueryTripleExtractor._ensure_asked(QueryTripleExtractor.parse(raw))
