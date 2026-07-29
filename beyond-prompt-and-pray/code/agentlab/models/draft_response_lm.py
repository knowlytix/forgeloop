"""Draft-response LM facade (Qwen3-4B-Instruct + LoRA).

Qwen port of the Chapter-31 draft-response model (replaces the TinyGPT LoRA).
The base Qwen3-4B-Instruct is frozen; a small LoRA adapter
(``data/draft_response_lm_qwen/``) trained by
``scripts/train_draft_response_lora.py`` on (complaint, issue, policy summary)
-> grounded reply pairs supplies the domain behavior.

This facade hides the prompt template, the policy-id-to-summary mapping and
greedy decoding so the agent's draft_response tool can call:

    text = get_default_lm().generate(category, issue, policy_evidence, message)

The prompt is built to match the SFT data exactly (``Complaint/Issue/Policy/
Response:`` then a trailing space), so inference stays on the training
distribution. Greedy decoding -> deterministic drafts. On the held-out eval
set this adapter cites the right policy and avoids forbidden phrases on 12/12
cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from agentlab._paths import data_path

_DEFAULT_DIR = data_path("draft_response_lm_qwen")
_BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

# One-line policy summaries taken verbatim from the SFT corpus
# (data/training/bank_policy/draft_response_sft*.jsonl). Keying off the
# policy_id keeps the prompt's "Policy:" line on the training distribution.
POLICY_SUMMARIES: dict[str, str] = {
    "overdraft":             "overdraft fee $35; one goodwill reversal per year",
    "disputes":              "dispute filed within 60 days; provisional credit",
    "fee_reversal":          "up to $35 reversal at rep discretion; manager approval above",
    "account_closure":       "customer-initiated closure needs identity; 30-day notice for bank",
    "pii_handling":          "PII redacted; exposure reported within 24 hours",
    "regulatory_escalation": "UDAAP escalates to Compliance within 1 business day",
}

# The agent's extract_facts emits a coarser issue taxonomy than the SFT corpus.
# Translate the agent's issue back onto the SFT-trained token via the policy_id
# (the policy choice from search_policy is the more reliable signal).
ISSUE_FOR_POLICY: dict[str, str] = {
    "overdraft":             "overdraft_fee",
    "disputes":              "unauthorized_transaction",
    "fee_reversal":          "fee_reversal_request",
    "account_closure":       "account_closure_request",
    "pii_handling":          "pii_exposed",
    "regulatory_escalation": "udaap_complaint",
}


@dataclass
class DraftResponseLM:
    """Frozen Qwen3-4B-Instruct base with a LoRA adapter that drafts a grounded complaint reply from the issue and retrieved policy evidence.

    Attributes:
        model: The LoRA-adapted causal language model.
        tokenizer: Tokenizer paired with the model.
        device: Torch device the model runs on.
        policy_summaries: Policy-id to one-line summary map used to build the prompt.
    """

    model: Any
    tokenizer: Any
    device: torch.device
    policy_summaries: dict[str, str] = field(default_factory=lambda: dict(POLICY_SUMMARIES))

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> "DraftResponseLM":
        """Load the frozen Qwen3-4B-Instruct base and apply the LoRA adapter from ``path``.

        Args:
            path: LoRA adapter directory; defaults to the bundled adapter dir.
            device: Torch device or device string; defaults to CUDA when available.

        Returns:
            A ready-to-use DraftResponseLM.
        """
        path = Path(path) if path is not None else _DEFAULT_DIR
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            _BASE_MODEL,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            device_map=device.type if device.type == "cuda" else None,
        )
        if device.type != "cuda":
            base = base.to(device)
        model = PeftModel.from_pretrained(base, path)
        model.eval()
        return cls(model=model, tokenizer=tokenizer, device=base.device)

    def _format_prompt(
        self,
        category: str,
        issue: str,
        policy_evidence: list[dict[str, Any]],
        message: str,
    ) -> tuple[str, str]:
        # Use the first policy_evidence id whose summary we know; else the first
        # id. An unknown id keeps the template but with an empty summary.
        policy_id = ""
        for ev in policy_evidence:
            pid = (ev or {}).get("id", "")
            if pid in self.policy_summaries:
                policy_id = pid
                break
        if not policy_id and policy_evidence:
            policy_id = (policy_evidence[0] or {}).get("id", "")
        summary = self.policy_summaries.get(policy_id, "")
        sft_issue = ISSUE_FOR_POLICY.get(policy_id, issue)
        # Matches the SFT "user" field exactly: it ends with "Response:" and
        # training appended a single trailing space before the completion.
        prompt = (
            f"Complaint: {message}\n"
            f"Issue: {sft_issue}\n"
            f"Policy: {summary}\n"
            "Response: "
        )
        return prompt, policy_id

    @torch.no_grad()
    def generate(
        self,
        category: str,
        issue: str,
        policy_evidence: list[dict[str, Any]],
        message: str,
        max_new_tokens: int = 80,
        max_sentences: int = 3,
    ) -> str:
        """Greedily decode a grounded reply and trim it to at most ``max_sentences`` sentences.

        Args:
            category: Complaint category from the agent's taxonomy.
            issue: Coarse issue label from the agent's taxonomy.
            policy_evidence: Retrieved policy records; the first known id sets the prompt summary.
            message: The customer complaint text.
            max_new_tokens: Decoding budget in new tokens.
            max_sentences: Maximum sentences retained in the reply.

        Returns:
            The drafted reply text.
        """
        prompt, _policy_id = self._format_prompt(category, issue, policy_evidence, message)
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        text = self.tokenizer.decode(
            out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        # Cut at a next-prompt boundary if the model runs on.
        for sep in ("\n\n", "\nComplaint:", "\nIssue:", "\nPolicy:"):
            if sep in text:
                text = text.split(sep, 1)[0].strip()
        # Keep at most ``max_sentences`` sentences; SFT replies are 1-2.
        sentences: list[str] = []
        buf = ""
        for ch in text:
            buf += ch
            if ch in ".!?" and len(buf.strip()) > 5:
                sentences.append(buf.strip())
                buf = ""
                if len(sentences) >= max_sentences:
                    break
        if buf.strip() and len(sentences) < max_sentences:
            sentences.append(buf.strip())
        return " ".join(sentences).strip()


_DEFAULT_LM: DraftResponseLM | None = None


def get_default_lm() -> DraftResponseLM:
    """Return the process-wide DraftResponseLM singleton, fetching the adapter artifact on first use.

    Returns:
        The lazily loaded DraftResponseLM.
    """
    global _DEFAULT_LM
    if _DEFAULT_LM is None:
        from agentlab._paths import ensure_default
        ensure_default("draft_response_lm_qwen")
        _DEFAULT_LM = DraftResponseLM.load()
    return _DEFAULT_LM
