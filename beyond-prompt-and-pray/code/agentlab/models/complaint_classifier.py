"""Banking-complaint classifier facade (Qwen3-4B + logit head).

A frozen Qwen3-4B-Instruct encoder turns a customer message into a feature
vector (the last non-pad token's hidden state); a single trained linear "logit
head" maps that to one of three labels. The head is the only learned part —
trained by ``scripts/train_complaint_classifier_qwen.py`` and stored at
``data/complaint_classifier_qwen/head.pt`` together with the feature
standardization so inference reproduces training exactly.

This replaces the earlier vendored TinyGPT classifier; the public API is
unchanged so ``classify_complaint`` keeps doing one call:

    label, confidence = get_default_classifier().classify(text)

Deterministic: a frozen encoder plus a fixed linear head means the same text
always yields the same label and confidence. ``confidence`` is the softmax
probability of the winning label.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DIR = _REPO_ROOT / "data" / "complaint_classifier_qwen"


@dataclass
class ComplaintClassifier:
    encoder: Any
    tokenizer: Any
    head_weight: torch.Tensor  # (num_labels, hidden)
    head_bias: torch.Tensor  # (num_labels,)
    mean: torch.Tensor  # (hidden,)
    std: torch.Tensor  # (hidden,)
    labels: list[str]
    device: torch.device
    max_length: int = 128

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> ComplaintClassifier:
        path = Path(path) if path is not None else _DEFAULT_DIR
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        ckpt = torch.load(path / "head.pt", map_location="cpu", weights_only=False)
        from transformers import AutoModel, AutoTokenizer

        model_id = ckpt["model_id"]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        encoder = AutoModel.from_pretrained(
            model_id,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            device_map=device.type if device.type == "cuda" else None,
        )
        if device.type != "cuda":
            encoder = encoder.to(device)
        encoder.eval()

        sd = ckpt["head_state_dict"]
        return cls(
            encoder=encoder,
            tokenizer=tokenizer,
            head_weight=sd["weight"].float().to(device),
            head_bias=sd["bias"].float().to(device),
            mean=ckpt["mean"].float().to(device),
            std=ckpt["std"].float().to(device),
            labels=list(ckpt["labels"]),
            device=device,
            max_length=int(ckpt.get("max_length", 128)),
        )

    @torch.no_grad()
    def _feature(self, text: str) -> torch.Tensor:
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        hidden = self.encoder(**enc).last_hidden_state  # (1, t, h)
        last = int(enc["attention_mask"].sum().item()) - 1
        return hidden[0, last].float()  # (h,)

    @torch.no_grad()
    def classify(self, text: str) -> tuple[str, float]:
        if not text or not text.strip():
            return self.labels[-1], 0.0
        feat = (self._feature(text) - self.mean) / self.std
        logits = feat @ self.head_weight.T + self.head_bias  # (num_labels,)
        probs = F.softmax(logits, dim=-1)
        idx = int(probs.argmax().item())
        return self.labels[idx], float(probs[idx].item())


@dataclass
class LoraComplaintClassifier:
    """LoRA-plus-head rung of the ladder: Qwen2 sequence-classification with a
    PEFT LoRA adapter and a trained classification head. The frozen logit head
    underfit hedged/ambiguous phrasing (Chapter 16 DoE testing); the LoRA rung
    reshapes the features instead of drawing a linear boundary on frozen ones.

    Same ``.classify(text) -> (label, confidence)`` API as ``ComplaintClassifier``,
    so ``classify_complaint`` is unchanged. The artifact is a PEFT adapter dir
    (``adapter_config.json`` + adapter weights + ``labels.json``).
    """

    model: Any
    tokenizer: Any
    labels: list[str]
    device: torch.device
    max_length: int = 128

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> LoraComplaintClassifier:
        path = Path(path) if path is not None else _DEFAULT_DIR
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        labels = list(json.loads((path / "labels.json").read_text()))
        cfg = json.loads((path / "adapter_config.json").read_text())
        base = cfg["base_model_name_or_path"]

        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(base)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForSequenceClassification.from_pretrained(
            base,
            num_labels=len(labels),
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            device_map=device.type if device.type == "cuda" else None,
        )
        if device.type != "cuda":
            model = model.to(device)
        model.config.pad_token_id = tokenizer.pad_token_id
        model = PeftModel.from_pretrained(model, str(path))  # restores adapter + score head
        model.eval()
        return cls(model=model, tokenizer=tokenizer, labels=labels, device=device)

    @torch.no_grad()
    def classify(self, text: str) -> tuple[str, float]:
        if not text or not text.strip():
            return self.labels[-1], 0.0
        enc = self.tokenizer(
            text, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self.device)
        logits = self.model(**enc).logits[0]
        probs = F.softmax(logits.float(), dim=-1)
        idx = int(probs.argmax())
        return self.labels[idx], float(probs[idx])


_DEFAULT_CLASSIFIER: Any = None


def get_default_classifier():
    """Lazy singleton accessor for the bundled classifier.

    Loads the LoRA-plus-head adapter if one is present in the default dir
    (``adapter_config.json``), else the frozen-encoder logit head (``head.pt``).
    """
    global _DEFAULT_CLASSIFIER
    if _DEFAULT_CLASSIFIER is None:
        if (_DEFAULT_DIR / "adapter_config.json").exists():
            _DEFAULT_CLASSIFIER = LoraComplaintClassifier.load()
        else:
            _DEFAULT_CLASSIFIER = ComplaintClassifier.load()
    return _DEFAULT_CLASSIFIER
