#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Generate train.jsonl and valid.jsonl for the complaint classifier.

Reads the committed ``data/eval_cases/cases.json`` (20 labeled banking
messages) and writes a stratified 80/20 split into
``data/training/complaint_classification/``. These seed files are the inputs
``train_complaint_classifier_qwen.py`` and ``augment_complaint_training_doe.py``
consume.

Run::

    python scripts/build_complaint_training_seeds.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_CASES = _REPO / "data" / "eval_cases" / "cases.json"
_OUT = _REPO / "data" / "training" / "complaint_classification"


def main() -> int:
    cases = json.loads(_CASES.read_text())

    by_label: dict[str, list[dict]] = {}
    for c in cases:
        label = c["expected_classification"]
        by_label.setdefault(label, []).append(
            {"id": c["id"], "message": c["message"], "label": label}
        )

    rng = random.Random(42)
    train, valid = [], []
    for label, rows in sorted(by_label.items()):
        rng.shuffle(rows)
        n_valid = max(1, len(rows) // 5)
        valid.extend(rows[:n_valid])
        train.extend(rows[n_valid:])

    rng.shuffle(train)
    rng.shuffle(valid)

    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / "train.jsonl").write_text(
        "\n".join(json.dumps(r) for r in train) + "\n"
    )
    (_OUT / "valid.jsonl").write_text(
        "\n".join(json.dumps(r) for r in valid) + "\n"
    )

    from collections import Counter
    print(f"train: {len(train)}  {dict(Counter(r['label'] for r in train))}")
    print(f"valid: {len(valid)}  {dict(Counter(r['label'] for r in valid))}")
    print(f"wrote {_OUT}/train.jsonl and valid.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
