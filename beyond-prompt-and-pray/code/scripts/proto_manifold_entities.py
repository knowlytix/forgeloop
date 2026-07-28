"""Smoke test for the windowed ManifoldEntityExtractor.

Loads the cap artifact, runs the extractor on a few contrasting messages, and
prints the fired entities + winning windows next to a naive substring baseline,
to show the context-sensitive (geometric) path against the literal one.

    python scripts/proto_manifold_entities.py
"""

from __future__ import annotations

import warnings

from agentlab.capstone.manifold_entities import ManifoldEntityExtractor

# A small, curated evidence vocabulary (canonical store entities).
VOCAB = [
    "hidden_fee", "unauthorized_transfer", "billing_error", "apr_dispute",
    "credit_report_dispute", "escrow_dispute", "udaap", "reg_x",
    "stolen_card", "interest_dispute",
]

# Contrasting messages: same surface tokens, different meaning in context.
MESSAGES = [
    "There was a $35 charge I never agreed to and it was never disclosed anywhere.",
    "I paid a $35 charge to the merchant at the store, that part is fine.",
    "Someone made transfers from my account that I did not authorize.",
    "I authorized the transfer myself but the bank says otherwise.",
    "My mortgage servicer mishandled the escrow account on my home loan.",
]


def baseline(text: str) -> set[str]:
    """Naive substring match on the underscore-spaced entity name."""
    t = text.lower()
    return {e for e in VOCAB if e.replace("_", " ") in t}


def main() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")              # un-calibrated default tau
        ext = ManifoldEntityExtractor.load(entities=VOCAB, default_tau=1.0)

    print(f"loaded: {len(ext.centers)} entity centers, m-sphere "
          f"dim={ext._matrix.shape[1]}, adapter={ext.embedder.adapter is not None}\n")

    for msg in MESSAGES:
        print("MSG:", msg)
        print("  regex/substr :", sorted(baseline(msg)) or "[]")
        fired = ext.explain(msg)
        if fired:
            for name, info in sorted(fired.items(), key=lambda kv: kv[1]["distance"]):
                print(f"  manifold     : {name:22s} d={info['distance']:.3f} "
                      f"(tau={info['tau']:.2f})  win={info['window']!r}")
        else:
            print("  manifold     : []")
        # Show the 3 nearest entities regardless of threshold, for calibration intuition.
        dists = sorted(ext.entity_distances(msg).items(), key=lambda kv: kv[1])[:3]
        print("  nearest      :", ", ".join(f"{n}={d:.3f}" for n, d in dists))
        print()


if __name__ == "__main__":
    main()
