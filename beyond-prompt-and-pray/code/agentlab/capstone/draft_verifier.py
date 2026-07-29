"""GMS verifier for the draft_response output.

The draft is the agent's externally visible commitment, and the one field a
generative model writes rather than transcribes --- so it is the highest-value
place to verify against the GMS substrate. This verifier does two things,
matching the two ways a draft can go wrong:

  * **Correct numbers (ENM).** A LoRA *generates* fee amounts; it can paraphrase
    a policy correctly yet drift a digit. We parse the dollar amount the draft
    states, look up the authoritative value in the store's Exact Numerical
    Memory (``lookup_enm``), and if it drifted we substitute the exact value.
    This is a safe, deterministic correction --- the byte-exact numeric
    guarantee a generative model cannot make on its own.

  * **Escalate claims.** A draft that makes an UNCONDITIONAL promise to waive or
    refund a fee asserts an action the policy does not authorize (fee reversals
    need manager approval). That is a policy contradiction, not a typo, so the
    safe disposition is to escalate to a human --- never silently rewrite the
    bank's commitment. The promise is detected by the semantic intent guard.

Fails loud: a store load error raises rather than skipping verification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentlab._paths import data_path

_DEFAULT_STORE = data_path("gms_banking_store")

# Policy id -> the ENM register/key holding that policy's authoritative dollar
# figure. Only policies whose draft cites a single primary fee are listed; the
# rest (disputes, pii, etc.) carry no dollar figure to verify.
_POLICY_FEE_ENM: dict[str, tuple[str, str]] = {
    "overdraft": ("fee_schedule", "overdraft/per_occurrence"),
    "fee_reversal": ("reversal_authority", "representative"),
}

_DOLLAR = re.compile(r"\$\s?(\d+(?:\.\d+)?)")


@dataclass
class DraftVerifier:
    """Verifies a generated draft against the GMS banking store, correcting drifted fee amounts via ENM and escalating unauthorized fee-waiver promises.

    Attributes:
        store: The loaded GMS banking expert store.
    """

    store: Any

    @classmethod
    def load(cls, store_path: Path | None = None) -> "DraftVerifier":
        """Load the GMS banking store backing the verifier.

        Args:
            store_path: Store directory; defaults to the bundled banking store.

        Returns:
            A ready-to-use DraftVerifier.
        """
        import torch
        from knowlytix.knowledge.query import DocGMSConfig, GMSExpertStore

        store_path = store_path or _DEFAULT_STORE
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        store = GMSExpertStore(DocGMSConfig(store_path=str(store_path)), device=device)
        if not store.load():
            raise RuntimeError(
                f"failed to load GMS banking store at {store_path!s}; "
                "run the store build first."
            )
        return cls(store=store)

    def verify(self, text: str, policy_id: str) -> dict[str, Any]:
        """Return the (possibly corrected) draft plus the verification verdict."""
        corrections: list[dict[str, Any]] = []
        new_text = text

        # 1. ENM numeric correction (only when a single dollar amount is present,
        #    so we never mis-correct a draft that legitimately names two figures).
        enm_key = _POLICY_FEE_ENM.get(policy_id)
        if enm_key is not None:
            authoritative = self.store.lookup_enm(*enm_key)
            if authoritative is not None:
                amounts = [float(m.group(1)) for m in _DOLLAR.finditer(text)]
                if len(amounts) == 1 and amounts[0] != float(authoritative):
                    fixed = f"${int(authoritative)}" if float(authoritative).is_integer() else f"${authoritative}"
                    new_text = _DOLLAR.sub(fixed, text)
                    corrections.append({
                        "field": f"{enm_key[0]}::{enm_key[1]}",
                        "draft_value": amounts[0],
                        "authoritative": float(authoritative),
                    })

        # 2. Policy-claim escalation: an unconditional waiver/refund promise.
        from agentlab.governance.semantic_guard import get_default_guard

        intent, _ = get_default_guard().classify(text)
        escalate = intent == "fee_waiver"

        return {
            "text": new_text,
            "corrected": bool(corrections),
            "corrections": corrections,
            "escalate": escalate,
            "reason": "draft makes an unauthorized fee-waiver promise" if escalate else "",
        }


_DEFAULT_VERIFIER: DraftVerifier | None = None


def get_default_verifier() -> DraftVerifier:
    """Return the process-wide DraftVerifier singleton, fetching the banking store on first use.

    Returns:
        The lazily loaded DraftVerifier.
    """
    global _DEFAULT_VERIFIER
    if _DEFAULT_VERIFIER is None:
        from agentlab._paths import ensure_default
        ensure_default("gms_banking_store")
        _DEFAULT_VERIFIER = DraftVerifier.load()
    return _DEFAULT_VERIFIER
