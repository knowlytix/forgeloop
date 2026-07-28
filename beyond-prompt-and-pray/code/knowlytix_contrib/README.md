# Knowlytix upstream contributions

Three small, additive, backward-compatible changes developed while building the
Chapter 16 capstone test harness. Each one promotes a *wrapper we had to write
in `agentlab`* into a first-class Knowlytix capability, so other users don't have
to rediscover it. The unified diffs in `patches/` apply against **Knowlytix
v0.2.0** (verified against the installed wheels — `knowlytix_core-0.2.0`,
`knowlytix_harness-0.2.0`).

> **Source note.** The diffs were authored against, and applied to, the
> *installed wheel* tree (`site-packages/knowlytix/...`) — that is the current
> v0.2.0 code. The `kg-memory/` checkout on this machine is an **older,
> divergent snapshot** (its `QuestionRephraser` still calls
> `self.llm_client.messages.create(...)`, the Anthropic-specific API that the
> shipped wheel replaced with a provider-agnostic `self.llm.complete(...)`), so
> it is **not** the right place to land these. Apply the patches to whichever
> repo currently builds `knowlytix_harness` / `knowlytix_core`.

Applying locally (already done in this environment, so the book/tests benefit now):

```bash
cd <site-packages>           # or the knowlytix source checkout root
patch -p1 < knowlytix_contrib/patches/core_llm_client.py.patch
patch -p1 < knowlytix_contrib/patches/harness_testing_judge.py.patch
patch -p1 < knowlytix_contrib/patches/harness_graphdoe_question_rephraser.py.patch
```

---

## 1. `LLMClient.complete_batch` — provider-agnostic batch completion
**File:** `knowlytix/core/llm/client.py` (package `knowlytix_core`)

Adds a `complete_batch(messages_list, **kwargs) -> list[str]` to `LLMClient`. The
base implementation dispatches sequentially, so it is correct for **every**
provider and for `DemoLLMClient` out of the box. Adapters backed by an engine
that decodes a batch in one forward pass (a local Hugging Face model, vLLM, TGI)
override it to exploit that parallelism — and callers see no difference because
they only depend on the signature.

This is the general form of the `agentlab` `QwenAdapter.complete_batch` wrapper:
the speedup belongs to *any* batch-capable LLM, not to one model.

## 2. `GMSJudge.apply_thresholds` — wire calibration into the live router
**File:** `knowlytix/harness/testing/judge.py` (package `knowlytix_harness`)

**Problem.** `GMSJudge.__init__` builds the `VerificationRouter` (and its
`GraphVerifier` / `LogicVerifier`) *before* any calibration exists. `calibrate()`
and `load()` populate `self._thresholds`, but nothing copies those values onto
the already-built verifiers — so a freshly calibrated or loaded judge still
routes claims through *uncalibrated* boundaries and typed verdicts come back
`NOT_CALIBRATED`. (We hit exactly this: 0 / N claims verified until we wired it
by hand.)

**Fix.** A public, idempotent `apply_thresholds()` that pushes `_thresholds` onto
the router using the documented mapping, and is now called automatically at the
end of both `calibrate()` and `load()`:

| threshold key | verifier attribute |
|---|---|
| `geodesic` | `router.graph.plausibility_threshold` |
| `holonomy` | `router.logic.config.tau_path` |
| `tau_ent` | `router.logic.config.tau_ent` |
| `tau_contra` | `router.logic.config.tau_contra` |

Missing keys / absent verifiers are skipped, so it is a safe no-op before any
calibration. This is the general form of our `_calibrate_and_wire` helper —
after this change, a user just calls `judge.calibrate()` (or `judge.load()`) and
the verifiers are ready; no private-attribute poking required.

## 3. `QuestionRephraser.expand` — batch the whole design in one LLM call
**File:** `knowlytix/harness/graphdoe/question_rephraser.py` (package `knowlytix_harness`)

`expand()` previously rephrased one (design-row × question) at a time — one
blocking `llm.complete` per item. It now builds every rephrase prompt up front
and submits them in a single `self.llm.complete_batch(...)` call, falling back to
sequential `complete` when the client doesn't implement it. Output order,
`doe_qid` numbering, and the ground-truth invariants are unchanged.

To keep the single- and batched paths from drifting, `_rephrase_llm` was split
into `_build_rephrase_messages` (prompt construction; returns `None` when no
text-modifying factor applies) and `_finish_rephrase` (parse → `GeneratedQuestion`);
both paths now share them.

Combined with (1), a local model rephrases an entire DOE design in batched
forward passes instead of one question at a time — the general form of our
`BatchedQwenRephraser`, which can now be deleted in favor of the stock
`QuestionRephraser(method="llm", llm=<batch-capable adapter>)`.

---

### Why these three
Each began as a wrapper in `agentlab/` that only existed because Knowlytix lacked
a hook a real user needs:

| `agentlab` wrapper | Promoted to |
|---|---|
| `QwenAdapter.complete_batch` | `LLMClient.complete_batch` (any provider) |
| `_calibrate_and_wire(bench)` | `GMSJudge.apply_thresholds()` (auto-called) |
| `BatchedQwenRephraser` | batched `QuestionRephraser.expand` |

All three are additive and behavior-preserving for existing callers.
