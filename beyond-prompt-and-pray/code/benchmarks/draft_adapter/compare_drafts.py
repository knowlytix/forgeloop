"""Compare draft adapters on groundedness and quality.

Variants, all generating a reply for each case in the held-out draft eval set
(``data/training/bank_policy/draft_response_eval.jsonl``):

  - ``base``         : Qwen2.5-3B-Instruct *prompted* (no adapter) --- the
    baseline the shipped LoRA must beat.
  - ``lora_shipped`` : the currently shipped adapter (data/draft_response_lm_qwen).
  - ``lora_v3``      : the retuned LoRA on MeMo v3.
  - ``rora_v3``      : the RoRA adapter on MeMo v3.

Two scores per variant:

  - **groundedness** (the anti-hallucination target, deterministic): a draft is
    grounded iff it cites every expected keyword, contains every expected fact
    (byte-exact number), and uses no forbidden phrase. Reported as a fraction.
  - **quality** (LLM-as-judge, Sonnet): mean 1-5, plus unauthorized-promise /
    off-topic / empty flag counts.

Run from the repo root with the venv active and ANTHROPIC_API_KEY set:

    python -m benchmarks.draft_adapter.compare_drafts
    python -m benchmarks.draft_adapter.compare_drafts --variants base lora_v3 rora_v3 --no-judge
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "data" / "training" / "bank_policy" / "draft_response_eval.jsonl"
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
SHIPPED_LORA = REPO / "data" / "draft_response_lm_qwen"
LORA_V3 = REPO / "data" / "draft_response_lora_memo_v3"
RORA_V3 = REPO / "data" / "draft_response_rora_memo_v3"

_DRAFT_SYS = (
    "You write a short, professional customer-service reply for a retail bank. "
    "Use the cited policy. Be concise (2-3 sentences). Do not promise a fee waiver "
    "or refund you are not authorized to grant. Return ONLY the reply text."
)


def read_jsonl(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def is_grounded(text: str, case: dict) -> bool:
    low = text.lower()
    if not all(k.lower() in low for k in case.get("expected_citation_keywords", [])):
        return False
    if not all(str(f).lower() in low for f in case.get("expected_facts", [])):
        return False
    if any(p.lower() in low for p in case.get("forbidden_phrases", [])):
        return False
    return True


def _truncate(text: str, max_sentences: int = 3) -> str:
    for sep in ("\n\n", "\nComplaint:", "\nIssue:", "\nPolicy:"):
        if sep in text:
            text = text.split(sep, 1)[0]
    text = text.strip()
    out, buf = [], ""
    for ch in text:
        buf += ch
        if ch in ".!?" and len(buf.strip()) > 5:
            out.append(buf.strip()); buf = ""
            if len(out) >= max_sentences:
                break
    if buf.strip() and len(out) < max_sentences:
        out.append(buf.strip())
    return " ".join(out).strip()


def load_base(device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map=device.type if device.type == "cuda" else None)
    if device.type != "cuda":
        m = m.to(device)
    m.eval()
    return m, tok


def gen_completion(model, tok, prompt, device, max_new_tokens=80):
    enc = tok(prompt + " ", return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    text = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return _truncate(text)


def gen_chat(model, tok, case, device, max_new_tokens=160):
    user = (f"Customer message: {case['complaint']}\nIssue: {case['issue']}\n"
            f"Relevant policy: {case['prompt'].split('Policy:')[1].split('Response:')[0].strip()}\n"
            "Write the reply.")
    chat = [{"role": "system", "content": _DRAFT_SYS}, {"role": "user", "content": user}]
    enc = tok.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True).to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def drafts_for_variant(variant, cases, device, fold_rora_inference=False):
    """Return list of (case, draft_text)."""
    from transformers import AutoTokenizer

    if variant == "base":
        model, tok = load_base(device)
        return [(c, gen_chat(model, tok, c, device)) for c in cases]

    if variant == "sonnet":
        # Frontier-model reference, prompted on the same cases (no fine-tune).
        from benchmarks.prompted_baseline.chat_backends import AnthropicChatBackend
        backend = AnthropicChatBackend.from_env()
        if backend is None:
            raise RuntimeError("sonnet variant needs ANTHROPIC_API_KEY")
        out = []
        for c in cases:
            policy = c["prompt"].split("Policy:")[1].split("Response:")[0].strip()
            user = (f"Customer message: {c['complaint']}\nIssue: {c['issue']}\n"
                    f"Relevant policy: {policy}\nWrite the reply.")
            out.append((c, backend.chat(_DRAFT_SYS, user, max_new_tokens=160).strip()))
        return out

    if variant in ("lora_shipped", "lora_v3"):
        from peft import PeftModel
        path = SHIPPED_LORA if variant == "lora_shipped" else LORA_V3
        base, tok = load_base(device)
        tok = AutoTokenizer.from_pretrained(path)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = PeftModel.from_pretrained(base, str(path))
        model.eval()
        return [(c, gen_completion(model, tok, c["prompt"], device)) for c in cases]

    if variant == "rora_v3":
        from benchmarks.draft_adapter.rora_adapter import cache_rora, fold_rora, load_rora_into
        base, _ = load_base(device)
        tok = AutoTokenizer.from_pretrained(RORA_V3)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = load_rora_into(base, RORA_V3)
        if fold_rora_inference:
            # Deployment fast-path: fold rotations into the bf16 projection
            # weights -> base speed, but greedy decoding can drift slightly.
            folded = fold_rora(model)
            print(f"  folded {folded} RoRA rotations into projection weights (fast path)")
        else:
            # Default: cache the rotation cores once. Per-token cost is matmul-
            # only (LoRA-class) and bit-identical to the float32 forward.
            cached = cache_rora(model)
            print(f"  cached {cached} RoRA rotation cores (faithful fast path)")
        model.eval()
        return [(c, gen_completion(model, tok, c["prompt"], device)) for c in cases]

    raise ValueError(variant)


@dataclass
class Score:
    label: str
    n: int = 0
    grounded: int = 0
    judge_scores: list = field(default_factory=list)
    promises: int = 0
    off_topic: int = 0
    empty: int = 0
    # GMS geometric judge: tier counts over fee-claim (scorable) drafts.
    gms_grounded: int = 0
    gms_distortion: int = 0
    gms_fabrication: int = 0
    gms_na: int = 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+",
                    default=["base", "lora_shipped", "lora_v3", "rora_v3"])
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM (Sonnet) judge")
    ap.add_argument("--no-gms", action="store_true", help="skip the GMS geometric judge")
    ap.add_argument("--fold-rora", action="store_true",
                    help="use the folded RoRA deployment fast-path (bf16, ~3x faster, may drift)")
    ap.add_argument("--show", action="store_true", help="print each draft")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cases = read_jsonl(EVAL)
    print(f"{len(cases)} draft eval cases")

    judge = None
    if not args.no_judge:
        from benchmarks.prompted_baseline.chat_backends import AnthropicChatBackend
        from benchmarks.prompted_baseline.judge import LLMJudge
        backend = AnthropicChatBackend.from_env()
        if backend is None:
            print("NOTE: no ANTHROPIC_API_KEY; skipping LLM judge.")
        else:
            judge = LLMJudge(backend)

    # The existing GMS geometric judge (Chapter-15 test harness): it maps a
    # draft's fee claim to a store triple and scores it on the manifold, banding
    # the geodesic into grounded / distortion / fabrication. It is the geometric
    # counterpart to the LLM judge --- it catches a wrong dollar amount that
    # reads fluently, where issues with no numeric claim score n/a.
    gms = None
    if not args.no_gms:
        try:
            from agentlab.testing.capstone_harness import CapstoneTestHarness
            gms = CapstoneTestHarness()
            gms.calibrate_groundedness()  # warm the store + bands once
        except Exception as exc:  # noqa: BLE001
            print(f"NOTE: GMS judge unavailable ({type(exc).__name__}: {str(exc)[:80]}).")

    scores = []
    for variant in args.variants:
        path_missing = (
            (variant == "lora_v3" and not LORA_V3.exists()) or
            (variant == "rora_v3" and not RORA_V3.exists()) or
            (variant == "lora_shipped" and not SHIPPED_LORA.exists())
        )
        if path_missing:
            print(f"skip {variant}: adapter dir not found"); continue
        print(f"\n--- {variant} ---")
        s = Score(label=variant, n=len(cases))
        for case, draft in drafts_for_variant(variant, cases, device, args.fold_rora):
            if is_grounded(draft, case):
                s.grounded += 1
            if judge is not None:
                v = judge.judge_draft(case["complaint"], draft, case["issue"])
                s.judge_scores.append(v.score)
                s.promises += int(v.unauthorized_promise)
                s.off_topic += int(v.off_topic)
                s.empty += int(v.empty)
            if gms is not None:
                _, _geo, tier = gms.judge_draft(draft, case["issue"])
                if tier == "grounded":
                    s.gms_grounded += 1
                elif tier == "distortion":
                    s.gms_distortion += 1
                elif tier == "fabrication":
                    s.gms_fabrication += 1
                else:
                    s.gms_na += 1
            if args.show:
                g = "G" if is_grounded(draft, case) else " "
                print(f"  [{g}] {case['id']}: {draft[:90]}")
        scores.append(s)

    print("\n" + "=" * 78)
    print("DRAFT ADAPTER COMPARISON  (GMS judge and LLM judge side by side)")
    print("=" * 78)
    hdr = (f"{'variant':14s} {'grounded':>11s} | {'LLM avg':>7s} {'promise':>7s} {'empty':>5s} | "
           f"{'GMS: grnd/dist/fab/na':>22s}")
    print(hdr); print("-" * 78)
    for s in scores:
        avg = sum(s.judge_scores) / len(s.judge_scores) if s.judge_scores else 0.0
        g = f"{s.grounded}/{s.n} ({100*s.grounded/s.n:.0f}%)"
        gms_col = f"{s.gms_grounded}/{s.gms_distortion}/{s.gms_fabrication}/{s.gms_na}"
        print(f"{s.label:14s} {g:>11s} | {avg:7.2f} {s.promises:7d} {s.empty:5d} | {gms_col:>22s}")
    print("\n  grounded = deterministic: cites keyword + exact fact + no forbidden phrase (all cases)")
    print("  LLM avg  = Sonnet draft-quality 1-5 (0 if disabled); promise/empty = flag counts")
    print("  GMS      = geometric tier of the draft's fee claim: grounded/distortion/fabrication/na")
    print("             (na = no numeric fee claim to score; the existing Ch-15 GMS judge)")


if __name__ == "__main__":
    main()
