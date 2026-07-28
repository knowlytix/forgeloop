"""Inject RoRA (Rotational Rank Adaptation) into a frozen causal LM.

RoRA replaces LoRA's additive low-rank update ``y = (W + BA)x`` with an
*orthogonal rotation of the input*, ``y = W(R^T x)``, where ``R = exp(UV^T -
VU^T)`` is built from a low-rank skew-symmetric generator. Because ``R`` is
orthogonal it preserves the norm of the hidden state, so the adapter cannot blow
up or collapse activations the way an unconstrained additive update can --- the
property we want after the additive LoRA degraded the base model below its own
prompted baseline.

The ``RoRALayer`` math lives in the model-merging research package; we import it
read-only and only add the injection/save/load glue needed to fine-tune Qwen's
attention projections. Each targeted ``nn.Linear`` is wrapped so its input is
rotated before the (frozen) projection runs; only the rotation generators ``U``,
``V`` train. The rotation is computed in float32 for numerical stability and
cast back to the model dtype.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Import the RoRALayer implementation from the model-merging package (read-only).
_RORA_SRC = "/path/to/model_merging/rora_merge/src"
if _RORA_SRC not in sys.path:
    sys.path.insert(0, _RORA_SRC)
from adapters.rora import RoRALayer  # noqa: E402

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


class RoRAInjected(nn.Module):
    """Wrap a frozen Linear so its input is rotated by a trainable RoRALayer."""

    def __init__(self, linear: nn.Linear, rank: int, init_scale: float = 0.01):
        super().__init__()
        self.linear = linear
        # Keep the rotation in float32 for numerical stability, but on the same
        # device as the (frozen) projection it wraps.
        self.rora = (
            RoRALayer(linear.in_features, rank, init_scale=init_scale)
            .float()
            .to(linear.weight.device)
        )
        self._cached = False  # inference: reuse Q, (Rcore-I) instead of rebuilding

    @torch.no_grad()
    def cache_rotation(self) -> None:
        """Precompute the rotation core once. Q and Rcore depend only on the
        frozen U,V --- not on x --- so during generation they are constant. The
        reference RoRALayer rebuilds them (QR + matrix_exp) on every forward;
        caching makes the per-token path matmul-only, exactly like LoRA, and
        bit-identical to the float32 forward (the apply still runs in float32)."""
        U, V = self.rora.U.float(), self.rora.V.float()
        Q, _ = torch.linalg.qr(torch.cat([U, V], dim=1))
        A, C = Q.T @ U, Q.T @ V
        M = A @ C.T - C @ A.T
        Rcore = torch.matrix_exp(-M)
        I = torch.eye(Rcore.shape[0], device=U.device)
        self._Q = Q
        self._RminusI = Rcore - I
        self._cached = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        if self._cached:
            # Pure matmul (LoRA-class): x + (xQ)(Rcore-I)Qᵀ, in float32.
            x32 = x.float()
            x_rot = x32 + (x32 @ self._Q) @ self._RminusI @ self._Q.T
            return self.linear(x_rot.to(dtype))
        # Training path: rebuild the rotation each step (U,V are updating).
        return self.linear(self.rora(x.float()).to(dtype))


def inject_rora(model: nn.Module, rank: int = 8, targets=DEFAULT_TARGETS) -> int:
    """Replace each targeted Linear with a RoRA-wrapped version; freeze the base.

    Returns the number of adapters injected."""
    to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.split(".")[-1] in targets:
            to_replace.append(name)

    for name in to_replace:
        *parents, attr = name.split(".")
        parent = model
        for p in parents:
            parent = getattr(parent, p)
        linear = getattr(parent, attr)
        setattr(parent, attr, RoRAInjected(linear, rank))

    for n, p in model.named_parameters():
        p.requires_grad = ".rora." in n

    return len(to_replace)


def rora_state(model: nn.Module) -> dict:
    """Collect the trainable rotation generators keyed by module name."""
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, RoRAInjected):
            state[name] = {
                "U": module.rora.U.detach().cpu(),
                "V": module.rora.V.detach().cpu(),
            }
    return state


def save_rora(model: nn.Module, out_dir: Path, rank: int, base_model: str,
              targets=DEFAULT_TARGETS) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(rora_state(model), out_dir / "rora_adapters.pt")
    (out_dir / "rora_config.json").write_text(json.dumps({
        "peft_type": "RORA",
        "base_model": base_model,
        "rank": rank,
        "targets": list(targets),
    }, indent=2))


def cache_rora(model: nn.Module) -> int:
    """Cache every RoRA rotation core for fast, faithful inference. Per-token
    cost drops to LoRA-class matmuls and output is bit-identical to the float32
    forward. Returns the number of adapters cached."""
    n = 0
    for m in model.modules():
        if isinstance(m, RoRAInjected):
            m.cache_rotation()
            n += 1
    return n


@torch.no_grad()
def fold_rora(model: nn.Module) -> int:
    """Fold each trained rotation into its projection weight for fast inference.

    The rotation R is constant once U,V are frozen, and ``y = W(R^T x) = (W R^T)
    x``. So we compute ``W' = W @ R^T`` once per layer and swap the RoRAInjected
    wrapper back to a plain nn.Linear holding W'. After folding there is zero
    per-token rotation cost --- generation runs at base-model speed --- and no
    QR / matrix_exp kernels at all.

    We avoid materializing the full d x d R^T: with R^T = I + Q(Rcore-I)Q^T,
    ``W R^T = W + (W Q)(Rcore - I) Q^T``, all done in float32 then cast back.
    Returns the number of layers folded.
    """
    targets = [(name, m) for name, m in model.named_modules() if isinstance(m, RoRAInjected)]
    for name, wrapped in targets:
        linear = wrapped.linear
        rora = wrapped.rora
        dev, dt = linear.weight.device, linear.weight.dtype

        U, V = rora.U.float(), rora.V.float()
        Q, _ = torch.linalg.qr(torch.cat([U, V], dim=1))      # (d, 2r)
        A, C = Q.T @ U, Q.T @ V                               # (2r, r)
        M = A @ C.T - C @ A.T                                 # (2r, 2r) skew
        Rcore = torch.matrix_exp(-M)                          # R^T core
        I = torch.eye(Rcore.shape[0], device=U.device)
        W = linear.weight.float()                             # (out, d)
        WQ = W @ Q.to(W.device)                               # (out, 2r)
        W_new = W + (WQ @ (Rcore - I)) @ Q.T                  # (out, d) = W R^T

        new_linear = nn.Linear(linear.in_features, linear.out_features,
                               bias=linear.bias is not None)
        new_linear.weight = nn.Parameter(W_new.to(dev, dt), requires_grad=False)
        if linear.bias is not None:
            new_linear.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)

        *parents, attr = name.split(".")
        parent = model
        for p in parents:
            parent = getattr(parent, p)
        setattr(parent, attr, new_linear.to(dev))
    return len(targets)


def load_rora_into(model: nn.Module, adapter_dir: Path) -> nn.Module:
    """Inject RoRA at the saved rank/targets and load the trained generators."""
    adapter_dir = Path(adapter_dir)
    cfg = json.loads((adapter_dir / "rora_config.json").read_text())
    inject_rora(model, rank=cfg["rank"], targets=tuple(cfg["targets"]))
    state = torch.load(adapter_dir / "rora_adapters.pt", map_location="cpu")
    by_name = {name: m for name, m in model.named_modules() if isinstance(m, RoRAInjected)}
    for name, tensors in state.items():
        m = by_name[name]
        with torch.no_grad():
            m.rora.U.copy_(tensors["U"].to(m.rora.U.device, m.rora.U.dtype))
            m.rora.V.copy_(tensors["V"].to(m.rora.V.device, m.rora.V.dtype))
    return model
