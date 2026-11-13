"""Qwen3 Task 1 model factory: full FT, LoRA, and projected LoRA.

No import-time downloads or training. No quantization.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import torch
from torch import nn
from torch.nn import functional as F

MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
SEED = 6304

TARGETS = (
    ("self_attn", "q_proj"),
    ("self_attn", "k_proj"),
    ("self_attn", "v_proj"),
    ("self_attn", "o_proj"),
    ("mlp", "gate_proj"),
    ("mlp", "up_proj"),
    ("mlp", "down_proj"),
)

METHOD_SETTINGS = {
    "full": (None, None, 2e-5),
    "l2sp": (None, None, 2e-5),
    "lora4": (4, 8, 2e-4),
    "lora16": (16, 32, 2e-4),
    "oplora16": (16, 32, 2e-4),
}

DROPOUT = 0.05
PROTECTION_RANK = 128


class AdapterLinear(nn.Module):
    """Frozen FP16 linear plus trainable FP32 A/B, optionally projected."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: int,
        init_seed: int,
        *,
        projection_rank: int | None = None,
        compute_bases: bool = True,
    ):
        super().__init__()

        if not isinstance(base, nn.Linear):
            raise TypeError("Expected an ordinary nn.Linear")

        if base.weight.dtype != torch.float16:
            raise TypeError("Adapter backbone must be FP16")

        if rank < 1 or alpha < 1:
            raise ValueError("Invalid rank/alpha")

        self.base = base

        for param in self.base.parameters():
            param.requires_grad_(False)

        input_size = base.in_features
        output_size = base.out_features

        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.protection_rank = projection_rank
        self.dropout = nn.Dropout(p=DROPOUT)

        # A private CPU generator makes the initial unprojected
        # A/B factors identical in LoRA-16 and OPLoRA-16.
        rng = torch.Generator(device="cpu").manual_seed(init_seed)

        a = torch.empty(
            rank, input_size, dtype=torch.float32
        )

        nn.init.kaiming_uniform_(
            a, a=math.sqrt(5), generator=rng
        )

        self.lora_A = nn.Parameter(
            a.to(base.weight.device)
        )

        self.lora_B = nn.Parameter(
            torch.zeros(
                output_size,
                rank,
                dtype=torch.float32,
                device=base.weight.device,
            )
        )

        if projection_rank is not None:
            if not 0 < projection_rank < min(
                input_size, output_size
            ):
                raise ValueError(
                    "Protection rank incompatible with target"
                )

            self.register_buffer(
                "basis_u",
                torch.zeros(
                    output_size,
                    projection_rank,
                    dtype=torch.float32,
                    device=base.weight.device,
                ),
                persistent=True,
            )

            self.register_buffer(
                "basis_v",
                torch.zeros(
                    input_size,
                    projection_rank,
                    dtype=torch.float32,
                    device=base.weight.device,
                ),
                persistent=True,
            )

            self.register_buffer(
                "basis_ready",
                torch.tensor(
                    False,
                    device=base.weight.device,
                ),
                persistent=True,
            )

            self._basis_ready_cached = False

            if compute_bases:
                self.compute_svd_bases()

    @torch.no_grad()
    def compute_svd_bases(self):
        if self.protection_rank is None:
            raise ValueError(
                "Standard LoRA has no projection bases"
            )

        w = self.base.weight.detach().float()

        u, _, vh = torch.linalg.svd(
            w, full_matrices=False,
            # The default CUDA driver produced insufficiently orthogonal
            # leading bases on the real Qwen down_proj preflight. Use the
            # QR-based full SVD; driver selection is CUDA-only in PyTorch.
            **({"driver": "gesvd"} if w.is_cuda else {}),
        )

        k = self.protection_rank

        self.basis_u.copy_(
            u[:, :k].contiguous()
        )

        self.basis_v.copy_(
            vh[:k, :].T.contiguous()
        )

        self.basis_ready.fill_(True)
        self._basis_ready_cached = True

    def effective_factors(self):
        if self.protection_rank is None:
            return self.lora_A, self.lora_B

        if not self._basis_ready_cached:
            # On checkpoint restore, check once instead of
            # synchronizing GPU-to-CPU on every forward.
            if not bool(self.basis_ready.item()):
                raise RuntimeError(
                    "OPLoRA bases not ready; initialize or resume"
                )

            self._basis_ready_cached = True

        with torch.autocast(
            device_type=self.lora_A.device.type,
            enabled=False,
        ):
            a = self.lora_A.float()
            b = self.lora_B.float()
            u = self.basis_u.float()
            v = self.basis_v.float()

            a_proj = a - (a @ v) @ v.T
            b_proj = b - u @ (u.T @ b)

        return a_proj, b_proj

    def forward(self, x):
        base_out = self.base(x)

        a, b = self.effective_factors()

        # CUDA autocast provides FP16 linear computation.
        # The adapter parameters remain in FP32.
        z = F.linear(
            self.dropout(x).float(), a
        )

        delta = F.linear(z, b) * self.scale

        return base_out + delta.to(
            dtype=base_out.dtype
        )

    @torch.no_grad()
    def exported_factors(self):
        """Get factors for merging and diagnostics."""
        a, b = self.effective_factors()

        return (
            a.detach().clone(),
            b.detach().clone(),
        )


@dataclass(frozen=True)
class ModelInfo:
    method: str
    peak_lr: float
    trainable_parameters: int
    total_parameters: int
    adapter_modules: int
    state_kind: str
    extra_state_keys: tuple[str, ...]
    basis_preprocessing_seconds: float


def _transformer_layers(model):
    if (
        not hasattr(model, "model")
        or not hasattr(model.model, "layers")
    ):
        raise ValueError(
            "Expected Qwen3 model.model.layers structure"
        )

    return model.model.layers


def configure_model(
    model: nn.Module,
    method: str,
    *,
    seed: int = SEED,
    precompute_bases: bool = True,
) -> ModelInfo:
    """Configure a fresh model without starting training."""

    if method not in METHOD_SETTINGS:
        raise ValueError(f"Unknown method: {method}")

    rank, alpha, lr = METHOD_SETTINGS[method]
    layers = _transformer_layers(model)

    if not layers:
        raise ValueError("No transformer layers")

    if getattr(
        model.config, "model_type", None
    ) == "qwen3":
        if len(layers) != 28:
            raise ValueError(
                "Expected 28 layers in Qwen3-0.6B"
            )

    model.config.use_cache = False

    if method in ("full", "l2sp"):
        if any(
            p.dtype != torch.float32
            for p in model.parameters()
        ):
            raise ValueError(
                "Full FT needs FP32 master parameters"
            )

        for p in model.parameters():
            p.requires_grad_(True)

        state_kind = "full"
        modules = 0
        elapsed = 0.0

    else:
        if any(
            p.dtype != torch.float16
            for p in model.parameters()
        ):
            raise ValueError(
                "Adapter backbone must be entirely FP16"
            )

        for p in model.parameters():
            p.requires_grad_(False)

        state_kind = "trainable"
        modules = 0
        t0 = time.perf_counter()

        for idx, layer in enumerate(layers):
            for j, (
                group_name, linear_name
            ) in enumerate(TARGETS):

                group = getattr(
                    layer, group_name
                )

                base = getattr(
                    group, linear_name
                )

                if not isinstance(
                    base, nn.Linear
                ):
                    raise TypeError(
                        f"Unexpected target: {idx} {linear_name}"
                    )

                projected = method == "oplora16"

                wrapped = AdapterLinear(
                    base,
                    rank,
                    alpha,
                    seed + idx * len(TARGETS) + j,
                    projection_rank=(
                        PROTECTION_RANK
                        if projected else None
                    ),
                    compute_bases=precompute_bases,
                )

                setattr(
                    group, linear_name, wrapped
                )

                modules += 1

        elapsed = (
            time.perf_counter() - t0
            if method == "oplora16"
            else 0.0
        )

        if modules != len(layers) * len(TARGETS):
            raise AssertionError(
                "Missing adapter targets"
            )

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        }
    )

    if not bool(model.is_gradient_checkpointing):
        raise RuntimeError(
            "Gradient checkpointing did not enable"
        )

    extra_keys = tuple(sorted(
        name for name, _ in model.named_buffers()
        if name.endswith(
            (".basis_u", ".basis_v", ".basis_ready")
        )
    ))

    if (
        method == "oplora16"
        and len(extra_keys) != 3 * modules
    ):
        raise AssertionError(
            "Missing OPLoRA persistent basis buffers"
        )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    if not trainable:
        raise AssertionError(
            "Model has no trainable parameters"
        )

    if (
        method in ("full", "l2sp")
        and trainable != total
    ):
        raise AssertionError(
            "Full FT must train every parameter"
        )

    if method not in ("full", "l2sp"):
        for name, p in model.named_parameters():
            expected = name.endswith(
                (".lora_A", ".lora_B")
            )

            if p.requires_grad != expected:
                raise AssertionError(
                    f"Unexpected trainability: {name}"
                )

            if (
                p.requires_grad
                and p.dtype != torch.float32
            ):
                raise AssertionError(
                    f"Non-FP32 adapter: {name}"
                )

    return ModelInfo(
        method=method,
        peak_lr=lr,
        trainable_parameters=trainable,
        total_parameters=total,
        adapter_modules=modules,
        state_kind=state_kind,
        extra_state_keys=extra_keys,
        basis_preprocessing_seconds=elapsed,
    )


def load_qwen_model(
    method: str,
    *,
    device: str | torch.device,
    resume_oplora=False,
):
    """Explicit model loading. Never trains by itself."""

    if method not in METHOD_SETTINGS:
        raise ValueError(method)

    if resume_oplora and method != "oplora16":
        raise ValueError(
            "resume_oplora is only valid for OPLoRA"
        )

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    dtype = (
        torch.float32
        if method in ("full", "l2sp")
        else torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        use_fast=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=dtype,
        attn_implementation="eager",
    )

    model.to(device=device)

    info = configure_model(
        model,
        method,
        precompute_bases=not resume_oplora,
    )

    if tokenizer.pad_token_id is None:
        raise ValueError(
            "Tokenizer has no padding token"
        )

    return model, tokenizer, info
