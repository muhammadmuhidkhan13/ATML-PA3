"""CPU-only model-factory tests; no Qwen downloads."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from atml_pa3.task1 import model_setup as ms


class Attn(nn.Module):
    def __init__(self, n):
        super().__init__()
        for name in (
            "q_proj", "k_proj", "v_proj", "o_proj"
        ):
            setattr(
                self, name,
                nn.Linear(n, n, bias=False)
            )


class Mlp(nn.Module):
    def __init__(self, n):
        super().__init__()
        for name in (
            "gate_proj", "up_proj", "down_proj"
        ):
            setattr(
                self, name,
                nn.Linear(n, n, bias=False)
            )


class Layer(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.self_attn = Attn(n)
        self.mlp = Mlp(n)


class FakeQwen(nn.Module):
    def __init__(self, n=8, layers=2):
        super().__init__()

        self.config = SimpleNamespace(
            model_type="tiny_fake",
            use_cache=True,
        )

        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            Layer(n) for _ in range(layers)
        )

        self.is_gradient_checkpointing = False
        self.kwargs = None

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs
    ):
        self.is_gradient_checkpointing = True
        self.kwargs = gradient_checkpointing_kwargs


def make_fake(seed=6304, dtype=torch.float16):
    torch.manual_seed(seed)
    return FakeQwen().to(dtype=dtype)


def named_adapters(model):
    return {
        k: v
        for k, v in model.named_modules()
        if isinstance(v, ms.AdapterLinear)
    }


def test_full_and_l2sp_all_float32_and_trainable():
    for method in ("full", "l2sp"):
        model = make_fake(dtype=torch.float32)
        info = ms.configure_model(model, method)

        assert info.state_kind == "full"
        assert info.peak_lr == 2e-5
        assert info.adapter_modules == 0
        assert (
            info.trainable_parameters
            == info.total_parameters
        )
        assert model.kwargs == {
            "use_reentrant": False
        }
        assert model.config.use_cache is False


def test_reject_wrong_full_precision():
    with pytest.raises(ValueError, match="FP32"):
        ms.configure_model(
            make_fake(), "full"
        )


def test_lora_ranks_and_frozen_base():
    for name, rank in (
        ("lora4", 4), ("lora16", 16)
    ):
        model = make_fake()
        info = ms.configure_model(
            model, name
        )

        assert info.adapter_modules == 14
        assert info.peak_lr == 2e-4
        assert info.state_kind == "trainable"
        assert (
            info.trainable_parameters
            == 14 * rank * (8 + 8)
        )
        assert not info.extra_state_keys

        for adapter in named_adapters(
            model
        ).values():
            assert (
                adapter.lora_A.dtype
                == torch.float32
            )
            assert (
                adapter.lora_B.dtype
                == torch.float32
            )
            assert not (
                adapter.base.weight.requires_grad
            )
            assert (
                adapter.base.weight.dtype
                == torch.float16
            )
            assert (
                torch.count_nonzero(
                    adapter.lora_B
                ).item() == 0
            )


def test_lora16_oplora16_exact_raw_init(
    monkeypatch
):
    monkeypatch.setattr(
        ms, "PROTECTION_RANK", 2
    )

    lora = make_fake()
    oplora = make_fake()

    ms.configure_model(
        lora, "lora16"
    )
    info = ms.configure_model(
        oplora, "oplora16"
    )

    assert info.adapter_modules == 14
    assert len(info.extra_state_keys) == 14 * 3

    for key, la in named_adapters(lora).items():
        oa = named_adapters(oplora)[key]

        torch.testing.assert_close(
            la.lora_A, oa.lora_A
        )
        torch.testing.assert_close(
            la.lora_B, oa.lora_B
        )
        torch.testing.assert_close(
            la.base.weight, oa.base.weight
        )
        assert oa.basis_ready.item()


def test_projected_factors_orthogonal_and_gradients(
    monkeypatch
):
    monkeypatch.setattr(
        ms, "PROTECTION_RANK", 2
    )

    model = make_fake()
    ms.configure_model(
        model, "oplora16"
    )

    adapter = named_adapters(model)[
        "model.layers.0.self_attn.q_proj"
    ]

    with torch.no_grad():
        adapter.lora_B.normal_(std=0.05)

    a, b = adapter.effective_factors()

    torch.testing.assert_close(
        a @ adapter.basis_v,
        torch.zeros(16, 2),
        atol=1e-5,
        rtol=0,
    )

    torch.testing.assert_close(
        adapter.basis_u.T @ b,
        torch.zeros(2, 16),
        atol=1e-5,
        rtol=0,
    )

    x = torch.randn(3, 8).half()

    adapter(x).float().square().mean().backward()

    assert adapter.lora_A.grad is not None
    assert adapter.lora_B.grad is not None
    assert adapter.base.weight.grad is None
    assert adapter(x).dtype == torch.float16


def test_restore_bases_without_recomputing(
    monkeypatch
):
    monkeypatch.setattr(
        ms, "PROTECTION_RANK", 2
    )

    model = make_fake()

    ms.configure_model(
        model,
        "oplora16",
        precompute_bases=False,
    )

    adapter = named_adapters(model)[
        "model.layers.0.self_attn.q_proj"
    ]

    with pytest.raises(
        RuntimeError,
        match="bases not ready",
    ):
        adapter.effective_factors()

    with torch.no_grad():
        adapter.compute_svd_bases()

    assert adapter.basis_ready.item()
    assert len(adapter.exported_factors()) == 2


def test_invalid_method_and_missing_target():
    with pytest.raises(
        ValueError,
        match="Unknown method",
    ):
        ms.configure_model(
            make_fake(), "nonexistent"
        )

    broken = make_fake()

    del (
        broken.model.layers[0]
        .self_attn.v_proj
    )

    with pytest.raises(AttributeError):
        ms.configure_model(
            broken, "lora4"
        )
