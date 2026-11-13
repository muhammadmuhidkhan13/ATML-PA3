"""Model reconstruction and projected-adapter merging for evaluation."""
from __future__ import annotations
from pathlib import Path
import torch
from torch import nn

from ..model_setup import (
    AdapterLinear, MODEL_ID, MODEL_REVISION, METHOD_SETTINGS,
    TARGETS, configure_model,
)
from .conditions import CONDITIONS
from .checkpoints import read_inference_state, verified_state


def load_model_for_evaluation(condition: str, *, device: str = 'cuda:0',
                              checkpoint: Path | None = None,
                              export: Path | None = None,
                              merge_adapters: bool = True):
    """Create original pinned Qwen and restore exact training checkpoint state.

    FP32 full-FT weights are loaded into an FP16 inference model. The original
    backbone must be fetched from pinned HF revision; never from a Task 1 run.
    """
    if condition not in CONDITIONS:
        raise ValueError(f'Unknown evaluation condition: {condition}')
    if checkpoint is not None and export is not None:
        raise ValueError('Choose checkpoint OR inference export, not both')
    c = CONDITIONS[condition]
    if c.method is None and (checkpoint is not None or export is not None):
        raise ValueError('Baseline must not specify a checkpoint')
    if c.method is not None and checkpoint is None and export is None:
        raise ValueError('Trained condition requires a checkpoint or export')

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                                dtype=torch.float16,
                                                attn_implementation='eager')
    if c.method not in (None, 'full', 'l2sp'):
        configure_model(model, c.method, precompute_bases=False)

    manifest = {'condition': condition, 'model_id': MODEL_ID, 'model_revision': MODEL_REVISION}
    if c.method is not None:
        if export is not None:
            state, saved = read_inference_state(export, condition)
            manifest['inference_export'] = str(export)
            manifest['checkpoint_sha256'] = saved['source_checkpoint_sha256']
        else:
            state, spec, saved = verified_state(checkpoint, condition)
            manifest['training_checkpoint'] = str(checkpoint)
            manifest['checkpoint_sha256'] = saved['state_sha256']
        # Strict full key matching for full FT, exact adapter/buffer matching for trainable-only.
        if c.method in ('full', 'l2sp'):
            model.load_state_dict(state, strict=True)
        else:
            expected = {name for name, p in model.named_parameters() if p.requires_grad}
            if c.method == 'oplora16':
                expected |= {name for name, _ in model.named_buffers()
                             if name.endswith(('.basis_u', '.basis_v', '.basis_ready'))}
            if set(state) != expected:
                missing, extra = sorted(expected - set(state)), sorted(set(state) - expected)
                raise ValueError(f'Adapter state mismatch: missing={missing[:8]} extra={extra[:8]}')
            # Only overwrite explicitly saved parameters and registered buffers.
            model.load_state_dict(state, strict=False)
            if c.method == 'oplora16':
                for module in model.modules():
                    if isinstance(module, AdapterLinear):
                        if not bool(module.basis_ready.item()):
                            raise ValueError('OPLoRA basis not restored')
                        module._basis_ready_cached = True
        del state

    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()
    if merge_adapters and c.method not in (None, 'full', 'l2sp'):
        merge_adapter_modules_(model)
        manifest['adapters_merged'] = True
    else:
        manifest['adapters_merged'] = False
    model.to(device)
    model.eval()
    return model, tokenizer, manifest


@torch.no_grad()
def merge_adapter_modules_(model: nn.Module):
    """Merge actual projected OPLoRA factors, not unprojected B@A.

    FP32 accumulation is cast back to the frozen base's FP16 weight.
    Replaces adapters in-place, so call only on a fresh evaluation model.
    """
    count = 0
    for layer in model.model.layers:
        for group_name, name in TARGETS:
            group = getattr(layer, group_name)
            module = getattr(group, name)
            if not isinstance(module, AdapterLinear):
                raise ValueError(f'Expected adapter at {group_name}.{name}')
            a, b = module.exported_factors()
            base = module.base
            update = (b.float() @ a.float()) * module.scale
            merged = base.weight.detach().float() + update
            base.weight.copy_(merged.to(base.weight.dtype))
            setattr(group, name, base)
            count += 1
    if count == 0:
        raise ValueError('No adapters to merge')
    return count


@torch.inference_mode()
def verify_merge_logits(model: nn.Module, token_ids: torch.Tensor,
                        *, atol: float = 0.15, rtol: float = 0.02) -> dict:
    """Compare nonmerged logits then merge this *evaluation* model in-place.

    FP16 merged rounding can differ from unmerged FP32 adapter accumulation;
    record the measured maximum error rather than assuming exact equality.
    """
    model.eval()
    before = model(input_ids=token_ids, use_cache=False).logits.float().detach()
    count = merge_adapter_modules_(model)
    after = model(input_ids=token_ids, use_cache=False).logits.float().detach()
    max_abs = (after - before).abs().max().item()
    if not torch.allclose(before, after, atol=atol, rtol=rtol):
        raise AssertionError(f'Merged and unmerged logits disagree (max abs {max_abs:.6g})')
    return {'adapter_modules_merged': count, 'max_abs_logit_delta': max_abs,
            'atol': atol, 'rtol': rtol}
