import json
from pathlib import Path
import pytest
import torch
from atml_pa3.task1.evaluation.checkpoints import (
    canonical_hash, export_inference_state, read_inference_state,
    validate_run_spec, validate_training_checkpoint, verified_state,
)
from atml_pa3.task1.evaluation.model import merge_adapter_modules_
from atml_pa3.task1.model_setup import MODEL_ID, MODEL_REVISION, AdapterLinear, TARGETS
from atml_pa3.task1.run_controller import FROZEN_DATA_SHA256
from atml_pa3.task1.recovery import save_checkpoint


def spec(method):
    return dict(method=method, seed=6304, model_id=MODEL_ID,
                model_revision=MODEL_REVISION, dataset_sha256=FROZEN_DATA_SHA256,
                total_examples=40000, checkpoint_updates=[312, 625, 937, 1250])


def checkpoint(tmp_path, method='full', kind='full', update=625):
    model = torch.nn.Linear(3, 2)
    if kind == 'trainable':
        model.weight.requires_grad_(False)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 1)
    scaler = torch.amp.GradScaler('cpu', enabled=False)
    path = tmp_path / 'checkpoint'
    save_checkpoint(path, model=model, optimizer=opt, scheduler=scheduler,
                    scaler=scaler, run_spec=spec(method), completed_updates=update,
                    state_kind=kind)
    return path


def test_full_checkpoint_roundtrip_and_export(tmp_path):
    source = checkpoint(tmp_path)
    metadata = validate_training_checkpoint(source, 'full_mid')
    assert metadata['completed_updates'] == 625
    state, run_spec, meta = verified_state(source, 'full_mid')
    assert set(state) == {'weight', 'bias'}
    assert canonical_hash(run_spec) == meta['run_spec_sha256']
    output = tmp_path / 'export'
    manifest = export_inference_state(source, 'full_mid', output)
    tensors, restored_manifest = read_inference_state(output, 'full_mid')
    assert manifest == restored_manifest
    assert set(tensors) == set(state)
    assert torch.equal(state['weight'], tensors['weight'])
    with pytest.raises(FileExistsError):
        export_inference_state(source, 'full_mid', output)


def test_corruption_and_wrong_update_are_rejected(tmp_path):
    source = checkpoint(tmp_path)
    with pytest.raises(ValueError, match='Incorrect checkpoint optimizer update'):
        validate_training_checkpoint(source, 'full_final')
    with (source / 'state.pt').open('ab') as f:
        f.write(b'corrupt')
    with pytest.raises(ValueError, match='SHA-256'):
        verified_state(source, 'full_mid')


def test_run_spec_mismatch_rejected():
    s = spec('full')
    validate_run_spec(s, 'full_mid', canonical_hash(s))
    s['seed'] = 1
    with pytest.raises(ValueError, match='seed'):
        validate_run_spec(s, 'full_mid')


def test_export_checksum_rejected(tmp_path):
    source = checkpoint(tmp_path)
    output = tmp_path / 'export'
    export_inference_state(source, 'full_mid', output)
    with (output / 'model_state.pt').open('ab') as f:
        f.write(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        read_inference_state(output, 'full_mid')


class ToyModel(torch.nn.Module):
    def __init__(self, protected=False):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList()
        for i in range(2):
            layer = torch.nn.Module()
            layer.self_attn = torch.nn.Module()
            layer.mlp = torch.nn.Module()
            for j, (group, name) in enumerate(TARGETS):
                raw = torch.nn.Linear(8, 8, bias=False, dtype=torch.float16)
                adapter = AdapterLinear(raw, rank=2, alpha=4, init_seed=100 + i * 7 + j,
                                        projection_rank=2 if protected else None,
                                        compute_bases=True)
                setattr(getattr(layer, group), name, adapter)
            self.model.layers.append(layer)


def test_merge_uses_projected_factors():
    model = ToyModel(protected=True)
    adapter = model.model.layers[0].mlp.down_proj
    with torch.no_grad():
        adapter.lora_B.normal_()
    a, b = adapter.exported_factors()
    expected = adapter.base.weight.float() + adapter.scale * (b @ a)
    count = merge_adapter_modules_(model)
    assert count == 14
    merged = model.model.layers[0].mlp.down_proj.weight.float()
    torch.testing.assert_close(merged, expected.half().float(), rtol=0, atol=0)
