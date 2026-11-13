"""Validate trusted training-recovery states, export compact inference states.

ATTENTION: recovery's state.pt contains Python pickles and optimizer states.
Only use this loader with checkpoints created by your own trusted training run.
"""
from __future__ import annotations
import hashlib
import json
import uuid
from pathlib import Path

from .conditions import CONDITIONS
from ..model_setup import MODEL_ID, MODEL_REVISION
from ..run_controller import FROZEN_DATA_SHA256


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_hash(obj: dict) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_training_checkpoint(directory: Path, condition: str) -> dict:
    if condition not in CONDITIONS or condition == 'baseline':
        raise ValueError('A nonbaseline evaluation condition is required')
    condition_spec = CONDITIONS[condition]
    directory = Path(directory)
    meta = _json(directory / 'metadata.json')
    if meta.get('format_version') != 1:
        raise ValueError('Unsupported training checkpoint format')
    if meta.get('completed_updates') != condition_spec.update:
        raise ValueError('Incorrect checkpoint optimizer update')
    if meta.get('next_example_offset') != condition_spec.update * 32:
        raise ValueError('Incorrect dataset position')
    expected_state = 'full' if condition_spec.method in ('full', 'l2sp') else 'trainable'
    if meta.get('state_kind') != expected_state:
        raise ValueError('Incorrect model state kind')
    if not isinstance(meta.get('saved_model_keys'), list) or not meta['saved_model_keys']:
        raise ValueError('Empty saved model key inventory')
    expected_statefile_hash = meta.get('state_sha256')
    if not isinstance(expected_statefile_hash, str) or sha256_file(directory / 'state.pt') != expected_statefile_hash:
        raise ValueError('Checkpoint SHA-256 mismatch')
    return meta


def validate_run_spec(spec: dict, condition: str, expected_hash: str | None = None) -> None:
    c = CONDITIONS[condition]
    if c.method is None:
        raise ValueError('Cannot attach a training run spec to baseline')
    required = {'method': c.method, 'seed': 6304, 'model_id': MODEL_ID,
                'model_revision': MODEL_REVISION, 'dataset_sha256': FROZEN_DATA_SHA256,
                'total_examples': 40000}
    for key, val in required.items():
        if spec.get(key) != val:
            raise ValueError(f'Run specification mismatch at {key}: {spec.get(key)!r}')
    if list(spec.get('checkpoint_updates', [])) != [312, 625, 937, 1250]:
        raise ValueError('Unexpected checkpoint schedule')
    if expected_hash is not None and canonical_hash(spec) != expected_hash:
        raise ValueError('Checkpoint run-spec hash mismatch')


def verified_state(directory: Path, condition: str) -> tuple[dict, dict, dict]:
    """Load *trusted* training recovery payload; returns (model, spec, metadata).

    Do not call on public downloads / untrusted checkpoints. Torch pickle loading
    is only permitted after explicit provenance validation and SHA-256 checking.
    """
    import torch
    meta = validate_training_checkpoint(directory, condition)
    payload = torch.load(Path(directory) / 'state.pt', map_location='cpu', weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError('Invalid checkpoint payload')
    embedded_meta = dict(meta)
    embedded_meta.pop('state_sha256', None)
    if payload.get('metadata') != embedded_meta:
        raise ValueError('Embedded checkpoint metadata mismatch')
    spec = payload.get('run_spec')
    if not isinstance(spec, dict):
        raise ValueError('No embedded training run spec')
    validate_run_spec(spec, condition, expected_hash=meta['run_spec_sha256'])
    state = payload.get('model')
    if not isinstance(state, dict) or set(state) != set(meta['saved_model_keys']):
        raise ValueError('Model state inventory mismatch')
    if not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError('Non-tensor model state entry')
    if meta['state_kind'] == 'trainable':
        expected_extra = set(meta.get('extra_state_keys', []))
        if condition == 'oplora16_final':
            if not expected_extra or not all(k.endswith(('.basis_u', '.basis_v', '.basis_ready')) for k in expected_extra):
                raise ValueError('Missing/invalid OPLoRA basis buffers')
            if not expected_extra.issubset(state):
                raise ValueError('Missing OPLoRA projection bases')
        elif expected_extra:
            raise ValueError('Unexpected protected subspace buffers')
    return state, spec, meta


def export_inference_state(source: Path, condition: str, destination: Path) -> dict:
    """Extract compact inference state to independent, atomic output directory.

    Does not modify the source checkpoint. Stored dtype remains unchanged;
    full-FT FP32 checkpoints are converted to FP16 only on evaluation load.
    """
    import torch
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(f'Output already exists: {destination}')
    state, spec, meta = verified_state(source, condition)
    temporary = destination.parent / f'.incomplete-{destination.name}-{uuid.uuid4().hex}'
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        torch.save(state, temporary / 'model_state.pt')
        manifest = {
            'format_version': 1, 'condition': condition,
            'source_checkpoint_sha256': meta['state_sha256'],
            'state_kind': meta['state_kind'],
            'state_keys': sorted(state),
            'run_spec': spec,
            'checkpoint_update': meta['completed_updates'],
            'model_state_sha256': sha256_file(temporary / 'model_state.pt'),
        }
        (temporary / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        temporary.rename(destination)
    except BaseException:
        import shutil
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def read_inference_state(directory: Path, condition: str) -> tuple[dict, dict]:
    """Read a verified *trusted/local* inference-only export."""
    import torch
    if condition == 'baseline' or condition not in CONDITIONS:
        raise ValueError('A nonbaseline condition is required')
    directory = Path(directory)
    manifest = _json(directory / 'manifest.json')
    if manifest.get('format_version') != 1 or manifest.get('condition') != condition:
        raise ValueError('Incorrect inference export identity')
    if manifest.get('checkpoint_update') != CONDITIONS[condition].update:
        raise ValueError('Incorrect inference export update')
    validate_run_spec(manifest.get('run_spec', {}), condition)
    if sha256_file(directory / 'model_state.pt') != manifest.get('model_state_sha256'):
        raise ValueError('Inference model-state hash mismatch')
    # Export has plain tensors only, so safe restricted tensor loading is possible.
    state = torch.load(directory / 'model_state.pt', map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or set(state) != set(manifest.get('state_keys', [])):
        raise ValueError('Inference model keys mismatch')
    return state, manifest
