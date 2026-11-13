"""Bounded OPLoRA checks on one cached Qwen weight, without model training."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .model_setup import AdapterLinear, MODEL_ID, MODEL_REVISION, SEED
from .recovery import load_checkpoint, save_checkpoint
from .train_core import make_adamw, make_scheduler


@torch.no_grad()
def projection_metrics(adapter: AdapterLinear) -> dict:
    """Measure relative constraint residuals in FP32 on nonzero factors."""
    a, b = adapter.exported_factors()
    u, v = adapter.basis_u, adapter.basis_v
    def relative(numerator, denominator):
        return float((torch.linalg.vector_norm(numerator)
                      / torch.linalg.vector_norm(denominator).clamp_min(1e-12)).item())
    return {
        "u_orthogonality_error": float((u.T @ u - torch.eye(u.shape[1], device=u.device)).abs().max().item()),
        "v_orthogonality_error": float((v.T @ v - torch.eye(v.shape[1], device=v.device)).abs().max().item()),
        "a_constraint_relative": relative(a @ v, a),
        "b_constraint_relative": relative(u.T @ b, b),
        "update_output_interference": relative(u.T @ (b @ a), b @ a) ** 2,
    }


def _tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def run_weight_preflight(output_dir: Path, *, layer: int = 0) -> dict:
    """SVD, nonzero projections, local backward, export, and fresh-object restore."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not 0 <= layer < 28:
        raise ValueError("Qwen layer must be in 0..27")
    if not torch.cuda.is_available():
        raise RuntimeError("Real-weight preflight requires CUDA")
    snapshot = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True))
    key = f"model.layers.{layer}.mlp.down_proj.weight"
    weight = None
    for path in sorted(snapshot.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            if key in handle.keys():
                weight = handle.get_tensor(key).half()
                break
    if weight is None:
        raise FileNotFoundError(f"Cached weight not found: {key}")
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        base = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False,
                               dtype=torch.float16, device=device)
        with torch.no_grad():
            base.weight.copy_(weight.to(device))
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        adapter = AdapterLinear(base, 16, 32, SEED + layer * 7 + 6,
                                projection_rank=128)
        torch.cuda.synchronize(device)
        svd_seconds = time.perf_counter() - started
        # B starts at zero in training. A synthetic nonzero B prevents vacuous
        # projection/export checks; it is not a learned or official adapter.
        with torch.no_grad():
            adapter.lora_B.normal_(std=0.01)
        adapter.eval()
        x = torch.randn(2, weight.shape[1], device=device).half()
        metrics = projection_metrics(adapter)
        if any(value > 1e-4 for value in metrics.values()):
            raise AssertionError(f"Projection residual too large: {metrics}")
        a, b = adapter.exported_factors()
        with torch.no_grad():
            factored = F.linear(F.linear(x.float(), a), b) * adapter.scale
            dense = F.linear(x.float(), (b @ a) * adapter.scale)
            torch.testing.assert_close(factored, dense, rtol=1e-4, atol=1e-5)
            with torch.autocast("cuda", dtype=torch.float16):
                expected_output = adapter(x).detach().clone()
        # Exercise autograd through both projected factors, not Qwen SFT.
        with torch.autocast("cuda", dtype=torch.float16):
            loss = adapter(x).float().square().mean()
        loss.backward()
        for name, param in adapter.named_parameters():
            if param.requires_grad:
                if param.grad is None or not torch.isfinite(param.grad).all():
                    raise AssertionError(f"Invalid projected gradient: {name}")
                if not torch.count_nonzero(param.grad):
                    raise AssertionError(f"Zero projected gradient: {name}")
            elif param.grad is not None:
                raise AssertionError(f"Frozen parameter has gradient: {name}")
        opt = make_adamw(adapter, 2e-4)
        sched = make_scheduler(opt)
        scaler = torch.amp.GradScaler("cuda", init_scale=256.0)
        spec = {"smoke_only": True, "kind": "single_weight_projection",
                "model_revision": MODEL_REVISION, "weight_key": key,
                "seed": SEED, "rank": 16, "alpha": 32, "protection_rank": 128,
                "svd_driver": "gesvd", "projection_tolerance": 1e-4,
                "weight_fp16_sha256": _tensor_hash(weight), "synthetic_nonzero_B": True}
        checkpoint = output_dir / "checkpoint"
        meta = save_checkpoint(checkpoint, model=adapter, optimizer=opt,
                               scheduler=sched, scaler=scaler, run_spec=spec,
                               completed_updates=0, state_kind="trainable",
                               extra_state_keys=("basis_u", "basis_v", "basis_ready"))
        # Fresh module with the same frozen weight, explicitly without SVD.
        restored = AdapterLinear(base, 16, 32, 77, projection_rank=128, compute_bases=False)
        opt2 = make_adamw(restored, 2e-4)
        load_checkpoint(checkpoint, model=restored, optimizer=opt2,
                        scheduler=make_scheduler(opt2), scaler=torch.amp.GradScaler("cuda"),
                        expected_run_spec=spec)
        for name in ("lora_A", "lora_B", "basis_u", "basis_v", "basis_ready"):
            torch.testing.assert_close(getattr(adapter, name), getattr(restored, name), rtol=0, atol=0)
        restored.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            torch.testing.assert_close(restored(x), expected_output, rtol=0, atol=0)
        result = {
            "status": "passed", "scope": "one real Qwen down_proj; synthetic local inputs/B",
            **spec, "shape": list(weight.shape), "svd_seconds": svd_seconds,
            "projection_metrics": metrics,
            "basis_storage_bytes": sum(t.numel() * t.element_size() for t in
                                       (adapter.basis_u, adapter.basis_v)),
            "peak_cuda_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "gpu": torch.cuda.get_device_name(device),
            "versions": {p: version(p) for p in ("torch", "safetensors", "huggingface_hub")},
            "checkpoint_sha256": meta["state_sha256"],
            "basis_u_sha256": _tensor_hash(adapter.basis_u),
            "basis_v_sha256": _tensor_hash(adapter.basis_v),
            "implementation_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                      for name in ("oplora_preflight.py", "model_setup.py", "recovery.py")},
            "optimizer_updates": 0, "fresh_object_restore_exact": True,
            "fp32_export_equivalence": True,
        }
        (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    args = parser.parse_args(argv)
    print(json.dumps(run_weight_preflight(args.output_dir, layer=args.layer), indent=2))
