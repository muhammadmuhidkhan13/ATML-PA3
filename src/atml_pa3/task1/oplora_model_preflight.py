"""All-target OPLoRA SVD/backward and fresh-model checkpoint validation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .model_setup import AdapterLinear, MODEL_REVISION, SEED, load_qwen_model
from .oplora_preflight import projection_metrics
from .qwen_preflight import check_label_roundtrip, select_short_examples, single_backward_probe
from .recovery import load_checkpoint, save_checkpoint
from .run_controller import FROZEN_DATA_SHA256
from .train_core import collate_microbatch, make_adamw, make_scheduler, verify_dataset


@torch.no_grad()
def check_all_projections(model, *, expected_modules=196) -> list[dict]:
    """Check nonzero corrections, then restore the zero-B initialization."""
    adapters = [(name, module) for name, module in model.named_modules()
                if isinstance(module, AdapterLinear)]
    if len(adapters) != expected_modules:
        raise AssertionError(f"Expected {expected_modules} adapters; found {len(adapters)}")
    results = []
    for name, adapter in adapters:
        if torch.count_nonzero(adapter.lora_B):
            raise AssertionError("Preflight expects untouched zero-B initialization")
        try:
            adapter.lora_B.normal_(std=0.01)
            metrics = projection_metrics(adapter)
            record = {"name": name, "shape": list(adapter.base.weight.shape), **metrics}
            results.append(record)
        finally:
            adapter.lora_B.zero_()
    return results


@torch.no_grad()
def prompt_logits(model, rows, pad_token_id, device):
    model.eval()
    batch = collate_microbatch(rows, pad_token_id, device)
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                       use_cache=False)
    return output.logits.detach().cpu()


def run_model_preflight(data_dir: Path, output_dir: Path) -> dict:
    """No optimizer updates; this is not an official run or evaluation."""
    output_dir, data_dir = Path(output_dir), Path(data_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("Full Qwen OPLoRA preflight requires CUDA")
    data_path = verify_dataset(data_dir, FROZEN_DATA_SHA256)
    rows = select_short_examples(data_path, max_tokens=128, count=2)
    output_dir.mkdir(parents=True, exist_ok=False)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda:0")
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        # Memory statistics require an initialized CUDA context.
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        print("Loading pinned Qwen and computing SVD bases for all 196 targets...", flush=True)
        model, tokenizer, info = load_qwen_model("oplora16", device=device)
        torch.cuda.synchronize(device)
        setup_seconds = time.perf_counter() - started
        setup_peak = torch.cuda.max_memory_allocated(device)
        check_label_roundtrip(rows, tokenizer)
        if info.adapter_modules != 196 or len(info.extra_state_keys) != 588:
            raise AssertionError("Unexpected OPLoRA module/basis count")
        print("Checking all bases and nonzero projected corrections...", flush=True)
        metrics = check_all_projections(model)
        (output_dir / "projection_metrics.json").write_text(
            json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        failed = [r["name"] for r in metrics if any(
            not np.isfinite(value) or value > 1e-4 for key, value in r.items()
            if key not in ("name", "shape"))]
        if failed:
            raise AssertionError(f"Projection checks failed for {len(failed)} targets: {failed[:5]}")
        basis_bytes = sum(b.numel() * b.element_size() for name, b in model.named_buffers()
                          if name.endswith((".basis_u", ".basis_v")))
        print("Running real-Qwen short forward/backward, with zero optimizer updates...", flush=True)
        backward = single_backward_probe(model, rows, tokenizer.pad_token_id, device)
        if backward["gradient_tensors"] != 392 or backward["nonzero_B_gradients"] != 196:
            raise AssertionError("Not all OPLoRA adapters received expected gradients")
        expected_logits = prompt_logits(model, rows, tokenizer.pad_token_id, device)
        optimizer = make_adamw(model, 2e-4)
        scheduler = make_scheduler(optimizer)
        scaler = torch.amp.GradScaler("cuda", init_scale=256.0)
        spec = {"smoke_only": True, "method": "oplora16", "seed": SEED,
                "model_revision": MODEL_REVISION, "dataset_sha256": FROZEN_DATA_SHA256,
                "source_indices": [row["source_index"] for row in rows],
                "max_tokens": 128, "optimizer_updates": 0, "svd_driver": "gesvd",
                "synthetic_nonzero_B_for_projection_checks_only": True,
                "projection_tolerance": 1e-4,
                "implementation_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                          for name in ("oplora_model_preflight.py", "oplora_preflight.py",
                                                       "model_setup.py", "qwen_preflight.py", "recovery.py")}}
        checkpoint = output_dir / "checkpoint"
        meta = save_checkpoint(checkpoint, model=model, optimizer=optimizer, scheduler=scheduler,
                               scaler=scaler, run_spec=spec, completed_updates=0,
                               state_kind=info.state_kind, extra_state_keys=info.extra_state_keys)
        # Free the original GPU model before restoring into a newly loaded one.
        del model, optimizer, scheduler, scaler
        gc.collect()
        torch.cuda.empty_cache()
        print("Reloading fresh pinned Qwen and restoring all saved bases without SVD...", flush=True)
        restored, tokenizer, restored_info = load_qwen_model("oplora16", device=device, resume_oplora=True)
        if any(bool(m.basis_ready.item()) for m in restored.modules() if isinstance(m, AdapterLinear)):
            raise AssertionError("Fresh resume model unexpectedly computed SVD bases")
        opt2 = make_adamw(restored, 2e-4)
        metadata = load_checkpoint(checkpoint, model=restored, optimizer=opt2,
                                   scheduler=make_scheduler(opt2), scaler=torch.amp.GradScaler("cuda"),
                                   expected_run_spec=spec)
        payload = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=False)
        actual = restored.state_dict()
        for name, expected in payload["model"].items():
            torch.testing.assert_close(actual[name].cpu(), expected, rtol=0, atol=0)
        if metadata["next_example_offset"] != 0 or metadata["completed_updates"] != 0:
            raise AssertionError("Unexpected restored cursor")
        torch.testing.assert_close(prompt_logits(restored, rows, tokenizer.pad_token_id, device),
                                   expected_logits, rtol=0, atol=0)
        result = {"status": "passed", **spec, "model_info": asdict(info),
                  "setup_seconds_including_load": setup_seconds,
                  "setup_peak_cuda_memory_gib": setup_peak / 1024**3,
                  "basis_storage_bytes": basis_bytes, "projection_targets_checked": len(metrics),
                  "max_projection_metrics": {key: max(row[key] for row in metrics)
                                             for key in metrics[0] if key not in ("name", "shape")},
                  "backward": backward, "restored_model_tensors": len(payload["model"]),
                  "all_checkpoint_tensors_exact": True, "fresh_model_logits_exact": True,
                  "resume_without_svd": True, "checkpoint_sha256": meta["state_sha256"],
                  "gpu": torch.cuda.get_device_name(device), "torch_version": torch.__version__}
        (output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return result
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run_model_preflight(args.data_dir, args.output_dir), indent=2))
