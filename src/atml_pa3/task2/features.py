"""FP16 SigLIP patch cache. The model applies final LayerNorm exactly once."""
from __future__ import annotations
import gc
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from .io import atomic_json, read_json, sha256

SIGLIP_ID = "google/siglip-base-patch16-224"
SIGLIP_REVISION = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
SIGLIP_WEIGHTS_SHA256 = "2c63cb7d1f2e95ba501893cbb8faeb4ea9a3af295498d35097126228659c2af8"
FEATURE_SHAPE = (196, 768)


class FeatureStore:
    """Part 2's integrity-checked, CPU FP16 feature interface."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = read_json(self.directory / "manifest.json")
        if not self.manifest.get("complete"):
            raise ValueError("Feature cache is incomplete")
        self.signature = self.manifest["signature"]

    def get(self, image_id):
        record = self.manifest["images"][str(image_id)]
        path = self.directory / record["path"]
        if not valid_feature(path, record, record["source_sha256"], self.signature):
            raise ValueError(f"Invalid cached feature for image {image_id}")
        return torch.from_numpy(np.load(path, allow_pickle=False))


def valid_feature(path, record, source_hash, signature):
    path = Path(path)
    if not path.is_file() or not record or record.get("source_sha256") != source_hash or record.get("signature") != signature:
        return False
    try:
        array = np.load(path, allow_pickle=False)
        return (array.shape == FEATURE_SHAPE and array.dtype == np.float16
                and np.isfinite(array).all() and sha256(path) == record["sha256"])
    except (ValueError, OSError, KeyError):
        return False


def write_feature(path, feature):
    path = Path(path)
    array = np.asarray(feature)
    if array.shape != FEATURE_SHAPE or array.dtype != np.float16 or not np.isfinite(array).all():
        raise ValueError("Expected finite FP16 [196,768] SigLIP features")
    temp = path.with_suffix(".partial")
    with temp.open("wb") as out:
        np.save(out, array, allow_pickle=False)
    temp.replace(path)
    return sha256(path)


def build_cache(root, cache_dir, model_path, batch_size=8, device="cuda:0"):
    from transformers import SiglipVisionModel, SiglipImageProcessorPil
    root, cache_dir, model_path = Path(root), Path(cache_dir), Path(model_path)
    manifest = read_json(root / "image_manifest.json")
    if not manifest["complete"]:
        raise ValueError("Required image acquisition is incomplete")
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    processor = SiglipImageProcessorPil.from_pretrained(model_path, local_files_only=True)
    config = processor.to_dict()
    signature_spec = {"model_id": SIGLIP_ID, "revision": SIGLIP_REVISION,
                      "model_files_sha256": {name: sha256(model_path / name) for name in
                                             ("model.safetensors", "config.json", "preprocessor_config.json")},
                      "versions": {package: version(package) for package in ("torch", "transformers", "Pillow")},
                      "preprocessing": config, "shape": list(FEATURE_SHAPE), "dtype": "float16",
                      "layernorm": "model last_hidden_state already post_layernorm; never apply again",
                      "spatial_order": "row-major 14x14 patch grid, no CLS token"}
    signature = hashlib.sha256(json.dumps(signature_spec, sort_keys=True).encode()).hexdigest()
    if signature_spec["model_files_sha256"]["model.safetensors"] != SIGLIP_WEIGHTS_SHA256:
        raise ValueError("SigLIP weights do not match the verified pinned snapshot")
    cache_dir.mkdir(parents=True, exist_ok=True)
    state_path = cache_dir / "manifest.json"
    state = read_json(state_path) if state_path.exists() else {"images": {}}
    if set(state["images"]) - set(manifest["images"]):
        raise ValueError("Cache contains images outside this selection")
    if state.get("signature") not in (None, signature):
        raise ValueError("Feature cache configuration differs; use a new directory")
    state.update({**signature_spec, "signature": signature,
                  "image_manifest_sha256": sha256(root / "image_manifest.json"), "complete": False})
    pending, skipped = [], 0
    for image_id, image in sorted(manifest["images"].items(), key=lambda item: int(item[0])):
        if sha256(root / image["relative_path"]) != image["sha256"]:
            raise ValueError(f"Source image hash changed: {image_id}")
        if valid_feature(cache_dir / f"{image_id}.npy", state["images"].get(image_id), image["sha256"], signature):
            skipped += 1
        else:
            pending.append((image_id, image))
    if not pending:
        state["complete"] = True
        atomic_json(state_path, state)
        return {"computed": 0, "reused": skipped, "complete": True}
    device = torch.device(device)
    started = time.perf_counter()
    model, loading = SiglipVisionModel.from_pretrained(model_path, local_files_only=True,
                         output_loading_info=True,
                         dtype=torch.float16 if device.type == "cuda" else torch.float32)
    if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise ValueError(f"SigLIP vision weights did not load completely: {loading}")
    model.requires_grad_(False).eval().to(device)
    model.use_head = False
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    layernorm_calls = [0]
    hook = model.post_layernorm.register_forward_hook(lambda *_: layernorm_calls.__setitem__(0, layernorm_calls[0]+1))
    repeats_verified = False
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start+batch_size]
        images = []
        for _, image in batch:
            with Image.open(root / image["relative_path"]) as loaded:
                images.append(loaded.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt")["pixel_values"].to(device=device, dtype=model.dtype)
        before = layernorm_calls[0]
        with torch.no_grad():
            features = model(pixel_values=inputs).last_hidden_state
            if layernorm_calls[0] != before + 1 or tuple(features.shape[1:]) != FEATURE_SHAPE:
                raise AssertionError("SigLIP shape or final-LayerNorm count mismatch")
            if not repeats_verified:
                repeated = model(pixel_values=inputs).last_hidden_state
                torch.testing.assert_close(features, repeated, rtol=0, atol=0)
                repeats_verified = True
            features = features.half().cpu().numpy()
        for (image_id, image), feature in zip(batch, features):
            digest = write_feature(cache_dir / f"{image_id}.npy", feature)
            state["images"][image_id] = {"path": f"{image_id}.npy", "sha256": digest,
                                         "source_sha256": image["sha256"], "signature": signature}
        atomic_json(state_path, state)
        if start % (batch_size * 20) == 0:
            print(f"Features computed {min(start+batch_size,len(pending))}/{len(pending)}; reused {skipped}", flush=True)
    hook.remove()
    state["complete"] = True
    state["verification"] = {"post_layernorm_once_per_forward": True, "deterministic_repeat_exact": repeats_verified,
                             "frozen_parameters": all(not p.requires_grad for p in model.parameters()),
                             "seconds_including_load": time.perf_counter()-started,
                             "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
    atomic_json(state_path, state)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"computed": len(pending), "reused": skipped, "complete": True, **state["verification"]}
