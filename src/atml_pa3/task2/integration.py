"""Bounded real-component Part 1 validation; zero optimizer updates."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import gc
import time
import numpy as np
import torch
from .features import FeatureStore
from .io import atomic_json, read_json, read_jsonl, sha256
from .model import load_vlm, encode_example, QWEN_REVISION


def validate_real_vlm(data_dir, cache_dir, output_dir, device="cuda:0"):
    data_dir, cache_dir, output_dir = Path(data_dir), Path(cache_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    store = FeatureStore(cache_dir)
    captions = read_jsonl(data_dir / "captions_train.jsonl")
    image_manifest = read_json(data_dir / "images.json")
    train_ids = sorted({row["image_id"] for row in captions})
    selected = [train_ids[int(i)] for i in np.random.default_rng(6304).permutation(len(train_ids))[:32]]
    if any(image_manifest[str(i)]["source_split"] != "train2014" for i in selected):
        raise ValueError("Calibration image split leakage")
    caption_map = {}
    for row in captions:
        caption_map.setdefault(row["image_id"], row)
    started = time.perf_counter()
    model, tokenizer = load_vlm(device)
    encodings = {i: encode_example(tokenizer, caption_map[i]["question"], caption_map[i]["target"], caption=True)
                 for i in selected}
    text_ids = []
    for i in selected:
        # Content only: caption IDs plus the capped captioning question.
        row = caption_map[i]
        text_ids.extend(tokenizer.encode(row["question"], add_special_tokens=False)[:80])
        text_ids.extend(tokenizer.encode(row["target"], add_special_tokens=False)[:64])
    calibration = model.calibrate([store.get(i) for i in selected], text_ids,
                                  split="train", image_ids=selected)
    calibration.update({"feature_manifest_sha256": sha256(cache_dir / "manifest.json"),
                        "dataset_manifest_sha256": sha256(data_dir / "manifest.json"),
                        "qwen_revision": QWEN_REVISION})
    torch.save({"connector": model.connector.state_dict(), "visual_scale": model.visual_scale.detach().cpu()},
               output_dir / "connector_initial_calibrated.pt")
    calibration["connector_state_sha256"] = sha256(output_dir / "connector_initial_calibrated.pt")
    atomic_json(output_dir / "calibration.json", calibration)
    examples = [encodings[i] for i in selected[:2]]
    features = [store.get(i) for i in selected[:2]]
    model.train()
    model.zero_grad(set_to_none=True)
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.AdamW(model.connector.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=256.0)
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        output = model(examples, features)
    if not torch.isfinite(output["loss"]):
        raise AssertionError("Nonfinite real VLM loss")
    scaler.scale(output["loss"]).backward()
    scaler.unscale_(optimizer)
    gradient_norm = 0.0
    for name, p in model.connector.named_parameters():
        if p.dtype != torch.float32 or p.grad is None or not torch.isfinite(p.grad).all() or not torch.count_nonzero(p.grad):
            raise AssertionError(f"Invalid connector gradient: {name}")
        gradient_norm += p.grad.float().square().sum().item()
    if any(p.requires_grad or p.grad is not None for p in model.language_model.parameters()):
        raise AssertionError("Original Qwen parameters are not frozen")
    loss, per_example = output["loss"].item(), output["per_example_loss"].detach().cpu().tolist()
    visual_length = output["attention_mask"].sum(1).cpu().tolist()
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    del output
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        no_image = model(examples, [None, None])
    no_image_length = no_image["attention_mask"].sum(1).cpu().tolist()
    if [a-b for a,b in zip(visual_length, no_image_length)] != [196,196]:
        raise AssertionError("No-image input did not remove exactly 196 visual positions")
    # Independently check each original supervised answer ends with assistant EOT.
    for example in examples:
        supervised = [i for i in example.labels if i != -100]
        if supervised[-1] != tokenizer.convert_tokens_to_ids("<|im_end|>"):
            raise AssertionError("Missing assistant EOT supervision")
    result = {"status": "passed", "qwen_revision": QWEN_REVISION,
              "image_ids": selected[:2], "loss": loss, "per_example_loss": per_example,
              "connector_parameters": sum(p.numel() for p in model.connector.parameters()),
              "connector_gradient_norm": gradient_norm ** 0.5,
              "all_connector_gradients_finite_nonzero": True, "qwen_frozen_no_gradients": True,
              "visual_sequence_lengths": visual_length, "no_image_sequence_lengths": no_image_length,
              "no_image_loss": no_image["loss"].item(), "peak_cuda_memory_bytes": peak,
              "seconds_including_model_load": time.perf_counter()-started,
              "calibration_scale": calibration["scale"], "optimizer_updates": 0,
              "tokenization": [asdict(e) for e in examples]}
    atomic_json(output_dir / "result.json", result)
    del model, no_image, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {k:v for k,v in result.items() if k != "tokenization"}
