"""Atomic hash-verified tensor-only Task 2 recovery and model exports."""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import random
import shutil
import uuid
import numpy as np
import torch
from .io import atomic_json, read_json, sha256


def identity(value):
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def rng_state():
    n = np.random.get_state()
    return {"python": random.getstate(), "numpy": [n[0], n[1].tolist(), n[2], n[3], n[4]],
            "cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["cpu"])
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG device count differs")
        torch.cuda.set_rng_state_all(state["cuda"])


@contextmanager
def preserve_rng(model=None):
    state = rng_state()
    training = model.training if model is not None else None
    try:
        yield
    finally:
        restore_rng(state)
        if model is not None:
            model.train(training)


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return type(value)(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def model_state(model):
    names = {name for name,p in model.named_parameters() if p.requires_grad}
    return {name: tensor.detach().cpu().clone() for name,tensor in model.state_dict().items()
            if name in names or name == "visual_scale"}


def restore_model(model, tensors):
    expected = set(model_state(model))
    if set(tensors) != expected:
        raise ValueError("Connector/adapter state keys differ from configured stage")
    current = model.state_dict()
    for key, tensor in tensors.items():
        if tensor.shape != current[key].shape or tensor.dtype != current[key].dtype:
            raise ValueError(f"Checkpoint tensor incompatible: {key}")
    model.load_state_dict(tensors, strict=False)


def publish(destination, payload):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / (".incomplete-"+uuid.uuid4().hex)
    temporary.mkdir()
    try:
        torch.save(cpu_copy(payload), temporary / "state.pt")
        atomic_json(temporary / "manifest.json", {"version": 1, "sha256": sha256(temporary / "state.pt"),
                                                  "spec_id": identity(payload["spec"]),
                                                  "stage": payload["progress"]["stage"],
                                                  "global_update": payload["progress"]["global_update"],
                                                  "kind": payload["kind"]})
        if destination.exists():
            raise FileExistsError(destination)
        temporary.rename(destination)
    except BaseException:
        resolved=temporary.resolve()
        if resolved.parent==destination.parent.resolve() and resolved.name.startswith(".incomplete-"):
            shutil.rmtree(resolved, ignore_errors=True)
        raise
    return read_json(destination / "manifest.json")


def read_payload(source, spec=None):
    source = Path(source)
    manifest = read_json(source / "manifest.json")
    if manifest["version"] != 1 or sha256(source / "state.pt") != manifest["sha256"]:
        raise ValueError("Checkpoint integrity failure")
    payload = torch.load(source / "state.pt", map_location="cpu", weights_only=True)
    if identity(payload["spec"]) != manifest["spec_id"] or (spec is not None and payload["spec"] != spec):
        raise ValueError("Checkpoint configuration differs")
    if any(payload["progress"][key] != manifest[key] for key in ("stage", "global_update")):
        raise ValueError("Checkpoint progress metadata differs")
    if payload["kind"] != manifest["kind"]:
        raise ValueError("Checkpoint kind differs")
    return payload
