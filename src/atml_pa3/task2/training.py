"""Continuous three-stage controller. Importing this module never trains."""
from collections import Counter
from pathlib import Path
import time
import uuid
import torch
from .adapters import attach_attention_lora, assert_trainables, make_stage_optimizer, backbone_hash
from .curriculum import Recipe, development_updates
from .io import atomic_json,read_json
from .state import model_state, restore_model, publish, read_payload, rng_state, restore_rng,identity


def event(path, record):
    import json
    with Path(path).open("a", encoding="utf-8", newline="\n") as out:
        out.write(json.dumps(record, sort_keys=True, allow_nan=False)+"\n")
        out.flush()


def attempt_update(model, records, optimizer, scheduler, scaler, loader, recipe=Recipe()):
    if len(records) != recipe.effective_batch:
        raise ValueError("Incomplete effective batch")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    measured = 0.
    device = next(model.connector.parameters()).device
    for start in range(0, len(records), recipe.microbatch):
        batch = records[start:start+recipe.microbatch]
        examples, features = loader(batch)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            output = model(examples, features)
            # Sum example means / effective batch, never normalize by microbatch/tokens.
            loss = output["per_example_loss"].sum()/len(records)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite supervised loss")
        measured += loss.detach().float().item()
        scaler.scale(loss).backward()
        del loss,output,examples,features
    scaler.unscale_(optimizer)
    params = [p for group in optimizer.param_groups for p in group["params"]]
    if any(p.grad is None for p in params):
        raise AssertionError("A trainable parameter is missing gradients")
    # FP64 norm accumulation avoids overflow of a norm of finite FP32 gradients.
    norm = torch.stack([p.grad.detach().double().square().sum() for p in params]).sum().sqrt()
    finite = bool(torch.isfinite(norm))
    if finite:
        factor = min(1., recipe.clip_norm/(norm.item()+1e-6))
        for p in params:
            p.grad.mul_(factor)
    elif not scaler.is_enabled():
        raise FloatingPointError("Nonfinite gradients without dynamic scaler")
    previous = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    completed = not scaler.is_enabled() or scaler.get_scale() >= previous
    if completed:
        scheduler.step()
        for state in optimizer.state.values():
            for key in ("exp_avg", "exp_avg_sq"):
                if state[key].dtype != torch.float32:
                    raise AssertionError("AdamW moments must remain FP32")
    return {"sft_loss": measured, "completed": completed,
            "grad_norm_before_clip": norm.item() if finite else None,
            "grad_norm_finite": finite, "scale_before": previous, "scale_after": float(scaler.get_scale())}


def run_curriculum(model, sampler, loader, run_dir, spec, *, recipe=Recipe(),
                   resume_from=None, stop_after=None, evaluate=None,
                   attach=attach_attention_lora, scaler=None, eval_updates=None,setup_metadata=None):
    """Authorized caller only. Toy tests inject small budgets/models; CLI locks recipe.

    Checkpoints belong to the completed stage. On resume, restore that stage
    before making its next transition. Optimizer moments reset at each stage.
    """
    if spec.get('task')=='task2':
        recipe.assert_official()
        if spec.get('recipe')!=recipe.serial():raise ValueError('Official controller recipe differs from recorded run spec')
    initialization_started=time.perf_counter()
    run_dir = Path(run_dir)
    if stop_after is None:
        stop_after = recipe.total_updates
    if stop_after not in recipe.boundaries and stop_after % recipe.checkpoint_interval:
        raise ValueError("Stopping requires a checkpoint boundary")
    if not 0 < stop_after <= recipe.total_updates:
        raise ValueError("Invalid stopping update")
    payload = read_payload(resume_from, spec) if resume_from else None
    if payload and payload["kind"] != "recovery":
        raise ValueError("A model export is not a recovery checkpoint")
    if payload:
        progress = payload["progress"]
        if Path(resume_from).resolve().parent != (run_dir / "checkpoints").resolve():
            raise ValueError("Resume checkpoint belongs to another run")
    else:
        if run_dir.exists():
            raise FileExistsError("New run must use an unused directory")
        progress = {"stage": 1, "global_update": 0, "stage_update": 0,
                    "completed_examples": 0, "skipped_attempts": 0, "evaluated": [],
                    "sft_loss_sum": 0., "training_seconds": 0., "mixture": {}}
        run_dir.mkdir(parents=True)
        atomic_json(run_dir / "run_spec.json", spec)
    if progress["global_update"] > stop_after:
        raise ValueError("Cannot stop before restored progress")
    recipe.validate_progress(progress)
    stage = progress["stage"]
    if stage >= 2:
        attach(model)
    initial_hash = backbone_hash(model)
    if payload and initial_hash != payload["backbone_hash"]:
        raise ValueError("Frozen starting backbone differs")
    optimizer, scheduler = make_stage_optimizer(model, stage, recipe)
    device = next(model.connector.parameters()).device
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=256.)
    if payload:
        restore_model(model, payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        scaler.load_state_dict(payload["scaler"])
        sampler.load_state_dict(payload["sampler"])
        if sampler.stage != stage or sampler.completed_examples != progress["completed_examples"]:
            raise ValueError("Sampler/checkpoint stage or cursor differs")
        restore_rng(payload["rng"])
    session = uuid.uuid4().hex
    log = run_dir / "updates.jsonl"
    journal_path=run_dir/"development_events.json"
    journal=read_json(journal_path) if journal_path.exists() else {"spec_id":identity(spec),"updates":[]}
    if journal["spec_id"]!=identity(spec): raise ValueError("Development event journal differs")
    events = set(development_updates() if eval_updates is None else eval_updates)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def evaluate_current():
        update = progress["global_update"]
        if evaluate is not None and update in events and update not in progress["evaluated"]:
            from .state import preserve_rng
            started = time.perf_counter()
            with preserve_rng(model):
                evaluate(model, update, stage)
            progress["evaluated"].append(update)
            if update not in journal["updates"]:
                journal["updates"].append(update)
                atomic_json(journal_path,journal)
                event(log, {"event": "development", "global_update": update,
                            "stage": stage, "seconds": time.perf_counter()-started, "session": session})

    def export_current():
        destination = run_dir / "exports" / f"stage_{stage}"
        if destination.exists():
            old = read_payload(destination, spec)
            if old["progress"]["global_update"] != progress["global_update"]:
                raise ValueError("Existing stage export differs")
            return
        if backbone_hash(model) != initial_hash:
            raise AssertionError("Original Qwen backbone changed")
        publish(destination, {"kind": "model_export", "spec": spec, "progress": dict(progress),
                              "backbone_hash": initial_hash, "model": model_state(model)})

    event(log, {"event": "resume" if payload else "start", "stage": stage,
                "global_update": progress["global_update"], "session": session})
    event(log,{"event":"stage_setup","stage":stage,"session":session,
               "controller_initialization_seconds":time.perf_counter()-initialization_started,
               "trainable_parameters":assert_trainables(model,stage),"runtime_setup":setup_metadata})
    if progress["stage_update"] == recipe.budgets[stage-1]:
        export_current()
    evaluate_current()
    while progress["global_update"] < stop_after:
        if progress["stage_update"] == recipe.budgets[stage-1]:
            started = time.perf_counter()
            stage += 1
            if stage == 2:
                attach(model)
            sampler.set_stage(stage)
            optimizer, scheduler = make_stage_optimizer(model, stage, recipe)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            progress.update(stage=stage, stage_update=0)
            event(log, {"event": "transition", "stage": stage, "global_update": progress["global_update"],
                        "seconds": time.perf_counter()-started, "trainable_parameters": assert_trainables(model, stage),
                        "optimizer_policy": "fresh AdamW moments each stage; weights continue", "session": session})
        before = sampler.state_dict()
        records = sampler.next_batch(recipe.effective_batch)
        attempt = 0
        while True:
            started = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            lrs = {group["name"]: float(group["lr"]) for group in optimizer.param_groups}
            try:
                metrics = attempt_update(model, records, optimizer, scheduler, scaler, loader, recipe)
            except torch.cuda.OutOfMemoryError as exc:
                # An optimizer OOM may partially mutate RAM state: never checkpoint or
                # retry it in-process. Restart from a verified complete checkpoint.
                optimizer.zero_grad(set_to_none=True)
                sampler.load_state_dict(before)
                event(log,{"event":"fatal_cuda_oom","stage":stage,"global_update":progress['global_update'],
                           "target_global_update":progress['global_update']+1,"session":session,
                           "microbatch":recipe.microbatch,"effective_batch":recipe.effective_batch,
                           "recovery":"restart fresh process from latest complete checkpoint; no examples skipped"})
                raise RuntimeError('Training CUDA OOM: no update committed. Restart from latest complete checkpoint '
                                   '(or a fresh run if none exists); do not reuse this in-memory model.') from exc
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds = time.perf_counter()-started
            event(log, {"event": "optimizer_attempt", "session": session,
                        "stage": stage, "target_global_update": progress["global_update"]+1,
                        "target_stage_update": progress["stage_update"]+1, "learning_rates": lrs,
                        "seconds": seconds, **metrics})
            progress["training_seconds"] += seconds
            if metrics["completed"]:
                break
            attempt += 1
            progress["skipped_attempts"] += 1
            if attempt >= recipe.max_skips:
                sampler.load_state_dict(before)
                raise RuntimeError("Too many skipped attempts; no next batch consumed")
        progress["global_update"] += 1
        progress["stage_update"] += 1
        progress["completed_examples"] += len(records)
        progress["sft_loss_sum"] += metrics["sft_loss"]
        progress["mixture"] = dict(sampler.mixture)
        assert_trainables(model, stage)
        event(log, {"event": "completed_update", "session": session, **progress,
                    "learning_rates": lrs, "sft_loss": metrics["sft_loss"],
                    "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                    "mixture_batch_examples": dict(Counter(r["sample_kind"] for r in records)),
                    "sample_ids": [{"image_id": r["image_id"], "caption_id": r.get("caption_id"),
                                    "question_id": r.get("question_id"), "pair_id": r.get("pair_id"),
                                    "draw_id": r["draw_id"]} for r in records],
                    "peak_gpu_allocated": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                    "peak_gpu_reserved": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None})
        if progress["global_update"] % recipe.checkpoint_interval == 0:
            started = time.perf_counter()
            destination = run_dir / "checkpoints" / f"update_{progress['global_update']:05d}"
            publish(destination, {"kind": "recovery", "spec": spec, "progress": dict(progress),
                                  "model": model_state(model), "optimizer": optimizer.state_dict(),
                                  "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                                  "sampler": sampler.state_dict(), "rng": rng_state(), "backbone_hash": initial_hash})
            event(log, {"event": "checkpoint", "global_update": progress["global_update"],
                        "seconds": time.perf_counter()-started,
                        "bytes": sum(p.stat().st_size for p in destination.iterdir() if p.is_file()), "session": session})
        if progress["stage_update"] == recipe.budgets[stage-1]:
            export_current()
        evaluate_current()
    atomic_json(run_dir / f"session_{session}.json", {"progress": progress, "finished": progress["global_update"] == recipe.total_updates})
    return progress
