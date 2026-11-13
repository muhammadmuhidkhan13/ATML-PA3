"""Bounded real-component checks: backwards only, never official optimizer steps."""
import copy
from pathlib import Path
import time
import torch
from .adapters import attach_attention_lora,assert_trainables,make_stage_optimizer,backbone_hash
from .curriculum import Recipe
from .runtime import validate_assets,restore_initialization,asset_paths,batch_loader
from .model import load_vlm
from .features import FeatureStore
from .data import draw_training_pair
from .io import read_jsonl,read_json,atomic_json
from .evaluation import ModelEvaluator,condition_records


def preflight(root,output,*,batch_size=2,device="cuda:0"):
    import numpy as np
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    if batch_size not in (1,2,8): raise ValueError("Bounded batch size must be 1, 2 or 8")
    if not torch.cuda.is_available(): raise RuntimeError("Bounded GPU preflight requires CUDA")
    assets=validate_assets(root)
    output.mkdir(parents=True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(6304)
    started=time.perf_counter()
    model,tokenizer=load_vlm(device)
    restore_initialization(model,root)
    frozen_before=backbone_hash(model)
    connector_before={k:v.detach().cpu().clone() for k,v in model.connector.state_dict().items()}
    data=asset_paths(root)["data"]
    store=FeatureStore(asset_paths(root)["features"])
    loader=batch_loader(tokenizer,store)
    captions=read_jsonl(data/"captions_train.jsonl")
    natural=read_jsonl(data/"vqa_train.jsonl")
    pools=read_json(data/"grounding_train_pools.json")
    rng=np.random.default_rng(6304)
    stages=[]
    adapter_ids=None
    for stage in (1,2,3):
        print(f"Bounded Stage {stage} backward check...",flush=True)
        if stage==2: attach_attention_lora(model)
        if stage>=2:
            current=[id(p) for name,p in model.named_parameters() if name.endswith((".lora_A",".lora_B"))]
            if stage==3 and current!=adapter_ids: raise AssertionError("Stage 3 reset adapters")
            adapter_ids=current
        trainable=assert_trainables(model,stage)
        optimizer,scheduler=make_stage_optimizer(model,stage)
        if stage==1: records=[{**r,"caption":True} for r in captions[:batch_size]]
        elif stage==2:
            records=[{**m,"caption":False} for _ in range((batch_size+1)//2)
                     for m in draw_training_pair(pools,"presence",rng)["members"]][:batch_size]
        else:
            records=[{**m,"caption":False} for pair in natural[:(batch_size+1)//2] for m in pair["members"]][:batch_size]
        model.train();model.zero_grad(set_to_none=True)
        examples,features=loader(records)
        torch.cuda.reset_peak_memory_stats(device)
        scaler=torch.amp.GradScaler("cuda",init_scale=256.)
        skipped=0
        while True:
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16): result=model(examples,features)
            if not torch.isfinite(result["loss"]): raise AssertionError("Nonfinite preflight loss")
            scaler.scale(result["loss"]).backward();scaler.unscale_(optimizer)
            grads=[p.grad for p in model.parameters() if p.requires_grad]
            if all(g is not None and torch.isfinite(g).all() for g in grads):
                break
            if any(g is None for g in grads): raise AssertionError("Disconnected trainable parameter")
            skipped+=1
            if skipped>=8: raise AssertionError("Preflight gradients remained nonfinite after scale backoff")
            # There are deliberately no optimizer steps in this preflight.
            # Back off the dynamic scale and reset its unscale-attempt state.
            scaler.update(new_scale=scaler.get_scale()/2)
            print(f"Stage {stage}: overflow; retrying same examples at scale {scaler.get_scale()}",flush=True)
        nonzero_b=0
        for name,p in model.named_parameters():
            if p.requires_grad:
                if p.grad is None or not torch.isfinite(p.grad).all(): raise AssertionError(f"Invalid gradient: {name}")
                if name.startswith("connector.") and not torch.count_nonzero(p.grad): raise AssertionError("Zero connector gradient")
                if name.endswith(".lora_B") and torch.count_nonzero(p.grad): nonzero_b+=1
            elif p.grad is not None: raise AssertionError("Frozen Qwen has gradient")
        if stage>=2 and nonzero_b!=112: raise AssertionError("Missing attention LoRA gradients")
        stages.append({"stage":stage,"loss":result["loss"].item(),"trainable_parameters":trainable,
                       "nonzero_lora_b_gradients":nonzero_b,"gradient_checkpointing":model.language_model.is_gradient_checkpointing,
                       "skipped_backward_attempts":skipped,"final_gradient_scale":scaler.get_scale(),
                       "peak_gpu_allocated_bytes":torch.cuda.max_memory_allocated(device),
                       "peak_gpu_reserved_bytes":torch.cuda.max_memory_reserved(device),
                       "image_ids":[r["image_id"] for r in records]})
        del result,optimizer,scheduler
        model.zero_grad(set_to_none=True)
    if backbone_hash(model)!=frozen_before: raise AssertionError("Backbone changed")
    for k,v in connector_before.items(): torch.testing.assert_close(model.connector.state_dict()[k].cpu(),v,rtol=0,atol=0)
    model.eval()
    backend=ModelEvaluator(model,tokenizer,store)
    references={}
    for condition in ("correct","complementary","no_image"):
        records=condition_records(natural[:1],"natural",condition)
        references[condition]=backend.reference(records)
    generated=backend.generate(condition_records(natural[:1],"natural","correct"),max_new_tokens=2)
    result={"status":"passed","scope":"bounded untrained real-component validation","official_optimizer_updates":0,
            "batch_size":batch_size,"gpu":torch.cuda.get_device_name(device),"assets":assets["identities"],
            "stages":stages,"reference_nll_smoke":references,"generation_smoke":generated,
            "backbone_unchanged":True,"connector_initialization_unchanged":True,
            "seconds":time.perf_counter()-started,"gpu_memory_bytes":torch.cuda.get_device_properties(device).total_memory}
    atomic_json(output/"result.json",result)
    return result
