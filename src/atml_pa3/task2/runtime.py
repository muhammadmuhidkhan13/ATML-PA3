"""Validated assets, calibrated initialization, and explicit official launch boundary."""
from importlib.metadata import version
from pathlib import Path
import platform
import random
import time
import numpy as np
import torch
from .adapters import attach_attention_lora, backbone_hash
from .curriculum import Recipe
from .features import FeatureStore, valid_feature, SIGLIP_ID, SIGLIP_REVISION
from .io import read_json, read_jsonl, sha256, atomic_json
from .model import load_vlm, encode_example, QWEN_REVISION
from .sampling import CurriculumSampler
from .state import identity, read_payload, restore_model
from .training import run_curriculum
from .validation import validate_prepared


def asset_paths(root):
    root = Path(root)
    return {"data":root/"data-v1", "features":root/"features-v1",
            "calibration":root/"vlm-validation-v1/calibration.json",
            "initial":root/"vlm-validation-v1/connector_initial_calibrated.pt"}


def validate_assets(root):
    paths = asset_paths(root)
    counts = validate_prepared(paths["data"])
    store = FeatureStore(paths["features"])
    if (store.manifest["model_id"]!=SIGLIP_ID or store.manifest["revision"]!=SIGLIP_REVISION
            or store.manifest["shape"]!=[196,768] or store.manifest["dtype"]!="float16"):
        raise ValueError("Feature cache does not use the required pinned SigLIP configuration")
    images = read_json(paths["data"] / "images.json")
    if set(store.manifest["images"]) != set(images) or len(images) != 6528:
        raise ValueError("Feature selection does not match dataset")
    for key, record in store.manifest["images"].items():
        if not valid_feature(paths["features"]/record["path"], record, record["source_sha256"], store.signature):
            raise ValueError(f"Corrupted required feature: {key}")
    calibration = read_json(paths["calibration"])
    identities = {"dataset_manifest":sha256(paths["data"]/"manifest.json"),
                  "feature_manifest":sha256(paths["features"]/"manifest.json"),
                  "calibration_metadata":sha256(paths["calibration"]),
                  "calibrated_connector":sha256(paths["initial"]), "qwen_revision":QWEN_REVISION}
    expected = {"dataset_manifest":"46200aa0b1fe350021683bbaea897310703942d621a8c43b0f651bcc39fded57",
                "feature_manifest":"1e7c1ce0d78f0805c6bdae88c2cfcb3781c352063c85aaf201de4b6f00fa8ddc",
                "calibration_metadata":"40da3931b76d9481c78320a77777cc7768be3a3c69ef0db0b7d1fb99a7c35adb",
                "calibrated_connector":"b704cc2925c1d6cd57c1ce6863b23783affd693725da66018ef42877a768dba0"}
    if any(identities[key]!=value for key,value in expected.items()):
        raise ValueError("Required immutable Part 1 artifact identities differ; no reselection or recalibration allowed")
    if (calibration["qwen_revision"] != QWEN_REVISION
            or calibration["dataset_manifest_sha256"] != identities["dataset_manifest"]
            or calibration["feature_manifest_sha256"] != identities["feature_manifest"]
            or calibration["connector_state_sha256"] != identities["calibrated_connector"]
            or calibration["split"] != "train"):
        raise ValueError("Calibrated initialization provenance differs")
    initial = torch.load(paths["initial"],map_location="cpu",weights_only=True)
    shapes = {"0.weight":(1024,768),"0.bias":(1024,),"2.weight":(1024,1024),"2.bias":(1024,)}
    if set(initial) != {"connector","visual_scale"} or set(initial["connector"]) != set(shapes):
        raise ValueError("Unexpected calibrated connector keys")
    for key, shape in shapes.items():
        tensor = initial["connector"][key]
        if tuple(tensor.shape) != shape or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
            raise ValueError("Invalid calibrated connector tensor")
    if initial["visual_scale"].item() != calibration["scale"] or calibration["scale"] <= 0:
        raise ValueError("Visual scale differs from saved calibration")
    from huggingface_hub import try_to_load_from_cache
    snapshot_hashes = {}
    for name in ("config.json","tokenizer.json","tokenizer_config.json","model.safetensors"):
        path = try_to_load_from_cache("Qwen/Qwen3-0.6B",name,revision=QWEN_REVISION)
        if not isinstance(path,str) or not Path(path).is_file():
            raise FileNotFoundError(f"Pinned original Qwen cache is missing {name}; supply the original snapshot before preflight")
        snapshot_hashes[name] = sha256(path)
    identities["qwen_files"] = snapshot_hashes
    return {"identities":identities,"counts":counts,"features":len(images),"calibration_verified":True}


def make_spec(root, assets=None, recipe=None):
    assets = validate_assets(root) if assets is None else assets
    source_files = ("model.py","curriculum.py","adapters.py","sampling.py","state.py","training.py",
                    "runtime.py","evaluation.py","result_validation.py","metrics.py","features.py","data.py","io.py","validation.py",
                    "normalization.py","vqa_tables.json","batching.py")
    return {"task":"task2", "recipe":(recipe or Recipe()).serial(), "assets":assets["identities"],
            "lora":{"rank":16,"alpha":32,"dropout":.05,"targets":["q_proj","k_proj","v_proj","o_proj"],"bias":False},
            "optimizer":{"name":"AdamW","betas":[.9,.999],"eps":1e-8,"stage_policy":"rebuild moments at every boundary"},
            "warmup_rounding":"ceil(0.05 * stage_budget)","precision":"FP16 autocast, FP32 trainables/moments; no quantization",
            "scaler":{"init_scale":256.,"growth_factor":2.,"backoff_factor":.5,"growth_interval":2000},
            "code_sha256":{name:sha256(Path(__file__).with_name(name)) for name in source_files},
            "packages":{name:version(name) for name in ("torch","transformers","numpy","Pillow","safetensors","tokenizers","huggingface_hub")}}


def restore_initialization(model, root):
    initial = torch.load(asset_paths(root)["initial"],map_location="cpu",weights_only=True)
    model.connector.load_state_dict(initial["connector"],strict=True)
    model.visual_scale.copy_(initial["visual_scale"].to(model.visual_scale.device))


def real_sampler(root):
    data = asset_paths(root)["data"]
    return CurriculumSampler(read_jsonl(data/"captions_train.jsonl"), read_jsonl(data/"vqa_train.jsonl"),
                             read_json(data/"grounding_train_pools.json"))


def batch_loader(tokenizer, store):
    def load(records):
        return ([encode_example(tokenizer,r["question"],r["target"],caption=r["caption"]) for r in records],
                [store.get(r["image_id"]) for r in records])
    return load


def validate_destination(run_dir,resume_from):
    run_dir = Path(run_dir)
    if resume_from is None:
        if run_dir.exists():
            raise FileExistsError("Use a new, separate Task 2 run directory")
    else:
        resume = Path(resume_from).resolve()
        if resume.parent != (run_dir/"checkpoints").resolve():
            raise ValueError("Resume checkpoint must belong to this run")
        checkpoints = sorted(p for p in (run_dir/"checkpoints").glob("update_*") if (p/"manifest.json").exists())
        if not checkpoints or resume != checkpoints[-1].resolve():
            raise ValueError("Resume must use the latest complete checkpoint")


def recipe_from_spec(spec):
    values=dict(spec['recipe'])
    for name in ('budgets','connector_lrs','adapter_lrs'):values[name]=tuple(values[name])
    recipe=Recipe(**values);recipe.assert_official();return recipe

def current_spec(root,assets,saved):
    recipe=recipe_from_spec(saved);spec=make_spec(root,assets,recipe)
    if 'batch_sizing' in saved:
        sizing=saved['batch_sizing']
        if sizing['selected_microbatch']!=recipe.microbatch:raise ValueError('Saved batch selection differs')
        spec['batch_sizing']=sizing
    return spec

def launch(root,run_dir,*,execute=False,resume_from=None,stop_after_stage=3,device="cuda:0",
           batch_options=None,batch_preflight=False,preflight_report=None):
    from dataclasses import replace
    from .batching import BatchOptions,detect_gpu,representative_batches,real_preflight
    options=batch_options or BatchOptions(batch_size='8')
    if options.effective_batch_size!=16:
        raise ValueError('ATML-PA3.pdf p8 requires effective batch16,4500 updates and72000 examples; effective32 is not permitted')
    if execute and batch_preflight:raise ValueError('Choose sizing preflight OR official training')
    if not batch_preflight and options.batch_size!='8':
        raise ValueError('Official Task 2 requires microbatch8 and accumulation2; auto/other sizes are diagnostic-only with --batch-preflight')
    if preflight_report is not None and Path(preflight_report).exists():raise FileExistsError('Use a new preflight report path')
    Recipe().assert_official()
    validate_destination(run_dir,resume_from)
    assets = validate_assets(root)
    spec = make_spec(root,assets)
    recipe=Recipe()
    if resume_from:
        saved=read_json(Path(run_dir)/'run_spec.json')
        recipe=recipe_from_spec(saved)
        if saved.get('batch_sizing',{}).get('request')!=options.serial():raise ValueError('Resume batch policy differs; configuration is immutable')
        spec=current_spec(root,assets,saved)
        payload=read_payload(resume_from,spec)
        recipe.validate_progress(payload['progress'])
        if read_json(Path(run_dir)/"run_spec.json") != spec:
            raise ValueError("Run spec changed")
    if not execute and not batch_preflight:
        return {"mode":"validation_only","training_started":False,"assets":assets,"spec":spec,
                "stop_after_stage":stop_after_stage,'batch_request':options.serial(),
                'batch_selection_pending':not bool(resume_from)}
    if stop_after_stage not in (1,2,3):
        raise ValueError("Invalid stop stage")
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise RuntimeError("Official mixed-precision training requires CUDA")
    gpu=detect_gpu(device)
    for package,expected in (("torch","2.14.1+cu130"),("transformers","5.22.1"),("tokenizers","0.23.3")):
        if version(package)!=expected:
            raise RuntimeError(f"Required {package}=={expected}; found {version(package)}. No installation attempted.")
    random.seed(6304)
    np.random.seed(6304)
    torch.manual_seed(6304)
    torch.cuda.manual_seed_all(6304)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    model,tokenizer = load_vlm(device)
    restore_initialization(model,root)
    store = FeatureStore(asset_paths(root)["features"])
    batches,representative=representative_batches(root,tokenizer,store)
    sizing=real_preflight(model,batches,options,device=device,locked=None if batch_preflight else 8)
    sizing['representative_batches']=representative
    if preflight_report is not None:
        if Path(preflight_report).exists():raise FileExistsError('Use a new preflight report path')
        atomic_json(Path(preflight_report),sizing)
    if batch_preflight:
        del model,store,batches
        import gc
        gc.collect();torch.cuda.empty_cache()
        return {'mode':'batch_preflight','training_started':False,**sizing}
    if not resume_from:
        recipe=replace(recipe,microbatch=sizing['selected_microbatch'],accumulation=sizing['accumulation'])
        recipe.assert_official()
        spec=make_spec(root,assets,recipe)
        spec['batch_sizing']={'request':options.serial(),'selected_microbatch':recipe.microbatch,
                              'policy':'PDF microbatch8/accumulation2; startup certification only'}
    setup = {"seconds":time.perf_counter()-started,"gpu":gpu,"python":platform.python_version(),
             "official_recipe":recipe.serial(),"model_revision":QWEN_REVISION,'batch_preflight':sizing}
    del batches
    from .evaluation import evaluate_development
    def evaluate(model,update,stage):
        return evaluate_development(model,tokenizer,store,asset_paths(root)["data"],
                                    Path(run_dir)/"development"/f"update_{update:05d}",
                                    model_identity={"run_spec":identity(spec),"global_update":update,"stage":stage})
    progress = run_curriculum(model,real_sampler(root),batch_loader(tokenizer,store),run_dir,spec,
                              resume_from=resume_from,stop_after=Recipe().boundaries[stop_after_stage-1],evaluate=evaluate,
                              setup_metadata=setup,recipe=recipe)
    import uuid
    atomic_json(Path(run_dir)/f"setup_{uuid.uuid4().hex}.json",setup)
    return progress


def load_evaluation_model(root, checkpoint, device="cuda:0", final=False):
    assets = validate_assets(root)
    saved_payload=read_payload(checkpoint)
    spec = current_spec(root,assets,saved_payload['spec']) if 'spec' in saved_payload else make_spec(root,assets)
    payload = read_payload(checkpoint,spec)
    if 'spec' in payload:
        recipe_from_spec(payload['spec']).validate_progress(payload['progress'])
    if final and payload["progress"]["global_update"] != 4500:
        raise ValueError("Final evaluation requires the completed Stage 3 model")
    model,tokenizer = load_vlm(device)
    if payload["progress"]["stage"] >= 2:
        attach_attention_lora(model)
    restore_model(model,payload["model"])
    if backbone_hash(model)!=payload["backbone_hash"]:
        raise ValueError("Reconstructed frozen backbone differs from checkpoint fingerprint")
    model.eval()
    return model,tokenizer,assets,payload
