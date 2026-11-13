"""Resumable development/reference and final four-condition evaluation."""
from dataclasses import replace
from pathlib import Path
import time
import torch
from .io import atomic_json, read_json, read_jsonl, sha256
from .model import encode_example, example_weighted_loss
from .metrics import score_record, summarize_suite
from .state import preserve_rng, identity

CONDITIONS = ("correct", "complementary", "mismatched", "no_image")


def mismatched_mapping(pairs, seed=6304):
    import numpy as np
    rng = np.random.default_rng(seed)
    result = {}
    for pair in sorted(pairs, key=lambda p:p["pair_id"]):
        excluded = {m["image_id"] for m in pair["members"]}
        candidates = sorted({m["image_id"] for p in pairs if p["pair_id"] != pair["pair_id"]
                             for m in p["members"]} - excluded)
        if not candidates:
            raise ValueError("Mismatched images need another unrelated pair")
        for index in (0,1):
            result[f"{pair['pair_id']}:{index}"] = int(rng.choice(candidates))
    return result


def condition_records(pairs, suite, condition, mapping=None):
    if condition not in CONDITIONS:
        raise ValueError("Unknown image condition")
    records = []
    for pair in pairs:
        for index, member in enumerate(pair["members"]):
            partner = pair["members"][1-index]
            actual = (member["image_id"] if condition == "correct" else partner["image_id"]
                      if condition == "complementary" else None if condition == "no_image"
                      else mapping[f"{pair['pair_id']}:{index}"])
            if condition == "mismatched" and actual in {member["image_id"],partner["image_id"]}:
                raise ValueError("Mismatched image belongs to original pair")
            records.append({"suite": suite, "question_type": pair.get("answer_type",pair.get("kind")),
                            "pair_id": pair["pair_id"], "member_index": index,
                            "question_id": member.get("question_id", f"{pair['pair_id']}:{index}"),
                            "original_image_id": member["image_id"], "actual_image_id": actual,
                            "condition": condition, "question": member["question"],
                            "original_target": member["target"], "canonical_answer": member["canonical_answer"],
                            "partner_canonical_answer": partner["canonical_answer"],
                            "references": member.get("answers", [{"answer": member["target"]}])})
    return records


class ModelEvaluator:
    def __init__(self, model, tokenizer, store):
        self.model, self.tokenizer, self.store = model, tokenizer, store

    @torch.no_grad()
    def reference(self, records, *, caption=False):
        examples = [encode_example(self.tokenizer, r["question"], r["original_target"], caption=caption) for r in records]
        features = [None if r["actual_image_id"] is None else self.store.get(r["actual_image_id"]) for r in records]
        batch = self.model.build_inputs(examples, features)
        device = batch["inputs_embeds"].device
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            output = self.model.language_model(inputs_embeds=batch["inputs_embeds"], attention_mask=batch["attention_mask"],
                                                position_ids=batch["position_ids"], use_cache=False)
        labels = batch["labels"].clone()
        if not caption:
            labels[labels == self.tokenizer.convert_tokens_to_ids("<|im_end|>")] = -100
        _, per_example = example_weighted_loss(output.logits, labels)
        return per_example.cpu().tolist()

    @torch.no_grad()
    def generate(self, records, *, max_new_tokens=16):
        # Explicit batched greedy decoding supports right-padded visual prompts.
        # No KV cache is used: gather each row's last valid logits, then append
        # a token before rebuilding masks/positions. This avoids padding/cache
        # ambiguity and is bounded to the prescribed 16 generated tokens.
        prompts = []
        for r in records:
            example = encode_example(self.tokenizer, r["question"], "x", caption=False)
            first = next(i for i,label in enumerate(example.labels) if label != -100)
            prompts.append(replace(example, input_ids=example.input_ids[:first], labels=[-100]*first,
                                   target_tokens=0, target_truncated=False))
        features = [None if r["actual_image_id"] is None else self.store.get(r["actual_image_id"]) for r in records]
        generated = [[] for _ in records]
        finished = [False]*len(records)
        eos = {self.tokenizer.eos_token_id}
        config = getattr(self.model.language_model, "generation_config", None)
        configured = getattr(config, "eos_token_id", None)
        eos.update(configured if isinstance(configured,list) else [configured])
        eos.discard(None)
        device = next(self.model.connector.parameters()).device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(max_new_tokens):
            examples = [replace(p, input_ids=p.input_ids+tokens, labels=[-100]*(len(p.input_ids)+len(tokens)))
                        for p,tokens in zip(prompts, generated)]
            batch = self.model.build_inputs(examples, features)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                output = self.model.language_model(inputs_embeds=batch["inputs_embeds"], attention_mask=batch["attention_mask"],
                                                    position_ids=batch["position_ids"], use_cache=False)
            indices = batch["attention_mask"].sum(1)-1
            tokens = output.logits[torch.arange(len(records), device=device), indices].argmax(-1).tolist()
            for i, token in enumerate(tokens):
                if not finished[i]:
                    generated[i].append(token)
                    finished[i] = token in eos
            if all(finished):
                break
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = (time.perf_counter()-started)/len(records)
        return [{"raw_response": self.tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False),
                 "prediction_text": self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                 "generated_token_ids": tokens, "generated_tokens": len(tokens),
                 "reached_token_cap": len(tokens) == max_new_tokens and not done,
                 "generation_seconds": seconds} for tokens,done in zip(generated,finished)]


def _persist_batches(destination, records, backend, *, spec, generate, caption=False, batch_size=8):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    spec_id = identity(spec)
    header = destination / "manifest.json"
    if header.exists() and read_json(header)["spec_id"] != spec_id:
        raise ValueError("Evaluation identity differs; cannot resume")
    atomic_json(header, {"spec_id": spec_id, "spec": spec, "expected_examples": len(records), "complete": False})
    all_records = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start+batch_size]
        path = destination / f"batch_{start//batch_size:05d}.json"
        checksum = path.with_suffix(".sha256")
        if path.exists() and checksum.exists():
            if checksum.read_text().strip() != sha256(path):
                raise ValueError("Evaluation batch integrity failure")
            saved = read_json(path)
            if saved["spec_id"] != spec_id or saved["inputs"] != batch:
                raise ValueError("Saved evaluation inputs differ")
            rows = saved["records"]
        else:
            reference_started = time.perf_counter()
            nll = backend.reference(batch, caption=caption)
            reference_seconds = (time.perf_counter()-reference_started)/len(batch)
            if len(nll) != len(batch):
                raise ValueError("Incomplete NLL batch")
            predictions = backend.generate(batch, max_new_tokens=16) if generate else [None]*len(batch)
            if len(predictions) != len(batch):
                raise ValueError("Incomplete generation batch")
            rows = []
            for record, likelihood, prediction in zip(batch,nll,predictions):
                row = {**record, "reference_nll": likelihood, "reference_seconds": reference_seconds,
                       "model_identity": spec["model_identity"]}
                if prediction is not None:
                    row.update(prediction)
                    row.update(score_record(record, prediction["prediction_text"]))
                rows.append(row)
            atomic_json(path, {"spec_id": spec_id, "inputs": batch, "records": rows})
            # Publish batch checksum atomically as well; interrupted writes fail
            # closed, never count as complete output.
            temporary = checksum.with_suffix(".partial")
            temporary.write_text(sha256(path)+"\n", encoding="utf-8")
            temporary.replace(checksum)
        if len(rows) != len(batch):
            raise ValueError("Saved evaluation batch is incomplete")
        all_records.extend(rows)
    atomic_json(header, {"spec_id": spec_id, "spec": spec, "expected_examples": len(records),
                         "inputs_id":identity(records),"batch_size":batch_size,
                         "complete": len(all_records) == len(records),
                         "batch_hashes": {p.name:sha256(p) for p in destination.glob("batch_*.json")}})
    return all_records


def evaluate_development(model, tokenizer, store, data_dir, output_dir, *, model_identity):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    summary = {"model_identity": model_identity, "complete": False, "suites": {}}
    output_dir.mkdir(parents=True,exist_ok=True)
    base = {"schema_version":2,"model_identity":model_identity,"dataset":sha256(data_dir / "manifest.json"),"kind":"development"}
    header=output_dir/'manifest.json'
    if header.exists() and read_json(header)['identity']!=identity(base):raise ValueError('Development identity differs')
    if header.exists() and read_json(header)['complete']:
        from .result_validation import verify_development
        verify_development(output_dir)
        return read_json(output_dir/'summary.json')
    atomic_json(header,{'identity':identity(base),'spec':base,'complete':False})
    with preserve_rng(model), torch.no_grad():
        model.eval()
        backend = ModelEvaluator(model, tokenizer, store)
        captions = [{"image_id":r["image_id"], "actual_image_id":r["image_id"], "question":r["question"],
                     "original_target":r["target"], "caption_id":r["caption_id"]}
                    for r in read_jsonl(data_dir / "captions_dev.jsonl")]
        rows = _persist_batches(output_dir / "captions", captions, backend, spec={**base,"suite":"captions"}, generate=False, caption=True)
        summary["caption_reference_loss"] = sum(r["reference_nll"] for r in rows)/len(rows)
        for suite, filename in (("grounding","grounding_dev.jsonl"),("natural","vqa_dev.jsonl")):
            pairs = read_jsonl(data_dir / filename)
            means = {}
            for condition in ("correct","complementary","no_image"):
                rows = _persist_batches(output_dir / suite / condition, condition_records(pairs,suite,condition), backend,
                                        spec={**base,"suite":suite,"condition":condition}, generate=False)
                means[condition] = sum(r["reference_nll"] for r in rows)/len(rows)
            summary["suites"][suite] = {"reference_nll":means,"swap_penalty":means["complementary"]-means["correct"]}
    summary["complete"] = True
    atomic_json(output_dir / "summary.json",summary)
    from .result_validation import payload_hashes,verify_development
    completion={'identity':identity(base),'spec':base,'complete':False,'files':payload_hashes(output_dir)}
    atomic_json(header,completion)
    verify_development(output_dir,_allow_incomplete=True)
    atomic_json(header,{**completion,'complete':True})
    return summary


def evaluate_final(backend, suites, output_dir, *, model_identity, dataset_identity, seed=6304, resamples=4000):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    plan_path = output_dir / "mismatched_images.json"
    plan = {suite:mismatched_mapping(pairs,seed) for suite,pairs in suites.items()}
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError("Frozen mismatched mapping differs")
    if not plan_path.exists():
        atomic_json(plan_path,plan)
    base = {"schema_version":2,"model_identity":model_identity,"dataset":dataset_identity,"kind":"final",
            "greedy":True,"new_token_cap":16,"batch_size":8,"mapping_sha256":sha256(plan_path),
            "bootstrap_seed":seed,"bootstrap_resamples":resamples,"suite_pairs_id":identity(suites)}
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and read_json(manifest_path)["identity"] != identity(base):
        raise ValueError("Final evaluation identity differs")
    if manifest_path.exists() and read_json(manifest_path)['complete']:
        from .result_validation import verify_final
        verify_final(output_dir,expected_suites=suites,model_identity=model_identity,dataset_identity=dataset_identity)
        return read_json(output_dir/'summary.json')
    atomic_json(output_dir/'expected_pairs.json',suites)
    atomic_json(manifest_path,{"identity":identity(base),"spec":base,"complete":False})
    results = {}
    for suite,pairs in suites.items():
        records = []
        for condition in CONDITIONS:
            records += _persist_batches(output_dir / suite / condition,
                                         condition_records(pairs,suite,condition,plan[suite]),backend,
                                         spec={**base,"suite":suite,"condition":condition}, generate=True)
        for condition in CONDITIONS:
            for pair in pairs:
                members = [r for r in records if r["condition"] == condition and r["pair_id"] == pair["pair_id"]]
                correct = all(m["canonical_correct"] for m in members)
                for member in members:
                    member["pair_correct"] = correct
        atomic_json(output_dir / suite / "records.json",records)
        results[suite] = summarize_suite(records,seed=seed,resamples=resamples)
        atomic_json(output_dir / suite / "summary.json",results[suite])
    atomic_json(output_dir / "summary.json",results)
    from .result_validation import payload_hashes,verify_final
    completion={"identity":identity(base),"spec":base,"complete":False,
                "summary_sha256":sha256(output_dir / "summary.json"),'files':payload_hashes(output_dir)}
    atomic_json(manifest_path,completion)
    verify_final(output_dir,expected_suites=suites,_allow_incomplete=True)
    atomic_json(manifest_path,{**completion,'complete':True})
    return results
