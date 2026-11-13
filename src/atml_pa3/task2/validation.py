"""Strict prepared-dataset validation, independent of selection ordering."""
from collections import Counter
from pathlib import Path
from .data import QUOTAS, pair_images
from .io import read_json, read_jsonl, sha256
from .normalization import consensus, normalize_answer


def assert_split_isolation(split_ids, images):
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        if split_ids[a] & split_ids[b]:
            raise ValueError("Cross-split image leakage")
    for split, ids in split_ids.items():
        required = "train2014" if split == "train" else "val2014"
        if any(images[str(i)]["source_split"] != required for i in ids):
            raise ValueError("COCO source split is incorrect")


def validate_pair(pair, images, *, natural, annotations=None):
    members = pair["members"]
    if len(members) != 2 or members[0]["image_id"] == members[1]["image_id"]:
        raise ValueError("Pair must have two distinct images")
    if members[0]["question"] != members[1]["question"]:
        raise ValueError("Complementary pair questions differ")
    if members[0]["canonical_answer"] == members[1]["canonical_answer"]:
        raise ValueError("Complementary answers must disagree")
    for member in members:
        if str(member["image_id"]) not in images:
            raise ValueError("Pair image missing from manifest")
        if natural:
            calculated = consensus(member["answers"])
            if calculated["agreement"] < 8 or any(member[k] != calculated[k] for k in calculated):
                raise ValueError("Consensus/8-of-10 agreement mismatch")
            annotation = member["annotation"]
            if (annotation["answer_type"] != pair["answer_type"] or annotation["answers"] != member["answers"]
                    or annotation["question_id"] != member["question_id"]
                    or annotation["image_id"] != member["image_id"]):
                raise ValueError("Natural VQA category or annotation mismatch")
        else:
            anns = [a for a in annotations[str(member["image_id"])] if a["category_id"] == pair["category_id"]]
            if member["annotation_ids"] != [a["id"] for a in anns]:
                raise ValueError("Grounding source annotation IDs disagree")
            if pair["kind"] == "presence":
                expected = "yes" if anns else "no"
                question = f"Does this image contain {pair['object']}?"
            elif pair["kind"] == "counting":
                if not 1 <= len(anns) <= 4 or any(a.get("iscrowd", 0) for a in anns):
                    raise ValueError("Invalid/ambiguous counting annotations")
                expected = str(len(anns))
                from .data import plural
                question = f"How many {plural(pair['object'])} can you see?"
            else:
                raise ValueError("Unknown grounding kind")
            if member["target"] != expected or member["canonical_answer"] != expected or member["question"] != question:
                raise ValueError("Grounding target/template does not match annotations")


def validate_prepared(data_dir, source_root=None):
    data_dir = Path(data_dir)
    manifest = read_json(data_dir / "manifest.json")
    for name, digest in manifest["files"].items():
        if sha256(data_dir / name) != digest:
            raise ValueError(f"Prepared artifact hash mismatch: {name}")
    images = read_json(data_dir / "images.json")
    split_ids = {split: set() for split in ("train", "dev", "test")}
    results = {}
    selected = {}
    for split, expected in QUOTAS.items():
        rows = read_jsonl(data_dir / f"vqa_{split}.jsonl")
        if dict(Counter(pair["answer_type"] for pair in rows)) != expected:
            raise ValueError(f"Natural VQA pair quotas differ: {split}")
        if len({p["pair_id"] for p in rows}) != len(rows):
            raise ValueError("Duplicate natural pair IDs")
        for pair in rows:
            validate_pair(pair, images, natural=True)
        selected[split] = rows
        split_ids[split] |= pair_images(rows)
        results[f"vqa_{split}_pairs"] = len(rows)
        results[f"vqa_{split}_categories"] = expected
    for split, expected in (("train", 18000), ("dev", 96)):
        rows = read_jsonl(data_dir / f"captions_{split}.jsonl")
        counts = Counter(row["image_id"] for row in rows)
        if len(rows) != expected or any(v != (3 if split == "train" else 1) for v in counts.values()):
            raise ValueError(f"Caption quotas differ: {split}")
        if len({row["caption_id"] for row in rows}) != len(rows):
            raise ValueError("Duplicate caption IDs")
        if split == "dev" and set(counts) & (split_ids["dev"] | split_ids["test"]):
            raise ValueError("Caption development images overlap another evaluation suite")
        split_ids[split] |= set(counts)
        results[f"captions_{split}_examples"] = len(rows)
    annotations = read_json(data_dir / "grounding_annotations_val.json")
    for split, expected in (("dev", 24), ("test", 64)):
        rows = read_jsonl(data_dir / f"grounding_{split}.jsonl")
        if len(rows) != expected or dict(Counter(p["kind"] for p in rows)) != {"presence": expected//2, "counting": expected//2}:
            raise ValueError("Grounding pair quotas differ")
        ids = pair_images(rows)
        if len(ids) != 2*len(rows) or ids & (split_ids["dev"] | split_ids["test"]):
            raise ValueError("Grounding evaluation image overlap")
        for pair in rows:
            validate_pair(pair, images, natural=False, annotations=annotations)
        split_ids[split] |= ids
        results[f"grounding_{split}_pairs"] = len(rows)
    assert_split_isolation(split_ids, images)
    if set(map(int, images)) != set.union(*split_ids.values()):
        raise ValueError("Image manifest coverage differs from selected data")
    train_annotations = read_json(data_dir / "grounding_annotations_train.json")
    pools = read_json(data_dir / "grounding_train_pools.json")
    if set(pools["image_ids"]) != split_ids["train"]:
        raise ValueError("Training grounding pool universe differs")
    for category in pools["categories"]:
        expected_positive, expected_counts = set(), {str(i): set() for i in range(1, 5)}
        for image_id in pools["image_ids"]:
            anns = [a for a in train_annotations[str(image_id)] if a["category_id"] == category["category_id"]]
            if anns:
                expected_positive.add(image_id)
            if 1 <= len(anns) <= 4 and not any(a.get("iscrowd", 0) for a in anns):
                expected_counts[str(len(anns))].add(image_id)
        if set(category["positive"]) != expected_positive or any(set(category["counts"][k]) != v for k, v in expected_counts.items()):
            raise ValueError("Training grounding category/count pool mismatch")
        if not set(category["positive"]) or not (set(pools["image_ids"])-set(category["positive"])):
            raise ValueError("Training presence pool has no complementary examples")
    if source_root is not None:
        source_root = Path(source_root)
        source_manifest = read_json(source_root / "source_manifest.json")
        if sha256(source_root / "source_manifest.json") != manifest["source_manifest_sha256"]:
            raise ValueError("Source manifest differs")
        for source in source_manifest.values():
            for name, digest in source.get("extracted", {}).items():
                if sha256(source_root / "annotations" / name) != digest:
                    raise ValueError(f"Original source metadata hash differs: {name}")
        for source_split in ("train", "val"):
            suffix = source_split + "2014"
            official_pairs = read_json(source_root / "annotations" / f"v2_mscoco_{suffix}_complementary_pairs.json")
            rows = selected["train"] if source_split == "train" else selected["dev"] + selected["test"]
            for pair in rows:
                if [m["question_id"] for m in pair["members"]] != official_pairs[pair["source_pair_index"]]:
                    raise ValueError("Selected pair is not its authentic source pair")
    return {"status": "passed", **results, "unique_images": len(images),
            "split_unique_images": {k: len(v) for k, v in split_ids.items()},
            "grounding_pool_categories": len(pools["categories"]), "all_artifact_hashes_verified": True,
            "all_pair_constraints_verified": True, "cross_suite_image_isolation": True}
