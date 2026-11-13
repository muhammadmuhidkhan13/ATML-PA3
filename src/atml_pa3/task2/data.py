"""Deterministic COCO/VQAv2 selections, genuine pairs, and image isolation."""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from .io import atomic_json, read_json, read_jsonl, sha256, write_jsonl
from .normalization import consensus, normalize_answer

SEED = 6304
QUOTAS = {"train": {"yes/no": 640, "number": 256, "other": 640},
          "dev": {"yes/no": 12, "number": 8, "other": 12},
          "test": {"yes/no": 36, "number": 24, "other": 36}}


def natural_candidates(questions, annotations, pairs):
    qmap = {q["question_id"]: q for q in questions}
    amap = {a["question_id"]: a for a in annotations}
    if len(qmap) != len(questions) or len(amap) != len(annotations):
        raise ValueError("Duplicate VQA question IDs")
    results, seen = [], set()
    for index, pair in enumerate(pairs):
        if not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(v, int) for v in pair):
            raise ValueError(f"Unexpected authentic complementary pair schema at {index}")
        identity = tuple(sorted(pair))
        if identity in seen:
            continue
        seen.add(identity)
        members = []
        for qid in pair:
            question, annotation = qmap[qid], amap[qid]
            if annotation["image_id"] != question["image_id"]:
                raise ValueError("Question/annotation image IDs disagree")
            members.append({"question_id": qid, "image_id": question["image_id"],
                            "question": question["question"], "annotation": annotation,
                            "answers": annotation["answers"], **consensus(annotation["answers"])})
        a, b = members
        if a["question"] != b["question"] or a["image_id"] == b["image_id"]:
            raise ValueError("Official pair does not share one question on distinct images")
        category = a["annotation"]["answer_type"]
        if category != b["annotation"]["answer_type"]:
            continue
        if min(a["agreement"], b["agreement"]) < 8 or a["canonical_answer"] == b["canonical_answer"]:
            continue
        results.append({"pair_id": f"vqa:{identity[0]}:{identity[1]}", "source_pair_index": index,
                        "answer_type": category, "members": members})
    return results


def select_natural(candidates, quotas, rng, forbidden=()):
    used = set(forbidden)
    selected = []
    for category, count in quotas.items():
        pool = sorted((p for p in candidates if p["answer_type"] == category), key=lambda p: p["pair_id"])
        accepted = 0
        for i in rng.permutation(len(pool)):
            pair = pool[int(i)]
            images = {m["image_id"] for m in pair["members"]}
            if images & used:
                continue
            selected.append(pair)
            # Within a split, pair images may recur; only external forbidden IDs
            # are excluded. Development/test isolation is enforced by callers.
            accepted += 1
            if accepted == count:
                break
        if accepted != count:
            raise ValueError(f"Not enough authentic {category} pairs: {accepted}/{count}")
    return selected


def pair_images(pairs):
    return {m["image_id"] for pair in pairs for m in pair["members"]}


def plural(noun):
    irregular = {"person": "people", "mouse": "mice", "knife": "knives", "sheep": "sheep",
                 "scissors": "scissors", "skis": "skis"}
    if noun in irregular:
        return irregular[noun]
    if noun.endswith(("s", "ch", "sh", "x")):
        return noun + "es"
    if noun.endswith("y") and noun[-2] not in "aeiou":
        return noun[:-1] + "ies"
    return noun + "s"


def annotation_index(instances):
    by_image = defaultdict(list)
    for annotation in instances["annotations"]:
        by_image[annotation["image_id"]].append(annotation)
    return by_image


def grounding_pools(image_ids, by_image, categories):
    universe = sorted(image_ids)
    pools = []
    for category in sorted(categories, key=lambda c: c["id"]):
        positives, counts = [], {str(i): [] for i in range(1, 5)}
        for image_id in universe:
            anns = [a for a in by_image.get(image_id, []) if a["category_id"] == category["id"]]
            if anns:
                positives.append(image_id)
            if 1 <= len(anns) <= 4 and not any(a.get("iscrowd", 0) for a in anns):
                counts[str(len(anns))].append(image_id)
        if positives:
            pools.append({"category_id": category["id"], "object": category["name"],
                          "plural": plural(category["name"]), "positive": positives,
                          "counts": counts})
    return {"image_ids": universe, "categories": pools}


def select_grounding(pools, by_image, count, rng, split):
    """Equal presence/count pair counts; evaluation images are never reused."""
    used, result = set(), []
    for kind in ("presence", "counting"):
        for _ in range(count // 2):
            found = False
            for index in rng.permutation(len(pools["categories"])):
                category = pools["categories"][int(index)]
                if kind == "presence":
                    left = sorted(set(category["positive"]) - used)
                    right = sorted(set(pools["image_ids"]) - set(category["positive"]) - used)
                    targets = ("yes", "no")
                else:
                    buckets = [k for k, ids in category["counts"].items() if set(ids) - used]
                    if len(buckets) < 2:
                        continue
                    ka, kb = rng.choice(buckets, 2, replace=False)
                    left = sorted(set(category["counts"][ka]) - used)
                    right = sorted(set(category["counts"][kb]) - used)
                    targets = (str(ka), str(kb))
                if not left or not right:
                    continue
                ids = [int(rng.choice(left)), int(rng.choice(right))]
                question = (f"Does this image contain {category['object']}?" if kind == "presence"
                            else f"How many {category['plural']} can you see?")
                result.append({"pair_id": f"ground:{split}:{len(result)}", "kind": kind,
                               "category_id": category["category_id"], "object": category["object"],
                               "members": [{"image_id": image_id, "question": question,
                                            "target": target, "canonical_answer": target,
                                            "annotation_ids": [a["id"] for a in by_image.get(image_id, [])
                                                               if a["category_id"] == category["category_id"]]}
                                           for image_id, target in zip(ids, targets)]})
                used.update(ids)
                found = True
                break
            if not found:
                raise ValueError(f"Insufficient image-disjoint {kind} pairs")
    if len(result) != count:
        raise ValueError("Grounding pair counts must be even")
    return result


def draw_training_pair(pools, kind, rng):
    """Draw one reusable complete training pair; mixture scheduling is Part 2."""
    if kind not in ("presence", "counting"):
        raise ValueError("Unknown grounding kind")
    for index in rng.permutation(len(pools["categories"])):
        category = pools["categories"][int(index)]
        if kind == "presence":
            left = category["positive"]
            right = sorted(set(pools["image_ids"])-set(left))
            targets = ("yes", "no")
            question = f"Does this image contain {category['object']}?"
        else:
            buckets = [k for k, v in category["counts"].items() if v]
            if len(buckets) < 2:
                continue
            a, b = rng.choice(buckets, 2, replace=False)
            left, right = category["counts"][a], category["counts"][b]
            targets = (str(a), str(b))
            question = f"How many {category['plural']} can you see?"
        if left and right:
            ids = [int(rng.choice(left)), int(rng.choice(right))]
            return {"pair_id": f"ground:train:{kind}:{category['category_id']}:{ids[0]}:{ids[1]}",
                    "kind": kind, "category_id": category["category_id"], "object": category["object"],
                    "members": [{"image_id": i, "question": question, "target": t,
                                 "canonical_answer": t} for i, t in zip(ids, targets)]}
    raise ValueError(f"No eligible training pool for {kind}")


def prepare(source_root: Path, output: Path):
    source_root, output = Path(source_root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    rng = np.random.default_rng(SEED)
    annotations = source_root / "annotations"
    natural = {}
    for source_split in ("train", "val"):
        suffix = source_split + "2014"
        q = read_json(annotations / f"v2_OpenEnded_mscoco_{suffix}_questions.json")
        a = read_json(annotations / f"v2_mscoco_{suffix}_annotations.json")
        pair_file = next(annotations.glob(f"*{source_split}*complementary*"), None)
        if pair_file is None:
            # The official filenames are inspected during source acquisition.
            pair_file = next(p for p in annotations.glob("*.json")
                             if "complementary" in p.name.lower() and source_split in p.name.lower())
        candidates = natural_candidates(q["questions"], a["annotations"], read_json(pair_file))
        if source_split == "train":
            natural["train"] = select_natural(candidates, QUOTAS["train"], rng)
        else:
            natural["dev"] = select_natural(candidates, QUOTAS["dev"], rng)
            natural["test"] = select_natural(candidates, QUOTAS["test"], rng, pair_images(natural["dev"]))
        print(f"Authentic {source_split} candidates: {len(candidates)}", flush=True)
        del q, a, candidates
    output.mkdir(parents=True)
    captions, grounding, all_images = {}, {}, {}
    reserved = pair_images(natural["dev"]) | pair_images(natural["test"])
    for source_split in ("train", "val"):
        suffix = source_split + "2014"
        cap = read_json(annotations / f"captions_{suffix}.json")
        inst = read_json(annotations / f"instances_{suffix}.json")
        image_map = {i["id"]: i for i in cap["images"]}
        by_image = annotation_index(inst)
        cap_by_image = defaultdict(list)
        for row in cap["annotations"]:
            cap_by_image[row["image_id"]].append(row)
        if source_split == "train":
            selected = sorted(pair_images(natural["train"]))
            remaining = sorted(set(image_map) - set(selected))
            selected += [remaining[int(i)] for i in rng.permutation(len(remaining))[:6000-len(selected)]]
            ncap, split = 3, "train"
        else:
            available = sorted(set(image_map) - reserved)
            selected = [available[int(i)] for i in rng.permutation(len(available))[:96]]
            reserved.update(selected)
            ncap, split = 1, "dev"
        captions[split] = []
        for image_id in selected:
            available_captions = sorted(cap_by_image[image_id], key=lambda row: row["id"])
            if len(available_captions) < ncap:
                raise ValueError("Insufficient captions for selected image")
            for i in rng.permutation(len(available_captions))[:ncap]:
                row = available_captions[int(i)]
                captions[split].append({"image_id": image_id, "caption_id": row["id"],
                                        "target": row["caption"], "question": "Describe this image."})
        if source_split == "train":
            pools = grounding_pools(selected, by_image, inst["categories"])
            atomic_json(output / "grounding_train_pools.json", pools)
        else:
            ground_dev = select_grounding(grounding_pools(set(image_map)-reserved, by_image, inst["categories"]),
                                          by_image, 24, rng, "dev")
            reserved.update(pair_images(ground_dev))
            ground_test = select_grounding(grounding_pools(set(image_map)-reserved, by_image, inst["categories"]),
                                           by_image, 64, rng, "test")
            grounding = {"dev": ground_dev, "test": ground_test}
        needed = set(selected) | pair_images(natural["train"] if source_split == "train" else natural["dev"]+natural["test"])
        if source_split == "val":
            needed |= pair_images(grounding["dev"]+grounding["test"])
        for image_id in sorted(needed):
            image = image_map[image_id]
            all_images[str(image_id)] = {**image, "source_split": suffix,
                                        "relative_path": f"images/{suffix}/{image['file_name']}"}
        atomic_json(output / f"grounding_annotations_{source_split}.json",
                    {str(i): by_image.get(i, []) for i in sorted(needed)})
        del cap, inst, by_image, cap_by_image
    for split, rows in natural.items():
        write_jsonl(output / f"vqa_{split}.jsonl", rows)
    for split, rows in grounding.items():
        write_jsonl(output / f"grounding_{split}.jsonl", rows)
    for split, rows in captions.items():
        write_jsonl(output / f"captions_{split}.jsonl", rows)
    atomic_json(output / "images.json", all_images)
    summary = {"seed": SEED, "natural_pairs": {s: len(v) for s, v in natural.items()},
               "caption_examples": {s: len(v) for s, v in captions.items()},
               "grounding_pairs": {s: len(v) for s, v in grounding.items()},
               "unique_images": len(all_images), "source_manifest_sha256": sha256(source_root / "source_manifest.json"),
               "selection_decisions": {"caption_train": "all natural training images plus seeded fill to 6000",
                                       "grounding_eval": "equal presence/count pairs; no repeated images",
                                       "grounding_train": "category pools over the 6000 caption training images",
                                       "counting": "1..4 instances, exclude image/category containing crowd annotations"}}
    atomic_json(output / "manifest.json", {**summary, "files": {p.name: sha256(p) for p in output.iterdir() if p.is_file()}})
    return summary
