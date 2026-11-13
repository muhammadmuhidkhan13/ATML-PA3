"""Local VQA scoring and paired, complete-pair bootstrap summaries."""
from collections import defaultdict
import numpy as np
from .normalization import normalize_answer


def vqa_score(prediction, answers):
    if len(answers) != 10:
        raise ValueError("VQA requires all ten human answers")
    prediction = normalize_answer(prediction)
    if not prediction:
        return 0.
    reference = [normalize_answer(a["answer"] if isinstance(a, dict) else a) for a in answers]
    # Exclude by annotator INDEX, not by value (identical answers remain distinct).
    return sum(min(1., sum(prediction == a for j,a in enumerate(reference) if j != i)/3.)
               for i in range(10))/10.


def score_record(record, prediction):
    normalized = normalize_answer(prediction)
    canonical = normalize_answer(record["canonical_answer"])
    score = (vqa_score(prediction, record["references"]) if record["suite"] == "natural"
             else float(bool(normalized) and normalized == canonical))
    return {"normalized_prediction": normalized, "score": score,
            "canonical_correct": bool(normalized) and normalized == canonical,
            "empty": not bool(normalized),
            "follows_partner_answer": bool(normalized) and normalized == normalize_answer(record["partner_canonical_answer"])}


def pair_vectors(records):
    by_pair = defaultdict(list)
    for record in records:
        by_pair[record["pair_id"]].append(record)
    vectors = []
    for pair_id, members in sorted(by_pair.items()):
        if len(members) != 2 or {m["member_index"] for m in members} != {0,1}:
            raise ValueError("Metrics require complete pairs with distinct members")
        predictions = [m["normalized_prediction"] for m in members]
        vectors.append({"pair_id": pair_id,
                        "accuracy": sum(m["score"] for m in members)/2,
                        "pair_accuracy": float(all(m["canonical_correct"] for m in members)),
                        "identical_answer_rate": float(predictions[0] == predictions[1]),
                        "empty_answer_rate": sum(m["empty"] for m in members)/2,
                        "partner_answer_rate": sum(m["follows_partner_answer"] for m in members)/2,
                        "neither_original_nor_partner_rate": sum(not m["canonical_correct"] and not m["follows_partner_answer"] for m in members)/2,
                        "reference_nll": sum(m["reference_nll"] for m in members)/2,
                        "generation_seconds_per_answer": sum(m["generation_seconds"] for m in members)/2})
    return vectors


def summarize_suite(records, *, seed=6304, resamples=4000):
    if resamples < 1:
        raise ValueError("Bootstrap resamples must be positive")
    conditions = ("correct", "complementary", "mismatched", "no_image")
    vectors = {condition: pair_vectors([r for r in records if r["condition"] == condition]) for condition in conditions}
    pair_ids = [r["pair_id"] for r in vectors["correct"]]
    if not pair_ids or any([r["pair_id"] for r in v] != pair_ids for v in vectors.values()):
        raise ValueError("Conditions must contain the same complete pairs")
    # One matrix of sampled pair indices reused across conditions AND differences.
    indices = np.random.default_rng(seed).integers(len(pair_ids), size=(resamples, len(pair_ids)))
    keys = [key for key in vectors["correct"][0] if key != "pair_id"]
    summary = {"pairs": len(pair_ids), "questions": 2*len(pair_ids), "bootstrap_resamples": resamples,
               "bootstrap_unit": "complete_pair", "seed": seed, "conditions": {}}
    arrays = {}
    for condition in conditions:
        values = {key: np.array([row[key] for row in vectors[condition]], dtype=np.float64) for key in keys}
        arrays[condition] = values
        condition_records = [r for r in records if r["condition"] == condition]
        by_type = {}
        for kind in sorted({r["question_type"] for r in condition_records}):
            members = [r for r in condition_records if r["question_type"] == kind]
            type_pairs = sorted({r["pair_id"] for r in members})
            per_pair = np.array([sum(r["score"] for r in members if r["pair_id"] == pid)/2 for pid in type_pairs])
            type_indices = np.random.default_rng(seed).integers(len(per_pair), size=(resamples, len(per_pair)))
            by_type[kind] = {"questions": len(members), "accuracy": float(per_pair.mean()),
                             "accuracy_ci95": np.quantile(per_pair[type_indices].mean(1), [.025,.975]).tolist()}
        summary["conditions"][condition] = {
            "metrics": {key: {"mean": float(value.mean()),
                               "ci95": np.quantile(value[indices].mean(1), [.025,.975]).tolist()}
                        for key, value in values.items()}, "question_types": by_type}
    differences = {
        "accuracy_advantage_correct_minus_complementary": arrays["correct"]["accuracy"]-arrays["complementary"]["accuracy"],
        "nll_advantage_complementary_minus_correct": arrays["complementary"]["reference_nll"]-arrays["correct"]["reference_nll"],
    }
    summary["paired_differences"] = {name: {"mean": float(value.mean()),
                                           "ci95": np.quantile(value[indices].mean(1), [.025,.975]).tolist()}
                                     for name,value in differences.items()}
    original = {(r["pair_id"],r["member_index"]):r["normalized_prediction"] for r in records if r["condition"] == "correct"}
    summary["prediction_change_rate_under_complementary"] = sum(
        r["normalized_prediction"] != original[(r["pair_id"],r["member_index"])]
        for r in records if r["condition"] == "complementary")/(2*len(pair_ids))
    return summary
