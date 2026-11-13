"""Exploratory ARC-Easy subgroup analysis of saved evaluation records."""

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


TOTAL = 2376
OFFICIAL = dict(baseline=55.98, full_mid=62.46, full_final=63.97,
                l2sp_final=63.68, lora4_final=62.92, lora16_final=63.34,
                oplora16_final=62.12)
QUANT = "quantitative-indicator"
OTHER = "other"
GROUPS = (QUANT, OTHER)
SEED = 6304
REPLICATES = 10000
RULES = {
    "calculation": r"\bcalculat(?:e|es|ed|ing|ion|ions)\b",
    "explicit_difference": r"\bhow (?:many|much) (?:more|less|fewer)\b",
    "multiplicative_comparison": r"\b(?:twice|half|\d+ times) (?:as (?:much|many|large|long|fast|heavy)|the (?:amount|number|mass|distance|length|speed|size))\b",
    "arithmetic_expression": r"(?<![\w./-])\d+(?:\.\d+)?\s*(?:[+×÷=−]|\s[-*/x]\s)\s*\d+(?:\.\d+)?(?![\w./-])",
    "ratio_or_percent_request": r"\b(?:what|which)\s+(?:is\s+the\s+)?(?:ratio|proportion|percentage|percent|fraction)\s+(?:of|between|would|is|represents?)\b",
    "average_or_rate_request": r"\b(?:what|which)\b[^?!.;]{0,75}\b(?:average (?:speed|velocity|rate|distance|temperature|mass)|rate of (?:change|acceleration))\b",
    "rate_formula": r"\b(?:speed|velocity)\s*(?:=|equals|is)\s*distance\s*(?:/|÷|divided by|per)\s*time\b",
    "numeric_measurement_comparison": r"\b(?:greater|less|longer|shorter|faster|slower|heavier|lighter) than\s+\d+(?:\.\d+)?\s*(?:km/h|m/s|kg|km|cm|mm|mL|°[CF]|meters?|grams?|liters?|seconds?|minutes?|hours?)\b",
}
PATTERNS = {key: re.compile(value, re.IGNORECASE) for key, value in RULES.items()}


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    labels: tuple
    options: tuple
    gold: str


def classify(text, options):
    return tuple(key for key, pattern in PATTERNS.items()
                 if any(pattern.search(field) for field in (text, *options)))


def read_condition(folder, condition):
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    expected = dict(status="COMPLETE_FULL_SPLIT", examples=TOTAL,
                    benchmark="arc_easy", condition=condition,
                    official_metric_key="acc_norm,none")
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(f"{condition}: invalid evaluation summary field {key}")

    questions, scores, source_ids = {}, {}, set()
    with (folder / "samples.jsonl").open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                doc_id, doc = row["doc_id"], row["doc"]
                question = Question(doc["id"], doc["question"], tuple(doc["choices"]["label"]),
                                    tuple(doc["choices"]["text"]), doc["answerKey"])
                score = row["acc_norm"]
                if (type(doc_id) is not int or doc_id < 0 or doc_id in questions
                        or not question.id or question.id in source_ids
                        or not question.text or len(question.labels) != len(question.options)
                        or len(set(question.labels)) != len(question.labels)
                        or any(not isinstance(field, str) or not field.strip()
                               for field in (question.id, question.text, *question.options))
                        or question.gold not in question.labels
                        or type(row["target"]) is not int
                        or row["target"] != question.labels.index(question.gold)
                        or row["filter"] != "none" or "acc_norm" not in row["metrics"]
                        or type(score) not in (int, float) or score not in (0, 1)):
                    raise ValueError("invalid or duplicate question, gold target, or acc_norm")
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{condition}: samples.jsonl line {line_number}: {exc}") from exc
            questions[doc_id], scores[doc_id] = question, int(score)
            source_ids.add(question.id)

    if len(questions) != TOTAL:
        raise ValueError(f"{condition}: expected {TOTAL} unique questions, got {len(questions)}")
    accuracy = sum(scores.values()) / TOTAL
    if (not np.isclose(accuracy, summary["primary_metric"], rtol=0, atol=1e-12)
            or round(100 * accuracy, 2) != OFFICIAL[condition]):
        raise ValueError(f"{condition}: acc_norm {accuracy:.8f} disagrees with official results")
    return questions, scores


def aligned_scores(reference, questions, scores, condition):
    if reference.keys() != questions.keys():
        raise ValueError(f"{condition}: document IDs differ from baseline")
    if any(question != questions[key] for key, question in reference.items()):
        raise ValueError(f"{condition}: question content or gold label differs from baseline")
    return np.array([scores[key] for key in reference], dtype=np.int8)


def subgroup_gains(baseline, scores, quantitative, rng, replicates=REPLICATES):
    delta = scores - baseline
    samples = (delta[quantitative], delta[~quantitative])
    if any(len(group) == 0 for group in samples):
        raise ValueError("Both subgroups must contain questions")
    point = [100 * group.mean() for group in samples]
    draws = []
    for group in samples:
        # Paired deltas are -1, 0 or +1. Multinomial counts give the same
        # bootstrap distribution as sampling matched question indices.
        counts = np.bincount(group + 1, minlength=3)
        resampled = rng.multinomial(len(group), counts / len(group), size=replicates)
        draws.append(100 * (resampled[:, 2] - resampled[:, 0]) / len(group))
    intervals = [np.percentile(values, (2.5, 97.5))
                 for values in (*draws, draws[0] - draws[1])]
    return dict(zip(("quantitative_gain", "other_gain", "d"),
                    zip((*point, point[0] - point[1]), intervals)))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    review = Path(__file__).resolve().parent
    source = review / "artifacts" / "task1" / "evaluation"
    output = review / "RQ1_arc_quantitative"
    folders = {condition: source / condition / "arc_easy" for condition in OFFICIAL}
    reference, baseline_scores = read_condition(folders["baseline"], "baseline")
    matches = [classify(question.text, question.options) for question in reference.values()]
    quantitative = np.array([bool(rules) for rules in matches])
    masks = (quantitative, ~quantitative)
    counts = [int(mask.sum()) for mask in masks]
    if not all(counts):
        raise ValueError("Both subgroups must contain questions")
    classification = [
        dict(doc_id=doc_id, question_id=q.id, classification=QUANT if rules else OTHER,
             matched_rule_ids="|".join(rules), question_text=q.text,
             answer_options=json.dumps(q.options, ensure_ascii=False))
        for (doc_id, q), rules in zip(reference.items(), matches, strict=True)
    ]
    baseline = aligned_scores(reference, reference, baseline_scores, "baseline")
    rng = np.random.Generator(np.random.PCG64(SEED))
    accuracy_rows, difference_rows = [], []
    print(f"ARC-Easy acc_norm | quantitative n={counts[0]} | other n={counts[1]}")
    print(f"{'Condition':16} {'Overall %':>9} {'Quant %':>9} {'Other %':>9} {'D pp [95% CI]'}")
    for condition, folder in folders.items():
        questions, scores = ((reference, baseline_scores) if condition == "baseline"
                             else read_condition(folder, condition))
        values = aligned_scores(reference, questions, scores, condition)
        correct = [int(values[mask].sum()) for mask in masks]
        if sum(correct) != int(values.sum()):
            raise ValueError(f"{condition}: subgroup totals do not reconstruct overall accuracy")
        subgroup_accuracy = [100 * c / n for c, n in zip(correct, counts)]
        for group, n, c, accuracy, mask in zip(GROUPS, counts, correct, subgroup_accuracy, masks):
            accuracy_rows.append(dict(condition=condition, subgroup=group, n=n, correct=c,
                                      accuracy_percent=accuracy,
                                      gain_pp=100 * (values[mask] - baseline[mask]).mean()))
        display = ""
        if condition != "baseline":
            results = subgroup_gains(baseline, values, quantitative, rng)
            row = dict(condition=condition)
            for name, (estimate, (low, high)) in results.items():
                row.update({f"{name}_pp": estimate, f"{name}_ci_low_pp": low,
                            f"{name}_ci_high_pp": high})
            difference_rows.append(row)
            d, (low, high) = results["d"]
            display = f"{d:+.2f} [{low:+.2f}, {high:+.2f}]"
        print(f"{condition:16} {100 * values.mean():9.2f} "
              f"{subgroup_accuracy[0]:9.2f} {subgroup_accuracy[1]:9.2f} {display}")

    output.mkdir(exist_ok=True)
    write_csv(output / "arc_question_classification.csv", classification)
    write_csv(output / "arc_subgroup_accuracy.csv", accuracy_rows)
    write_csv(output / "arc_subgroup_differences.csv", difference_rows)
    inputs = {str(path.relative_to(review).as_posix()): hashlib.sha256(path.read_bytes()).hexdigest()
              for folder in folders.values()
              for path in (folder / "samples.jsonl", folder / "summary.json")}
    rules = dict(version="1.0.0", flags=["IGNORECASE"], patterns=RULES,
                 categories={QUANT: "At least one rule matches", OTHER: "No rules match"},
                 scope="Search question text and each answer option separately",
                 bootstrap=dict(seed=SEED, replicates=REPLICATES, generator="PCG64",
                                numpy_version=np.__version__),
                 input_sha256=inputs)
    (output / "classification_rules.json").write_text(
        json.dumps(rules, indent=2) + "\n", encoding="utf-8")
    small = [group for group, n in zip(GROUPS, counts) if n < 100]
    if small:
        print(f"Small subgroup (<100 questions): {', '.join(small)}. See confidence intervals.")
    (output / "appendix_methodology.md").write_text(
        f"""# ARC-Easy subgroup analysis method

- Scope. Exploratory, post hoc analysis of {TOTAL:,} saved ARC-Easy test records per condition.
- Proxy. Case-insensitive regex rules in `classification_rules.json` were fixed before inspecting subgroup results. Each rule searches the question and each option separately. Any match assigns the quantitative-indicator category ({counts[0]:,} questions). All remaining questions are other ({counts[1]:,}). Standalone numbers, dates, identifiers, grade levels and unit mentions do not qualify on their own.
- Pairing and score. Match by harness document ID and verify source ID, question, ordered choices and gold label. Use saved `acc_norm` outcomes for filter `none`. Verify overall accuracy against the saved summary and the official rounded result.
- Estimates. Subtract baseline accuracy within each subgroup, in percentage points. Compute D as the quantitative-indicator gain minus the other gain.
- Uncertainty. For each contrast, draw {REPLICATES:,} paired question bootstraps independently within each subgroup, holding subgroup sizes fixed. Multinomial counts over the paired correctness deltas implement sampling with replacement. Use percentile intervals at 2.5 and 97.5 percent for both gains and D. The PCG64 seed is {SEED}. Intervals are pointwise, with no multiplicity adjustment, and reflect question variation conditional on the saved models.
- Limitations. The proxy can miss implicit quantitative reasoning or match a cue in a distractor. It does not measure internal or multi-step reasoning. Subgroups below 100 questions are flagged for limited precision. Small subgroups here: {', '.join(small) or 'none'}.
""", encoding="utf-8")


if __name__ == "__main__":
    main()
