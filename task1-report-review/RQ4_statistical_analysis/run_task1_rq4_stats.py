"""ATML PA3 Task 1: verified paired benchmark inference; reads only original evidence."""
import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
except ImportError:
    sys.exit("Missing NumPy: run with the atml-pa3 Conda Python environment.")

EXPECTED = {"gsm8k_cot": 1319, "hellaswag": 10042,
            "boolq": 3270, "arc_easy": 2376}
CONDITIONS = ("baseline", "full_final", "l2sp_final", "lora4_final",
              "lora16_final", "oplora16_final")
PRIMARY = (("Full FT -> L2-SP", "full_final", "l2sp_final"),
           ("LoRA-4 -> LoRA-16", "lora4_final", "lora16_final"),
           ("LoRA-16 -> OPLoRA-16", "lora16_final", "oplora16_final"),
           ("Full FT -> LoRA-16", "full_final", "lora16_final"))
SECONDARY = tuple(("Baseline -> " + b, "baseline", b)
                  for b in CONDITIONS if b != "baseline")


def die(s):
    raise ValueError(s)


def read_json(p):
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def read_score_csv(p):
    if not p.is_file():
        return {}
    with p.open(encoding="utf-8-sig", newline="") as f:
        entries = list(csv.DictReader(f))
    scores = {}
    for r in entries:
        k = (r["condition"], r["benchmark"])
        if k in scores:
            die(f"Duplicate exported score: {k}")
        scores[k] = (float(r["score"]), r.get("primary_metric_key", ""))
    return scores


def find_evidence(project):
    roots = [project / "task1-report-review" / "artifacts" / "task1",
             project / "artifacts" / "task1"]
    for base in roots:
        p = base / "evaluation"
        if p.is_dir():
            return base
    return None


def read_samples(base, condition, bench, exported_scores):
    folder = base / "evaluation" / condition / bench
    if not folder.is_dir():
        die(f"Missing evaluation folder: {folder}")
    if bench == "gsm8k_cot":
        preferred = folder / "metric_filter_samples.jsonl"
        samples = preferred if preferred.is_file() else folder / "samples.jsonl"
    else:
        samples = folder / "samples.jsonl"
    if not samples.is_file():
        die(f"Missing per-example data: {samples}")
    summary_file = folder / "summary.json"
    if not summary_file.is_file():
        die(f"Missing official evaluation summary: {summary_file}")
    summary = read_json(summary_file)
    key = "exact_match,flexible-extract" if bench == "gsm8k_cot" else (
        "acc,none" if bench == "boolq" else "acc_norm,none")
    if summary.get("official_metric_key") != key:
        die(f"Unexpected primary metric {condition}/{bench}: "
            f"{summary.get('official_metric_key')!r}; expected {key!r}")
    if int(summary.get("examples", -1)) != EXPECTED[bench]:
        die(f"Unexpected official sample count {condition}/{bench}: "
            f"{summary.get('examples')}")
    result = {}
    signatures = {}
    count_skipped_other_filters = 0
    with samples.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                die(f"Invalid JSON {samples}:{line_no}: {e}")
            if bench == "gsm8k_cot":
                if r.get("filter") != "flexible-extract":
                    count_skipped_other_filters += 1
                    continue
                field = "exact_match"
            else:
                field = "acc" if bench == "boolq" else "acc_norm"
            if "doc_id" not in r:
                die(f"Missing doc_id in {samples}:{line_no}")
            k = str(r["doc_id"])
            v = r.get(field)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v not in (0, 0.0, 1, 1.0):
                die(f"Missing/nonbinary primary metric '{field}' for {condition}/{bench} doc_id={k}: {v!r}")
            if k in result:
                die(f"Duplicate official sample {condition}/{bench} doc_id={k}")
            result[k] = int(v)
            signatures[k] = (str(r.get("doc_hash", "")), str(r.get("target_hash", "")))
    if len(result) != EXPECTED[bench]:
        note = (" (GSM8K samples.jsonl may contain only strict-match rows; "
                "metric_filter_samples.jsonl with flexible-extract rows is required)"
                if bench == "gsm8k_cot" else "")
        die(f"Wrong number of selected primary-metric samples {condition}/{bench}: "
            f"{len(result)} versus {EXPECTED[bench]}{note}")
    observed = sum(result.values()) / len(result)
    official = float(summary.get("primary_metric", float("nan")))
    if not math.isfinite(official) or abs(observed - official) > 1e-8:
        die(f"Raw mean {observed:.9f} != official summary {official:.9f} "
            f"for {condition}/{bench}; refuse incorrect inference")
    if (condition, bench) in exported_scores:
        exported, metric = exported_scores[(condition, bench)]
        if abs(observed - exported) > 1e-8 or metric != key:
            die(f"Raw score / metric != sealed export for {condition}/{bench}: "
                f"raw {observed:.9f}, export {exported:.9f} ({metric})")
    return result, signatures, {"samples": str(samples), "summary": str(summary_file),
                                "N": len(result), "primary_score": observed,
                                "discarded_nonprimary_filter_records": count_skipped_other_filters}


def exact_mcnemar_p(a_only, b_only):
    """Exact two-sided McNemar = two-sided Binomial(n, 0.5) on discordant pairs."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    logp = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            - n * math.log(2.0) for i in range(k + 1)]
    m = max(logp)
    return min(1.0, 2 * math.exp(m) * math.fsum(math.exp(x - m) for x in logp))


def paired_bootstrap_ci(a_only, b_only, n, seed, reps):
    """Multinomial 3-cell resampling == resampling paired correctness outcomes."""
    probs = [a_only / n, b_only / n, 1.0 - (a_only + b_only) / n]
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n, probs, size=reps)
    deltas = (counts[:, 1] - counts[:, 0]) * (100.0 / n)
    return tuple(float(x) for x in np.percentile(deltas, [2.5, 97.5]))


def compare(family, name, bench, left, right, data, seed, reps):
    a, b = data[(left, bench)], data[(right, bench)]
    if set(a) != set(b):
        die(f"Doc-ID sets differ {bench}: {left} vs {right}")
    n = len(a)
    a_only = sum(a[k] == 1 and b[k] == 0 for k in a)
    b_only = sum(a[k] == 0 and b[k] == 1 for k in a)
    cilo, cihi = paired_bootstrap_ci(a_only, b_only, n, seed, reps)
    return {"family": family, "comparison": name, "benchmark": bench,
            "A": left, "B": right, "N": n,
            "accuracy_A_pct": round(100 * sum(a.values()) / n, 5),
            "accuracy_B_pct": round(100 * sum(b.values()) / n, 5),
            "difference_B_minus_A_pp": round(100 * (b_only - a_only) / n, 5),
            "ci95_low_pp": round(cilo, 5), "ci95_high_pp": round(cihi, 5),
            "A_only_correct": a_only, "B_only_correct": b_only,
            "discordant_pairs": a_only + b_only,
            "exact_mcnemar_p": exact_mcnemar_p(a_only, b_only),
            "holm_adjusted_p": None, "holm_significant_0_05": None,
            "ci95_excludes_zero_unadjusted": cilo > 0 or cihi < 0}


def apply_holm(rows):
    ordered = sorted(range(len(rows)), key=lambda i: rows[i]["exact_mcnemar_p"])
    previous = 0.0
    m = len(rows)
    for rank, idx in enumerate(ordered):
        p = rows[idx]["exact_mcnemar_p"]
        previous = max(previous, min(1.0, (m - rank) * p))
        rows[idx]["holm_adjusted_p"] = previous
        rows[idx]["holm_significant_0_05"] = previous < 0.05


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=10000)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    audit = {"created_utc": datetime.now(timezone.utc).isoformat(),
             "project": str(args.project), "bootstrap_reps": args.bootstrap,
             "bootstrap_seed": 6304, "units": "accuracy percentage points",
             "primary_family_holm_tests": 16, "secondary_family_holm_tests": 20,
             "available_inputs": {}, "issues": [], "checks": []}
    try:
        root = find_evidence(args.project)
        if root is None:
            die("Could not locate artifacts/task1/evaluation under project or "
                "task1-report-review. Restore the original Task 1 evaluation evidence first.")
        score_path = root / "report-data-v1" / "export" / "benchmark_scores.csv"
        scores = read_score_csv(score_path)
        audit["evidence_root"] = str(root)
        audit["sealed_benchmark_scores"] = str(score_path) if score_path.is_file() else "not found"
        data, hashes = {}, {}
        for bench in EXPECTED:
            for condition in CONDITIONS:
                try:
                    rows, signatures, details = read_samples(root, condition, bench, scores)
                    data[(condition, bench)] = rows
                    hashes[(condition, bench)] = signatures
                    audit["available_inputs"][f"{condition}/{bench}"] = details
                except (ValueError, OSError, KeyError, TypeError) as e:
                    audit["issues"].append(str(e))
        if audit["issues"]:
            die(f"Evidence validation failed for {len(audit['issues'])} conditions; "
                "see audit.json for exact missing or mismatched files")
        # Identity check: doc IDs AND any common available document/target hashes.
        for bench in EXPECTED:
            ids = set(data[("baseline", bench)])
            for condition in CONDITIONS[1:]:
                if set(data[(condition, bench)]) != ids:
                    die(f"Different doc IDs: {condition}/{bench}")
                for doc in ids:
                    s0 = hashes[("baseline", bench)][doc]
                    s1 = hashes[(condition, bench)][doc]
                    if any(x and y and x != y for x, y in zip(s0, s1)):
                        die(f"Different question/target hashes: {condition}/{bench} doc_id={doc}")
            audit["checks"].append(f"{bench}: {len(ids)} paired documents, signatures verified when present")
        primary, secondary = [], []
        for bench_idx, bench in enumerate(EXPECTED):
            for j, (name, left, right) in enumerate(PRIMARY):
                primary.append(compare("primary", name, bench, left, right, data,
                                       6304 + bench_idx * 100 + j, args.bootstrap))
            for j, (name, left, right) in enumerate(SECONDARY):
                secondary.append(compare("baseline_secondary", name, bench, left, right, data,
                                         16304 + bench_idx * 100 + j, args.bootstrap))
        apply_holm(primary)
        apply_holm(secondary)
        write_csv(args.output / "rq4_primary_16_comparisons.csv", primary)
        write_csv(args.output / "baseline_20_comparisons.csv", secondary)
        audit["status"] = "COMPLETE"
        audit["primary_comparisons"] = len(primary)
        audit["secondary_comparisons"] = len(secondary)
        # no interpretation; plain tabular numerical report
        lines = ["TASK 1 RQ4: PAIRED INFERENCE", "",
                 "Difference = B - A, in accuracy percentage points",
                 "95% paired percentile bootstrap CI (10,000 resamples by default)",
                 "Exact two-sided McNemar p, Holm adjustment within each family",
                 "NOTE: 95% CIs are unadjusted (not simultaneous); Holm applies to p-values.",
                 "Do not read 'not significant' as proven equivalence.", ""]
        for title, rows in (("PRIMARY: 16 CONTROLLED COMPARISONS", primary),
                            ("SECONDARY: 20 BASELINE COMPARISONS", secondary)):
            lines.extend([title, "-"])
            for row in rows:
                lines.append(
                    f"{row['benchmark']:10s} | {row['comparison']:26s} | "
                    f"{row['difference_B_minus_A_pp']:+7.3f} pp "
                    f"[{row['ci95_low_pp']:+7.3f}, {row['ci95_high_pp']:+7.3f}] "
                    f"p={row['exact_mcnemar_p']:.4g} "
                    f"Holm={row['holm_adjusted_p']:.4g} "
                    f"significant={row['holm_significant_0_05']}")
            lines.append("")
        (args.output / "results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        print("Saved: rq4_primary_16_comparisons.csv")
        print("Saved: baseline_20_comparisons.csv")
        print("Saved: results.txt")
    except Exception as e:
        audit["status"] = "FAILED"
        if str(e) not in audit["issues"]:
            audit["issues"].append(str(e))
        print(f"ERROR: {e}", file=sys.stderr)
        print("See audit.json for details.", file=sys.stderr)
        return 1
    finally:
        (args.output / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        print("Saved: audit.json")
    return 0

if __name__ == "__main__":
    sys.exit(main())
