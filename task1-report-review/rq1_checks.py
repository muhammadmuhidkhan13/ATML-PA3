from pathlib import Path
from collections import defaultdict
from math import comb
import csv, json
import numpy as np

review = Path(__file__).resolve().parent
source = review / "artifacts" / "task1"
out = review / "RQ1_checks"
out.mkdir(exist_ok=True)

conditions = [
    "baseline", "full_mid", "full_final", "l2sp_final",
    "lora4_final", "lora16_final", "oplora16_final"
]

pairs = [("baseline", m) for m in conditions[1:]] + [
    ("full_mid", "full_final"),
    ("full_final", "l2sp_final"),
    ("full_final", "lora16_final"),
    ("lora4_final", "lora16_final"),
    ("lora16_final", "oplora16_final")
]

def write_csv(name, rows):
    if not rows:
        raise ValueError(f"No rows for {name}")
    with (out / name).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def read_samples(name):
    folder = source / "evaluation" / name / "gsm8k_cot"
    summary = json.loads(
        (folder / "summary.json").read_text(encoding="utf-8")
    )

    if (summary.get("status") != "COMPLETE_FULL_SPLIT"
            or summary.get("examples") != 1319):
        raise ValueError(f"{name}: incomplete evaluation")

    if "flexible-extract" not in str(summary.get("official_metric_key")):
        raise ValueError(f"{name}: unexpected primary metric")

    by_id = {}

    with (folder / "metric_filter_samples.jsonl").open(
        encoding="utf-8"
    ) as f:
        for line in f:
            r = json.loads(line)

            if r.get("filter") != "flexible-extract":
                continue

            k = str(r["doc_id"])

            if k in by_id or r["exact_match"] not in (0, 1, 0.0, 1.0):
                raise ValueError(f"{name}: invalid score or duplicate {k}")

            by_id[k] = (
                int(r["exact_match"]),
                str(r.get("doc_hash", "")),
                str(r.get("target_hash", ""))
            )

    if len(by_id) != 1319:
        raise ValueError(f"{name}: expected 1319 unique questions")

    score = sum(x[0] for x in by_id.values()) / 1319

    if abs(score - float(summary["primary_metric"])) > 1e-10:
        raise ValueError(f"{name}: score differs from official summary")

    print(
        f"{name:16} {score*100:6.2f}% "
        f"batch={summary.get('batch_size')}"
    )

    return by_id, summary.get("batch_size")

print("=== VERIFY OFFICIAL GSM8K RESULTS ===")

loaded = {name: read_samples(name) for name in conditions}

ids = sorted(
    loaded["baseline"][0],
    key=lambda s: (0, int(s)) if s.isdigit() else (1, s)
)

for name in conditions[1:]:
    x = loaded[name][0]

    if set(x) != set(ids):
        raise ValueError(f"{name}: question IDs differ")

    for k in ids:
        for col in (1, 2):
            a = loaded["baseline"][0][k][col]
            b = x[k][col]

            if a and b and a != b:
                raise ValueError(
                    f"{name}: document or target hash differs for {k}"
                )

if len(set(str(v[1]) for v in loaded.values())) > 1:
    print("WARNING: GSM8K batch sizes differ between conditions.")

print("\n=== PAIRED GSM8K COMPARISONS ===")

rng = np.random.default_rng(6304)
results, changes = [], []

for a, b in pairs:
    A = np.array([loaded[a][0][k][0] for k in ids])
    B = np.array([loaded[b][0][k][0] for k in ids])

    d = B - A

    a_only = int(((A == 1) & (B == 0)).sum())
    b_only = int(((A == 0) & (B == 1)).sum())
    both = int(((A == 1) & (B == 1)).sum())

    # Paired bootstrap: resample questions, not models independently.
    means = []

    for _ in range(50):
        ix = rng.integers(0, len(d), size=(100, len(d)))
        means.extend(d[ix].mean(axis=1))

    lo, hi = np.percentile(means, [2.5, 97.5]) * 100

    # Exact two-sided McNemar test.
    n = a_only + b_only

    if n:
        p = min(
            1.0,
            2 * sum(
                comb(n, j)
                for j in range(min(a_only, b_only) + 1)
            ) / 2**n
        )
    else:
        p = 1.0

    record = dict(
        comparison=f"{b} minus {a}",
        N=len(d),
        A_correct=int(A.sum()),
        B_correct=int(B.sum()),
        A_only_correct=a_only,
        B_only_correct=b_only,
        both_correct=both,
        both_wrong=len(d) - both - a_only - b_only,
        difference_pp=round(float(d.mean() * 100), 3),
        CI95_low_pp=round(float(lo), 3),
        CI95_high_pp=round(float(hi), 3),
        exact_McNemar_p=f"{p:.6g}"
    )

    results.append(record)

    for k, da in zip(ids, d):
        if da:
            changes.append(dict(
                comparison=record["comparison"],
                doc_id=k,
                direction="B_gained" if da > 0 else "B_lost"
            ))

    print(
        f"{record['comparison']:38} "
        f"{record['difference_pp']:+6.2f} pp "
        f"95% CI [{lo:+.2f}, {hi:+.2f}] "
        f"A-only={a_only} B-only={b_only} p={p:.3g}"
    )

write_csv("gsm8k_paired_comparisons.csv", results)
write_csv("gsm8k_changed_questions.csv", changes)

print("\n=== TRAINING LOSS CHECK ===")

with (
    source / "report-data-v1" / "export" / "training_curves.csv"
).open(newline="", encoding="utf-8-sig") as f:
    raw = list(csv.DictReader(f))

by_method = defaultdict(list)

for r in raw:
    by_method[r["method"]].append(r)

loss_table = []
curves = {}

for method in ["full", "l2sp", "lora4", "lora16", "oplora16"]:
    rows = sorted(
        by_method[method],
        key=lambda r: int(r["update"])
    )

    steps = np.array([int(r["update"]) for r in rows])
    loss = np.array(
        [float(r["sft_loss"]) for r in rows]
    )

    if (
        not np.array_equal(steps, np.arange(1, 1251))
        or not np.isfinite(loss).all()
    ):
        raise ValueError(f"{method}: invalid training curve")

    row = dict(
        method=method,
        first100_mean=round(float(loss[:100].mean()), 5),
        middle100_mean=round(float(loss[575:675].mean()), 5),
        last100_mean=round(float(loss[-100:].mean()), 5),
        last_update_loss=round(float(loss[-1]), 5)
    )

    loss_table.append(row)
    curves[method] = (steps, loss)
    print(row)

write_csv("training_loss_windows.csv", loss_table)

try:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5))

    for method, (steps, loss) in curves.items():
        smooth = np.convolve(
            loss, np.ones(50) / 50, mode="valid"
        )
        plt.plot(steps[49:], smooth, label=method)

    plt.xlabel("Optimizer update")
    plt.ylabel("50-update moving average SFT loss")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()

    plt.savefig(out / "training_loss_curves.png", dpi=180)
    plt.close()

except ImportError:
    print("Matplotlib unavailable; CSV results saved.")

print(f"\nFinished. Outputs: {out}")
