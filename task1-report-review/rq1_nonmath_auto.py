#!/usr/bin/env python3
"""Automatic, read-only RQ1 non-math analysis. stdlib only.

Run from task1-report-review or pass --root PATH.
Outputs go to --out PATH; does not change training/evaluation data.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
from collections import Counter
from pathlib import Path

BENCHMARKS = {"hellaswag": ("acc_norm", 10042), "boolq": ("acc", 3270), "arc_easy": ("acc_norm", 2376)}
METHODS = ("baseline", "full_mid", "full_final", "l2sp_final", "lora4_final", "lora16_final", "oplora16_final")
OFFICIAL = {
    "baseline": (47.31,64.22,55.98),
    "full_mid": (45.68,53.33,62.46),
    "full_final": (46.00,55.75,63.97),
    "l2sp_final": (46.47,66.48,63.68),
    "lora4_final": (47.50,69.76,62.92),
    "lora16_final": (47.14,71.13,63.34),
    "oplora16_final": (47.14,70.31,62.12),
}

def benchmark_name(name):
    s = re.sub(r"[^a-z0-9]", "", name.lower())
    if "hellaswag" in s: return "hellaswag"
    if "boolq" in s: return "boolq"
    if "arceasy" in s or "arceasy" in s.replace("ai2", ""): return "arc_easy"
    return None

def method_name(name):
    s = re.sub(r"[^a-z0-9]", "", name.lower())
    for m in METHODS:
        if s == re.sub(r"[^a-z0-9]", "", m): return m
    # allow conventional names that encode 'final'
    aliases = {"fullftfinal":"full_final", "fullftmid":"full_mid", "l2sp":"l2sp_final", "lora4":"lora4_final", "lora16":"lora16_final", "oplora16":"oplora16_final", "original":"baseline"}
    return aliases.get(s)

def search_samples(root):
    ev = root / "artifacts" / "task1" / "evaluation"
    if not ev.is_dir():
        raise SystemExit(f"Cannot find evaluation directory: {ev}")
    found = {}
    for path in sorted(ev.rglob("*.jsonl")):
        if path.name not in ("samples.jsonl", "metric_filter_samples.jsonl"):
            continue
        parts = path.relative_to(ev).parts[:-1]
        bs = [benchmark_name(x) for x in parts]
        ms = [method_name(x) for x in parts]
        bm = next((x for x in bs if x), None)
        m = next((x for x in ms if x), None)
        if not bm or not m or (m, bm) not in [(a,b) for a in METHODS for b in BENCHMARKS]: continue
        # For these benchmarks prefer samples.jsonl if both exist
        old = found.get((m,bm))
        if old is None or (path.name == "samples.jsonl" and old.name != "samples.jsonl"):
            found[(m,bm)] = path
    return found

def correctness(row, metric):
    v = row.get(metric)
    if v is None and isinstance(row.get("metrics"),dict): v = row["metrics"].get(metric)
    if isinstance(v,bool): return int(v)
    if isinstance(v,(int,float)) and v in (0, 0.0, 1, 1.0): return int(v)
    raise ValueError(f"Metric {metric!r} missing or not 0/1; keys: {list(row)[:25]}")

def document(row):
    d = row.get("doc")
    return d if isinstance(d,dict) else {}

def doc_id(row):
    k = row.get("doc_id")
    if k is None: raise ValueError("record missing doc_id")
    return str(k)

def question(doc, bm):
    if bm == "hellaswag":
        return str(doc.get("ctx", doc.get("ctx_a", "")))
    return str(doc.get("question", doc.get("query", "")))

def word_count(t):
    return len(re.findall(r"\b[\w'-]+\b", t))

def yes_answer(doc):
    v=doc.get("answer")
    if isinstance(v,bool): return v
    if isinstance(v,str) and v.strip().lower() in ("yes","true","no","false"):
        return v.strip().lower() in ("yes","true")
    return None

def load(path, bm):
    metric=BENCHMARKS[bm][0]
    out={}
    with path.open(encoding="utf-8") as f:
        for line_num,line in enumerate(f,1):
            if not line.strip(): continue
            r=json.loads(line)
            k=doc_id(r)
            if k in out: raise ValueError(f"Duplicate doc_id={k} in {path} line {line_num}")
            d=document(r)
            out[k]={"ok":correctness(r,metric), "hash":r.get("doc_hash"), "question":question(d,bm), "passage":str(d.get("passage","")), "gold_yes":yes_answer(d)}
    if not out: raise ValueError(f"No records: {path}")
    return out

def check_alignment(a, b, bm, m):
    if a.keys()!=b.keys():
        missing = list(a.keys()-b.keys())[:5]
        extra = list(b.keys()-a.keys())[:5]
        raise ValueError(f"Misaligned {bm} {m}: missing={missing}, extra={extra}")
    for k in a:
        x,y=a[k],b[k]
        if x["hash"] and y["hash"] and x["hash"]!=y["hash"]:
            raise ValueError(f"doc_hash mismatch: {bm} {m} doc_id={k}")
        if x["question"] and y["question"] and x["question"]!=y["question"]:
            raise ValueError(f"Question mismatch: {bm} {m} doc_id={k}")

def ci_for_paired(base, other, seed, reps=4000):
    # SE estimated on per-question paired differences, normal CI, conservative for large N;
    # permutation-based sign-flip p-value for paired null, reproducibly seeded.
    n=len(base)
    d=[b-a for a,b in zip(base,other)]
    pp=100*sum(d)/n
    se=100*statistics.stdev(d)/math.sqrt(n) if n>1 else 0.
    low,hi=pp-1.96*se,pp+1.96*se
    plus=d.count(1); minus=d.count(-1)
    changed=plus+minus
    if changed:
        rng=random.Random(seed)
        extreme=0
        # Under equal probabilities of plus/minus discordances, 2-sided permutation
        observed=abs(plus-minus)
        for _ in range(reps):
            nplus=sum(rng.random()<0.5 for _ in range(changed))
            if abs(2*nplus-changed)>=observed: extreme+=1
        p=(extreme+1)/(reps+1)
    else:p=1.0
    return dict(n=n, delta_pp=round(pp,3), ci95_low_pp=round(low,3), ci95_high_pp=round(hi,3), baseline_wrong_model_right=plus, baseline_right_model_wrong=minus, permutation_p=round(p,6))

def write_csv(path, rows):
    if not rows:return
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.cwd())
    ap.add_argument("--out", type=Path)
    args=ap.parse_args()
    root=args.root.resolve()
    out=(args.out or root/"RQ1_nonmath_auto").resolve()
    if out==root or out==root/"artifacts":raise SystemExit("Unsafe output path")
    out.mkdir(parents=True,exist_ok=True)
    paths=search_samples(root)
    print("Discovered evaluation sample files:")
    for (m,bm),path in sorted(paths.items()): print(f"  {m:15} {bm:12} {path}")
    absent=[(m,bm) for m in METHODS for bm in BENCHMARKS if (m,bm) not in paths]
    if absent:
        print("WARNING missing condition/benchmark pairs:",absent)
        print("If names differ, inspect the printed directory paths and the actual filenames.")
    data={}
    scores=[]
    for (m,bm),path in paths.items():
        d=load(path,bm)
        n=len(d); score=100*sum(r["ok"] for r in d.values())/n
        expected=OFFICIAL[m][list(BENCHMARKS).index(bm)]
        if abs(score-expected)>0.15:
            raise ValueError(f"Score mismatch {m} {bm}: read {score:.3f}% from {path}, expected {expected:.2f}%. Wrong metric/filter/file?")
        if n!=BENCHMARKS[bm][1]: raise ValueError(f"Unexpected sample count {m} {bm}: {n}; expected {BENCHMARKS[bm][1]}")
        data[(m,bm)]=d
        scores.append(dict(method=m,benchmark=bm,n=n,accuracy_pct=round(score,4)))
    write_csv(out/"verified_scores.csv",scores)

    paired=[]; strata=[]; bias=[]
    for bm in BENCHMARKS:
        if ("baseline",bm) not in data:continue
        base=data[("baseline",bm)]
        keys=sorted(base.keys(),key=lambda z:(not z.isdigit(),int(z) if z.isdigit() else z))
        bl=[base[k]["ok"] for k in keys]
        for mi,m in enumerate(METHODS):
            if (m,bm) not in data or m=="baseline":continue
            d=data[(m,bm)]
            check_alignment(base,d,bm,m)
            target=[d[k]["ok"] for k in keys]
            paired.append(dict(benchmark=bm,comparison=f"{m} minus baseline",**ci_for_paired(bl,target,6304+mi*10+list(BENCHMARKS).index(bm))))

        # Objectively measurable *text-length* strata. NOT annotated reasoning steps.
        if bm in ("boolq","arc_easy"):
            qlength={k:word_count(base[k]["question"]) for k in keys}
            threshold=statistics.median(qlength.values())
            for label, subset in (
                ("shorter_or_equal_question",[k for k in keys if qlength[k]<=threshold]),
                ("longer_question",[k for k in keys if qlength[k]>threshold]),
            ):
                if not subset:
                    print(f"WARNING {bm}: {label} empty; all questions may have identical lengths")
                    continue
                for mi,m in enumerate(METHODS):
                    if (m,bm) not in data or m=="baseline":continue
                    d=data[(m,bm)]
                    b0=[base[k]["ok"] for k in subset]
                    t0=[d[k]["ok"] for k in subset]
                    result=ci_for_paired(b0,t0,113+mi+len(subset))
                    strata.append(dict(benchmark=bm,method=m,stratum=label,median_question_words=threshold,**result))

        if bm=="boolq":
            for m in METHODS:
                if (m,bm) not in data:continue
                d=data[(m,bm)]
                if not all(d[k]["gold_yes"] is not None for k in keys):
                    print(f"WARNING {m} BoolQ documents have no usable gold Yes/No labels; skipping answer-bias check")
                    continue
                n_yes=sum(d[k]["gold_yes"] for k in keys)
                n_no=len(keys)-n_yes
                correct_yes=sum(d[k]["ok"] for k in keys if d[k]["gold_yes"])
                correct_no=sum(d[k]["ok"] for k in keys if not d[k]["gold_yes"])
                # Binary forced-choice: an incorrect decision is necessarily the opposite option.
                predicted_yes=sum((d[k]["gold_yes"] == bool(d[k]["ok"])) for k in keys)
                bias.append(dict(method=m,n=len(keys),gold_yes_pct=round(100*n_yes/len(keys),3),predicted_yes_pct=round(100*predicted_yes/len(keys),3),accuracy_on_gold_yes_pct=round(100*correct_yes/n_yes,3),accuracy_on_gold_no_pct=round(100*correct_no/n_no,3)))

    write_csv(out/"paired_accuracy.csv",paired)
    write_csv(out/"question_length_strata.csv",strata)
    write_csv(out/"boolq_answer_bias.csv",bias)
    notes=(
        "All calculations use the official sample-level acc (BoolQ) or acc_norm (ARC-Easy/HellaSwag),"
        " paired by doc_id with question/doc-hash alignment checks.\n"
        "Paired 95% confidence intervals are normal approximations of paired Bernoulli differences;"
        " p-values are two-sided randomization tests under paired symmetry (4000 iterations),"
        " so very small p-values are limited by simulation resolution.\n"
        "Question-length strata are AUTOMATIC TEXT-LENGTH PROXIES ONLY. A longer question is NOT"
        " necessarily multi-step, and improvements in that stratum do NOT establish transfer of"
        " multi-step reasoning. Passage length is deliberately not used for BoolQ complexity.\n"
        "BoolQ predicted Yes rate is inferred from binary correctness plus gold label; this is"
        " valid only because the benchmark is forced-choice Yes/No.\n"
        "No reasoning traces are generated during these evaluations. Neither transfer mechanisms"
        " nor actual internal reasoning steps can be identified from these metrics.\n"
        "The analyses are exploratory, without multiplicity correction.\n"
    )
    (out/"READ_ME_FIRST.txt").write_text("".join(notes),encoding="utf-8")
    print("\n=== VERIFIED SCORES ===")
    for r in scores: print(f"{r['method']:15} {r['benchmark']:12} {r['accuracy_pct']:.2f}%")
    print("\n=== PAIRED VERSUS BASELINE ===")
    for r in paired: print(f"{r['benchmark']:12} {r['comparison']:29} {r['delta_pp']:+6.2f} pp  CI [{r['ci95_low_pp']:+6.2f},{r['ci95_high_pp']:+6.2f}] p~{r['permutation_p']:.4g}")
    print("\n=== BOOLQ ANSWER-BIAS CHECK ===")
    for r in bias: print(f"{r['method']:15} PredYes={r['predicted_yes_pct']:6.2f}% YesAcc={r['accuracy_on_gold_yes_pct']:6.2f}% NoAcc={r['accuracy_on_gold_no_pct']:6.2f}%")
    print("\n=== AUTOMATIC QUESTION-LENGTH PROXY (NOT REASONING LABELS) ===")
    for r in strata: print(f"{r['benchmark']:9} {r['method']:15} {r['stratum']:26} n={r['n']:5d} delta={r['delta_pp']:+6.2f} pp")
    print(f"\nSaved outputs to {out}")

if __name__=="__main__":main()
