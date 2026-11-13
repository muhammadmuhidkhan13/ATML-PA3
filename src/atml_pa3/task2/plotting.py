"""Export saved results and figures; never invent missing experiment values."""
from collections import defaultdict
import csv
import html
import json
import os
from pathlib import Path
from .io import atomic_json,read_json


def csv_rows(path,rows):
    if not rows: return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w",encoding="utf-8",newline="") as out:
        writer=csv.DictWriter(out,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def export_artifacts(output,*,run_dir=None,evaluation_dir=None,root=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    from .result_validation import verify_final,verify_development
    if evaluation_dir is not None:verify_final(evaluation_dir)
    if run_dir is not None:
        for path in (Path(run_dir)/'development').glob('update_*/summary.json'):verify_development(path.parent)
    output.mkdir(parents=True)
    generated=[]
    def save(fig,name):
        fig.tight_layout();fig.savefig(output/(name+".png"),dpi=160);plt.close(fig)
        generated.append(name+".png")
    if run_dir is not None:
        run_dir=Path(run_dir)
        events=[json.loads(line) for line in (run_dir/"updates.jsonl").read_text(encoding="utf-8").splitlines()]
        # Resume can replay work after the last durable checkpoint. Keep the
        # latest completed realization for curves; charge all attempts as cost.
        updates={r["global_update"]:r for r in events if r["event"]=="completed_update"}
        rows=[{"global_update":u,"stage":r["stage"],"sft_loss":r["sft_loss"],
               "connector_lr":r["learning_rates"]["connector"],"lora_lr":r["learning_rates"].get("lora",0.),
               "completed_examples":r["completed_examples"],"peak_gpu_allocated":r["peak_gpu_allocated"],
               "peak_gpu_reserved":r["peak_gpu_reserved"]} for u,r in sorted(updates.items())]
        csv_rows(output/"training.csv",rows)
        if rows:
            fig,ax=plt.subplots(figsize=(7,3));ax.plot([r["global_update"] for r in rows],[r["sft_loss"] for r in rows])
            for boundary in (2250,3750):ax.axvline(boundary,color="gray",linestyle="--")
            ax.set(xlabel="Global optimizer update",ylabel="Example-weighted SFT loss");save(fig,"training_loss")
        resources=[]
        for stage in (1,2,3):
            completed=[r for r in events if r["event"]=="completed_update" and r["stage"]==stage]
            attempts=[r for r in events if r["event"]=="optimizer_attempt" and r["stage"]==stage]
            resources.append({"stage":stage,"attempt_seconds_including_replayed_work":sum(r["seconds"] for r in attempts),
                              "checkpoint_seconds":sum(r["seconds"] for r in events if r["event"]=="checkpoint" and
                                                       (0 if stage==1 else (2250,3750)[stage-2])<r["global_update"]<=(2250,3750,4500)[stage-1]),
                              "development_seconds":sum(r["seconds"] for r in events if r["event"]=="development" and r["stage"]==stage),
                              "transition_seconds":sum(r["seconds"] for r in events if r["event"]=="transition" and r["stage"]==stage),
                              "setup_seconds":sum(r["controller_initialization_seconds"]+(r.get("runtime_setup") or {}).get("seconds",0.)
                                                  for r in events if r["event"]=="stage_setup" and r["stage"]==stage),
                              "attempts":len(attempts),"skipped_attempts":sum(not r["completed"] for r in attempts),
                              "trainable_parameters":completed[-1]["trainable_parameters"] if completed else None,
                              "peak_allocated_bytes":max((r["peak_gpu_allocated"] or 0 for r in completed),default=0),
                              "peak_reserved_bytes":max((r["peak_gpu_reserved"] or 0 for r in completed),default=0)})
        csv_rows(output/"resources.csv",resources)
        if any(r["attempts"] for r in resources):
            fig,axes=plt.subplots(1,3,figsize=(11,3))
            for ax,key,label in zip(axes,("attempt_seconds_including_replayed_work","peak_allocated_bytes","trainable_parameters"),
                                    ("Attempt time (seconds)","Peak allocated bytes","Trainable parameters")):
                ax.bar([1,2,3],[r[key] or 0 for r in resources]);ax.set(xlabel="Stage",ylabel=label)
            save(fig,"resources")
        dev=[]
        for path in sorted((run_dir/"development").glob("update_*/summary.json")):
            summary=read_json(path)
            if not summary["complete"]: raise ValueError("Incomplete development results")
            row={"global_update":int(path.parent.name.split("_")[1]),"caption_loss":summary["caption_reference_loss"]}
            for suite,values in summary["suites"].items():
                for condition,value in values["reference_nll"].items():row[f"{suite}_{condition}_nll"]=value
                row[f"{suite}_swap_penalty"]=values["swap_penalty"]
            dev.append(row)
        csv_rows(output/"development.csv",dev)
        if dev:
            fig,axes=plt.subplots(1,3,figsize=(13,3))
            x=[r["global_update"] for r in dev]
            axes[0].plot(x,[r["caption_loss"] for r in dev]);axes[0].set_ylabel("Caption reference loss")
            for suite in ("grounding","natural"):
                for condition in ("correct","complementary","no_image"):
                    axes[1].plot(x,[r[f"{suite}_{condition}_nll"] for r in dev],label=f"{suite} {condition}")
                axes[2].plot(x,[r[f"{suite}_swap_penalty"] for r in dev],label=suite)
            axes[1].set_ylabel("Answer NLL (no EOT)");axes[2].set_ylabel("Swap NLL minus correct NLL")
            axes[1].legend(fontsize=6);axes[2].legend(fontsize=7)
            for ax in axes:
                ax.set_xlabel("Global optimizer update")
                for boundary in (2250,3750):ax.axvline(boundary,color="gray",linestyle="--")
            save(fig,"development_curves")
    if evaluation_dir is not None:
        evaluation_dir=Path(evaluation_dir)
        if not read_json(evaluation_dir/"manifest.json")["complete"]: raise ValueError("Final outputs are incomplete")
        summaries=read_json(evaluation_dir/"summary.json")
        rows=[];qualitative=[];inference=[]
        for suite,summary in summaries.items():
            for condition,values in summary["conditions"].items():
                for metric,result in values["metrics"].items():
                    rows.append({"suite":suite,"condition":condition,"metric":metric,"mean":result["mean"],
                                 "ci95_low":result["ci95"][0],"ci95_high":result["ci95"][1]})
                for kind,result in values["question_types"].items():
                    rows.append({"suite":suite,"condition":condition,"metric":"accuracy","question_type":kind,
                                 "mean":result["accuracy"],"ci95_low":result["accuracy_ci95"][0],"ci95_high":result["accuracy_ci95"][1]})
            for name,value in summary["paired_differences"].items():
                rows.append({"suite":suite,"condition":"paired_difference","metric":name,"mean":value["mean"],
                             "ci95_low":value["ci95"][0],"ci95_high":value["ci95"][1]})
            conditions=list(summary["conditions"])
            fig,axes=plt.subplots(1,3,figsize=(12,3))
            for ax,metric in zip(axes,("accuracy","pair_accuracy","reference_nll")):
                values=[summary["conditions"][c]["metrics"][metric] for c in conditions]
                ax.bar(conditions,[v["mean"] for v in values])
                ax.tick_params(axis="x",labelrotation=25)
                ax.set_title(suite+": "+metric)
                ax.set_ylabel("NLL" if metric=="reference_nll" else "Fraction")
            save(fig,suite+"_conditions")
            fig,axes=plt.subplots(1,2,figsize=(9,3))
            for metric in ("identical_answer_rate","empty_answer_rate"):
                axes[0].plot(conditions,[summary["conditions"][c]["metrics"][metric]["mean"] for c in conditions],marker="o",label=metric)
            axes[0].legend(fontsize=6);axes[0].tick_params(axis="x",labelrotation=25);axes[0].set_ylabel("Fraction")
            differences=summary["paired_differences"]
            for i,(name,value) in enumerate(differences.items()):
                axes[1].plot(i,value["mean"],"o");axes[1].vlines(i,*value["ci95"])
            axes[1].set_xticks(range(len(differences)),["Accuracy advantage","NLL advantage"],rotation=15)
            axes[1].axhline(0,color="gray",linestyle="--");save(fig,suite+"_pair_diagnostics")
            kinds=list(summary["conditions"]["correct"]["question_types"])
            fig,ax=plt.subplots(figsize=(7,3))
            for condition in conditions:
                ax.plot(kinds,[summary["conditions"][condition]["question_types"][k]["accuracy"] for k in kinds],marker="o",label=condition)
            ax.legend(fontsize=7);ax.set_ylabel("Accuracy (fraction)");save(fig,suite+"_question_types")
            records=read_json(evaluation_dir/suite/"records.json")
            for condition in conditions:
                subset=[r for r in records if r["condition"]==condition]
                inference.append({"suite":suite,"condition":condition,"answers":len(subset),
                                  "generation_seconds":sum(r["generation_seconds"] for r in subset),
                                  "reference_seconds":sum(r["reference_seconds"] for r in subset),
                                  "generated_tokens":sum(r["generated_tokens"] for r in subset),
                                  "token_cap_hits":sum(r["reached_token_cap"] for r in subset)})
            selected=sorted({r["pair_id"] for r in records})[:3]
            qualitative.extend(r for r in records if r["pair_id"] in selected)
        csv_rows(output/"final_metrics.csv",rows)
        csv_rows(output/"inference_resources.csv",inference)
        atomic_json(output/"qualitative.json",{"selection_rule":"first three pair IDs in sorted order per suite",
                                               "records":qualitative})
        if root is not None:
            root=Path(root);images=read_json(root/"data-v1/images.json")
            panels=["<!doctype html><meta charset='utf-8'><title>Task 2 aligned examples</title>"]
            for suite,pair_id in sorted({(r["suite"],r["pair_id"]) for r in qualitative}):
                members=[r for r in qualitative if r["suite"]==suite and r["pair_id"]==pair_id]
                panels.append(f"<h2>{html.escape(suite+' '+pair_id)}</h2>")
                panels.append("<p>"+html.escape(members[0]["question"])+"</p>")
                for member_index in (0,1):
                    item=next(r for r in members if r["member_index"]==member_index)
                    image=root/images[str(item["original_image_id"])]["relative_path"]
                    if not image.exists():raise FileNotFoundError("Qualitative images are local-only; provide the original image root")
                    src=os.path.relpath(image,output).replace(os.sep,"/")
                    panels.append(f"<img width='300' src='{html.escape(src)}'><p>Reference: {html.escape(item['original_target'])}</p>")
                    for row in members:
                        if row["member_index"]==member_index:
                            panels.append("<p>"+html.escape(row["condition"]+": "+row["raw_response"])+"</p>")
            (output/"qualitative.html").write_text("\n".join(panels),encoding="utf-8")
    atomic_json(output/"artifacts.json",{"figures":generated,"source_run":str(run_dir) if run_dir else None,
                                         "source_evaluation":str(evaluation_dir) if evaluation_dir else None})
    return generated
