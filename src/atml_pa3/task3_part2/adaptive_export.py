"""Saved-result plots, fixed offline comparisons and deterministic evidence cases."""
from pathlib import Path
import csv
import numpy as np
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from atml_pa3.task3.io import read_json,atomic_json,identity,sha256
from atml_pa3.task3.experiment import csv_rows
from atml_pa3.task3.data import DatasetSource

def export(project,output):
    project,output=Path(project),Path(output)
    if not (output/'EVALUATED.json').exists():raise ValueError('Held-out experiment incomplete')
    figs=output/'figures'
    if figs.exists():raise FileExistsError('Preserve completed figures; use a new export destination/version')
    figs.mkdir();figures=[];stops=[];comparisons=[]
    for ds in ('math','mathvista'):
        dev=read_json(output/ds/'development/aggregate.json')['rows']
        table=read_json(output/ds/'evaluation/aggregate.json');rr=table['rows'];by={r['policy']:r for r in rr}
        fixed=by['Fixed32']
        for r in rr:
            for n in (4,8,16,32):stops.append({'dataset':ds,'policy':r['policy'],'checkpoint':n,
                'fraction':r[f'stop{n}'],'order_events':int(round(r[f'stop{n}']*8000)),
                'denominator_question_order_events':8000,'question_averaging':'ten orders then 800 questions'})
            comparisons.append({'dataset':ds,'policy':r['policy'],'accuracy':r['accuracy'],
                'accuracy_difference_pp':100*(r['accuracy']-fixed['accuracy']),
                'mean_retained_generated_tokens':r['token_cost'],'relative_token_savings':r['relative_token_savings'],
                'mean_rollouts':r['rollouts'],'incorrect_early_frequency':r['incorrect_early'],
                'fixed32_recovery_frequency':r['recoverable_early']})
        def save(name,title,xlabel,ylabel):
            plt.title(ds.upper()+': '+title);plt.xlabel(xlabel);plt.ylabel(ylabel);plt.grid(alpha=.2)
            plt.tight_layout();path=figs/f'{ds}_{name}.png';plt.savefig(path,dpi=160);plt.close();figures.append(path.name)
        plt.figure(figsize=(7,4))
        for family,marker in [('agreement','o'),('quality','x'),('always32','s')]:
            x=[r for r in dev if r['family']==family];plt.scatter([r['token_cost'] for r in x],[r['accuracy'] for r in x],label=family,marker=marker)
        plt.axhline(next(r for r in dev if r['policy']=='Fixed32')['accuracy']-.01,color='gray',linestyle='--',label='eligibility boundary')
        selected=read_json(output/'frozen_config.json')['selected'][ds]
        for family,r in selected.items():plt.annotate('selected '+family,(r['token_cost'],r['accuracy']),xytext=(3,5),textcoords='offset points',fontsize=8)
        plt.legend(fontsize=8);save('development_grid','development threshold grid','Mean retained generated tokens (offline proxy)','Answer accuracy (fraction)')
        plt.figure(figsize=(7,4));fr=[by[f'Fixed{n}'] for n in (4,8,16,32)]
        plt.plot([r['token_cost'] for r in fr],[r['accuracy'] for r in fr],'o-',label='fixed SC')
        for name in ('agreement','quality'):
            r=by[name];plt.errorbar(r['token_cost'],r['accuracy'],yerr=[[r['accuracy']-r['accuracy_ci_low']],[r['accuracy_ci_high']-r['accuracy']]],fmt='s',capsize=3,label=name)
        plt.legend();save('heldout_accuracy_cost','held-out adaptive vs fixed SC','Mean retained generated tokens (offline proxy)','Answer accuracy (fraction)')
        names=[r['policy'] for r in rr];xx=np.arange(len(names))
        plt.figure(figsize=(8,4));bottom=np.zeros(len(rr))
        for n in (4,8,16,32):
            y=np.array([r[f'stop{n}'] for r in rr]);plt.bar(xx,y,bottom=bottom,label=str(n));bottom+=y
        plt.xticks(xx,names,rotation=20);plt.legend(title='Consumed');save('stopping_distribution','stopping distribution','Policy','Fraction of question-order outcomes')
        for metric,title,unit in [('rollouts','mean rollouts consumed','Rollouts'),('incorrect_early','incorrect early stops','Fraction of question-order outcomes'),
                                  ('recoverable_early','early errors recovered by fixed-32 SC','Fraction of question-order outcomes')]:
            plt.figure(figsize=(8,4));plt.bar(xx,[r[metric] for r in rr]);plt.xticks(xx,names,rotation=20);save(metric,title,'Policy',unit)
        for metric,title,scale in [('accuracy','accuracy difference vs fixed-32 SC',100),('token_cost','token cost difference vs fixed-32 SC',1)]:
            plt.figure(figsize=(8,4));diff=[next(d for d in table['differences'] if d['policy']==name and d['reference']=='Fixed32' and d['metric']==metric) for name in names]
            y=[d['mean']*scale for d in diff];lo=[(d['mean']-d['ci95'][0])*scale for d in diff];hi=[(d['ci95'][1]-d['mean'])*scale for d in diff]
            plt.errorbar(xx,y,yerr=[lo,hi],fmt='o',capsize=3);plt.axhline(0,color='gray');plt.xticks(xx,names,rotation=20)
            save(metric+'_difference',title,'Policy','Percentage points' if scale==100 else 'Retained generated tokens (offline proxy)')
        plt.figure(figsize=(8,4));plt.bar(xx,[100*r['relative_token_savings'] for r in rr]);plt.xticks(xx,names,rotation=20)
        save('token_savings','token savings vs fixed-32 SC','Policy','Relative token savings (%)')
        plt.figure(figsize=(7,4))
        for name in ('agreement','quality'):
            r=by[name];plt.scatter(r['token_cost'],r['accuracy'],s=70,label=f"{name}; {r['rollouts']:.2f} rollouts")
        plt.legend();save('controller_comparison','development-selected controllers','Mean retained generated tokens (offline proxy)','Answer accuracy (fraction)')
        offline=read_json(project/'artifacts/task3/offline-v1'/ds/'evaluation/aggregate.json')['metrics']
        # Reuse tables read-only; never regenerate completed Part 1 figures.
        for mode in ('fixed_count','oracle_gap','integrated_cost'):
            plt.figure(figsize=(8,4))
            for method in ('FirstAns','SC','HighestK','K_RWC')+ (('HighestVisConf','VisConf_RWC') if ds=='mathvista' else ()):
                ms=sorted([r for r in offline if r['method']==method],key=lambda r:r['budget'])
                x=[r['budget'] if mode=='fixed_count' else r['oracle'] if mode=='oracle_gap' else r['token_cost'] for r in ms]
                plt.plot(x,[r['accuracy'] for r in ms],'o-',label=method,markersize=3)
            if mode=='integrated_cost':
                for name in ('agreement','quality'):plt.scatter(by[name]['token_cost'],by[name]['accuracy'],marker='s',label=name)
            plt.legend(fontsize=7);save(mode,mode.replace('_',' '),'Candidate count' if mode=='fixed_count' else 'Oracle coverage (fraction)' if mode=='oracle_gap' else 'Mean retained generated tokens (offline proxy)','Answer accuracy (fraction)')
    csv_rows(output/'stopping_counts.csv',stops);csv_rows(output/'heldout_comparison.csv',comparisons)
    examples=adaptive_examples(project,output)
    atomic_json(output/'exports.json',{'figures':figures,'live_artifacts':'PENDING; no real generation executed',
        'example_count':len(examples),'sources':{ds:sha256(output/ds/'evaluation/manifest.json') for ds in ('math','mathvista')}})
    return {'figures':len(figures),'live':'pending'}

def adaptive_examples(project,output):
    """First lexicographic question then order then controller; savings >=50%."""
    examples={};a=Path(project)/'artifacts/task3'
    for ds in ('math','mathvista'):
        source=None;directory=output/ds/'evaluation'
        for sid in sorted(read_json(directory/'manifest.json')['question_ids']):
            rows=pq.read_table(directory/'questions'/f'{identity(sid)}.parquet').to_pylist()
            for r in sorted(rows,key=lambda r:(r['order'],r['policy'])):
                if r['policy'] not in ('agreement','quality'):continue
                fixed=next(x for x in rows if x['order']==r['order'] and x['policy']=='Fixed32')
                flags={'correct_saves_half':r['correct_early'] and r['token_cost']<=.5*fixed['token_cost'],
                       'incorrect_fixed32_recovers':r['recoverable_early']}
                for category,flag in flags.items():
                    key=ds+':'+category
                    if flag and key not in examples:
                        if source is None:source=DatasetSource(Path(project)/'ATML-PA3-Data',ds)
                        raw=source.question(sid);orders=read_json(a/'prepared-v1'/ds/'orders.json')['orders'][sid]
                        examples[key]={'dataset':ds,'sample_id':sid,'question':raw[0]['question_text'],'gold':raw[0]['gold_answer'],
                            'candidate_order':orders[r['order']],'decision':r,'fixed32':fixed,'reason':category,
                            'candidates':[{k:raw[i][k] for k in ('rollout_id','generated_text','extracted_answer','n_generated_text_tokens')} for i in orders[r['order']]]}
            if all(ds+':'+c in examples for c in ('correct_saves_half','incorrect_fixed32_recovers')):break
    atomic_json(output/'examples.json',{'rule':'first lexicographic question, then order, then controller; correct savings >=50%',
        'examples':examples,'part1_selection_examples':'../offline-v1/illustrations_evaluation.json',
        'live_revision_examples':'PENDING; requires real live records'})
    return examples
