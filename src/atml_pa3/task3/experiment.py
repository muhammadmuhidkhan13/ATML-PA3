"""Frozen, resumable offline fixed-budget experiment; no adaptive/live inference."""
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
import csv
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from .data import DatasetSource,assert_source_identity,validate_mathvista_metadata
from .cache import load_semantics
from .equivalence import prefix_groups
from .preparation import BUDGETS
from .selection import Candidate,select,methods
from .scoring import base_scores
from .visconf import calibrate
from .statistics import METRICS,failure_decomposition,paired_bootstrap
from .diagnostics import question_diagnostics,illustration_candidates
from .io import atomic_json,read_json,sha256,identity,save_npz

def csv_rows(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)

def evaluate_question(rows:list[dict],semantics:dict,orders:list[list[int]])->tuple:
    """Selection receives only the current prefix's group labels/quality values."""
    vision='img_attn_total' in rows[0];base=[base_scores(r) for r in rows]
    fullfusion=calibrate([b['kl'] for b in base],[b['v'] for b in base]) if vision else None
    result=[];prefix_z=np.full((len(orders),len(BUDGETS),len(rows)),np.nan)
    prefix_zt=prefix_z.copy();quantiles=np.full((len(orders),len(BUDGETS),2),np.nan)
    for order_id,order in enumerate(orders):
        for budget_index,n in enumerate(BUDGETS):
            prefix=order[:n]
            groups=prefix_groups(prefix,semantics['equivalent'],semantics['missing'])
            candidates=[Candidate(i,g,base[i]['length']) for i,g in zip(prefix,groups)]
            quality={name:[base[i][name] for i in prefix] for name in ('mean_logp','self_certainty','vision')}
            fusion=None
            if vision:
                fusion=calibrate([base[i]['kl'] for i in prefix],[base[i]['v'] for i in prefix])
                quality.update(visconf=fusion['visconf'],tokenwise=fusion['tokenwise'])
                prefix_z[order_id,budget_index,:n]=fusion['visconf'];prefix_zt[order_id,budget_index,:n]=fusion['tokenwise']
                quantiles[order_id,budget_index]=[fusion['q05'],fusion['q95']]
            correct=[bool(semantics['correct'][i]) for i in prefix]
            for method,(rule,score) in methods(vision).items():
                chosen=select(candidates,rule,quality[score] if score else None)
                selected=prefix[chosen.index]
                result.append({'order':order_id,'budget':n,'method':method,'selected_rollout_id':selected,
                    'selected_answer_group':chosen.answer_group,'selected_extracted_answer':rows[selected]['extracted_answer'],
                    'selected_normalized_answer':str(semantics['normalized'][selected]),'score':chosen.score,
                    'consumed_rollouts':chosen.consumed,'token_cost':chosen.token_cost,'distinct_groups':len(set(groups)),
                    **failure_decomposition(correct,chosen.index)})
    diagnostic=question_diagnostics(base,semantics['correct'],fullfusion)
    return result,diagnostic,{'visconf':prefix_z,'tokenwise':prefix_zt,'quantiles':quantiles},base,fullfusion

def frozen_config(prepared,semantics,metadata):
    source=Path(__file__).parent
    return {'version':1,'steps':[1,2,3],'seed':6304,'budgets':list(BUDGETS),'orders':10,'bootstrap_resamples':1000,
            'bootstrap_unit':'question after ten-order mean','percentiles':'linear',
            'tie_break':'earliest supporting rollout in saved prefix; average tied RWC ranks',
            'quartile_ties':'stable ascending score then rollout_id; disjoint bottom8/top8; all-equal undefined',
            'token_correlations':'Pearson per rollout; mean valid rollouts within question; mean questions',
            'signal_correlations':'Spearman within question; undefined variance recorded',
            'invalid_metrics':'unaligned=>unavailable; K/logp mean finite aligned entries or -inf; invalid vision ratio0; invalid calibrated KL0',
            'illustrations':'retrospective; first lexicographic question per category; first canonical rollout(s); no configuration changes',
            'source_manifest':sha256(Path(prepared)/'source_manifest.json'),'semantics_manifest':sha256(Path(semantics)/'manifest.json'),
            'metadata':sha256(metadata),'scorer':read_json(Path(semantics)/'scorer_config.json'),
            'split_and_orders':{ds:{name:sha256(Path(prepared)/ds/name) for name in ('split.json','orders.json')} for ds in ('math','mathvista')},
            'code':{p.name:sha256(p) for p in sorted(source.glob('*.py'))},
            'packages':{name:version(name) for name in ('numpy','pyarrow','math-verify','sympy','matplotlib')}}

def run(root:Path,prepared:Path,semantics:Path,metadata:Path,output:Path,*,split:str='development',limit:int|None=None)->dict:
    root,prepared,semantics,output=map(Path,(root,prepared,semantics,output))
    if split not in ('development','evaluation'):raise ValueError('Unknown split')
    assert_source_identity(root,prepared)
    validate_mathvista_metadata(read_json(metadata))
    if not read_json(semantics/'manifest.json')['complete']:raise ValueError('Semantics cache incomplete')
    for name,digest in read_json(semantics/'manifest.json')['files'].items():
        if sha256(semantics/name)!=digest:raise ValueError('Corrupt equivalence cache')
    config=frozen_config(prepared,semantics,metadata);output.mkdir(parents=True,exist_ok=True)
    path=output/'frozen_config.json'
    if path.exists() and read_json(path)!=config:raise ValueError('Frozen experiment configuration changed; use another output')
    if not path.exists():atomic_json(path,config)
    config_id=identity(config)
    for dataset in ('math','mathvista'):
        source=DatasetSource(root,dataset);ids=read_json(prepared/dataset/'split.json')[split]
        if limit is not None:ids=ids[:limit]
        orders=read_json(prepared/dataset/'orders.json')['orders']
        directory=output/dataset/split;directory.mkdir(parents=True,exist_ok=True)
        manifest_path=directory/'manifest.json'
        state=read_json(manifest_path) if manifest_path.exists() else {'config_id':config_id,'question_ids':ids,'files':{},'complete':False}
        if state['config_id']!=config_id or state['question_ids']!=ids:raise ValueError('Output selection/configuration differs')
        for position,sid in enumerate(ids):
            stem=identity(sid);dest=directory/'questions'/stem
            names=[f'questions/{stem}{suffix}' for suffix in ('.parquet','.diagnostics.json','.scores.npz')]
            if all(name in state['files'] for name in names):
                if any(sha256(directory/name)!=state['files'][name] for name in names):raise ValueError('Corrupt completed question output')
                continue
            rows=source.question(sid);sem=load_semantics(semantics,dataset,sid)
            results,diagnostic,prefix_scores,base,fusion=evaluate_question(rows,sem,orders[sid])
            for row in results:row.update(dataset=dataset,sample_id=sid,split=split,config_id=config_id)
            dest.parent.mkdir(parents=True,exist_ok=True)
            temporary=dest.with_suffix('.parquet.partial');pq.write_table(pa.Table.from_pylist(results),temporary,compression='zstd')
            temporary.replace(dest.with_suffix('.parquet'))
            atomic_json(dest.with_suffix('.diagnostics.json'),{'sample_id':sid,**diagnostic})
            save_npz(dest.with_suffix('.scores.npz'),**prefix_scores,
                     base=np.array([[b[k] for k in ('length','mean_logp','self_certainty','vision')] for b in base]))
            for name in names:state['files'][name]=sha256(directory/name)
            atomic_json(manifest_path,state)
            if position%25==0:print(dataset,split,position+1,len(ids),flush=True)
        state['complete']=True;atomic_json(manifest_path,state)
        summarize(directory,dataset,split)
    return {'status':'complete','split':split,'official_inference_calls':0}

def summarize(directory,dataset,split):
    directory=Path(directory);manifest=read_json(directory/'manifest.json')
    if not manifest['complete']:raise ValueError('Incomplete experiment')
    names=list(methods(dataset=='mathvista'));ids=manifest['question_ids']
    means=np.empty((len(ids),len(BUDGETS),len(names),len(METRICS)))
    diag=[];associations=[];vision=[]
    for q,sid in enumerate(ids):
        stem=identity(sid)
        rows=pq.read_table(directory/'questions'/f'{stem}.parquet').to_pylist()
        if len(rows)!=10*len(BUDGETS)*len(names):raise ValueError('Missing order/budget/method outcomes')
        seen={(r['order'],r['budget'],r['method']) for r in rows}
        if len(seen)!=len(rows):raise ValueError('Duplicate outcome')
        for j,n in enumerate(BUDGETS):
            for m,name in enumerate(names):
                selected=[r for r in rows if r['budget']==n and r['method']==name]
                means[q,j,m]=[np.mean([r[k] for r in selected]) for k in METRICS]
        d=read_json(directory/'questions'/f'{stem}.diagnostics.json')
        for signal,values in d['quartiles'].items():diag.append({'sample_id':sid,'signal':signal,**{k:v for k,v in values.items() if not isinstance(v,list)}})
        for pair,value in d['spearman'].items():associations.append({'sample_id':sid,'signals':pair,'spearman':value})
        if 'question_mean_token_pearson' in d:vision.append({'sample_id':sid,'pearson':d['question_mean_token_pearson'],'undefined_rollouts':d['undefined_rollout_correlations']})
    aggregate=[];differences=[]
    for j,n in enumerate(BUDGETS):
        for m,name in enumerate(names):
            accuracy=means[:,j,m,0];interval=paired_bootstrap(accuracy,np.zeros(len(ids)))
            aggregate.append({'dataset':dataset,'split':split,'method':name,'budget':n,'questions':len(ids),
                              **{k:float(means[:,j,m,t].mean()) for t,k in enumerate(METRICS)},
                              'accuracy_ci95_low':interval['ci95'][0],'accuracy_ci95_high':interval['ci95'][1]})
            references=['FirstAns','SC']
            counterparts={'K_RWC':'HighestK','V_RWC':'HighestV','VisConf_RWC':'HighestVisConf',
                          'HighestTokenFusion':'HighestVisConf','TokenFusion_RWC':'VisConf_RWC'}
            if name in counterparts:references.append(counterparts[name])
            for reference in references:
                ci=paired_bootstrap(accuracy,means[:,j,names.index(reference),0])
                differences.append({'dataset':dataset,'split':split,'budget':n,'method':name,'reference':reference,
                                    'difference':ci['mean'],'ci95_low':ci['ci95'][0],'ci95_high':ci['ci95'][1]})
    save_npz(directory/'question_means.npz',values=means,question_ids=np.array(ids),methods=np.array(names),metrics=np.array(METRICS),budgets=np.array(BUDGETS))
    csv_rows(directory/'aggregate.csv',aggregate);csv_rows(directory/'paired_differences.csv',differences)
    csv_rows(directory/'quartiles.csv',diag);csv_rows(directory/'signal_associations.csv',associations);csv_rows(directory/'confidence_vision.csv',vision)
    atomic_json(directory/'aggregate.json',{'dataset':dataset,'split':split,'question_count':len(ids),'orders':10,
                                          'metrics':aggregate,'bootstrap_resamples':1000,'unit':'questions after order averaging',
                                          'cost_units':'retained generated tokens; offline computational proxy'})
