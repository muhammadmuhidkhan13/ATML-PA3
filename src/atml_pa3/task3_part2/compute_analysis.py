"""Post-freeze numeric compute-allocation diagnostics; never used for tuning."""
from pathlib import Path
from collections import defaultdict
import numpy as np
import pyarrow.parquet as pq
from atml_pa3.task3.io import read_json,identity,atomic_json
from atml_pa3.task3.experiment import csv_rows
from atml_pa3.task3.statistics import paired_bootstrap

def export(output):
    output=Path(output);comparisons=[];by_fraction=[];savings=[]
    for ds in ('math','mathvista'):
        d=output/ds/'evaluation';state=read_json(d/'manifest.json');table=read_json(d/'aggregate.json')['rows'];ref=next(r for r in table if r['policy']=='Fixed32')
        with np.load(d/'question_means.npz',allow_pickle=False) as f:means=f['values'];names=f['policies'].tolist();metrics=f['metrics'].tolist()
        for policy in ('agreement','quality'):
            p=next(r for r in table if r['policy']==policy)
            closest=min([r for r in table if r['family']=='fixed'],key=lambda r:(abs(r['token_cost']-p['token_cost']),r['fixed']))
            pi,fi=names.index(policy),names.index(closest['policy']);ai,ti=metrics.index('accuracy'),metrics.index('token_cost')
            comparisons.append({'dataset':ds,'policy':policy,'nearest_fixed_policy':closest['policy'],
                'adaptive_mean_tokens':p['token_cost'],'fixed_mean_tokens':closest['token_cost'],
                'adaptive_accuracy':p['accuracy'],'fixed_accuracy':closest['accuracy'],
                'accuracy_difference':paired_bootstrap(means[:,pi,ai],means[:,fi,ai]),
                'token_difference':paired_bootstrap(means[:,pi,ti],means[:,fi,ti]),
                'rule':'nearest held-out mean retained tokens, descriptive comparison only; no policy retuning'})
            indices=np.random.default_rng(6304).integers(len(means),size=(1000,len(means)))
            left,right=means[:,pi,ti],means[:,names.index('Fixed32'),ti]
            boot=1-left[indices].mean(1)/right[indices].mean(1)
            savings.append({'dataset':ds,'policy':policy,'relative_savings':1-left.mean()/right.mean(),
                'ci95_low':float(np.quantile(boot,.025)),'ci95_high':float(np.quantile(boot,.975)),
                'resamples':1000,'seed':6304,'unit':'paired questions after ten-order means'})
        bins=defaultdict(list)
        for sid in state['question_ids']:
            rows=pq.read_table(d/'questions'/f'{identity(sid)}.parquet').to_pylist();frac=rows[0]['full_pool_correct_fraction']
            label='0' if frac==0 else '(0,.25]' if frac<=.25 else '(.25,.5]' if frac<=.5 else '(.5,.75]' if frac<=.75 else '(.75,1]'
            for policy in ('agreement','quality'):
                rr=[r for r in rows if r['policy']==policy]
                bins[policy,label].append({m:float(np.mean([r[m] for r in rr])) for m in ('accuracy','token_cost','rollouts','leader_support','incorrect_early','recoverable_early')})
        for (policy,label),qq in sorted(bins.items()):by_fraction.append({'dataset':ds,'policy':policy,
            'full_pool_correct_fraction_bin':label,'questions':len(qq),**{m:float(np.mean([q[m] for q in qq])) for m in qq[0]}})
    atomic_json(output/'similar_cost_comparisons.json',comparisons)
    csv_rows(output/'similar_cost_comparisons.csv',comparisons);csv_rows(output/'relative_savings_intervals.csv',savings)
    csv_rows(output/'stopping_by_pool_correct_fraction.csv',by_fraction)
    return {'nearest_cost_comparisons':len(comparisons),'fraction_bins':len(by_fraction),'savings_intervals':len(savings)}
