"""Read-only verification of fixed-budget outcome counts, ranges and identities."""
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from .io import read_json,sha256,identity
from .preparation import BUDGETS
from .selection import methods

def validate_run(output,*,split='evaluation',expected_questions=800):
    output=Path(output);report={};config=read_json(output/'frozen_config.json');config_id=identity(config)
    for dataset in ('math','mathvista'):
        directory=output/dataset/split;manifest=read_json(directory/'manifest.json')
        if not manifest['complete'] or manifest['config_id']!=config_id:raise ValueError('Incomplete or incompatible run')
        ids=manifest['question_ids'];names=list(methods(dataset=='mathvista'))
        if len(ids)!=expected_questions:raise ValueError('Question count differs')
        count=0
        for sid in ids:
            stem=identity(sid)
            for suffix in ('.parquet','.diagnostics.json','.scores.npz'):
                name=f'questions/{stem}{suffix}'
                if sha256(directory/name)!=manifest['files'][name]:raise ValueError('Question output integrity failure')
            rows=pq.read_table(directory/'questions'/f'{stem}.parquet').to_pylist()
            expected={(order,n,method) for order in range(10) for n in BUDGETS for method in names}
            if {(r['order'],r['budget'],r['method']) for r in rows}!=expected or len(rows)!=len(expected):raise ValueError('Outcome coverage differs')
            for row in rows:
                if row['sample_id']!=sid or row['config_id']!=config_id:raise ValueError('Wrong question/config identity')
                if row['accuracy']>row['oracle'] or not 0<=row['accuracy']<=1:raise ValueError('Accuracy/oracle range')
                if row['sampling_failure']+row['selection_failure']+row['selection_success']!=1:raise ValueError('Failure decomposition differs')
                if row['consumed_rollouts']!=(1 if row['method']=='FirstAns' else row['budget']):raise ValueError('Rollout cost differs')
                if not 0<=row['token_cost']<=2048*row['consumed_rollouts']:raise ValueError('Generated token cost outside cap')
            for order in range(10):
                first=[r for r in rows if r['order']==order and r['method']=='FirstAns']
                first.sort(key=lambda r:r['budget'])
                if any(a['oracle']>b['oracle'] for a,b in zip(first,first[1:])):raise ValueError('Oracle non-monotonic')
                if len({(r['selected_rollout_id'],r['accuracy'],r['token_cost']) for r in first})!=1:raise ValueError('FirstAns depends on N')
                one=[r for r in rows if r['order']==order and r['budget']==1]
                if len({(r['selected_rollout_id'],r['accuracy']) for r in one})!=1:raise ValueError('N=1 selectors differ')
            count+=len(rows)
        with np.load(directory/'question_means.npz',allow_pickle=False) as f:
            values=f['values']
            if values.shape!=(expected_questions,6,len(names),7):raise ValueError('Question-mean array shape differs')
            if not np.isfinite(values).all():raise ValueError('Nonfinite aggregate outcomes')
        report[dataset]={'questions':len(ids),'orders':10,'budgets':6,'methods_including_controls':len(names),'outcomes':count,'status':'passed'}
    return report
