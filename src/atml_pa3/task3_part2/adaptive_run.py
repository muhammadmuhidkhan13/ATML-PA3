"""Read-only reuse of Part 1, immutable dev selection, then held-out evaluation."""
from pathlib import Path
from importlib.metadata import version
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from atml_pa3.task3.io import read_json, atomic_json, sha256, identity, save_npz
from atml_pa3.task3.cache import load_semantics
from atml_pa3.task3.equivalence import prefix_groups
from atml_pa3.task3.selection import Candidate
from atml_pa3.task3.statistics import paired_bootstrap
from atml_pa3.task3.experiment import csv_rows
from atml_pa3.task3.data import assert_source_identity
from .adaptive import CHECKPOINTS, TAUS, GAMMAS, Policy, policies, evidence, decide, choose_policy, policy_dict, from_dict

FIXED = [Policy('fixed', fixed=n) for n in (4, 8, 16, 32)]
METRICS = ('accuracy', 'token_cost', 'rollouts', 'oracle', 'incorrect_early', 'recoverable_early',
           'correct_early', 'answer_changes_at32', 'new_groups_same_winner', 'unseen_correct', 'no_answer')

def verify_part1(project):
    project = Path(project); a = project / 'artifacts/task3'
    frozen = read_json(a / 'offline-v1/frozen_config.json')
    for name, digest in frozen['code'].items():
        if sha256(project / 'src/atml_pa3/task3' / name) != digest: raise ValueError('Part 1 source differs: ' + name)
    if sha256(a/'semantics-v2/manifest.json') != frozen['semantics_manifest']: raise ValueError('Semantic manifest differs')
    for name, digest in read_json(a/'semantics-v2/manifest.json')['files'].items():
        if sha256(a/'semantics-v2'/name) != digest: raise ValueError('Semantic record corrupt')
    for ds in ('math', 'mathvista'):
        for name, digest in frozen['split_and_orders'][ds].items():
            if sha256(a/'prepared-v1'/ds/name) != digest: raise ValueError('Prepared identity differs')
    assert_source_identity(project/'ATML-PA3-Data', a/'prepared-v1')
    return frozen

def specification(project):
    a = Path(project)/'artifacts/task3'
    return {'version': 'adaptive-v1', 'seed': 6304, 'checkpoints': list(CHECKPOINTS), 'fallback': 32,
            'tau_grid': list(TAUS), 'gamma_grid': list(GAMMAS), 'accuracy_tolerance': .01,
            'selection': 'minimum tokens eligible; higher accuracy, smaller tau, smaller gamma; fallback last on exact tie',
            'bootstrap': {'resamples': 1000, 'seed': 6304, 'unit': 'question after ten-order mean', 'percentiles': 'linear'},
            'part1_config_sha256': sha256(a/'offline-v1/frozen_config.json'),
            'source_manifest_sha256': sha256(a/'prepared-v1/source_manifest.json'),
            'semantics_manifest_sha256': sha256(a/'semantics-v2/manifest.json'),
            'scorer_sha256': sha256(a/'semantics-v2/scorer_config.json'),
            'splits_orders': {ds: {n: sha256(a/'prepared-v1'/ds/n) for n in ('split.json', 'orders.json')} for ds in ('math', 'mathvista')},
            'controller_code': {n: sha256(Path(__file__).with_name(n)) for n in ('adaptive.py','adaptive_run.py')},
            'packages': {n: version(n) for n in ('numpy','pyarrow','math-verify','sympy')}}

def question_snapshots(project, ds, split, sid, orders):
    a = Path(project)/'artifacts/task3'; d = a/'offline-v1'/ds/split
    manifest = read_json(d/'manifest.json'); stem = identity(sid)
    if not manifest['complete'] or manifest['config_id'] != identity(read_json(a/'offline-v1/frozen_config.json')): raise ValueError('Part 1 incomplete')
    for suffix in ('.parquet', '.scores.npz'):
        name = f'questions/{stem}{suffix}'
        if sha256(d/name) != manifest['files'][name]: raise ValueError('Part 1 output corruption')
    sem = load_semantics(a/'semantics-v2', ds, sid)
    with np.load(d/'questions'/f'{stem}.scores.npz', allow_pickle=False) as f:
        base, z = f['base'], f['visconf']
    fixed = {(r['order'],r['budget']): r for r in pq.read_table(d/'questions'/f'{stem}.parquet').to_pylist() if r['method']=='SC'}
    all_snapshots = []
    for oi, order in enumerate(orders):
        snapshots = {}
        for n in (4,8,16,32):
            prefix = order[:n]; groups = prefix_groups(prefix,sem['equivalent'],sem['missing'])
            quality = base[prefix,2] if ds=='math' else z[oi,(1,2,4,8,16,32).index(n),:n]
            if np.isnan(quality).any(): raise ValueError('Missing prefix quality')
            candidates = [Candidate(rid,g,int(base[rid,0])) for rid,g in zip(prefix,groups)]
            s = evidence(candidates,quality)
            old = fixed[oi,n]
            if (s['selected_rollout_id'],s['token_cost'],s['answer_group']) != (old['selected_rollout_id'],old['token_cost'],old['selected_answer_group']): raise ValueError('SC reuse differs')
            snapshots[n] = s
        all_snapshots.append(snapshots)
    return all_snapshots, sem

def retrospective(chosen, full, sem):
    rid = chosen['selected_rollout_id']; n=chosen['n']; early=n<32
    unseen = set(full['consumed_ids'])-set(chosen['consumed_ids'])
    correct = bool(sem['correct'][rid]); full_correct=bool(sem['correct'][full['selected_rollout_id']])
    changed = chosen['answer_group'] != full['answer_group']
    return {'accuracy': int(correct), 'rollouts':n,'oracle':int(any(sem['correct'][chosen['consumed_ids']])),
            'incorrect_early':int(early and not correct),'recoverable_early':int(early and not correct and full_correct),
            'correct_early':int(early and correct),'answer_changes_at32':int(early and changed),
            'new_groups_same_winner':int(early and not changed and len(full['votes'])>len(chosen['votes'])),
            'unseen_correct':int(any(sem['correct'][list(unseen)])) if unseen else 0,
            'full32_correct':full_correct,'full_pool_correct_fraction':float(sem['correct'].mean()),
            'selected_normalized_answer':str(sem['normalized'][rid]),'full32_group':full['answer_group']}

def run_split(project, output, split, selected=None):
    a=Path(project)/'artifacts/task3'; output=Path(output)
    spec=read_json(output/'specification.json'); results={}
    for ds in ('math','mathvista'):
        ids=read_json(a/'prepared-v1'/ds/'split.json')[split]
        orders=read_json(a/'prepared-v1'/ds/'orders.json')['orders']
        active=policies()+FIXED if split=='development' else [from_dict(selected[ds][f]) for f in ('agreement','quality')]+FIXED
        # Family labels remain distinct even if both selected policies are fallback.
        labels=[p.id for p in active] if split=='development' else ['agreement','quality']+[p.id for p in FIXED]
        directory=output/ds/split; directory.mkdir(parents=True,exist_ok=True)
        mp=directory/'manifest.json'
        config_id=identity(read_json(output/'frozen_config.json')) if split=='evaluation' else identity(spec)
        initial={'config_id':config_id,'question_ids':ids,'policy_labels':labels,'files':{},'complete':False}
        state=read_json(mp) if mp.exists() else initial
        if any(state[k]!=initial[k] for k in ('config_id','question_ids','policy_labels')): raise ValueError('Incompatible adaptive resume')
        for qi,sid in enumerate(ids):
            name=f'questions/{identity(sid)}.parquet'; dest=directory/name
            if name in state['files']:
                if sha256(dest)!=state['files'][name]:raise ValueError('Corrupt adaptive output')
                continue
            snapshots,sem=question_snapshots(project,ds,split,sid,orders[sid]); rows=[]
            for oi,ss in enumerate(snapshots):
                for label,p in zip(labels,active):
                    s=decide(p,ss)
                    rows.append({'dataset':ds,'sample_id':sid,'order':oi,'policy':label,**policy_dict(p),
                        **s,**retrospective(s,ss[32],sem),
                        'checkpoint_trace':[ss[n] for n in CHECKPOINTS if n<=s['n']], 'config_id':config_id})
            dest.parent.mkdir(parents=True,exist_ok=True); temp=dest.with_suffix('.partial')
            pq.write_table(pa.Table.from_pylist(rows),temp,compression='zstd');temp.replace(dest)
            state['files'][name]=sha256(dest);atomic_json(mp,state)
            if qi%100==0:print(ds,split,qi+1,len(ids),flush=True)
        state['complete']=True;atomic_json(mp,state)
        results[ds]=summarize(directory,active,labels)
    return results

def summarize(directory,active,labels):
    state=read_json(directory/'manifest.json'); ids=state['question_ids']
    metrics=list(METRICS)+[f'stop{n}' for n in (4,8,16,32)]
    means=np.zeros((len(ids),len(labels),len(metrics)))
    for qi,sid in enumerate(ids):
        rows=pq.read_table(directory/'questions'/f'{identity(sid)}.parquet').to_pylist()
        if len(rows)!=10*len(labels) or len({(r['order'],r['policy']) for r in rows})!=len(rows):raise ValueError('Outcome coverage differs')
        for pi,label in enumerate(labels):
            rr=[r for r in rows if r['policy']==label]
            if {r['order'] for r in rr}!=set(range(10)): raise ValueError('Order coverage differs')
            means[qi,pi]=[np.mean([r[m] if m in METRICS else int(r['n']==int(m[4:])) for r in rr]) for m in metrics]
    table=[]; differences=[]; ref=labels.index('Fixed32')
    for pi,(label,p) in enumerate(zip(labels,active)):
        row={'policy':label,**policy_dict(p),'questions':len(ids),**dict(zip(metrics,means[:,pi].mean(0).tolist()))}
        for metric in ('accuracy','token_cost','rollouts'):
            ix=metrics.index(metric);ci=paired_bootstrap(means[:,pi,ix],np.zeros(len(ids)))
            row[metric+'_ci_low'],row[metric+'_ci_high']=ci['ci95']
            diff=paired_bootstrap(means[:,pi,ix],means[:,ref,ix])
            differences.append({'policy':label,'reference':'Fixed32','metric':metric,**diff})
        row['relative_token_savings']=1-row['token_cost']/means[:,ref,metrics.index('token_cost')].mean()
        table.append(row)
    if 'agreement' in labels:
        for metric in ('accuracy','token_cost','rollouts'):
            ix=metrics.index(metric)
            for left in ('agreement','quality'):
                for right in labels:
                    if left==right:continue
                    differences.append({'policy':left,'reference':right,'metric':metric,
                        **paired_bootstrap(means[:,labels.index(left),ix],means[:,labels.index(right),ix])})
    csv_rows(directory/'aggregate.csv',table);csv_rows(directory/'paired_differences.csv',
        [{**r,'ci95_low':r['ci95'][0],'ci95_high':r['ci95'][1]} for r in differences])
    atomic_json(directory/'aggregate.json',{'rows':table,'differences':differences,'unit':'question after ten-order mean',
                                          'cost_units':'retained generated tokens; offline proxy'})
    save_npz(directory/'question_means.npz',values=means,question_ids=np.array(ids),policies=np.array(labels),metrics=np.array(metrics))
    return table

def run(project,output,stage):
    project,output=Path(project),Path(output);verify_part1(project)
    spec=specification(project);output.mkdir(parents=True,exist_ok=True)
    sp=output/'specification.json'
    if sp.exists() and read_json(sp)!=spec:raise ValueError('Specification changed; use new experiment version')
    if not sp.exists():atomic_json(sp,spec)
    if stage=='development':
        if (output/'frozen_config.json').exists():raise ValueError('Development already frozen; do not reselect')
        grids=run_split(project,output,'development');selected={}; references={}
        for ds,grid in grids.items():
            reference=next(r for r in grid if r['policy']=='Fixed32')['accuracy']; references[ds]=reference
            selected[ds]={f:choose_policy(grid,f,reference) for f in ('agreement','quality')}
        config={'specification':spec,'selected':selected,'fixed32_development_accuracy':references,'development_grid':grids,
                'development_manifests':{ds:sha256(output/ds/'development/manifest.json') for ds in grids}}
        atomic_json(output/'frozen_config.json',config)
        atomic_json(output/'FROZEN.json',{'sha256':sha256(output/'frozen_config.json'),'heldout_executed':False})
        return selected
    frozen=read_json(output/'FROZEN.json')
    if frozen['sha256']!=sha256(output/'frozen_config.json'):raise ValueError('Frozen configuration corrupt')
    config=read_json(output/'frozen_config.json')
    if config['specification']!=spec:raise ValueError('Frozen configuration incompatible')
    for ds,digest in config['development_manifests'].items():
        if sha256(output/ds/'development/manifest.json')!=digest:raise ValueError('Development manifest changed')
    result=run_split(project,output,'evaluation',config['selected'])
    atomic_json(output/'EVALUATED.json',{'frozen_config_sha256':frozen['sha256'],
        'manifests':{ds:sha256(output/ds/'evaluation/manifest.json') for ds in result}})
    return result
