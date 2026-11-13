"""Independent arithmetic review of ALL saved adaptive outcomes, read-only."""
from pathlib import Path
from collections import Counter
import numpy as np
import pyarrow.parquet as pq
from atml_pa3.task3.io import read_json,sha256,identity,atomic_json
from atml_pa3.task3.cache import load_semantics
from atml_pa3.task3.equivalence import prefix_groups,NO_ANSWER
from atml_pa3.task3.statistics import paired_bootstrap
from .adaptive_run import verify_part1,specification

def independent_snapshot(prefix,groups,quality,lengths):
    votes=Counter(groups);leader=max(votes,key=votes.get)
    # Independent O(n^2) average-rank arithmetic, not the production RWC function.
    ranks=[1+sum(x>q for x in quality)+(sum(x==q for x in quality)-1)/2 for q in quality]
    weights=[2*(len(prefix)-r) for r in ranks];weighted=dict.fromkeys(groups,0.)
    for g,w in zip(groups,weights):weighted[g]+=w
    winner=max(weighted,key=weighted.get);ranked=sorted(weighted.values(),reverse=True)
    return {'group':leader,'rid':prefix[groups.index(leader)],'support':votes[leader]/len(prefix),
        'unique':sum(v==max(votes.values()) for v in votes.values())==1,'votes':dict(votes),
        'agrees':winner==leader,'margin':(ranked[0]-(ranked[1] if len(ranked)>1 else 0))/sum(ranked),
        'token_cost':int(sum(lengths[i] for i in prefix))}

def validate(project,output):
    project,output=Path(project),Path(output);a=project/'artifacts/task3';verify_part1(project)
    config=read_json(output/'frozen_config.json');spec=specification(project)
    if config['specification']!=spec or read_json(output/'specification.json')!=spec:raise ValueError('Frozen code/specification differs')
    if sha256(output/'frozen_config.json')!=read_json(output/'FROZEN.json')['sha256']:raise ValueError('Freeze hash differs')
    report={};total=0
    for ds in ('math','mathvista'):
        grid=config['development_grid'][ds];ref=config['fixed32_development_accuracy'][ds]
        for family in ('agreement','quality'):
            eligible=[r for r in grid if r['family'] in (family,'always32') and r['accuracy']+1e-12>=ref-.01]
            best=sorted(eligible,key=lambda r:(r['token_cost'],-r['accuracy'],r['tau'] if r['tau'] is not None else np.inf,r['gamma'] if r['gamma'] is not None else np.inf))[0]
            if best!=config['selected'][ds][family]:raise ValueError('Development selection differs')
        orders=read_json(a/'prepared-v1'/ds/'orders.json')['orders'];splits=read_json(a/'prepared-v1'/ds/'split.json')
        for split in ('development','evaluation'):
            d=output/ds/split;state=read_json(d/'manifest.json');ids=state['question_ids'];count=0
            if ids!=splits[split] or not state['complete']:raise ValueError('Wrong split/complete status')
            with np.load(d/'question_means.npz',allow_pickle=False) as f:means=f['values'];labels=f['policies'].tolist();metrics=f['metrics'].tolist()
            for qi,sid in enumerate(ids):
                name=f'questions/{identity(sid)}.parquet'
                if sha256(d/name)!=state['files'][name]:raise ValueError('Adaptive question hash differs')
                rows=pq.read_table(d/name).to_pylist();expected={(o,p) for o in range(10) for p in labels}
                if len(rows)!=len(expected) or {(r['order'],r['policy']) for r in rows}!=expected:raise ValueError('Outcome coverage differs')
                sem=load_semantics(a/'semantics-v2',ds,sid)
                with np.load(a/'offline-v1'/ds/split/'questions'/f'{identity(sid)}.scores.npz',allow_pickle=False) as f:base=f['base'];z=f['visconf']
                snapshots={}
                for oi,order in enumerate(orders[sid]):
                    for n in (4,8,16,32):
                        prefix=order[:n];groups=prefix_groups(prefix,sem['equivalent'],sem['missing'])
                        q=base[prefix,2] if ds=='math' else z[oi,(1,2,4,8,16,32).index(n),:n]
                        snapshots[oi,n]=independent_snapshot(prefix,groups,q,base[:,0])
                for r in rows:
                    oi,n=r['order'],r['n'];s=snapshots[oi,n];order=orders[sid][oi]
                    if r['family']=='fixed':wanted=r['fixed']
                    else:
                        wanted=32
                        for checkpoint in (4,8,16):
                            e=snapshots[oi,checkpoint]
                            agreement=e['unique'] and e['group']!=NO_ANSWER and r['tau'] is not None and e['support']>=r['tau']
                            passed=agreement if r['family']=='agreement' else agreement and e['agrees'] and e['margin']>=r['gamma'] if r['family']=='quality' else False
                            if passed:wanted=checkpoint;break
                    if wanted!=n:raise ValueError('Wrong stopping checkpoint')
                    if r['selected_rollout_id']!=s['rid'] or r['answer_group']!=s['group'] or r['token_cost']!=s['token_cost']:raise ValueError('Selection/cost differs')
                    if r['supporting_ids']!=[i for i,g in zip(order[:n],prefix_groups(order[:n],sem['equivalent'],sem['missing'])) if g==s['group']]:raise ValueError('Support IDs differ')
                    if r['consumed_ids']!=order[:n] or r['votes']!={k:v for k,v in s['votes'].items() if v is not None}:
                        # Arrow structs union all possible group fields, filling absent keys with None.
                        if r['consumed_ids']!=order[:n] or {k:v for k,v in r['votes'].items() if v is not None}!=s['votes']:raise ValueError('Prefix/votes differ')
                    if not np.isclose(r['rwc_margin'],s['margin']) or r['rwc_agrees']!=s['agrees']:raise ValueError('RWC evidence differs')
                    full=snapshots[oi,32];correct=bool(sem['correct'][s['rid']]);fullcorrect=bool(sem['correct'][full['rid']])
                    checks={'accuracy':int(correct),'oracle':int(any(sem['correct'][order[:n]])),
                        'incorrect_early':int(n<32 and not correct),'recoverable_early':int(n<32 and not correct and fullcorrect),
                        'correct_early':int(n<32 and correct),'answer_changes_at32':int(n<32 and s['group']!=full['group']),
                        'new_groups_same_winner':int(n<32 and s['group']==full['group'] and len(full['votes'])>len(s['votes'])),
                        'unseen_correct':int(any(sem['correct'][order[n:]])),'no_answer':s['group']==NO_ANSWER}
                    if any(r[k]!=v for k,v in checks.items()):raise ValueError('Retrospective metric differs')
                for pi,label in enumerate(labels):
                    rr=[r for r in rows if r['policy']==label]
                    expected=[np.mean([r[m] if m in r else int(r['n']==int(m[4:])) for r in rr]) for m in metrics]
                    if not np.allclose(means[qi,pi],expected):raise ValueError('Question/order means differ')
                count+=len(rows)
                if qi%200==0:print('audit',ds,split,qi+1,len(ids),flush=True)
            aggregate=read_json(d/'aggregate.json')
            for pi,row in enumerate(aggregate['rows']):
                if not np.allclose([row[m] for m in metrics],means[:,pi].mean(0)):raise ValueError('Aggregate differs')
            # Separate NumPy reconstruction of a main paired interval; shared question indices.
            left=labels.index('agreement') if split=='evaluation' else 0;right=labels.index('Fixed32')
            delta=means[:,left,metrics.index('accuracy')]-means[:,right,metrics.index('accuracy')]
            indices=np.random.default_rng(6304).integers(len(ids),size=(1000,len(ids)))
            ci=np.quantile(delta[indices].mean(1),[.025,.975],method='linear')
            stored=next(x for x in aggregate['differences'] if x['policy']==labels[left] and x['reference']=='Fixed32' and x['metric']=='accuracy')
            if not np.allclose(ci,stored['ci95']):raise ValueError('Bootstrap differs')
            report[ds+':'+split]={'questions':len(ids),'outcomes':count,'all_raw_decisions_independently_checked':True}
            total+=count
    atomic_json(output/'validation.json',{'status':'PASSED','collections':report,'total_outcomes':total,
        'checks':['freeze identity','dev-only selection','every stopping decision','independent ranks/margins','prefix token costs',
                  'support and consumed IDs','retrospective correctness','question means','aggregate means','paired bootstrap'],
        'official_live_calls':0})
    return report
