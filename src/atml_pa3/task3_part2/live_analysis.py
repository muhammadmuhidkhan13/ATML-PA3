"""Real saved-call grading, attributed costs and paired 50-question comparisons."""
from pathlib import Path
import numpy as np
from atml_pa3.task3.io import atomic_json,identity,sha256
from .live_store import read_json
from atml_pa3.task3.equivalence import prefix_groups,NO_ANSWER
from atml_pa3.task3.selection import Candidate,select
from atml_pa3.task3.statistics import paired_bootstrap
from atml_pa3.task3.experiment import csv_rows
from .live_spec import LIMITS,STRATEGIES

def truncated(record):
    return record['hit_cap'] and not record['ended_with_eos']

def reserved_peak(records):
    peaks=[r.get('peak_reserved_bytes') for r in records]
    return max(peaks) if all(p is not None for p in peaks) else None

def strategy_results(question,records,sem):
    """Selection uses the gold-free relation; correctness is read afterwards."""
    calls=list(LIMITS);results=[]
    initial=calls.index('parallel0');revision=calls.index('revision')
    for name,names in STRATEGIES.items():
        indices=[calls.index(c) for c in names]
        groups=prefix_groups(indices,sem['equivalent'],sem['missing'])
        cs=[Candidate(i,g,records[c]['output_tokens']) for i,g,c in zip(indices,groups,names)]
        chosen=select(cs,'consistency') if name=='parallel' else select(cs[-1:],'first')
        selected=cs[chosen.index].rollout_id if name=='parallel' else indices[-1]
        rr=[records[c] for c in names];final=records[calls[selected]]
        result={'sample_id':question['sample_id'],'strategy':name,'selected_call':calls[selected],
            'selected_answer_group':chosen.answer_group,'selected_extraction':final['extracted_answer'],
            'correct':bool(sem['correct'][selected]),'extraction_failed':final['extracted_answer'] is None,
            'selected_hit_cap':final['hit_cap'],'selected_truncated':truncated(final),'attributed_call_ids':names,
            'call_record_hashes':{c:identity(records[c]) for c in names},
            'output_tokens':sum(r['output_tokens'] for r in rr),'input_tokens':sum(r['input_tokens'] for r in rr),
            'inference_seconds':sum(r['generation_seconds'] for r in rr),
            'peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in rr),
            'peak_reserved_bytes':reserved_peak(rr),
            'call_extraction_failures':sum(r['extracted_answer'] is None for r in rr),
            'call_truncations':sum(truncated(r) for r in rr),'unique_calls_attributed':len(rr),
            'nominal_output_allowance':3072,'answers':[r['extracted_answer'] for r in rr],'answer_groups':groups}
        if name=='refinement':
            a,b=bool(sem['correct'][initial]),bool(sem['correct'][revision])
            result.update(initial_correct=a,transition=('correct' if a else 'wrong')+'_to_'+('correct' if b else 'wrong'),
                initial_missing=bool(sem['missing'][initial]),revised_missing=bool(sem['missing'][revision]),
                answer_changed=not bool(sem['equivalent'][initial][revision]),
                revision_hit_cap=final['hit_cap'],revision_truncated=truncated(final))
        results.append(result)
    return results

def grade_source(config_id,question,records):
    return {'sample_id':question['sample_id'],'call_order':list(LIMITS),
            'calls':{c:identity(records[c]) for c in LIMITS},'gold':question['gold_answer'],'config_id':config_id}

def grade_calls(store,questions,records):
    from atml_pa3.task3.cache import EquivalenceWorker
    worker=None;results=[];grading_files={}
    try:
        for q in questions:
            rr={c:records[q['sample_id'],c] for c in LIMITS}
            source=grade_source(store.config_id,q,rr)
            path=store.root/'grading'/f'{identity(q["sample_id"])}.json'
            if path.exists():
                cached=read_json(path)
                if cached['source']!=source or cached['sha256']!=identity(cached['semantics']):raise ValueError('Corrupt/incompatible live grades')
                sem=cached['semantics']
            else:
                if worker is None:worker=EquivalenceWorker(store.root/'live_grading.log')
                sem=worker.compute(dataset='math',answers=[rr[c]['extracted_answer'] for c in LIMITS],gold=q['gold_answer'],metadata=None)
                atomic_json(path,{'source':source,'semantics':sem,'sha256':identity(sem)})
            grading_files[path.name]=sha256(path)
            results.extend(strategy_results(q,rr,sem))
    finally:
        if worker is not None:worker.close()
    atomic_json(store.root/'strategy_results.json',{'rows':results,'grading_files':grading_files,
        'config_id':store.config_id,'kind':'real controlled FP16 generation; not instructor offline rollouts'})

def validate_strategy_results(questions,rows,records):
    """Check attributed call accounting alone; official completion also requires grades."""
    expected={(q['sample_id'],s) for q in questions for s in STRATEGIES}
    if len(rows)!=len(expected) or {(r['sample_id'],r['strategy']) for r in rows}!=expected:raise ValueError('Strategy result coverage differs')
    for r in rows:
        calls=STRATEGIES[r['strategy']];rr=[records[r['sample_id'],c] for c in calls]
        for name in ('output_tokens','input_tokens'):
            if r[name]!=sum(x[name] for x in rr):raise ValueError('Attributed token costs differ')
        if r['inference_seconds']!=sum(x['generation_seconds'] for x in rr):raise ValueError('Attributed time differs')
        if r['peak_allocated_bytes']!=max(x['peak_allocated_bytes'] for x in rr):raise ValueError('Peak memory differs')
        if r['call_record_hashes']!={c:identity(records[r['sample_id'],c]) for c in calls}:raise ValueError('Shared call identity differs')
        if r['selected_call'] not in calls:raise ValueError('Selected an unavailable call')
        if r['strategy']=='refinement' and r['selected_call']!='revision':raise ValueError('Revision substituted')
        if r['selected_extraction']!=records[r['sample_id'],r['selected_call']]['extracted_answer']:raise ValueError('Selected extraction differs')
        final=records[r['sample_id'],r['selected_call']]
        if (r['selected_hit_cap']!=final['hit_cap'] or r['selected_truncated']!=truncated(final)
                or r['call_truncations']!=sum(truncated(x) for x in rr)):
            raise ValueError('Truncation accounting differs')
    return True

def export_live_tables(output,questions,rows,records):
    metrics=('correct','output_tokens','input_tokens','inference_seconds','extraction_failed','selected_hit_cap','selected_truncated')
    names=list(STRATEGIES);ids=[q['sample_id'] for q in questions];means=np.empty((len(ids),3,len(metrics)))
    table=[];diffs=[]
    for ni,name in enumerate(names):
        rr=[next(r for r in rows if r['strategy']==name and r['sample_id']==sid) for sid in ids]
        means[:,ni]=[[r[m] for m in metrics] for r in rr]
        row={'strategy':name,'questions':len(ids),'nominal_allowance':3072,
             **{m:float(means[:,ni,mi].mean()) for mi,m in enumerate(metrics)},
             'total_output_tokens':sum(r['output_tokens'] for r in rr),'total_input_tokens':sum(r['input_tokens'] for r in rr),
             'total_inference_seconds':sum(r['inference_seconds'] for r in rr),
             'peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in rr),
             'peak_reserved_bytes':reserved_peak(rr),
             'call_extraction_failure_rate':sum(r['call_extraction_failures'] for r in rr)/sum(r['unique_calls_attributed'] for r in rr),
             'call_truncation_rate':sum(r['call_truncations'] for r in rr)/sum(r['unique_calls_attributed'] for r in rr)}
        if name=='refinement':
            row['initial_accuracy']=float(np.mean([r['initial_correct'] for r in rr]))
            for t in ('wrong_to_correct','correct_to_wrong','correct_to_correct','wrong_to_wrong'):
                row[t+'_count']=sum(r['transition']==t for r in rr);row[t+'_rate']=row[t+'_count']/len(rr)
            wrong=sum(not r['initial_correct'] for r in rr);correct=len(rr)-wrong
            row['wrong_to_correct_given_wrong']=row['wrong_to_correct_count']/wrong if wrong else None
            row['correct_to_wrong_given_correct']=row['correct_to_wrong_count']/correct if correct else None
            for m in ('answer_changed','initial_missing','revised_missing','revision_hit_cap','revision_truncated'):row[m+'_rate']=float(np.mean([r[m] for r in rr]))
            for before in (False,True):
                for after in (False,True):row[f'missing_{before}_to_{after}_count']=sum(r['initial_missing']==before and r['revised_missing']==after for r in rr)
        table.append(row)
    for left in range(3):
        for right in range(left):
            for mi,m in enumerate(metrics):diffs.append({'left':names[left],'right':names[right],'metric':m,
                **paired_bootstrap(means[:,left,mi],means[:,right,mi])})
    physical={'unique_calls':len(records),'output_tokens':sum(r['output_tokens'] for r in records.values()),
              'input_tokens':sum(r['input_tokens'] for r in records.values()),
              'inference_seconds':sum(r['generation_seconds'] for r in records.values()),
              'peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in records.values()),
              'peak_reserved_bytes':reserved_peak(records.values()),
              'model_loading_excluded':True,'shared_initial_generated_once':True}
    atomic_json(Path(output)/'aggregate.json',{'strategies':table,'paired_differences':diffs,'physical_execution':physical,
        'bootstrap':{'resamples':1000,'seed':6304,'unit':'question','interval':'95% linear percentile'}})
    csv_rows(Path(output)/'aggregate.csv',table)
    csv_rows(Path(output)/'paired_differences.csv',[{**r,'ci_low':r['ci95'][0],'ci_high':r['ci95'][1]} for r in diffs])

def select_live_examples(output,questions,rows,records):
    chosen={};qmap={q['sample_id']:q for q in questions}
    for sid in sorted(qmap):
        r={x['strategy']:x for x in rows if x['sample_id']==sid};ref=r['refinement'];parallel=r['parallel'];longer=r['longer']
        flags={'refinement_wrong_to_correct':ref['transition']=='wrong_to_correct',
               'refinement_correct_to_wrong':ref['transition']=='correct_to_wrong',
               'parallel_disagreement_correct':len(set(parallel['answer_groups']))>1 and parallel['correct'],
               'longer_truncated_or_missing_other_succeeds':(longer['selected_truncated'] or longer['extraction_failed']) and (parallel['correct'] or ref['correct'])}
        for category,flag in flags.items():
            if flag and category not in chosen:chosen[category]={'question':qmap[sid],'results':r,
                'calls':{c:records[sid,c] for c in LIMITS},'reason':category}
    categories={'refinement_wrong_to_correct','refinement_correct_to_wrong',
                'parallel_disagreement_correct','longer_truncated_or_missing_other_succeeds'}
    if qmap and not {'refinement_wrong_to_correct','refinement_correct_to_wrong'}.intersection(chosen):
        sid=min(qmap)
        chosen['refinement_unchanged_fallback']={'question':qmap[sid],
            'results':{x['strategy']:x for x in rows if x['sample_id']==sid},
            'calls':{c:records[sid,c] for c in LIMITS},'reason':'no correctness-changing refinement in this run'}
    atomic_json(Path(output)/'examples.json',{'rule':'first lexicographic question per category; first question as refinement fallback if neither correctness change occurred; retrospective after freeze',
        'examples':chosen,'unavailable_categories':sorted(categories-chosen.keys())})
