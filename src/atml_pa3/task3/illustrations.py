"""Fixed retrospective trajectory selection; no experimental rules are changed."""
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from .io import read_json,atomic_json,identity
from .data import DatasetSource
from .cache import load_semantics
from .scoring import base_scores
from .visconf import calibrate
from .equivalence import prefix_groups
from .diagnostics import illustration_candidates

def export_illustrations(root,prepared,semantics,output,*,split='evaluation'):
    output,prepared=Path(output),Path(prepared);examples=[];selection={}
    for dataset in ('math','mathvista'):
        directory=output/dataset/split;ids=sorted(read_json(directory/'manifest.json')['question_ids'])
        source=DatasetSource(root,dataset);orders=read_json(prepared/dataset/'orders.json')['orders']
        categories=['high_confidence_incorrect','low_confidence_correct','rwc_vs_highest']
        if dataset=='mathvista':categories+=['high_vision_low_confidence','high_confidence_low_vision','fusion_disagreement']
        pending=set(categories)
        for sid in ids:
            d=read_json(directory/'questions'/f'{identity(sid)}.diagnostics.json')
            k=np.asarray(d['signals']['self_certainty']);v=np.asarray(d['signals'].get('vision',np.zeros(32)))
            correct=np.asarray(d['correctness']);choices={
                'high_confidence_incorrect':np.flatnonzero((k>=np.quantile(k,.75))&~correct).tolist(),
                'low_confidence_correct':np.flatnonzero((k<=np.quantile(k,.25))&correct).tolist(),
                'high_vision_low_confidence':np.flatnonzero((v>=np.quantile(v,.75))&(k<=np.quantile(k,.25))).tolist(),
                'high_confidence_low_vision':np.flatnonzero((k>=np.quantile(k,.75))&(v<=np.quantile(v,.25))).tolist()}
            records=pq.read_table(directory/'questions'/f'{identity(sid)}.parquet').to_pylist()
            selected={r['method']:r for r in records if r['order']==0 and r['budget']==32}
            hi='HighestVisConf' if dataset=='mathvista' else 'HighestK';rwc='VisConf_RWC' if dataset=='mathvista' else 'K_RWC'
            if selected[hi]['selected_answer_group']!=selected[rwc]['selected_answer_group']:
                choices['rwc_vs_highest']=[selected[hi]['selected_rollout_id'],selected[rwc]['selected_rollout_id']]
            if dataset=='mathvista' and selected[hi]['selected_rollout_id']!=selected['HighestTokenFusion']['selected_rollout_id']:
                choices['fusion_disagreement']=[selected[hi]['selected_rollout_id'],selected['HighestTokenFusion']['selected_rollout_id']]
            chosen=[category for category in categories if category in pending and choices.get(category)]
            if not chosen:continue
            rows=source.question(sid);base=[base_scores(row) for row in rows];sem=load_semantics(semantics,dataset,sid)
            fusion=calibrate([b['kl'] for b in base],[b['v'] for b in base]) if dataset=='mathvista' else None
            groups=prefix_groups(orders[sid][0],sem['equivalent'],sem['missing']);group_by_id=dict(zip(orders[sid][0],groups))
            for category in chosen:
                rollout_ids=choices[category][:2] if 'disagreement' in category or category=='rwc_vs_highest' else choices[category][:1]
                items=[]
                for rid in rollout_ids:
                    item={'source_record':rows[rid],'answer_group':group_by_id[rid],'normalized_answer':str(sem['normalized'][rid]),
                          'correct':bool(sem['correct'][rid]),'base_scores':{name:base[rid][name] for name in ('length','mean_logp','self_certainty','vision')}}
                    if fusion is not None:item.update(calibrated_confidence=fusion['calibrated'][rid],token_visual_engagement=base[rid]['v'],
                                                      visconf=fusion['visconf'][rid],tokenwise_fusion=fusion['tokenwise'][rid],
                                                      population_covariance=fusion['covariance'][rid],q05=fusion['q05'],q95=fusion['q95'])
                    items.append(item)
                examples.append({'dataset':dataset,'split':split,'category':category,'sample_id':sid,'order':0,'budget':32,'trajectories':items})
                selection[dataset+':'+category]=sid;pending.remove(category)
            if not pending:break
        for category in sorted(pending):selection[dataset+':'+category]=None
    atomic_json(output/f'illustrations_{split}.json',{'selection_rule':'first lexicographic qualifying question per category; first canonical rollout or two disagreeing selected rollouts; retrospective only',
                                                   'selection':selection,'examples':examples})
    return selection
