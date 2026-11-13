"""Exact-key joins with scalar indexes and bounded per-question Parquet access."""
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from .io import sha256,read_json

KEYS=('dataset','sample_id','rollout_id')
FOLDERS={'math':'qwen25-math','mathvista':'qwen25vl-mathvista'}

def assert_source_identity(root:str|Path,prepared:str|Path)->None:
    manifest=read_json(Path(prepared)/'source_manifest.json')
    for dataset,audit in manifest['datasets'].items():
        for name,record in audit['files'].items():
            if sha256(Path(root)/FOLDERS[dataset]/f'{name}.parquet')!=record['sha256']:
                raise ValueError('Source Parquet changed after preparation')
    if sha256(Path(root)/'README.MD')!=manifest['readme_sha256']:raise ValueError('Source README changed')

def validate_mathvista_metadata(metadata:dict)->None:
    for sid,row in metadata.items():
        required={'question_type','answer_type','choices','precision','answer'}
        if not required<=row.keys():raise ValueError(f'Missing released metadata fields:{sid}')
        if row['answer'] is None or not str(row['answer']).strip():raise ValueError('Missing released gold answer')
        if row['question_type']=='multi_choice':
            if not isinstance(row['choices'],list) or not 1<=len(row['choices'])<=26:
                raise ValueError(f'Missing/invalid choices:{sid}')
        elif row['question_type']=='free_form':
            if row['answer_type'] not in ('integer','float','list','text','string'):raise ValueError('Unknown released answer type')
            if row['answer_type']=='float' and (not isinstance(row['precision'],int) or row['precision']<0):
                raise ValueError(f'Missing numeric precision:{sid}')
        else:raise ValueError('Unknown released question type')

def key(row):return tuple(row[k] for k in KEYS)

def index_table(table):
    """Read keys only; duplicate detection occurs before any join."""
    index={};questions=defaultdict(set)
    for group in range(table.num_row_groups):
        for row in table.read_row_group(group,columns=list(KEYS)).to_pylist():
            k=key(row)
            if k in index:raise ValueError(f'Duplicate key {k}')
            if not isinstance(k[1],str) or not isinstance(k[2],int):raise ValueError('Invalid key types')
            index[k]=group;questions[k[1]].add(group)
    return index,dict(questions)

class DatasetSource:
    def __init__(self,root:str|Path,dataset:str,*,expected_questions:int=1000,rollouts:int=32)->None:
        self.root=Path(root);self.dataset=dataset;self.rollouts=rollouts
        folder=self.root/FOLDERS[dataset]
        self.paths={name:folder/f'{name}.parquet' for name in ('generations','probability_metrics')}
        if dataset=='mathvista':self.paths['attention_metrics']=folder/'attention_metrics.parquet'
        self.tables={k:pq.ParquetFile(p) for k,p in self.paths.items()}
        self.indexes={};self.groups={}
        for name,table in self.tables.items():self.indexes[name],self.groups[name]=index_table(table)
        expected=set(self.indexes['generations'])
        for name,index in self.indexes.items():
            if set(index)!=expected:raise ValueError(f'Unmatched join keys: {name},missing={len(expected-set(index))},extra={len(set(index)-expected)}')
        by_question=defaultdict(set)
        for ds,sid,rid in expected:
            if ds!=dataset:raise ValueError('Dataset key disagrees with source folder')
            by_question[sid].add(rid)
        if len(by_question)!=expected_questions:raise ValueError('Question count differs')
        if any(ids!=set(range(rollouts)) for ids in by_question.values()):raise ValueError('Expected unique rollout IDs0..31')
        self.ids=sorted(by_question)

    def question(self,sample_id:str)->list[dict]:
        joined={}
        for name,table in self.tables.items():
            rows={key(row):row for group in sorted(self.groups[name][sample_id])
                  for row in table.read_row_group(group).to_pylist() if row['sample_id']==sample_id}
            joined[name]=rows
        result=[]
        for rid in range(self.rollouts):
            k=(self.dataset,sample_id,rid);row=dict(joined['generations'][k])
            row.update({field:value for field,value in joined['probability_metrics'][k].items() if field not in KEYS})
            if 'attention_metrics' in joined:
                attention=joined['attention_metrics'][k]
                row['preceding_generated_counts']=attention['n_generated_text_tokens']
                row.update({field:value for field,value in attention.items() if field not in (*KEYS,'n_generated_text_tokens')})
            result.append(row)
        if len({r['gold_answer'] for r in result})!=1 or len({r['question_text'] for r in result})!=1:
            raise ValueError('Question/reference changed within rollout pool')
        return result

    def audit(self)->dict:
        counts=Counter();nonfinite=Counter();invalid=Counter();metadata={}
        for name,table in self.tables.items():
            metadata[name]={k.decode():v.decode() for k,v in (table.schema_arrow.metadata or {}).items()}
        for sid in self.ids:
            for row in self.question(sid):
                n=len(row['generated_token_ids'])
                if n!=row['n_generated_text_tokens'] or any(not isinstance(t,int) or t<0 for t in row['generated_token_ids']):
                    raise ValueError('Retained token IDs/count mismatch')
                counts['rows']+=1;counts['tokens']+=n;counts['capped']+=int(row['hit_max_new_tokens'])
                counts['missing_extracted']+=int(row['extracted_answer'] is None or not row['extracted_answer'].strip())
                if not row['gold_answer']:raise ValueError('Missing gold reference')
                fields=['logp','kl_u_p']
                if self.dataset=='mathvista':
                    fields+=['img_attn_total','prompt_text_attn_total','generated_text_attn_total']
                    if row['preceding_generated_counts']!=list(range(n)):raise ValueError('Attention is shifted relative to tokens')
                    if row['n_image_tokens']<=0 or row['n_prompt_text_tokens']<=0:raise ValueError('Empty attention groups')
                    if n and row['generated_text_attn_total'][0]!=0:raise ValueError('First prediction attends to generated tokens')
                for field in fields:
                    values=row[field]
                    if len(values)!=n:raise ValueError(f'Token/metric alignment mismatch: {field}')
                    array=np.array(values,dtype=float);nonfinite[field]+=int((~np.isfinite(array)).sum())
                    invalid[field]+=int(((array>1e-6) if field=='logp' else (array< -1e-6)).sum())
        if any(invalid.values()):raise ValueError(f'Invalid probability/attention values:{invalid}')
        return {'dataset':self.dataset,'questions':len(self.ids),'rollouts_per_question':self.rollouts,
                'counts':dict(counts),'nonfinite_token_measurements':dict(nonfinite),'invalid_values':dict(invalid),
                'duplicate_keys':0,'unmatched_keys':0,'token_alignment':'corresponding prediction; no shift; termination excluded',
                'files':{name:{'path':str(path.resolve()),'bytes':path.stat().st_size,'sha256':sha256(path),
                                'schema':str(self.tables[name].schema_arrow),'metadata':metadata[name]} for name,path in self.paths.items()}}
