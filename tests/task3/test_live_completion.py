"""CPU-only tests of the actual launcher, seal and final export paths."""
import hashlib
import json
from pathlib import Path

import pytest
from transformers import GenerationConfig

from atml_pa3.task3.io import atomic_json,identity,sha256
from atml_pa3.task3_part2 import live_backend as backend
from atml_pa3.task3_part2 import live_run
from atml_pa3.task3_part2.live_analysis import select_live_examples,strategy_results
from atml_pa3.task3_part2.live_export import export,validate_live_results
from atml_pa3.task3_part2.live_spec import LIMITS,MODEL,REVISION,last_boxed
from atml_pa3.task3_part2.live_store import read_json

from .test_live import assets_path

# This serialization has the exact pinned Git blob hash, checked by validate_setup.
NATIVE={'bos_token_id':151643,'pad_token_id':151643,'do_sample':True,'eos_token_id':[151645,151643],
        'repetition_penalty':1.05,'temperature':.7,'top_p':.8,'top_k':20,'transformers_version':'4.37.0'}


@pytest.fixture
def launcher(tmp_path,monkeypatch):
    from atml_pa3.task3 import cache
    assets=assets_path()
    lock=read_json(assets/'model_files.json')
    hardware={'name':'NVIDIA GeForce RTX 3090','total_bytes':24*backend.GIB,
              'free_bytes':23*backend.GIB,'capability':[8,6]}
    backend_id={'model':MODEL,'revision':REVISION,'model_lock':identity(lock),'device':'cuda:0',
                'dtype':'float16','attention':'sdpa','batch_size':1,'torch':'2.14.1+cu130',
                'transformers':GenerationConfig().to_dict()['transformers_version'],'python':'3.12.15',
                'numpy':'2.4.6','math_verify':'0.9.0','sympy':'1.14.0','cuda':'13.0',
                'gpu_uuid':'synthetic-test-device','gpu_name':hardware['name'],
                'gpu_total_bytes':hardware['total_bytes'],'gpu_capability':[8,6]}
    info=read_json(assets/'qwen25_model_info.json')
    template=info['config']['tokenizer_config']['chat_template']
    calls=[];fail_at=None

    class SyntheticBackend:
        def __init__(self,*_):
            self.setup={'hardware':hardware,'backend':backend_id,'loading_seconds':.1+len(calls),
                        'chat_template_sha256':hashlib.sha256(template.encode()).hexdigest(),
                        'native_generation_config':json.dumps(NATIVE,indent=2),
                        'weights_trainable':False,'quantized':False,'offloaded':False,
                        'padding_side':'right','generation_api':'GenerationConfig + generate; no training'}

        def generate(self,request):
            if len(calls)==fail_at:raise RuntimeError('synthetic interruption')
            calls.append(identity(request))
            cap=request['max_new_tokens'];stress='min_new_tokens' in request
            ids=[5]*cap if stress else [5,151645]
            n=backend.INPUT_LIMITS[cap]+16 if stress else 12
            options={**request['sampling'],'max_new_tokens':cap,'min_new_tokens':request.get('min_new_tokens'),
                     **{k:NATIVE[k] for k in ('bos_token_id','pad_token_id','eos_token_id')}}
            prompt=''.join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in request['messages'])
            return {'text':r'\boxed{2}','input_tokens':n,'input_token_ids':[4]*n,'output_tokens':len(ids),
                    'token_ids':ids,'retained_token_ids':[i for i in ids if i not in NATIVE['eos_token_id']],
                    'ended_with_eos':not stress,'hit_cap':stress,'rendered_prompt':prompt+'<|im_start|>assistant\n',
                    'setup_id':identity(self.setup),'backend':backend_id,'hardware':hardware,
                    'effective_generation_config':GenerationConfig(**options).to_dict(),'padding_side':'right',
                    'attention_implementation':'sdpa','timing_scope':backend.TIMING_SCOPE,
                    'generation_seconds':.5,'cuda_generation_seconds':.25,'peak_allocated_bytes':6*backend.GIB,
                    'peak_reserved_bytes':8*backend.GIB,'free_after_bytes':10*backend.GIB}

        def close(self):pass

    class SyntheticWorker:
        def __init__(self,*_):pass
        def compute(self,**request):
            answers=request['answers'];n=len(answers)
            return {'missing':[False]*n,'parsed':[True]*n,'correct':[True]*n,
                    'equivalent':[[True]*n for _ in answers],'normalized':answers,'operation_timeouts':[]}
        def close(self):pass

    monkeypatch.setattr(backend,'validate_model_files',lambda *_:None)
    monkeypatch.setattr(backend,'check_gpu',lambda:hardware)
    monkeypatch.setattr(backend,'backend_identity',lambda _:backend_id)
    monkeypatch.setattr(backend,'QwenBackend',SyntheticBackend)
    monkeypatch.setattr(cache,'EquivalenceWorker',SyntheticWorker)
    preflight=tmp_path/'preflight';run=tmp_path/'run'
    backend.preflight(assets,tmp_path/'model',preflight,execute=True)
    calls.clear()

    def launch(interrupt=None):
        nonlocal fail_at
        fail_at=interrupt
        return live_run.launch(assets,tmp_path/'model',run,execute=True,preflight_path=preflight)

    return assets,run,launch,calls


@pytest.mark.parametrize('field,value',[('gpu_name','NVIDIA GeForce RTX 3090 Ti'),
                                        ('gpu_uuid','different-test-device')])
def test_official_launch_rejects_preflight_from_another_gpu(launcher,monkeypatch,field,value):
    _,run,launch,calls=launcher
    proof=read_json(run.parent/'preflight'/'PASSED.json')
    other={**proof['backend'],field:value}
    monkeypatch.setattr(backend,'backend_identity',lambda _:other)
    with pytest.raises(ValueError,match='incompatible preflight evidence'):launch()
    assert not run.exists() and not calls


def test_valid_complete_run_resumes_and_exports(launcher,tmp_path):
    assets,run,launch,calls=launcher
    with pytest.raises(RuntimeError,match='synthetic interruption'):
        launch(interrupt=7)
    committed={p.name:sha256(p) for p in (run/'calls').glob('*.json')}
    assert len(committed)==7 and not (run/'COMPLETE.json').exists() and not (run/'RUNNING.lock').exists()
    assert launch()=={'status':'COMPLETE','unique_calls':250}
    assert len(calls)==len(set(calls))==250
    assert all(sha256(run/'calls'/name)==digest for name,digest in committed.items())
    complete=read_json(run/'COMPLETE.json')
    assert len(complete['calls'])==250 and len(complete['grades'])==50 and len(complete['setups'])==2
    before=sha256(run/'COMPLETE.json')
    launch()
    assert sha256(run/'COMPLETE.json')==before and len(calls)==250
    qs,records,rows=validate_live_results(assets,run)
    assert (len(qs),len(records),len(rows))==(50,250,150)
    assert rows[1]['call_record_hashes']['parallel0']==rows[2]['call_record_hashes']['parallel0']
    out=tmp_path/'export'
    assert export(assets,run,out)=={'live':'complete','figures':7}
    assert read_json(out/'aggregate.json')['physical_execution']['peak_reserved_bytes']==8*backend.GIB
    fallback=read_json(out/'examples.json')['examples']['refinement_unchanged_fallback']
    assert fallback['question']['sample_id']==min(q['sample_id'] for q in qs)
    assert fallback['results']['refinement']['transition']=='correct_to_correct'

def test_inconsistent_grade_output_cannot_write_completion(launcher,monkeypatch):
    from atml_pa3.task3_part2 import live_analysis
    _,run,launch,calls=launcher
    grade=live_analysis.grade_calls

    def inconsistent(*args):
        grade(*args)
        path=run/'strategy_results.json';result=read_json(path)
        result['rows'][0]['correct']=False
        atomic_json(path,result)

    with monkeypatch.context() as patch:
        patch.setattr(live_analysis,'grade_calls',inconsistent)
        with pytest.raises(ValueError,match='Strategy results differ'):launch()
    assert not (run/'COMPLETE.json').exists() and not (run/'aggregate.json').exists()
    assert len(calls)==250
    assert launch()=={'status':'COMPLETE','unique_calls':250}
    assert len(calls)==250


@pytest.mark.parametrize('damage,error',[
    ('call_index','call hash index'),('grade_index','grade index'),('missing_call','Incomplete unique calls'),
    ('missing_grade','grade index'),('completion_hash','completion hashes'),
    ('corrupt_call','Corrupt call record'),('duplicate_json','Duplicate JSON key'),
    ('duplicate_row','Strategy results differ'),('reordered_rows','Strategy results differ'),
    ('incorrect','Strategy results differ'),('shared_cost','Strategy results differ'),
    ('grade_order','grade provenance'),('gpu_evidence','CUDA resource'),('config','generation configuration'),
    ('seed','Call prompt/config changed'),('setup','Missing call setup'),('preflight','run configuration'),
    ('device','Call setup'),('precision','Call setup'),
])
def test_invalid_completed_runs_fail_closed(launcher,tmp_path,damage,error):
    assets,run,launch,_=launcher
    launch()
    marker_path=run/'COMPLETE.json';marker=read_json(marker_path)
    call_path=run/'calls'/next(iter(marker['calls']))
    strategy_path=run/'strategy_results.json';result=read_json(strategy_path)
    if damage=='call_index':
        marker['calls']={}
    elif damage=='grade_index':
        result['grading_files']={}
    elif damage=='missing_call':
        call_path.unlink()
    elif damage=='missing_grade':
        (run/'grading'/next(iter(marker['grades']))).unlink()
    elif damage=='completion_hash':
        marker['calls'][call_path.name]='0'*64
    elif damage=='duplicate_json':
        marker_path.write_text('{"calls":{},"calls":{}}',encoding='utf-8')
    elif damage=='setup':
        (run/'setups'/next(iter(marker['setups']))).unlink()
    elif damage=='preflight':
        atomic_json(run/'preflight.json',{})
    elif damage=='grade_order':
        grade=run/'grading'/next(iter(result['grading_files']))
        saved=read_json(grade);saved['source']['call_order'].reverse();atomic_json(grade,saved)
        result['grading_files'][grade.name]=sha256(grade)
        marker['grades']=result['grading_files']
    elif damage in ('corrupt_call','gpu_evidence','config','seed','device','precision'):
        envelope=read_json(call_path);record=envelope['record']
        if damage=='corrupt_call':record['text']='corrupted'
        elif damage=='gpu_evidence':record['peak_reserved_bytes']=0
        elif damage=='config':record['effective_generation_config']['temperature']=.5
        elif damage=='device':record['backend']['device']='cpu'
        elif damage=='precision':record['backend']['dtype']='bfloat16'
        else:record['request']['seed']+=1
        if damage!='corrupt_call':envelope['sha256']=identity(record)
        atomic_json(call_path,envelope);marker['calls'][call_path.name]=sha256(call_path)
    else:
        rows=result['rows']
        if damage=='duplicate_row':rows[1]=rows[0]
        elif damage=='reordered_rows':rows[0],rows[1]=rows[1],rows[0]
        elif damage=='incorrect':rows[0]['correct']=False
        else:rows[2]['input_tokens']-=1
    atomic_json(strategy_path,result)
    marker['strategy_results_sha256']=sha256(strategy_path)
    if damage!='duplicate_json':atomic_json(marker_path,marker)
    out=tmp_path/'invalid-export'
    with pytest.raises(ValueError,match=error):export(assets,run,out)
    assert not out.exists()


def test_semantic_aliases_are_not_disagreements_and_raw_strings_survive(launcher,tmp_path):
    assets,run,launch,_=launcher
    launch()
    questions,records,_=validate_live_results(assets,run)
    q=questions[0];sid=q['sample_id'];rr={c:dict(records[sid,c]) for c in LIMITS}
    rr['parallel1']['text']=r'\boxed{02}'
    rr['parallel1']['extracted_answer']=last_boxed(rr['parallel1']['text'])
    sem={'equivalent':[[True]*5 for _ in LIMITS],'missing':[False]*5,'correct':[True]*5}
    rows=strategy_results(q,rr,sem)
    records={(sid,c):r for c,r in rr.items()}
    select_live_examples(tmp_path,[q],rows,records)
    examples=read_json(tmp_path/'examples.json')['examples']
    assert set(examples)=={'refinement_unchanged_fallback'}
    assert examples['refinement_unchanged_fallback']['results']['parallel']['answers']==['2','02','2']
    assert examples['refinement_unchanged_fallback']['calls']['parallel1']['text']==r'\boxed{02}'
    sem['equivalent'][2]=[False,False,True,False,False]
    for i in (0,1,3,4):sem['equivalent'][i][2]=False
    rows=strategy_results(q,rr,sem)
    select_live_examples(tmp_path,[q],rows,records)
    assert 'parallel_disagreement_correct' in read_json(tmp_path/'examples.json')['examples']
