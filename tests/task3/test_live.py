from pathlib import Path
import copy,json
import pytest
from atml_pa3.task3.io import read_json,identity
from atml_pa3.task3_part2.live_spec import last_boxed,LIMITS,SAMPLING,seed_for,messages,INSTRUCTION,REVIEW,SYSTEM,load_assets
from atml_pa3.task3_part2.live_store import CallStore,RunLock
from atml_pa3.task3_part2.live_run import execute_calls,verify_calls,request_for,launch
from atml_pa3.task3_part2.live_analysis import strategy_results,export_live_tables

def assets_path():
    root=Path(__file__).resolve().parents[2]
    return root/'artifacts/task3/live-assets-v1' if (root/'artifacts/task3/live-assets-v1').exists() else root/'assets'

@pytest.mark.parametrize('text,expected',[(r'\boxed{\frac{1}{2}}',r'\frac{1}{2}'),
    (r'\boxed{1} then \boxed{2}','2'),(r'\boxed{1} then \boxed{bad',None),
    (r'\boxed{{a}+{b}}',r'{a}+{b}'),(r'\boxed{  }',None),('unboxed',None)])
def test_last_literal_balanced_box(text,expected):assert last_boxed(text)==expected

class MockBackend:
    def __init__(self,fail_after=None):self.requests=[];self.fail_after=fail_after
    def generate(self,request):
        if self.fail_after is not None and len(self.requests)>=self.fail_after:raise RuntimeError('simulated interruption')
        self.requests.append(request)
        assert 'gold_answer' not in request and all('SECRET_GOLD' not in m['content'] for m in request['messages'])
        return {'text':r'\boxed{2}','token_ids':[1,2,3],'input_token_ids':[4,5],
            'input_tokens':2,'output_tokens':3,'hit_cap':False,'ended_with_eos':True,'generation_seconds':.5,
            'peak_allocated_bytes':1024,'rendered_prompt':json.dumps(request['messages'])}

def test_resume_shared_initial_and_attributed_costs(tmp_path):
    q={'sample_id':'synthetic-q','problem':'one plus one?','gold_answer':'SECRET_GOLD'}
    store=CallStore(tmp_path,{'synthetic_test':True});backend=MockBackend(fail_after=3)
    with pytest.raises(RuntimeError):execute_calls(store,[q],backend)
    first=store.read(q['sample_id'],'parallel0');backend=MockBackend()
    execute_calls(store,[q],backend)
    assert len(backend.requests)==2 and store.read(q['sample_id'],'parallel0')==first
    records=verify_calls(store,[q]);rr={c:records[q['sample_id'],c] for c in LIMITS}
    revision=rr['revision']['request']
    assert revision['messages']==messages(q['problem'],first['text'])
    assert revision['shared_initial_sha256']==identity(first)
    sem={'equivalent':[[True]*5 for _ in range(5)],'missing':[False]*5,'correct':[True]*5}
    rows=strategy_results(q,rr,sem);r={x['strategy']:x for x in rows}
    assert [r[s]['output_tokens'] for s in ('longer','parallel','refinement')]==[3,9,6]
    assert [r[s]['inference_seconds'] for s in ('longer','parallel','refinement')]==[.5,1.5,1.]
    assert r['parallel']['call_record_hashes']['parallel0']==r['refinement']['call_record_hashes']['parallel0']
    export_live_tables(tmp_path,[q],rows,records)
    agg=read_json(tmp_path/'aggregate.json')
    assert agg['physical_execution']['unique_calls']==5 and agg['physical_execution']['output_tokens']==15
    assert r['refinement']['transition']=='correct_to_correct'
    before=sum(len(p.read_bytes()) for p in (tmp_path/'calls').glob('*.json'))
    execute_calls(store,[q],MockBackend(fail_after=0))
    assert before==sum(len(p.read_bytes()) for p in (tmp_path/'calls').glob('*.json'))

def test_250_unique_calls_mock_only(tmp_path):
    qs=[{'sample_id':f'SYNTHETIC-{i}','problem':'1+1?'} for i in range(50)]
    store=CallStore(tmp_path,{'synthetic_test':True});backend=MockBackend();execute_calls(store,qs,backend)
    assert len(backend.requests)==250 and len(verify_calls(store,qs))==250
    assert not (tmp_path/'COMPLETE.json').exists()
    assert len({seed_for(q['sample_id'],c) for q in qs for c in LIMITS})==250

def test_truncation_excludes_eos_at_cap(tmp_path):
    from atml_pa3.task3_part2.live_analysis import validate_strategy_results
    q={'sample_id':'q','problem':'x'}
    store=CallStore(tmp_path,{'test':True})

    class CapBackend(MockBackend):
        def generate(self,request):
            record=super().generate(request)
            cap=request['max_new_tokens']
            return {**record,'output_tokens':cap,'token_ids':[1]*cap,
                    'hit_cap':True,'ended_with_eos':cap==3072}

    execute_calls(store,[q],CapBackend())
    records=verify_calls(store,[q])
    sem={'equivalent':[[True]*5 for _ in range(5)],'missing':[False]*5,'correct':[True]*5}
    rows=strategy_results(q,{c:records['q',c] for c in LIMITS},sem)
    assert [r['selected_hit_cap'] for r in rows]==[True,True,True]
    assert [r['selected_truncated'] for r in rows]==[False,True,True]
    assert [r['call_truncations'] for r in rows]==[0,3,2]
    assert validate_strategy_results([q],rows,records)
    export_live_tables(tmp_path,[q],rows,records)
    longer,_,refinement=read_json(tmp_path/'aggregate.json')['strategies']
    assert longer['selected_hit_cap']==1 and longer['call_truncation_rate']==0
    assert refinement['revision_truncated_rate']==1

def test_exact_prompts_limits_and_native_template():
    from transformers import PreTrainedTokenizerFast
    from tokenizers import Tokenizer,models
    info=read_json(assets_path()/'qwen25_model_info.json')
    template=info['config']['tokenizer_config']['chat_template']
    # Real pinned native Jinja template, artificial vocabulary: no model/official call.
    tok=PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel({'[UNK]':0},unk_token='[UNK]')),chat_template=template)
    ms=messages('ORIGINAL PROBLEM')
    assert ms==[{'role':'system','content':SYSTEM},{'role':'user','content':'ORIGINAL PROBLEM\n\n'+INSTRUCTION}]
    expected='<|im_start|>system\n'+SYSTEM+'<|im_end|>\n<|im_start|>user\nORIGINAL PROBLEM\n\n'+INSTRUCTION+'<|im_end|>\n<|im_start|>assistant\n'
    assert tok.apply_chat_template(ms,tokenize=False,add_generation_prompt=True)==expected
    assert messages('problem','prior')[-1]=={'role':'user','content':REVIEW}
    assert LIMITS=={'longer':3072,'parallel0':1024,'parallel1':1024,'parallel2':1024,'revision':2048}

def test_corrupt_incompatible_and_missing_records(tmp_path):
    q={'sample_id':'q','problem':'x'};store=CallStore(tmp_path,{'test':True})
    execute_calls(store,[q],MockBackend())
    with pytest.raises(ValueError):CallStore(tmp_path,{'test':False})
    with pytest.raises(ValueError):store.read('q','longer',{'different':True})
    with pytest.raises(FileExistsError):store.write('q','longer',{})
    p=store.path('q','parallel0');r=read_json(p);r['record']['text']='tampered';p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='hash'):store.read('q','parallel0')
    p.unlink()
    with pytest.raises(ValueError,match='Incomplete'):verify_calls(store,[q])

def test_run_lock_and_foreign_directory(tmp_path):
    with RunLock(tmp_path):
        with pytest.raises(ValueError,match='locked'):
            with RunLock(tmp_path):pass
    assert not (tmp_path/'RUNNING.lock').exists()
    (tmp_path/'foreign.txt').write_text('keep')
    with pytest.raises(ValueError):CallStore(tmp_path,{'new':True})

def test_default_does_not_load_model_or_create_run(tmp_path,monkeypatch):
    import atml_pa3.task3_part2.live_run as module
    assets=assets_path()
    report=launch(assets,tmp_path/'MISSING_MODEL',tmp_path/'output')
    assert report['unique_calls_planned']==250 and not (tmp_path/'output').exists()
    spec,qs,_=load_assets(assets)
    assert len(qs)==50 and spec['protocol']['revision']=='aa8e72537993ba99e69dfaafa59ed015b17504d1'
    from atml_pa3.task3_part2.live_backend import validate_model_files
    with pytest.raises(ValueError,match='Missing/incomplete'):validate_model_files(tmp_path/'missing',read_json(assets/'model_files.json'))

def test_refinement_never_substitutes_correct_initial(tmp_path):
    q={'sample_id':'q','problem':'x'};store=CallStore(tmp_path,{'test':True});execute_calls(store,[q],MockBackend())
    rr={c:store.read('q',c) for c in LIMITS}
    sem={'equivalent':[[i==j for j in range(5)] for i in range(5)],'missing':[False]*5,'correct':[False,True,False,False,False]}
    rows=strategy_results(q,rr,sem);r=next(x for x in rows if x['strategy']=='refinement')
    assert not r['correct'] and r['transition']=='correct_to_wrong' and r['selected_call']=='revision'
    assert next(x for x in rows if x['strategy']=='parallel')['selected_call']=='parallel0'

def test_tiny_real_transformers_generation_contract_cpu():
    # Fresh tiny random model, artificial input; no weights downloaded, no training.
    import torch
    from transformers import Qwen2Config,Qwen2ForCausalLM,GenerationConfig
    torch.manual_seed(6304)
    model=Qwen2ForCausalLM(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=64)).eval()
    before=[p.detach().clone() for p in model.parameters()]
    config=GenerationConfig(**SAMPLING,max_new_tokens=2,eos_token_id=31,pad_token_id=0)
    with torch.inference_mode():out=model.generate(input_ids=torch.tensor([[1,2]]),attention_mask=torch.ones(1,2,dtype=torch.long),generation_config=config)
    assert out.shape[1] in (3,4) and all(torch.equal(a,b) for a,b in zip(before,model.parameters()))

def test_invalid_numerics_and_interrupted_atomic_write(tmp_path):
    from atml_pa3.task3_part2.live_store import validate_record
    q={'sample_id':'q','problem':'x'};store=CallStore(tmp_path,{'test':True})
    execute_calls(store,[q],MockBackend());r=store.read('q','longer')
    for value in (float('nan'),float('inf'),-1):
        broken={**r,'generation_seconds':value}
        with pytest.raises(ValueError):validate_record(broken)
    path=store.path('q','revision');path.unlink();path.with_suffix('.json.partial').write_text('{incomplete')
    backend=MockBackend();execute_calls(store,[q],backend)
    assert len(backend.requests)==1 and len(verify_calls(store,[q]))==5

def test_pending_live_export_has_no_simulated_figures(tmp_path):
    from atml_pa3.task3_part2.live_export import export
    assert export(assets_path(),tmp_path/'absent',tmp_path/'figures')=={'live':'pending','figures':0}
    assert list((tmp_path/'figures').glob('*.png'))==[]
