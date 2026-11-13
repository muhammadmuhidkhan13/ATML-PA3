import copy,json
from pathlib import Path
import pytest
from atml_pa3.task3.io import read_json,atomic_json,identity
from atml_pa3.task3_part2.live_spec import load_assets,LIMITS
from atml_pa3.task3_part2.live_backend import validate_model_files
from atml_pa3.task3_part2.live_run import request_for
from atml_pa3.task3_part2.live_store import CallStore

def test_corrupt_frozen_question_asset_is_rejected(tmp_path):
    import shutil
    from .test_live import assets_path
    shutil.copytree(assets_path(),tmp_path/'assets')
    p=tmp_path/'assets/questions.json';rows=read_json(p);rows[0]['sample_id']='other';atomic_json(p,rows)
    with pytest.raises(ValueError,match='Corrupt live asset'):load_assets(tmp_path/'assets')

def test_model_checksums_before_loading(tmp_path):
    import hashlib
    from atml_pa3.task3_part2.live_spec import MODEL,REVISION
    files={'config.json':{'model_type':'qwen2'},'model.safetensors.index.json':{'weight_map':{'a':'model.safetensors'}}}
    for name,value in files.items():atomic_json(tmp_path/name,value)
    (tmp_path/'model.safetensors').write_bytes(b'fake-small-fixture-not-real-weights')
    records={}
    for p in tmp_path.iterdir():
        b=p.read_bytes();records[p.name]={'size':len(b),'algorithm':'sha256','digest':hashlib.sha256(b).hexdigest()}
    lock={'model':MODEL,'revision':REVISION,'files':records}
    assert validate_model_files(tmp_path,lock)
    (tmp_path/'chat_template.jinja').write_text('unexpected override')
    with pytest.raises(ValueError,match='override'):validate_model_files(tmp_path,lock)
    (tmp_path/'chat_template.jinja').unlink()
    (tmp_path/'model.safetensors').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='Missing/incomplete'):validate_model_files(tmp_path,lock)
    lock['revision']='bad'
    with pytest.raises(ValueError,match='identity'):validate_model_files(tmp_path,lock)

def test_shared_cost_validation_rejects_wrong_totals(tmp_path):
    from .test_live import MockBackend
    from atml_pa3.task3_part2.live_run import execute_calls,verify_calls
    from atml_pa3.task3_part2.live_analysis import strategy_results,validate_strategy_results
    q={'sample_id':'SYNTHETIC','problem':'1+1?'};store=CallStore(tmp_path,{'test':True});execute_calls(store,[q],MockBackend())
    records=verify_calls(store,[q]);rr={c:records[q['sample_id'],c] for c in LIMITS}
    sem={'equivalent':[[True]*5 for _ in range(5)],'missing':[False]*5,'correct':[True]*5}
    rows=strategy_results(q,rr,sem);assert validate_strategy_results([q],rows,records)
    rows[-1]['output_tokens']-=3
    with pytest.raises(ValueError,match='token costs'):validate_strategy_results([q],rows,records)

def test_missing_revision_initial_and_completion_rejected(tmp_path):
    from atml_pa3.task3_part2.live_export import validate_live_results
    from .test_live import assets_path
    with pytest.raises(ValueError):request_for({'sample_id':'q','problem':'x'},'revision')
    with pytest.raises(FileNotFoundError):validate_live_results(assets_path(),tmp_path)

@pytest.mark.parametrize('name,capability,total,free,error',[
    ('NVIDIA GeForce RTX 3090',(8,6),24,23,None),
    ('NVIDIA GeForce RTX 3090 Ti',(8,6),24,23,None),
    ('NVIDIA A100-SXM4-80GB',(8,0),80,79,None),
    ('NVIDIA A100-SXM4-40GB',(8,0),40,39,'eligible'),
    ('NVIDIA GeForce RTX 3090',(8,6),22,22,'eligible'),
    ('NVIDIA GeForce RTX 3090',(8,0),24,23,'eligible'),
    ('NVIDIA GeForce RTX 3090',(8,6),24,19,'already in use'),
    ('NVIDIA GeForce RTX 3090',(8,6),24,11,'12 GiB'),
    ('NVIDIA GeForce RTX 3090 Laptop GPU',(8,6),24,23,'eligible'),
    ('NVIDIA GeForce RTX 3090 Ti Laptop GPU',(8,6),24,23,'eligible'),
    ('NVIDIA GeForce RTX 3080',(8,6),24,23,'eligible'),
    ('NVIDIA GeForce RTX 4090',(8,9),24,23,'eligible'),
    ('NVIDIA GeForce RTX 3090',(8,6),24,25,'Invalid GPU memory'),
    ('NVIDIA GeForce RTX 3090',(8,6),24,-1,'Invalid GPU memory'),
    ('NVIDIA GeForce RTX 3090 Ti',(8,6),22,22,'eligible'),
    ('NVIDIA GeForce RTX 3090 Ti',(8,6),24,11,'12 GiB'),
    ('NVIDIA GeForce RTX 3090 Ti',(8,6),24,19,'already in use'),
])
def test_gpu_eligibility(monkeypatch,name,capability,total,free,error):
    from types import SimpleNamespace
    import torch
    from atml_pa3.task3_part2.live_backend import check_gpu,GIB
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'get_device_properties',
                        lambda _:SimpleNamespace(name=name,total_memory=total*GIB))
    monkeypatch.setattr(torch.cuda,'get_device_capability',lambda _:capability)
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda _:(free*GIB,total*GIB))
    if error:
        with pytest.raises(ValueError,match=error):check_gpu()
    else:
        assert check_gpu()['name']==name

def test_gpu_eligibility_requires_cuda(monkeypatch):
    import torch
    from atml_pa3.task3_part2.live_backend import check_gpu
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(ValueError,match='CUDA GPU required'):check_gpu()

@pytest.mark.parametrize('fail', [None,'revision_memory'])
def test_full_preflight_requires_two_probes_and_never_certifies_oom(tmp_path,monkeypatch,fail):
    import torch
    from .test_live import assets_path,MockBackend
    from atml_pa3.task3 import cache
    from atml_pa3.task3_part2 import live_backend as b
    from atml_pa3.task3_part2.live_run import verify_preflight

    assets=assets_path();out=tmp_path/'preflight';seen=[];closed=[]
    backend_id={'device':'mock CUDA'}
    monkeypatch.setattr(b,'validate_model_files',lambda *_:None)
    monkeypatch.setattr(b,'backend_identity',lambda _:backend_id)

    class Worker:
        def __init__(self,*_):pass
        def compute(self,**_):return {'correct':[True]}
        def close(self):pass

    class Backend(MockBackend):
        setup={}
        def __init__(self,*_):super().__init__()
        def generate(self,request):
            r=super().generate(request)
            cap=request['max_new_tokens'];seen.append(cap)
            if cap==64:return r
            if cap==2048 and fail:raise torch.cuda.OutOfMemoryError('synthetic OOM')
            n=b.INPUT_LIMITS[cap]+32
            return {**r,'input_tokens':n,'input_token_ids':[1]*n,'output_tokens':cap,
                    'token_ids':[2]*cap,'hit_cap':True,'ended_with_eos':False,
                    'effective_generation_config':{'min_new_tokens':request['min_new_tokens']},
                    'peak_reserved_bytes':2048,'free_after_bytes':b.MIN_HEADROOM}
        def close(self):closed.append(True)

    monkeypatch.setattr(cache,'EquivalenceWorker',Worker)
    monkeypatch.setattr(b,'QwenBackend',Backend)
    if fail:
        with pytest.raises(RuntimeError,match='CUDA out of memory'):
            b.preflight(assets,tmp_path/'model',out,execute=True)
        assert not (out/'PASSED.json').exists()
        assert not (out/'RUNNING.lock').exists()
    else:
        assert b.preflight(assets,tmp_path/'model',out,execute=True)['official_questions_generated']==0
        proof=read_json(out/'PASSED.json')
        config={'backend':backend_id,'asset_manifest':proof['assets'],
                'code':{'live_backend.py':proof['backend_code_sha256']}}
        verify_preflight(out,config)
        proof['stress_call_hashes'].pop('revision_memory')
        atomic_json(out/'PASSED.json',proof)
        with pytest.raises(ValueError,match='Full-length'):verify_preflight(out,config)
    assert seen==[64,3072,2048] and closed==[True]

def test_probe_rejects_short_generation_low_headroom_and_oversized_input():
    from atml_pa3.task3_part2.live_backend import check_input_length,stress_requests,validate_stress,MIN_HEADROOM
    request=stress_requests()['revision_memory']
    record={'input_tokens':8192,'output_tokens':2048,'effective_generation_config':{'min_new_tokens':2048},
            'free_after_bytes':MIN_HEADROOM,'peak_allocated_bytes':1024,'peak_reserved_bytes':2048}
    validate_stress(record,request)
    with pytest.raises(ValueError,match='generation cap'):
        validate_stress({**record,'output_tokens':2047},request)
    with pytest.raises(ValueError,match='headroom'):
        validate_stress({**record,'free_after_bytes':MIN_HEADROOM-1},request)
    with pytest.raises(ValueError,match='bounded'):check_input_length(8449,2048,True)
    with pytest.raises(ValueError,match='certified'):check_input_length(8193,2048,False)
    assert 'min_new_tokens' not in request_for({'sample_id':'q','problem':'x'},'longer')
