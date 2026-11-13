import copy,json
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from atml_pa3.task2.batching import BatchOptions,candidates,detect_gpu,probe_candidates,select_trial,real_preflight
from atml_pa3.task2.curriculum import Recipe
from atml_pa3.task2.adapters import make_stage_optimizer,AttentionLoRA,backbone_hash
from atml_pa3.task2.training import attempt_update,run_curriculum
from atml_pa3.task2.state import rng_state
from .test_training import components,attach_toy

def test_preferred_first_and_independent_effective_batch():
    assert candidates(BatchOptions())==[16,8,4,2,1]
    assert candidates(BatchOptions(effective_batch_size=32))==[32,16,8,4,2,1]
    assert candidates(BatchOptions(effective_batch_size=64))==[32,64,16,8,4,2,1]
    assert candidates(BatchOptions(batch_size='4'))==[4]
    with pytest.raises(ValueError):replace(Recipe(),microbatch=16,accumulation=1).assert_official()
    with pytest.raises(ValueError):replace(Recipe(),microbatch=1,accumulation=16).assert_official()
    with pytest.raises(ValueError):replace(Recipe(),microbatch=32,accumulation=1).assert_official()

@pytest.mark.parametrize('kwargs',[{'batch_size':'3'},{'batch_size':'no'},{'preferred_batch_size':0},
    {'memory_headroom_fraction':1},{'memory_headroom_fraction':-.1},{'effective_batch_size':15}])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):BatchOptions(**kwargs)

def test_gpu_detection_no_fixed_size_or_architecture(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    class Context:
        def __enter__(self):pass
        def __exit__(self,*args):pass
    monkeypatch.setattr(torch.cuda,'device',lambda d:Context())
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda d:(4*1024**3,6*1024**3))
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda d:SimpleNamespace(name='small GPU',major=7,minor=5))
    result=detect_gpu();assert result['free_bytes']==4*1024**3 and result['capability']==[7,5]
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='requires CUDA'):detect_gpu()

def test_fallback_benchmark_and_advisory_headroom():
    called=[]
    def probe(n):
        called.append(n)
        return {'status':'oom'} if n>=16 else {'status':'fit','headroom_fraction':.15,'examples_per_second':10 if n==8 else 10.5}
    result=probe_candidates(BatchOptions(effective_batch_size=32),probe)
    assert called==[32,16,8,4] and result['selected_microbatch']==8 and result['accumulation']==4
    fast=[{'status':'fit','microbatch':16,'headroom_fraction':.1,'examples_per_second':10},
          {'status':'fit','microbatch':8,'headroom_fraction':.2,'examples_per_second':12}]
    assert select_trial(fast)['microbatch']==8
    for r in fast:r['headroom_fraction']=.05
    assert not select_trial(fast)['headroom_target_met']
    with pytest.raises(RuntimeError,match='No tested microbatch'):select_trial([{'status':'oom'}])

@pytest.mark.parametrize('microbatch',[1,2,4,8,16])
def test_accumulation_loss_parameters_scheduler_and_accounting(microbatch):
    model,sampler,loader=components();ref=copy.deepcopy(model);records=sampler.next_batch(16)
    recipe=replace(Recipe(),microbatch=microbatch,accumulation=16//microbatch)
    opt,scheduler=make_stage_optimizer(model,1,recipe);ropt,rs=make_stage_optimizer(ref,1,recipe)
    scaler=lambda:torch.amp.GradScaler('cuda',enabled=False)
    result=attempt_update(model,records,opt,scheduler,scaler(),loader,recipe)
    full=attempt_update(ref,records,ropt,rs,scaler(),loader,replace(recipe,microbatch=16,accumulation=1))
    assert result['sft_loss']==pytest.approx(full['sft_loss'],rel=1e-6)
    assert scheduler.last_epoch==rs.last_epoch==1 and sampler.completed_examples==16
    for a,b in zip(model.connector.parameters(),ref.connector.parameters()):torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)

def test_training_oom_aborts_without_committing_or_skipping(tmp_path,monkeypatch):
    import atml_pa3.task2.training as training
    model,sampler,loader=components();before=sampler.state_dict()
    def oom(*args):raise torch.cuda.OutOfMemoryError('synthetic OOM during update')
    monkeypatch.setattr(training,'attempt_update',oom)
    with pytest.raises(RuntimeError,match='no update committed'):
        run_curriculum(model,sampler,loader,tmp_path/'run',{'test':True},
                       recipe=Recipe(budgets=(1,1,1),checkpoint_interval=1),stop_after=1)
    assert sampler.state_dict()==before
    assert not (tmp_path/'run/checkpoints').exists()
    events=[json.loads(line) for line in (tmp_path/'run/updates.jsonl').read_text().splitlines()]
    assert events[-1]['event']=='fatal_cuda_oom' and events[-1]['global_update']==0

def test_unrelated_probe_exception_not_hidden():
    with pytest.raises(ValueError,match='not an OOM'):
        probe_candidates(BatchOptions(),lambda n:(_ for _ in ()).throw(ValueError('not an OOM')))

def test_effective32_rejected_before_assets_or_model(tmp_path):
    from atml_pa3.task2.runtime import launch
    with pytest.raises(ValueError,match='effective32 is not permitted'):
        launch(tmp_path,tmp_path/'run',batch_options=BatchOptions(effective_batch_size=32))

def test_full_probe_rollback_on_oom_and_real_optimizer_step(monkeypatch):
    import atml_pa3.task2.batching as batching
    import atml_pa3.task2.training as training
    model,sampler,loader=components();state=copy.deepcopy(model.state_dict());backbone=backbone_hash(model)
    records=sampler.next_batch(16);examples,features=loader(records);batches={s:list(zip(examples,features)) for s in (1,2,3)}
    sampler_before=sampler.state_dict();rng=torch.get_rng_state().clone();calls=[]
    monkeypatch.setattr(batching,'detect_gpu',lambda *a:{'total_bytes':1000,'free_bytes':900})
    for name in ('empty_cache','reset_peak_memory_stats','synchronize'):monkeypatch.setattr(torch.cuda,name,lambda *a:None)
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda *a:(900,1000))
    monkeypatch.setattr(torch.cuda,'memory_allocated',lambda *a:100)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda *a:150)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda *a:160)
    original_scaler=torch.amp.GradScaler
    monkeypatch.setattr(batching.torch.amp,'GradScaler',lambda *a,**kw:original_scaler('cuda',enabled=False))
    original_update=training.attempt_update
    def update(model,records,opt,sched,scaler,loader,recipe):
        calls.append(recipe.microbatch)
        if recipe.microbatch>=16:
            with torch.no_grad():next(model.connector.parameters()).add_(5)
            raise torch.cuda.OutOfMemoryError('synthetic after partial mutation')
        return original_update(model,records,opt,sched,scaler,loader,recipe)
    monkeypatch.setattr(training,'attempt_update',update)
    result=real_preflight(model,batches,BatchOptions(),device='cpu',attach=attach_toy,repeats=1)
    assert result['selected_microbatch'] in (4,8) and calls[0]==16
    assert backbone_hash(model)==backbone and sampler.state_dict()==sampler_before and torch.equal(torch.get_rng_state(),rng)
    assert not any(isinstance(m,AttentionLoRA) for m in model.modules())
    for name,tensor in state.items():torch.testing.assert_close(model.state_dict()[name],tensor,rtol=0,atol=0)

def test_runtime_uses_selected_recipe_and_locks_it_on_resume(tmp_path,monkeypatch):
    from atml_pa3.task2 import runtime,batching
    from atml_pa3.task2.io import atomic_json
    model,_,_=components();captured=[];locked=[]
    monkeypatch.setattr(runtime,'validate_assets',lambda root:{'identities':{}})
    monkeypatch.setattr(runtime,'make_spec',lambda root,assets,recipe=None:{'recipe':(recipe or Recipe()).serial()})
    monkeypatch.setattr(runtime,'load_vlm',lambda device:(model,None))
    monkeypatch.setattr(runtime,'restore_initialization',lambda *a:None)
    monkeypatch.setattr(runtime,'FeatureStore',lambda p:None)
    monkeypatch.setattr(runtime,'real_sampler',lambda p:None)
    monkeypatch.setattr(runtime,'batch_loader',lambda *a:None)
    monkeypatch.setattr(runtime,'version',lambda name:{'torch':'2.14.1+cu130','transformers':'5.22.1','tokenizers':'0.23.3'}[name])
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'manual_seed_all',lambda seed:None)
    monkeypatch.setattr(batching,'detect_gpu',lambda *a:{'name':'small CUDA GPU'})
    monkeypatch.setattr(batching,'representative_batches',lambda *a:({},{}))
    def preflight(*a,**kw):
        locked.append(kw['locked']);return {'selected_microbatch':8,'accumulation':2}
    monkeypatch.setattr(batching,'real_preflight',preflight)
    def controller(*args,**kwargs):
        captured.append((args[4],kwargs['recipe']));return {'global_update':0}
    monkeypatch.setattr(runtime,'run_curriculum',controller)
    run=tmp_path/'run'
    runtime.launch(tmp_path,run,execute=True)
    spec,recipe=captured[-1]
    assert (recipe.microbatch,recipe.accumulation,recipe.effective_batch)==(8,2,16)
    assert spec['batch_sizing']['selected_microbatch']==8 and locked==[8]
    checkpoint=run/'checkpoints/update_00250';checkpoint.mkdir(parents=True);atomic_json(checkpoint/'manifest.json',{})
    atomic_json(run/'run_spec.json',spec)
    monkeypatch.setattr(runtime,'read_payload',lambda *a:{'progress':{'stage':1,'stage_update':250,'global_update':250,'completed_examples':4000}})
    runtime.launch(tmp_path,run,execute=True,resume_from=checkpoint)
    assert locked==[8,8] and captured[-1][0]==spec
    with pytest.raises(ValueError,match='Resume batch policy differs'):
        runtime.launch(tmp_path,run,resume_from=checkpoint,batch_options=BatchOptions(batch_size='8',memory_headroom_fraction=.15))

def test_representative_batches_use_long_training_content_and_reuse_features(tmp_path,monkeypatch):
    from atml_pa3.task2.batching import representative_batches
    from atml_pa3.task2 import io
    from .test_model import Tokenizer
    captions=[{'question':'Q','target':'a','image_id':1},
              {'question':'Q','target':'b'*100,'image_id':2}]
    natural=[{'members':[{'question':'x'*200,'target':'c'*30,'image_id':3}]}]
    pools={'categories':[{'object':'cat','plural':'cats','positive':[4],'counts':{'1':[5],'4':[6]}}]}
    monkeypatch.setattr(io,'read_jsonl',lambda p:captions if p.name=='captions_train.jsonl' else natural)
    monkeypatch.setattr(io,'read_json',lambda p:pools)
    class Store:
        calls=[]
        def get(self,image):self.calls.append(image);return torch.zeros(196,768,dtype=torch.float16)
    store=Store();batches,details=representative_batches(tmp_path,Tokenizer(),store)
    assert batches[1][0][0].target_tokens==64 and details['3']['max_sequence_tokens']>=details['1']['max_sequence_tokens']
    assert len(store.calls)==len(set(store.calls)) and 4 in store.calls and 6 in store.calls
    assert details['3']['question_tokens'][0]==80 and details['3']['target_tokens'][0]==16
