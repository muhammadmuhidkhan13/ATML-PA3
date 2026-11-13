"""Independent regressions for PDF compliance and malformed checkpoint progress."""
from dataclasses import replace
from pathlib import Path
import runpy
import pytest
from atml_pa3.task2.curriculum import Recipe
from atml_pa3.task2.batching import BatchOptions
from atml_pa3.task2 import runtime
from atml_pa3.task2.state import publish

@pytest.mark.parametrize('micro,accum',[(16,1),(4,4),(2,8),(1,16)])
def test_pdf_requires_physical_eight_and_accumulation_two(micro,accum):
    with pytest.raises(ValueError,match='microbatch8'):
        replace(Recipe(),microbatch=micro,accumulation=accum).assert_official()

def test_default_cli_is_fixed_eight(monkeypatch,tmp_path,capsys):
    captured={}
    def launch(*args,**kwargs):captured.update(kwargs);return {'training_started':False}
    monkeypatch.setattr(runtime,'launch',launch)
    cli=runpy.run_path(str(Path(__file__).resolve().parents[2]/'scripts/task2_train.py'))
    cli['main'](['--run-dir',str(tmp_path/'never-created')])
    assert captured['batch_options'].batch_size=='8'
    assert not captured['execute']

@pytest.mark.parametrize('size',['auto','16','4','2','1'])
def test_nonstandard_physical_size_rejected_before_assets(monkeypatch,tmp_path,size):
    def forbidden(*a):raise AssertionError('Assets/model must not load for noncompliant official settings')
    monkeypatch.setattr(runtime,'validate_assets',forbidden)
    with pytest.raises(ValueError,match='microbatch8'):
        runtime.launch(tmp_path,tmp_path/'run',batch_options=BatchOptions(batch_size=size))

@pytest.mark.parametrize('progress',[
    {'stage':2,'stage_update':2250,'global_update':4500,'completed_examples':72000},
    {'stage':1,'stage_update':4500,'global_update':4500,'completed_examples':72000},
    {'stage':3,'stage_update':750,'global_update':4500,'completed_examples':71984},
])
def test_semantically_invalid_checkpoint_rejected_before_model(tmp_path,monkeypatch,progress):
    assets={'identities':{}};spec=runtime.make_spec(tmp_path,assets)
    publish(tmp_path/'checkpoint',{'spec':spec,'progress':progress,'kind':'model_export','model':{},'backbone_hash':'fixture'})
    monkeypatch.setattr(runtime,'validate_assets',lambda root:assets)
    def forbidden(*a):raise AssertionError('Invalid stage/cursor must be rejected before model loading')
    monkeypatch.setattr(runtime,'load_vlm',forbidden)
    with pytest.raises(ValueError,match='progress'):
        runtime.load_evaluation_model(tmp_path,tmp_path/'checkpoint',final=True)

def test_official_micro8_oom_does_not_fallback_to_noncompliant_size():
    from atml_pa3.task2.batching import probe_candidates
    calls=[]
    def oom(n):calls.append(n);return {'status':'oom'}
    with pytest.raises(RuntimeError,match='No tested microbatch fits'):
        probe_candidates(BatchOptions(batch_size='8'),oom,locked=8)
    assert calls==[8]

def test_final_batch8_and_token16_protocol_on_multiple_batches(tmp_path):
    from .test_evaluation import Backend,pairs
    from atml_pa3.task2.evaluation import evaluate_final
    class Capture(Backend):
        sizes=[]
        def generate(self,records,**kwargs):
            self.sizes.append(len(records));return super().generate(records,**kwargs)
    source=pairs();source+= [{**p,'pair_id':str(i+3),'members':[{**m,'image_id':m['image_id']+6} for m in p['members']]} for i,p in enumerate(pairs())]
    backend=Capture();evaluate_final(backend,{'grounding':source},tmp_path,model_identity='test',dataset_identity='test',resamples=20)
    assert backend.sizes==[8,4]*4

@pytest.mark.parametrize('completed,stage,within',[(0,1,0),(250,1,250),(2250,1,2250),(2251,2,1),(3750,2,1500),(3751,3,1),(4500,3,750)])
def test_real_budget_checkpoint_counter_boundaries(completed,stage,within):
    Recipe().validate_progress({'global_update':completed,'stage':stage,'stage_update':within,'completed_examples':completed*16})

@pytest.mark.parametrize('recipe,recorded',[
    (replace(Recipe(),microbatch=16,accumulation=1),Recipe().serial()),
    (Recipe(),replace(Recipe(),microbatch=16,accumulation=1).serial()),
])
def test_official_controller_cannot_disagree_with_recorded_recipe(tmp_path,monkeypatch,recipe,recorded):
    from .test_training import components
    from atml_pa3.task2 import training
    model,sampler,loader=components()
    def forbidden(*a):raise AssertionError('Invalid official recipe must fail before optimizer setup')
    monkeypatch.setattr(training,'make_stage_optimizer',forbidden)
    with pytest.raises(ValueError,match='(microbatch8|recipe)'):
        training.run_curriculum(model,sampler,loader,tmp_path/'run',{'task':'task2','recipe':recorded},recipe=recipe,stop_after=250)
    assert not (tmp_path/'run').exists()
