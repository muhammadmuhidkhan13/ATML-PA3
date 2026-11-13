import copy
import json
import pytest
import torch
from atml_pa3.task2 import runtime
from atml_pa3.task2.adapters import backbone_hash
from atml_pa3.task2.state import model_state
from atml_pa3.task2.io import atomic_json
from atml_pa3.task2.evaluation import evaluate_final
from atml_pa3.task2.plotting import export_artifacts
from .test_evaluation import Backend,pairs
from .test_training import components,attach_toy
from atml_pa3.task2.result_validation import verify_final,payload_hashes


@pytest.mark.parametrize('stage',[1,2,3])
def test_evaluation_loader_rejects_reconstructed_backbone_mismatch(monkeypatch,stage):
    model,_,_=components()
    if stage>=2:attach_toy(model)
    payload={'progress':{'stage':stage,'global_update':4500},'model':model_state(model),
             'backbone_hash':backbone_hash(model)}
    monkeypatch.setattr(runtime,'validate_assets',lambda *a:{})
    monkeypatch.setattr(runtime,'make_spec',lambda *a:{})
    monkeypatch.setattr(runtime,'read_payload',lambda *a:payload)
    monkeypatch.setattr(runtime,'load_vlm',lambda *a:(model,None))
    monkeypatch.setattr(runtime,'attach_attention_lora',attach_toy)
    assert runtime.load_evaluation_model('unused','unused',device='cpu')[0] is model
    with torch.no_grad():model.language_model.embedding.weight[0,0]+=1
    with pytest.raises(ValueError,match='backbone'):
        runtime.load_evaluation_model('unused','unused',device='cpu')


@pytest.mark.parametrize('target',['summary','records','raw','mapping','missing','duplicate'])
def test_plot_rejects_corrupt_complete_evaluation(tmp_path,target):
    destination=tmp_path/'final'
    evaluate_final(Backend(),{'grounding':pairs()},destination,model_identity='toy',dataset_identity='toy',resamples=20)
    if target=='summary':
        path=destination/'summary.json';value=json.loads(path.read_text());value['grounding']['questions']=999
    elif target in ('records','duplicate'):
        path=destination/'grounding/records.json';value=json.loads(path.read_text())
        if target=='duplicate':value.append(copy.deepcopy(value[0]))
        else:value[0]['reference_nll']=999.
    elif target=='raw':
        path=destination/'grounding/correct/batch_00000.json';value=json.loads(path.read_text());value['records'][0]['score']=999
    elif target=='mapping':
        path=destination/'mismatched_images.json';value=json.loads(path.read_text());value['grounding']['0:0']=0
    else:
        (destination/'grounding/no_image/batch_00000.json').unlink();path=None
    if path is not None:atomic_json(path,value)
    with pytest.raises((ValueError,FileNotFoundError),match='.'):
        export_artifacts(tmp_path/'plots',evaluation_dir=destination)


@pytest.mark.parametrize('target',['summary','records'])
def test_recomputation_rejects_inconsistent_derived_files_even_with_updated_hashes(tmp_path,target):
    evaluate_final(Backend(),{'grounding':pairs()},tmp_path,model_identity='toy',dataset_identity='toy',resamples=20)
    path=tmp_path/('summary.json' if target=='summary' else 'grounding/records.json')
    value=json.loads(path.read_text())
    if target=='summary':value['grounding']['questions']=999
    else:value[0]['reference_nll']=999.
    atomic_json(path,value)
    header=json.loads((tmp_path/'manifest.json').read_text());header['files']=payload_hashes(tmp_path)
    header['summary_sha256']=header['files']['summary.json']
    atomic_json(tmp_path/'manifest.json',header)
    with pytest.raises(ValueError,match='recomputation|records differ'):verify_final(tmp_path)


def test_completed_validation_is_read_only_and_never_regenerates(tmp_path):
    expected={'grounding':pairs()}
    evaluate_final(Backend(),expected,tmp_path,model_identity='toy',dataset_identity='toy',resamples=20)
    before=payload_hashes(tmp_path)
    assert verify_final(tmp_path,expected_suites=expected,model_identity='toy')['inference_performed'] is False
    backend=Backend()
    evaluate_final(backend,expected,tmp_path,model_identity='toy',dataset_identity='toy',resamples=20)
    assert backend.calls==0 and payload_hashes(tmp_path)==before
    with pytest.raises(ValueError,match='Checkpoint'):verify_final(tmp_path,model_identity='different')
    with pytest.raises(ValueError,match='suite/pair'):verify_final(tmp_path,expected_suites={'grounding':pairs()[:2]})


def test_development_derived_integrity_and_read_only_reuse(tmp_path,monkeypatch):
    from atml_pa3.task2 import evaluation
    from atml_pa3.task2.result_validation import verify_development
    from .test_model import fixture
    data=tmp_path/'data';data.mkdir();output=tmp_path/'dev'
    atomic_json(data/'manifest.json',{'fixture':True})
    (data/'captions_dev.jsonl').write_text(json.dumps({'image_id':0,'question':'Q','target':'caption','caption_id':1})+'\n')
    for suite in ('grounding','vqa'):
        (data/f'{suite}_dev.jsonl').write_text('\n'.join(json.dumps(p) for p in pairs())+'\n')
    class ReferenceOnly:
        calls=0
        def __init__(self,*a):pass
        def reference(self,rows,**kwargs):
            ReferenceOnly.calls+=1
            return [2. if r.get('condition')=='complementary' else 1. for r in rows]
    monkeypatch.setattr(evaluation,'ModelEvaluator',ReferenceOnly)
    model,_=fixture()
    evaluation.evaluate_development(model,None,None,data,output,model_identity='toy')
    assert verify_development(output)['status']=='passed'
    before=ReferenceOnly.calls
    evaluation.evaluate_development(model,None,None,data,output,model_identity='toy')
    assert ReferenceOnly.calls==before
    value=json.loads((output/'summary.json').read_text());value['caption_reference_loss']=999.
    atomic_json(output/'summary.json',value)
    with pytest.raises(ValueError,match='integrity'):verify_development(output)


def test_independent_teacher_reference_matches_reduction_without_production_loss_helper():
    from .test_model import fixture
    from atml_pa3.task2.audit_integration import _teacher_reference
    from atml_pa3.task2.evaluation import ModelEvaluator
    model,tokenizer=fixture();model.eval()
    class Store:
        def get(self,key):return torch.ones(4,3)
    rows=[{'question':'Q','original_target':'a','actual_image_id':1},
          {'question':'long Q','original_target':'bbb','actual_image_id':None}]
    expected=_teacher_reference(model,tokenizer,Store(),rows,amp=False)
    assert ModelEvaluator(model,tokenizer,Store()).reference(rows)==pytest.approx(expected)


def test_bounded_smoke_cli_needs_explicit_opt_in(monkeypatch,capsys,tmp_path):
    import runpy,sys
    from pathlib import Path
    from atml_pa3.task2 import audit_integration
    def forbidden(*a,**kw):raise AssertionError('No GPU work in default mode')
    monkeypatch.setattr(audit_integration,'optimizer_smoke',forbidden)
    monkeypatch.setattr(audit_integration,'evaluation_equivalence',forbidden)
    monkeypatch.setattr(sys,'argv',['task2_audit_integration.py','--mode','optimizer','--output',str(tmp_path/'smoke')])
    runpy.run_path(str(Path(__file__).resolve().parents[2]/'scripts/task2_audit_integration.py'),run_name='__main__')
    assert json.loads(capsys.readouterr().out)['smoke_started'] is False
    assert not (tmp_path/'smoke').exists()


def test_invalid_backend_output_cannot_publish_complete_marker(tmp_path):
    class InvalidBackend(Backend):
        def generate(self,records,**kwargs):
            values=super().generate(records,**kwargs)
            for value in values:value['generated_tokens']=0
            return values
    with pytest.raises(ValueError,match='token counts'):
        evaluate_final(InvalidBackend(),{'grounding':pairs()},tmp_path,model_identity='toy',dataset_identity='toy',resamples=20)
    assert json.loads((tmp_path/'manifest.json').read_text())['complete'] is False
