import copy
import json
import numpy as np
import pytest
import torch
from atml_pa3.task2.metrics import vqa_score,summarize_suite,score_record
from atml_pa3.task2.evaluation import condition_records,mismatched_mapping,evaluate_final,ModelEvaluator
from atml_pa3.task2.state import preserve_rng
from .test_model import fixture


@pytest.mark.parametrize("matches,expected",[(0,0.),(1,.3),(2,.6),(3,.9),(4,1.),(10,1.)])
def test_vqa_formula_independent_known_scores(matches,expected):
    assert vqa_score("two",[{"answer":"2"}]*matches+[{"answer":"3"}]*(10-matches))==pytest.approx(expected)
    assert vqa_score("",[{"answer":""}]*10)==0.


def pairs():
    return [{"pair_id":str(i),"kind":"presence","members":[
        {"image_id":2*i,"question":"Q","target":"yes","canonical_answer":"yes"},
        {"image_id":2*i+1,"question":"Q","target":"no","canonical_answer":"no"}]} for i in range(3)]


class Backend:
    def __init__(self,fail=False): self.calls=0;self.fail=fail
    def reference(self,records,**kwargs):
        return [1. if r["condition"]=="correct" else 2. for r in records]
    def generate(self,records,**kwargs):
        self.calls+=1
        if self.fail and self.calls==2: raise RuntimeError("interrupted")
        assert kwargs["max_new_tokens"]==16
        return [{"raw_response":r["original_target"],"prediction_text":r["original_target"],
                 "generated_token_ids":[1],"generated_tokens":1,"reached_token_cap":False,
                 "generation_seconds":.01} for r in records]


def test_four_conditions_keep_original_references_and_frozen_mapping():
    p=pairs(); mapping=mismatched_mapping(p)
    assert mapping==mismatched_mapping(p)
    before=copy.deepcopy(p)
    for condition in ("correct","complementary","mismatched","no_image"):
        records=condition_records(p,"grounding",condition,mapping)
        assert [r["original_target"] for r in records]==["yes","no"]*3
        if condition=="no_image": assert all(r["actual_image_id"] is None for r in records)
        if condition=="mismatched":
            assert all(r["actual_image_id"] not in {2*int(r['pair_id']),2*int(r['pair_id'])+1} for r in records)
    assert p==before


def test_resumable_evaluation_never_marks_partial_complete(tmp_path):
    backend=Backend(fail=True)
    with pytest.raises(RuntimeError,match="interrupted"):
        evaluate_final(backend,{"grounding":pairs()},tmp_path,model_identity="toy",dataset_identity="toy",resamples=30)
    assert read(tmp_path/"manifest.json")["complete"] is False
    backend2=Backend()
    result=evaluate_final(backend2,{"grounding":pairs()},tmp_path,model_identity="toy",dataset_identity="toy",resamples=30)
    assert backend2.calls==3  # first condition's completed batch reused
    assert read(tmp_path/"manifest.json")["complete"]
    assert result["grounding"]["questions"]==6
    assert result["grounding"]["paired_differences"]["nll_advantage_complementary_minus_correct"]["mean"]==1.
    with pytest.raises(ValueError,match="identity"):
        evaluate_final(Backend(),{"grounding":pairs()},tmp_path,model_identity="different",dataset_identity="toy")


def read(path): return json.loads(path.read_text())


def test_complete_pair_bootstrap_and_paired_difference(tmp_path):
    evaluate_final(Backend(),{"grounding":pairs()},tmp_path,model_identity="toy",dataset_identity="toy",resamples=100)
    records=read(tmp_path/"grounding/records.json")
    # Within each pair both scores move together. Independent-example
    # bootstrap would produce different quantiles on this fixture.
    for r in records:
        r["score"]=float(int(r["pair_id"])>0)
        r["canonical_correct"]=bool(r["score"])
        r["reference_nll"]=int(r["pair_id"])+(0.5 if r["condition"]=="complementary" else 0.)
    summary=summarize_suite(records,seed=12,resamples=4000)
    indices=np.random.default_rng(12).integers(3,size=(4000,3))
    expected=np.quantile(np.array([0.,1.,1.])[indices].mean(1),[.025,.975])
    np.testing.assert_array_equal(summary["conditions"]["correct"]["metrics"]["accuracy"]["ci95"],expected)
    assert summary["paired_differences"]["nll_advantage_complementary_minus_correct"]["ci95"]==[.5,.5]
    with pytest.raises(ValueError,match="complete pairs"):
        summarize_suite(records[1:])


def test_empty_output_remains_in_denominators(tmp_path):
    evaluate_final(Backend(),{"grounding":pairs()},tmp_path,model_identity="toy",dataset_identity="toy",resamples=20)
    records=read(tmp_path/"grounding/records.json")
    for record in records:
        record.update(score_record(record,""))
    summary=summarize_suite(records,resamples=20)
    metrics=summary["conditions"]["correct"]["metrics"]
    assert metrics["accuracy"]["mean"]==0 and metrics["pair_accuracy"]["mean"]==0
    assert metrics["empty_answer_rate"]["mean"]==1 and metrics["identical_answer_rate"]["mean"]==1


def test_reference_nll_excludes_eot_and_weights_examples():
    model,tokenizer=fixture()
    class Store:
        def get(self,key): return torch.ones(4,3)
    backend=ModelEvaluator(model,tokenizer,Store())
    records=[{"question":"Q","original_target":"a","actual_image_id":1},
             {"question":"Q","original_target":"bbb","actual_image_id":None}]
    values=backend.reference(records)
    from atml_pa3.task2.model import encode_example
    enc=[encode_example(tokenizer,r["question"],r["original_target"]) for r in records]
    batch=model.build_inputs(enc,[torch.ones(4,3),None])
    logits=model.language_model(inputs_embeds=batch["inputs_embeds"]).logits
    expected=[]
    for i in range(2):
        labels=batch["labels"][i,1:]; mask=(labels!=-100)&(labels!=1)
        expected.append(torch.nn.functional.cross_entropy(logits[i,:-1][mask],labels[mask]).item())
    assert values==pytest.approx(expected)


def test_evaluation_rng_context_restores_training_randomness():
    torch.manual_seed(12); np.random.seed(12)
    expected=torch.rand(3);expected_np=np.random.rand(3)
    torch.manual_seed(12); np.random.seed(12)
    model,_=fixture(); torch.manual_seed(12)
    with preserve_rng(model):
        model.eval(); torch.rand(50);np.random.rand(50)
    assert model.training
    torch.testing.assert_close(torch.rand(3),expected,rtol=0,atol=0)
    np.testing.assert_array_equal(np.random.rand(3),expected_np)


def test_greedy_batch_handles_eos_cap_and_no_image():
    model,tokenizer=fixture()
    tokenizer.eos_token_id=1
    tokenizer.decode=lambda tokens,**kw: "" if kw["skip_special_tokens"] and tokens==[1] else str(tokens)
    with torch.no_grad():
        model.language_model.head.weight.zero_()
        model.language_model.head.bias.zero_()
        model.language_model.head.bias[1]=1
    class Store:
        def get(self,key): return torch.ones(4,3)
    rows=[{"question":"long Q","actual_image_id":1},{"question":"Q","actual_image_id":None}]
    backend=ModelEvaluator(model,tokenizer,Store())
    result=backend.generate(rows)
    assert all(r["generated_token_ids"]==[1] and not r["reached_token_cap"] for r in result)
    with torch.no_grad(): model.language_model.head.bias[2]=2
    result=backend.generate(rows)
    assert all(r["generated_token_ids"]==[2]*16 and r["reached_token_cap"] for r in result)


def test_uncommitted_batch_is_recomputed(tmp_path):
    from atml_pa3.task2.evaluation import _persist_batches
    rows=condition_records(pairs(),"grounding","correct")
    spec={"model_identity":"toy"}
    _persist_batches(tmp_path,rows,Backend(),spec=spec,generate=True)
    (tmp_path/"batch_00000.sha256").unlink()
    backend=Backend()
    _persist_batches(tmp_path,rows,backend,spec=spec,generate=True)
    assert backend.calls==1
