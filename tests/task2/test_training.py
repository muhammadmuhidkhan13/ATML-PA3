import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from atml_pa3.task2.model import VisualLanguageModel, encode_example
from atml_pa3.task2.adapters import attach_attention_lora, make_stage_optimizer, backbone_hash, AttentionLoRA
from atml_pa3.task2.curriculum import Recipe, lr_factor, development_updates
from atml_pa3.task2.sampling import CurriculumSampler
from atml_pa3.task2.training import run_curriculum, attempt_update
from atml_pa3.task2.state import read_payload
from .test_model import Tokenizer


class ToyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Embedding(256,8)
        self.model=nn.Module()
        layer=nn.Module(); layer.self_attn=nn.Module()
        for name in ("q_proj","k_proj","v_proj","o_proj"):
            setattr(layer.self_attn,name,nn.Linear(8,8,bias=False))
        self.model.layers=nn.ModuleList([layer])
        self.head=nn.Linear(8,256)
    def get_input_embeddings(self):
        return self.embedding
    def forward(self,inputs_embeds,**kwargs):
        x=inputs_embeds.cumsum(1)/10
        for layer in self.model.layers:
            a=layer.self_attn
            x=a.o_proj(torch.tanh(a.q_proj(x)+a.k_proj(x)+a.v_proj(x)))
        return SimpleNamespace(logits=self.head(x))


def components():
    torch.manual_seed(6304)
    model=VisualLanguageModel(ToyLM().half(),vision_size=3,hidden_size=8,visual_tokens=4)
    captions=[{"image_id":i,"caption_id":i,"question":"Q","target":"a"*(i%3+1)} for i in range(16)]
    natural=[{"pair_id":"n:1","members":[{"image_id":20,"question_id":1,"question":"Q","target":"a"},
                                           {"image_id":21,"question_id":2,"question":"Q","target":"b"}]}]
    pools={"image_ids":list(range(40)),"categories":[{"category_id":1,"object":"cat","plural":"cats",
            "positive":list(range(1,40)),"counts":{str(i):list(range(i,40,4)) for i in range(1,5)}}]}
    sampler=CurriculumSampler(captions,natural,pools)
    tokenizer=Tokenizer()
    generator=torch.Generator().manual_seed(123)
    features={i:torch.randn(4,3,generator=generator) for i in range(40)}
    def loader(rows):
        return ([encode_example(tokenizer,r["question"],r["target"],caption=r["caption"]) for r in rows],
                [features[r["image_id"]] for r in rows])
    return model,sampler,loader


def attach_toy(model):
    attach_attention_lora(model,expected_layers=1)


@pytest.mark.parametrize("stop",[1,2,3,4,5])
def test_exact_resume_inside_and_between_all_stages(tmp_path,stop):
    recipe=Recipe(budgets=(2,2,2),checkpoint_interval=1)
    model,sampler,loader=components()
    before=backbone_hash(model)
    full=run_curriculum(model,sampler,loader,tmp_path/"full",{"toy":True},recipe=recipe,attach=attach_toy)
    expected=copy.deepcopy(model.state_dict()); expected_sampler=sampler.state_dict()
    model2,sampler2,loader2=components()
    run_curriculum(model2,sampler2,loader2,tmp_path/"partial",{"toy":True},recipe=recipe,
                   attach=attach_toy,stop_after=stop)
    fresh,sampler3,loader3=components()
    torch.manual_seed(99); np.random.seed(99)
    final=run_curriculum(fresh,sampler3,loader3,tmp_path/"partial",{"toy":True},recipe=recipe,
                        attach=attach_toy,resume_from=tmp_path/"partial/checkpoints"/f"update_{stop:05d}")
    for key,value in expected.items():
        torch.testing.assert_close(fresh.state_dict()[key],value,rtol=0,atol=0)
    assert final["global_update"]==6 and final["completed_examples"]==96
    assert sampler3.state_dict()==expected_sampler
    assert backbone_hash(fresh)==before
    def samples(path):
        return [r["sample_ids"] for r in map(json.loads,path.read_text().splitlines()) if r["event"]=="completed_update"]
    assert samples(tmp_path/"full/updates.jsonl")==samples(tmp_path/"partial/updates.jsonl")
    for stage in (1,2,3):
        payload=read_payload(tmp_path/"partial/exports"/f"stage_{stage}")
        assert payload["kind"]=="model_export"
        assert payload["progress"]["global_update"]==2*stage


def test_stage1_exact_two_passes_and_pair_sampling_state():
    _,sampler,_=components()
    records=sampler.next_batch(32)
    assert [r["caption_id"] for r in records]==list(range(16))*2
    sampler.set_stage(2)
    state=sampler.state_dict()
    a=sampler.next_batch()
    sampler.load_state_dict(state)
    assert sampler.next_batch()==a
    assert all(a[i]["draw_id"]==a[i+1]["draw_id"] for i in range(0,16,2))
    assert {r["sample_kind"] for r in a} <= {"caption","presence","counting"}
    sampler.set_stage(3)
    assert {r["sample_kind"] for r in sampler.next_batch(200)} <= {"caption","presence","counting","natural"}


@pytest.mark.parametrize("stage,probabilities",[(2,{"presence":.5,"counting":.25,"caption":.25}),
                                               (3,{"natural":.7,"presence":.1,"counting":.1,"caption":.1})])
def test_seeded_mixture_probabilities(stage,probabilities):
    from collections import Counter
    _,sampler,_=components()
    for next_stage in range(2,stage+1):sampler.set_stage(next_stage)
    rows=sampler.next_batch(40000)
    counts=Counter(r["sample_kind"] for r in rows[::2])
    assert set(counts)==set(probabilities)
    for kind,probability in probabilities.items():
        assert abs(counts[kind]/20000-probability)<.015


def test_failed_checkpoint_publication_never_creates_complete_destination(tmp_path,monkeypatch):
    from atml_pa3.task2.state import publish
    def fail(*a,**k):raise OSError("interrupted save")
    monkeypatch.setattr(torch,"save",fail)
    with pytest.raises(OSError,match="interrupted"):
        publish(tmp_path/"checkpoint",{})
    assert not (tmp_path/"checkpoint").exists()
    assert not list(tmp_path.glob(".incomplete-*"))


def test_example_weighted_accumulation_matches_single_batch():
    model,sampler,loader=components(); reference=copy.deepcopy(model)
    records=sampler.next_batch()
    recipe=Recipe(budgets=(2,2,2),checkpoint_interval=1)
    opt,sched=make_stage_optimizer(model,1,recipe)
    refopt,_=make_stage_optimizer(reference,1,recipe)
    result=attempt_update(model,records,opt,sched,torch.amp.GradScaler("cuda",enabled=False),loader,recipe)
    enc,features=loader(records)
    loss=reference(enc,features)["per_example_loss"].mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(reference.connector.parameters(),1.)
    refopt.step()
    assert result["sft_loss"]==pytest.approx(loss.item(),rel=1e-6)
    for a,b in zip(model.connector.parameters(),reference.connector.parameters()):
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)


class SkipOnceScaler:
    def __init__(self): self.attempts=0; self.scale_value=2.
    def scale(self,loss): return loss
    def unscale_(self,opt): pass
    def step(self,opt):
        self.attempts+=1
        if self.attempts>1: opt.step()
    def update(self):
        if self.attempts==1: self.scale_value/=2
    def get_scale(self): return self.scale_value
    def is_enabled(self): return True
    def state_dict(self): return {"attempts":self.attempts,"scale":self.scale_value}


def test_skipped_update_reuses_examples_and_scheduler(tmp_path):
    model,sampler,loader=components()
    calls=[]
    def tracked(rows):
        calls.append([r["caption_id"] for r in rows])
        return loader(rows)
    result=run_curriculum(model,sampler,tracked,tmp_path/"run",{"toy":True},
                           recipe=Recipe(budgets=(2,2,2),checkpoint_interval=1),stop_after=1,
                           attach=attach_toy,scaler=SkipOnceScaler())
    assert calls[:2]==calls[2:]
    assert result["skipped_attempts"]==1 and sampler.completed_examples==16
    payload=read_payload(tmp_path/"run/checkpoints/update_00001")
    assert payload["scheduler"]["last_epoch"]==1
    assert payload["scaler"]["attempts"]==2


def test_schedule_endpoints_and_development_events():
    recipe=Recipe()
    assert recipe.boundaries==(2250,3750,4500)
    assert [__import__('math').ceil(.05*n) for n in recipe.budgets]==[113,75,38]
    for n in recipe.budgets:
        warm=__import__('math').ceil(.05*n)
        assert lr_factor(warm,n)==1.
        assert lr_factor(n,n)==pytest.approx(.1)
    assert development_updates()==(0,500,1000,1500,2000,2250,2500,3000,3500,3750,4000,4500)


def test_checkpoint_integrity_and_overwrite(tmp_path):
    model,sampler,loader=components()
    recipe=Recipe(budgets=(2,2,2),checkpoint_interval=1)
    run_curriculum(model,sampler,loader,tmp_path/"run",{"toy":True},recipe=recipe,stop_after=1)
    checkpoint=tmp_path/"run/checkpoints/update_00001"
    from atml_pa3.task2.state import publish
    with pytest.raises(FileExistsError):
        publish(checkpoint,read_payload(checkpoint))
    with pytest.raises(ValueError,match="configuration"):
        read_payload(checkpoint,{"toy":False})
    with (checkpoint/"state.pt").open("ab") as f: f.write(b"bad")
    with pytest.raises(ValueError,match="integrity"):
        read_payload(checkpoint)


def test_attention_only_lora_preserves_connector_and_is_idempotent():
    model,_,_=components(); before=copy.deepcopy(model.connector.state_dict())
    attach_toy(model)
    assert len([m for m in model.modules() if isinstance(m,AttentionLoRA)])==4
    ids=[id(p) for p in model.parameters()]
    attach_toy(model)
    assert [id(p) for p in model.parameters()]==ids
    for k,v in before.items(): torch.testing.assert_close(model.connector.state_dict()[k],v,rtol=0,atol=0)


def test_development_journal_deduplicates_resume_and_preserves_rng(tmp_path):
    recipe=Recipe(budgets=(2,2,2),checkpoint_interval=1)
    def evaluate(model,update,stage):
        model.eval(); torch.rand(30); np.random.rand(30)
    model,sampler,loader=components()
    run_curriculum(model,sampler,loader,tmp_path/"run",{"toy":True},recipe=recipe,
                   attach=attach_toy,evaluate=evaluate,eval_updates=(0,2,4,6),stop_after=2)
    model,sampler,loader=components()
    run_curriculum(model,sampler,loader,tmp_path/"run",{"toy":True},recipe=recipe,
                   attach=attach_toy,evaluate=evaluate,eval_updates=(0,2,4,6),
                   resume_from=tmp_path/"run/checkpoints/update_00002")
    reference,rs,rl=components()
    run_curriculum(reference,rs,rl,tmp_path/"ref",{"toy":True},recipe=recipe,attach=attach_toy)
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,reference.state_dict()[key],rtol=0,atol=0)
    events=[json.loads(line) for line in (tmp_path/"run/updates.jsonl").read_text().splitlines()]
    assert [r["global_update"] for r in events if r["event"]=="development"]==[0,2,4,6]
