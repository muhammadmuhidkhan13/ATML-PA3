"""Non-official toy recovery and complete real-asset/schedule verification."""
import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from .adapters import attach_attention_lora
from .curriculum import Recipe, development_updates, lr_factor
from .model import VisualLanguageModel, EncodedExample
from .runtime import validate_assets, real_sampler, make_spec
from .sampling import CurriculumSampler
from .training import run_curriculum
from .metrics import vqa_score
from .io import atomic_json


class TinyValidationLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Embedding(16,8)
        self.head=nn.Linear(8,16)
        self.model=nn.Module()
        layer=nn.Module();layer.self_attn=nn.Module()
        for name in ("q_proj","k_proj","v_proj","o_proj"):
            setattr(layer.self_attn,name,nn.Linear(8,8,bias=False))
        self.model.layers=nn.ModuleList([layer])
    def get_input_embeddings(self): return self.embed
    def forward(self,inputs_embeds,**kwargs):
        x=inputs_embeds.cumsum(1)/10
        a=self.model.layers[0].self_attn
        x=a.o_proj(torch.tanh(a.q_proj(x)+a.k_proj(x)+a.v_proj(x)))
        return SimpleNamespace(logits=self.head(x))


def toy_components():
    torch.manual_seed(6304)
    model=VisualLanguageModel(TinyValidationLM().half(),vision_size=3,hidden_size=8,visual_tokens=4)
    captions=[{"image_id":i,"caption_id":i,"question":"q","target":"a"} for i in range(16)]
    natural=[{"pair_id":"toy","members":[{"image_id":1,"question":"q","target":"a"},
                                            {"image_id":2,"question":"q","target":"b"}]}]
    pools={"image_ids":[0,1,2,3],"categories":[{"category_id":1,"object":"cat","plural":"cats",
            "positive":[1,2,3],"counts":{"1":[1],"2":[2],"3":[3],"4":[]}}]}
    sampler=CurriculumSampler(captions,natural,pools)
    def loader(records):
        examples=[EncodedExample([2,3,4],[-100,3,4],1,1,1,False,False) for _ in records]
        features=[torch.full((4,3),float(r["image_id"]+1)/10) for r in records]
        return examples,features
    return model,sampler,loader


def toy_recovery_check(output):
    output=Path(output)
    recipe=Recipe(budgets=(2,2,2),checkpoint_interval=1)
    attach=lambda model:attach_attention_lora(model,expected_layers=1)
    model,sampler,loader=toy_components()
    full=run_curriculum(model,sampler,loader,output/"continuous",{"toy_only":True},recipe=recipe,attach=attach)
    expected=copy.deepcopy(model.state_dict())
    partial,sampler2,loader2=toy_components()
    run_curriculum(partial,sampler2,loader2,output/"resumed",{"toy_only":True},recipe=recipe,attach=attach,stop_after=3)
    restored,sampler3,loader3=toy_components()
    run_curriculum(restored,sampler3,loader3,output/"resumed",{"toy_only":True},recipe=recipe,attach=attach,
                   resume_from=output/"resumed/checkpoints/update_00003")
    for key,value in expected.items():
        torch.testing.assert_close(restored.state_dict()[key],value,rtol=0,atol=0)
    if sampler.state_dict()!=sampler3.state_dict():
        raise AssertionError("Toy sampler continuation differs")
    return {"scope":"tiny synthetic CPU models only","exact_resume":True,"toy_updates_per_run":6,
            "official_optimizer_updates":0,"completed_examples":full["completed_examples"]}


def verify(root,output,*,simulate=True,toy=True):
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    assets=validate_assets(root)
    summary={"status":"passed","assets":assets,"official_training_started":False,"official_optimizer_updates":0,
             "warmups":[__import__('math').ceil(.05*n) for n in Recipe().budgets],
             "development_updates":list(development_updates()),"recipe":Recipe().serial(),"spec":make_spec(root,assets)}
    if simulate:
        sampler=real_sampler(root);stage=1
        caption_ids=[]
        for completed in range(Recipe().total_updates):
            if completed in (2250,3750):
                stage+=1;sampler.set_stage(stage)
            records=sampler.next_batch()
            if stage==1: caption_ids.extend(r["caption_id"] for r in records)
            if len(records)!=16 or any(records[i]["draw_id"]!=records[i+1]["draw_id"] for i in range(0,16,2)):
                raise AssertionError("Simulation split a pair")
        expected=[r["caption_id"] for r in sampler.captions]*2
        if caption_ids!=expected or sampler.completed_examples!=72000:
            raise AssertionError("Full schedule data budget/traversal differs")
        summary["simulation"]={"official_optimizer_updates":0,"example_presentations":72000,
                               "stage1_exact_two_passes":True,"mixture_examples":dict(sampler.mixture),
                               "draw_position":sampler.draw_position}
    if toy: summary["toy_recovery"]=toy_recovery_check(output/"toy")
    from .runtime import launch
    validation = launch(root,output/"validation_only_run")
    if validation["training_started"] or (output/"validation_only_run").exists():
        raise AssertionError("Default invocation crossed the training boundary")
    summary["default_launcher_training_started"] = False
    known=[vqa_score("2",[{"answer":"two"}]*k+[{"answer":"3"}]*(10-k)) for k in (0,1,2,3,4)]
    if not np.allclose(known,[0,.3,.6,.9,1.]): raise AssertionError("VQA self-check failed")
    summary["known_vqa_scores"]=known
    atomic_json(output/"result.json",summary)
    return summary
