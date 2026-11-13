"""Bounded, explicitly non-official real-model optimizer and decoder audit."""
import copy
from dataclasses import replace
import gc
from pathlib import Path
import shutil
import time
import torch
from .adapters import attach_attention_lora,backbone_hash,make_stage_optimizer
from .curriculum import Recipe
from .evaluation import ModelEvaluator,condition_records,mismatched_mapping
from .features import FeatureStore
from .io import atomic_json,read_jsonl,sha256
from .model import load_vlm,encode_example
from .runtime import validate_assets,restore_initialization,asset_paths,real_sampler,batch_loader,make_spec
from .state import read_payload,model_state,preserve_rng
from .training import run_curriculum,attempt_update


def _release():
    gc.collect();torch.cuda.empty_cache()


def _compare(left,right,*,rtol=1e-5,atol=2e-6):
    if torch.is_tensor(left):
        torch.testing.assert_close(left,right,rtol=rtol,atol=atol)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:_compare(left[key],right[key],rtol=rtol,atol=atol)
    elif isinstance(left,(tuple,list)):
        assert len(left)==len(right)
        for a,b in zip(left,right):_compare(a,b,rtol=rtol,atol=atol)
    else:assert left==right,(left,right)


def optimizer_smoke(root,output,*,microbatch=1,device='cuda:0'):
    """Six synthetic-budget real-data updates plus bounded recovery branches.

    Uses production optimizer/controller/checkpoint code; never the official
    launcher or 4500-update configuration. Writes only inside the audit output.
    """
    root,output=Path(root),Path(output)
    if output.exists():raise FileExistsError(output)
    if microbatch not in (1,2,8):raise ValueError('Bounded microbatch must be 1,2,8')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for real optimizer audit')
    assets=validate_assets(root);output.mkdir(parents=True)
    started=time.perf_counter();torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    recipe=Recipe(budgets=(2,2,2),microbatch=microbatch,accumulation=2,checkpoint_interval=1)
    spec={'scope':'NON-OFFICIAL bounded real-model audit','official_optimizer_updates':0,
          'recipe':recipe.serial(),'production_spec':make_spec(root,assets)}
    store=FeatureStore(asset_paths(root)['features'])
    def initialize():
        model,tokenizer=load_vlm(device);restore_initialization(model,root)
        return model,real_sampler(root),batch_loader(tokenizer,store)
    model,sampler,loader=initialize();initial=model_state(model);frozen=backbone_hash(model)
    torch.cuda.reset_peak_memory_stats(device)
    progress=run_curriculum(model,sampler,loader,output/'continuous',spec,recipe=recipe)
    expected=read_payload(output/'continuous/checkpoints/update_00006')
    assert expected['backbone_hash']==frozen and backbone_hash(model)==frozen
    changed=[];previous=initial
    for stage,update in ((1,2),(2,4),(3,6)):
        payload=read_payload(output/'continuous/checkpoints'/f'update_{update:05d}')
        current=payload['model']
        connector=any(not torch.equal(value,previous[name]) for name,value in current.items() if name.startswith('connector.'))
        lora=any(name.endswith('lora_B') and (bool(torch.count_nonzero(value)) if name not in previous
                                              else not torch.equal(value,previous[name])) for name,value in current.items())
        assert connector and (lora if stage>=2 else not lora)
        assert payload['optimizer']['state'] and payload['scaler']
        for state in payload['optimizer']['state'].values():
            assert state['exp_avg'].dtype==state['exp_avg_sq'].dtype==torch.float32
        changed.append({'stage':stage,'connector_changed':connector,'lora_B_changed':lora})
        previous=current
    peak=torch.cuda.max_memory_allocated(device)
    del model,sampler,loader;_release()
    recovery=[]
    for update in (1,2,3,4,5):
        branch=output/f'resume_{update}';checkpoint=branch/'checkpoints'/f'update_{update:05d}'
        checkpoint.parent.mkdir(parents=True)
        shutil.copytree(output/'continuous/checkpoints'/checkpoint.name,checkpoint)
        shutil.copyfile(output/'continuous/run_spec.json',branch/'run_spec.json')
        model,sampler,loader=initialize()
        run_curriculum(model,sampler,loader,branch,spec,recipe=recipe,resume_from=checkpoint)
        actual=read_payload(branch/'checkpoints/update_00006')
        # Compare all mathematical recovery state; elapsed-time counters differ.
        for key in ('model','optimizer','scheduler','scaler','sampler','rng'):_compare(expected[key],actual[key])
        assert backbone_hash(model)==frozen
        max_abs=max((actual['model'][k]-v).abs().max().item() for k,v in expected['model'].items())
        recovery.append({'resume_update':update,'all_recovery_state_close':True,'max_parameter_abs_difference':max_abs,
                         'parameters_bitwise_equal':all(torch.equal(v,actual['model'][k]) for k,v in expected['model'].items())})
        peak=max(peak,torch.cuda.max_memory_allocated(device))
        del model,sampler,loader,actual;_release()
    # Exercise genuine GradScaler overflow handling without advancing a step.
    model,sampler,loader=initialize();optimizer,scheduler=make_stage_optimizer(model,1,recipe)
    scaler=torch.amp.GradScaler('cuda',init_scale=256.)
    records=sampler.next_batch(recipe.effective_batch);before=model_state(model);epoch=scheduler.last_epoch
    handle=next(model.connector.parameters()).register_hook(lambda grad:torch.full_like(grad,float('inf')))
    skipped=attempt_update(model,records,optimizer,scheduler,scaler,loader,recipe);handle.remove()
    assert not skipped['completed'] and scheduler.last_epoch==epoch and not optimizer.state
    _compare(before,model_state(model),rtol=0,atol=0)
    for _ in range(8):
        successful=attempt_update(model,records,optimizer,scheduler,scaler,loader,recipe)
        if successful['completed']:break
    assert successful['completed'] and scheduler.last_epoch==epoch+1 and backbone_hash(model)==frozen
    result={'status':'passed','scope':'NON-OFFICIAL bounded optimizer/recovery integration',
            'official_optimizer_updates':0,'baseline_successful_smoke_updates':6,'microbatch':microbatch,
            'gpu':torch.cuda.get_device_name(device),'a100_verified':'A100' in torch.cuda.get_device_name(device),
            'stage_parameter_changes':changed,'recovery':recovery,'forced_overflow':skipped,
            'subsequent_update':successful,'peak_gpu_allocated_bytes':peak,'seconds':time.perf_counter()-started,
            'comparison_tolerance':{'rtol':1e-5,'atol':2e-6},'assets':assets['identities']}
    atomic_json(output/'result.json',result)
    del model,optimizer,scheduler,loader;_release()
    return result


@torch.no_grad()
def _reference_prompt(model,tokenizer,record):
    """Independently assemble segmented non-thinking text + projected image."""
    question='AUDIT_QUESTION_MARKER';answer='AUDIT_ANSWER_MARKER'
    template=tokenizer.apply_chat_template([{'role':'user','content':question},{'role':'assistant','content':answer}],
                                         tokenize=False,enable_thinking=False,add_generation_prompt=False)
    prefix,rest=template.split(question);middle,_=rest.split(answer)
    ids=lambda s:tokenizer.encode(s,add_special_tokens=False)
    before=ids(prefix);text=before+ids(record['question'])[:80]+ids(middle)
    embedding=model.language_model.get_input_embeddings();device=embedding.weight.device
    vector=embedding(torch.tensor(text,device=device))
    if record['actual_image_id'] is not None:
        visual=model.connector(record['_feature'].to(device=device,dtype=torch.float32))*model.visual_scale
        vector=torch.cat((vector[:len(before)],visual.to(vector.dtype),vector[len(before):]))
    return vector[None]


@torch.no_grad()
def _teacher_reference(model,tokenizer,store,rows,*,amp=True):
    """Independent teacher forcing with the same full sequence/padding shape.

    Only answer-content log probabilities are reduced. Keeping EOT/suffix input
    positions matches production GEMM dimensions without supervising termination.
    """
    vectors=[];targets=[];starts=[]
    for row in rows:
        item=dict(row)
        if item['actual_image_id'] is not None:item['_feature']=store.get(item['actual_image_id'])
        prompt=_reference_prompt(model,tokenizer,item)
        target=tokenizer.encode(row['original_target'],add_special_tokens=False)[:16]
        marker='AUDIT_TARGET_MARKER'
        template=tokenizer.apply_chat_template([{'role':'user','content':'Q'},{'role':'assistant','content':marker}],
                                             tokenize=False,enable_thinking=False,add_generation_prompt=False)
        suffix=tokenizer.encode(template.split(marker)[1],add_special_tokens=False)
        ids=torch.tensor(target+suffix,device=prompt.device)
        vectors.append(torch.cat((prompt[0],model.language_model.get_input_embeddings()(ids))))
        targets.append(target);starts.append(prompt.shape[1]-1)
    length=max(len(v) for v in vectors)
    inputs=torch.stack([torch.nn.functional.pad(v,(0,0,0,length-len(v))) for v in vectors])
    mask=torch.stack([torch.arange(length,device=inputs.device)<len(v) for v in vectors]).long()
    positions=(mask.cumsum(1)-1).clamp_min(0)
    with torch.autocast('cuda',dtype=torch.float16,enabled=amp):
        logits=model.language_model(inputs_embeds=inputs,attention_mask=mask,position_ids=positions,use_cache=False).logits
    return [torch.nn.functional.cross_entropy(logits[i,start:start+len(target)].float(),torch.tensor(target,device=inputs.device)).item()
            for i,(start,target) in enumerate(zip(starts,targets))]


@torch.no_grad()
def _verify_greedy_tokens(model,tokenizer,store,rows,outputs):
    """Teacher-force every generated prefix using independently built embeddings.

    This checks the argmax and padding index at every active decoding step at
    the prescribed batch shape, independently of production build_inputs.
    """
    prompts=[]
    for row in rows:
        item=dict(row)
        if item['actual_image_id'] is not None:item['_feature']=store.get(item['actual_image_id'])
        prompts.append(_reference_prompt(model,tokenizer,item)[0])
    for step in range(max(len(tokens) for tokens in outputs)):
        sequences=[]
        for prompt,tokens in zip(prompts,outputs):
            previous=torch.tensor(tokens[:step],device=prompt.device,dtype=torch.long)
            sequences.append(torch.cat((prompt,model.language_model.get_input_embeddings()(previous))))
        length=max(len(v) for v in sequences)
        inputs=torch.stack([torch.nn.functional.pad(v,(0,0,0,length-len(v))) for v in sequences])
        mask=torch.stack([torch.arange(length,device=inputs.device)<len(v) for v in sequences]).long()
        with torch.autocast('cuda',dtype=torch.float16):
            logits=model.language_model(inputs_embeds=inputs,attention_mask=mask,position_ids=(mask.cumsum(1)-1).clamp_min(0),use_cache=False).logits
        for i,tokens in enumerate(outputs):
            if step<len(tokens):assert logits[i,len(sequences[i])-1].argmax().item()==tokens[step],(i,step)
    return True


@torch.no_grad()
def evaluation_equivalence(root,output,*,batch_sizes=(1,2),device='cuda:0'):
    root,output=Path(root),Path(output)
    if output.exists():raise FileExistsError(output)
    if any(b not in (1,2,8) for b in batch_sizes):raise ValueError('Audit batch sizes are 1,2,8')
    assets=validate_assets(root);output.mkdir(parents=True);started=time.perf_counter()
    model,tokenizer=load_vlm(device);restore_initialization(model,root);model.eval()
    store=FeatureStore(asset_paths(root)['features']);backend=ModelEvaluator(model,tokenizer,store)
    # Training pairs only. Neither final-test questions nor final-test answers enter this audit.
    pairs=read_jsonl(asset_paths(root)['data']/'vqa_train.jsonl')[:4]
    mapping=mismatched_mapping(pairs);comparisons=[]
    configured=model.language_model.generation_config.eos_token_id
    eos=set(configured if isinstance(configured,list) else [configured])
    torch.cuda.reset_peak_memory_stats(device)
    for condition in ('correct','complementary','mismatched','no_image'):
        rows=condition_records(pairs,'natural',condition,mapping)
        rows=rows[::2]+rows[1::2]  # Different questions/lengths occur within batch2.
        independent=[];reference_nll=[]
        for row in rows[:max(batch_sizes)]:
            item=dict(row)
            if item['actual_image_id'] is not None:item['_feature']=store.get(item['actual_image_id'])
            prompt=_reference_prompt(model,tokenizer,item)
            # Check the independent segmented assembly against production input
            # construction, including exactly196 omitted no-image positions.
            encoded=encode_example(tokenizer,row['question'],'x')
            first=next(i for i,v in enumerate(encoded.labels) if v!=-100)
            production=model.build_inputs([replace(encoded,input_ids=encoded.input_ids[:first],labels=[-100]*first)],
                                          [item.get('_feature')])
            torch.testing.assert_close(prompt,production['inputs_embeds'],rtol=0,atol=0)
            assert prompt.shape[1]==first+(196 if item.get('_feature') is not None else 0)
            # Hugging Face's independent, uncached generation loop; same embeddings/settings.
            with torch.autocast('cuda',dtype=torch.float16):
                generated=model.language_model.generate(inputs_embeds=prompt,attention_mask=torch.ones(prompt.shape[:2],dtype=torch.long,device=device),
                    max_new_tokens=16,do_sample=False,use_cache=False,eos_token_id=sorted(eos),pad_token_id=tokenizer.pad_token_id)
            tokens=generated[0].tolist();independent.append(tokens)
            target=tokenizer.encode(row['original_target'],add_special_tokens=False)[:16]
            appended=model.language_model.get_input_embeddings()(torch.tensor(target,device=device))[None]
            inputs=torch.cat((prompt,appended),dim=1)
            with torch.autocast('cuda',dtype=torch.float16):
                logits=model.language_model(inputs_embeds=inputs,attention_mask=torch.ones(inputs.shape[:2],dtype=torch.long,device=device),
                    position_ids=torch.arange(inputs.shape[1],device=device)[None],use_cache=False).logits
            selected=logits[0,prompt.shape[1]-1:prompt.shape[1]+len(target)-1].float()
            value=torch.nn.functional.cross_entropy(selected,torch.tensor(target,device=device)).item();reference_nll.append(value)
        for batch_size in batch_sizes:
            selected=rows[:batch_size];actual=backend.generate(selected,max_new_tokens=16)
            repeated=backend.generate(selected,max_new_tokens=16)
            tokens=[r['generated_token_ids'] for r in actual]
            equal=tokens==independent[:batch_size]
            if not equal:
                atomic_json(output/f'cross_batch_difference_{condition}_{batch_size}.json',
                            {'condition':condition,'batch':batch_size,'actual':tokens,'reference':independent[:batch_size],
                             'note':'Material cross-batch difference retained for audit; same-batch greedy oracle must pass.'})
            oracle=_verify_greedy_tokens(model,tokenizer,store,selected,tokens)
            assert tokens==[r['generated_token_ids'] for r in repeated]
            for row in actual:
                assert row['generated_tokens']==len(row['generated_token_ids'])<=16
                assert row['reached_token_cap']==(len(row['generated_token_ids'])==16 and row['generated_token_ids'][-1] not in eos)
            nll=backend.reference(selected)
            independent_batch=_teacher_reference(model,tokenizer,store,selected)
            torch.testing.assert_close(torch.tensor(nll),torch.tensor(independent_batch),rtol=1e-5,atol=1e-5)
            comparisons.append({'condition':condition,'batch_size':batch_size,'single_vs_batch_tokens_exact':equal,
                                'same_batch_greedy_oracle_passed':oracle,'repeat_exact':True,
                                'max_nll_abs_difference':max(abs(a-b) for a,b in zip(nll,independent_batch)),
                                'max_nll_difference_from_unpadded_fp16':max(abs(a-b) for a,b in zip(nll,reference_nll)),
                                'generated_tokens':tokens,'independent_single_tokens':independent[:batch_size],
                                'reference_nll':reference_nll[:batch_size],'production_nll':nll})
            atomic_json(output/'progress.json',comparisons)
    fp16_peak=torch.cuda.max_memory_allocated(device)
    # Diagnostic only: promote a disposable model copy's half-valued weights to
    # FP32 to distinguish batch-shape rounding from a mask/position defect.
    model.language_model.float();torch.backends.cuda.matmul.allow_tf32=False
    control_rows=condition_records(pairs,'natural','correct',mapping)[::2][:2]
    control_batch=_teacher_reference(model,tokenizer,store,control_rows,amp=False)
    control_single=[_teacher_reference(model,tokenizer,store,[row],amp=False)[0] for row in control_rows]
    torch.testing.assert_close(torch.tensor(control_batch),torch.tensor(control_single),rtol=1e-4,atol=1e-4)
    differences=[{'condition':r['condition'],'batch_size':r['batch_size']} for r in comparisons if not r['single_vs_batch_tokens_exact']]
    result={'status':'passed_with_cross_batch_numerical_differences' if differences else 'passed',
            'cross_batch_differences':differences,
            'scope':'untrained real-model same-batch inference equivalence; training pairs only','assets':assets['identities'],
            'comparisons':comparisons,'gpu':torch.cuda.get_device_name(device),'batch_sizes':list(batch_sizes),
            'official_optimizer_updates':0,'seconds':time.perf_counter()-started,
            'peak_gpu_allocated_bytes':fp16_peak,'nll_tolerance':{'rtol':1e-5,'atol':1e-5},
            'fp32_batch_shape_control':{'batch':control_batch,'single':control_single,
                                       'max_abs_difference':max(abs(a-b) for a,b in zip(control_batch,control_single))}}
    atomic_json(output/'result.json',result)
    return result
