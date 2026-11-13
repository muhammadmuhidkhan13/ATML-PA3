"""Startup-only performance sizing with real updates, rollback and fixed effective batches."""
from dataclasses import dataclass,asdict,replace
from statistics import median
import gc,math,time
import torch
from .curriculum import Recipe
from .adapters import attach_attention_lora,AttentionLoRA,TARGETS,make_stage_optimizer
from .state import model_state,restore_model,rng_state,restore_rng

@dataclass(frozen=True)
class BatchOptions:
    batch_size: str = 'auto'
    preferred_batch_size: int = 32
    effective_batch_size: int = 16  # PDF p8: 4500 updates *16 =72000 presentations.
    max_batch_size: int = 64
    memory_headroom_fraction: float = .10

    def __post_init__(self):
        if min(self.preferred_batch_size,self.effective_batch_size,self.max_batch_size)<1:
            raise ValueError('Batch sizes must be positive')
        if self.effective_batch_size%2:raise ValueError('Effective batch must preserve complete pairs')
        if not 0<=self.memory_headroom_fraction<1:raise ValueError('Headroom fraction must be in [0,1)')
        if self.batch_size!='auto':
            try:n=int(self.batch_size)
            except (TypeError,ValueError):raise ValueError('Batch size must be auto or a positive integer')
            if n<1 or n>self.max_batch_size or self.effective_batch_size%n:
                raise ValueError('Microbatch must divide the effective batch and respect max-batch-size')

    def serial(self):return asdict(self)

def candidates(options):
    if options.batch_size!='auto':return [int(options.batch_size)]
    upper=min(options.max_batch_size,options.effective_batch_size)
    sizes=[2**i for i in range(upper.bit_length()) if 2**i<=upper and options.effective_batch_size%(2**i)==0]
    preferred=max([n for n in sizes if n<=options.preferred_batch_size],default=1)
    # Preferred first; larger optional probes only when effective/max constraints permit.
    return [preferred]+sorted([n for n in sizes if n>preferred],reverse=True)+sorted([n for n in sizes if n<preferred],reverse=True)

def detect_gpu(device='cuda:0'):
    device=torch.device(device)
    if device.type!='cuda' or not torch.cuda.is_available():raise RuntimeError('Batch preflight requires CUDA')
    with torch.cuda.device(device):free,total=torch.cuda.mem_get_info(device)
    props=torch.cuda.get_device_properties(device)
    return {'name':props.name,'total_bytes':total,'free_bytes':free,'capability':[props.major,props.minor]}

def select_trial(trials,headroom=.10):
    fits=[t for t in trials if t['status']=='fit']
    if not fits:raise RuntimeError('No tested microbatch fits; no official updates performed')
    reliable=[t for t in fits if t['headroom_fraction']>=headroom]
    pool=reliable or fits  # Target is advisory when no fitting candidate achieves it.
    largest=max(pool,key=lambda t:t['microbatch'])
    fastest=max(pool,key=lambda t:t['examples_per_second'])
    # Avoid choosing a small batch based on marginal/noisy timing differences.
    winner=fastest if fastest['examples_per_second']>1.10*largest['examples_per_second'] else largest
    return {**winner,'headroom_target_met':bool(reliable),
            'selection_reason':'throughput >10% faster' if winner is not largest else 'largest fitting with practical headroom'}

def probe_candidates(options,probe,*,locked=None):
    trials=[];fit_count=0
    for n in ([locked] if locked is not None else candidates(options)):
        trial=probe(n);trials.append({'microbatch':n,**trial})
        if trial['status']=='fit' and trial['headroom_fraction']>=options.memory_headroom_fraction:fit_count+=1
        # Benchmark two viable sizes, rather than exhaustively probing tiny batches.
        if fit_count>=2:break
    result=select_trial(trials,options.memory_headroom_fraction)
    return {'selected_microbatch':result['microbatch'],'accumulation':options.effective_batch_size//result['microbatch'],
            'effective_batch_size':options.effective_batch_size,'selection':result,'trials':trials,
            'request':options.serial(),'official_optimizer_updates':0}

def representative_batches(root,tokenizer,store):
    """Longest actual encoded TRAIN records, including every grounding template.

    No development examples, padded fake token lengths or easy short-only batches.
    CPU features/encodings are reused across probes; production still transfers its
    ordinary cached FP16 features in each actual microbatch.
    """
    from .runtime import asset_paths
    from .io import read_json,read_jsonl
    from .model import encode_example
    data=asset_paths(root)['data'];captions=[];natural=[];ground=[]
    def encode(r,caption):
        e=encode_example(tokenizer,r['question'],r['target'],caption=caption)
        return (len(e.input_ids)+196,e,r['image_id'])
    captions=[encode(r,True) for r in read_jsonl(data/'captions_train.jsonl')]
    natural=[encode(m,False) for p in read_jsonl(data/'vqa_train.jsonl') for m in p['members']]
    pools=read_json(data/'grounding_train_pools.json')
    for c in pools['categories']:
        rows=[(f"Does this image contain {c['object']}?",'yes',c['positive'][0])]
        counts=[(k,images) for k,images in c['counts'].items() if images]
        if counts:
            k,images=max(counts,key=lambda item:int(item[0]))
            rows.append((f"How many {c['plural']} can you see?",k,images[0]))
        for question,target,image in rows:
            ground.append(encode({'question':question,'target':target,'image_id':image},False))
    batches={};details={};features={}
    for stage,pool in ((1,captions),(2,captions+ground),(3,captions+ground+natural)):
        top=sorted(pool,key=lambda x:x[0],reverse=True)[:64]
        for _,_,image in top:
            if image not in features:features[image]=store.get(image)
        batches[stage]=[(e,features[image]) for _,e,image in top]
        details[str(stage)]={'selection':'longest encoded training examples/templates; descending stable order',
            'pool_examples':len(pool),'min_selected_sequence_tokens':min(x[0] for x in top),
            'max_sequence_tokens':top[0][0],'visual_tokens':196,'image_ids':[x[2] for x in top],
            'question_tokens':[x[1].question_tokens for x in top],'target_tokens':[x[1].target_tokens for x in top]}
    return batches,details

def real_preflight(model,batches,options,*,device='cuda:0',locked=None,attach=attach_attention_lora,repeats=2):
    """Temporarily test all three stages, then restore weights/architecture/RNG.

    Each timed sample is a COMPLETE effective-batch forward/backward/AdamW update,
    so moment initialization, accumulated gradients and clipping are included.
    Optimizers are disposable; official scheduler/sampler state is never touched.
    """
    from .training import attempt_update
    gpu=detect_gpu(device);rng=rng_state();original=model_state(model)
    modes={name:m.training for name,m in model.named_modules()}
    had_lora=any(isinstance(m,AttentionLoRA) for m in model.modules())
    if had_lora:raise ValueError('Probe a clean original model, not an attached/running training model')
    if repeats<1:raise ValueError('At least one timed repeat is required')
    stage_states={1:original};start=time.perf_counter()
    try:
        attach(model);stage_states[2]=model_state(model)
        wrappers=[(layer.self_attn,name,getattr(layer.self_attn,name))
                  for layer in model.language_model.model.layers for name in TARGETS]
        adapters=[p for name,p in model.named_parameters() if name.endswith(('.lora_A','.lora_B'))]
        def probe(n):
            print(f'Batch sizing: probing microbatch{n}, effective{options.effective_batch_size}',flush=True)
            optimizer=scheduler=scaler=None
            details=[];status='fit'
            try:
                for stage in (1,2,3):
                    for p in adapters:p.requires_grad_(stage>=2)
                    for attention,name,wrapper in wrappers:setattr(attention,name,wrapper.base if stage==1 else wrapper)
                    restore_model(model,stage_states[1 if stage==1 else 2])
                    restore_rng(rng)
                    recipe=replace(Recipe(),microbatch=n,accumulation=options.effective_batch_size//n)
                    optimizer,scheduler=make_stage_optimizer(model,stage,recipe)
                    scaler=torch.amp.GradScaler('cuda',init_scale=256.)
                    pool=batches[stage]
                    records=[pool[i%len(pool)] for i in range(options.effective_batch_size)]
                    loader=lambda rows:([r[0] for r in rows],[r[1] for r in rows])
                    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
                    free_before,total=torch.cuda.mem_get_info(device);allocated_before=torch.cuda.memory_allocated(device)
                    times=[]
                    # Warmup initializes AdamW moments; subsequent samples see steady-state state memory.
                    for repeat in range(repeats+1):
                        torch.cuda.synchronize(device);tick=time.perf_counter()
                        result=attempt_update(model,records,optimizer,scheduler,scaler,loader,recipe)
                        torch.cuda.synchronize(device)
                        if not result['completed']:raise FloatingPointError('Probe scaler skipped an update; batch not certified')
                        if repeat:times.append(time.perf_counter()-tick)
                    peak=torch.cuda.max_memory_allocated(device);torch.cuda.empty_cache();free_after,_=torch.cuda.mem_get_info(device)
                    # Peak allocated rather than cached reserved bytes measures actual activation pressure;
                    # initial free VRAM accounts for other processes and driver/context allocations.
                    spare=max(0,min(free_after,free_before-max(0,peak-allocated_before)))
                    details.append({'stage':stage,'median_update_seconds':median(times),'peak_allocated_bytes':peak,
                        'peak_reserved_bytes':torch.cuda.max_memory_reserved(device),'headroom_fraction':spare/total,
                        'timed_updates':repeats,'warmup_updates':1})
                    optimizer.zero_grad(set_to_none=True);optimizer=scheduler=scaler=None
                    gc.collect();torch.cuda.empty_cache()
                seconds=sum(d['median_update_seconds'] for d in details)/len(details)
                return {'status':'fit','headroom_fraction':min(d['headroom_fraction'] for d in details),
                        'examples_per_second':options.effective_batch_size/seconds,'stages':details}
            except torch.cuda.OutOfMemoryError:
                status='oom'
                return {'status':status,'stages':details}
            finally:
                model.zero_grad(set_to_none=True);optimizer=scheduler=scaler=None
                gc.collect();torch.cuda.empty_cache()
        report=probe_candidates(options,probe,locked=locked)
        report.update(gpu=gpu,seconds=time.perf_counter()-start,scope='restored startup sizing; no sampler draws or official updates')
    finally:
        model.zero_grad(set_to_none=True)
        # Remove ONLY the temporary wrappers installed on the clean model.
        for layer in model.language_model.model.layers:
            for name in TARGETS:
                module=getattr(layer.self_attn,name)
                if isinstance(module,AttentionLoRA):setattr(layer.self_attn,name,module.base)
        restore_model(model,original);restore_rng(rng)
        for name,module in model.named_modules():module.training=modes[name]
        restored=model_state(model)
        if set(restored)!=set(original) or any(not torch.equal(restored[k],original[k]) for k in original):
            raise AssertionError('Preflight did not exactly restore trainable initialization')
        if not torch.equal(torch.get_rng_state(),rng['cpu']):raise AssertionError('CPU RNG restoration failed')
        if any(not torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),rng['cuda'])):raise AssertionError('CUDA RNG restoration failed')
        gc.collect();torch.cuda.empty_cache()
    report.update(initialization_restored=True,temporary_adapters_removed=True,rng_restored=True)
    return report
