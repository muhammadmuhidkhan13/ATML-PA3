"""Isolated native Transformers FP16/CUDA adapter; imported only on explicit opt-in."""
from pathlib import Path
from importlib.metadata import version
import hashlib,json,time,gc
from atml_pa3.task3.io import read_json,sha256,identity
from .live_spec import MODEL,REVISION,messages,SAMPLING

GIB=1024**3
MIN_FREE=12*GIB
MAX_OCCUPIED=4*GIB
MIN_HEADROOM=2*GIB
INPUT_LIMITS={1024:4096,3072:4096,2048:8192}
TIMING_SCOPE='native prompt render, tokenize, transfer, CUDA-synchronized generate, decode; excludes loading/scoring'

def check_input_length(n,cap,stress):
    if stress:
        minimum=INPUT_LIMITS[cap]
        if not minimum<=n<=minimum+256:
            raise ValueError('Synthetic prompt is outside the bounded memory probe envelope')
    elif cap in INPUT_LIMITS and n>INPUT_LIMITS[cap]:
        raise ValueError(f'Prompt length {n} exceeds the certified {INPUT_LIMITS[cap]}-token input envelope')

def validate_model_files(directory,lock):
    directory=Path(directory)
    if lock['model']!=MODEL or lock['revision']!=REVISION:raise ValueError('Model lock identity differs')
    for p in directory.iterdir() if directory.exists() else ():
        if p.is_file() and p.suffix in ('.json','.jinja','.safetensors','.bin','.model') and p.name not in lock['files']:
            raise ValueError('Unverified model/tokenizer override: '+p.name)
    for name,expected in lock['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Unsafe model path')
        p=directory/name
        if not p.is_file() or p.stat().st_size!=expected['size']:raise ValueError('Missing/incomplete model asset: '+name)
        if expected['algorithm']=='sha256':actual=sha256(p)
        elif expected['algorithm']=='git-sha1':
            content=p.read_bytes();actual=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        else:raise ValueError('Unsupported model checksum')
        if actual!=expected['digest']:raise ValueError('Pinned model file hash differs: '+name)
    config=read_json(directory/'config.json')
    if config.get('model_type')!='qwen2' or config.get('quantization_config'):raise ValueError('Unexpected or quantized model')
    index=read_json(directory/'model.safetensors.index.json')
    if not set(index['weight_map'].values())<=set(lock['files']):raise ValueError('Unverified weight shards')
    return True

def validate_gpu(hardware):
    name,total,free,capability=(hardware[k] for k in ('name','total_bytes','free_bytes','capability'))
    if type(total) is not int or type(free) is not int or not 0<=free<=total:
        raise ValueError('Invalid GPU memory evidence')
    a100='A100' in name and capability==[8,0] and total>=70*GIB
    rtx=name in ('NVIDIA GeForce RTX 3090','NVIDIA GeForce RTX 3090 Ti') and capability==[8,6] and total>=23*GIB
    if not (a100 or rtx):
        raise ValueError('Expected an A100 80GB, RTX 3090 24GB or RTX 3090 Ti 24GB with eligible CUDA capability and VRAM')
    if free<MIN_FREE:raise ValueError('At least 12 GiB of free GPU memory is required')
    # Any other active process using significant GPU memory must be handled externally.
    if total-free>MAX_OCCUPIED:raise ValueError('GPU memory already in use; ensure competing jobs have finished')

def check_gpu():
    import torch
    if not torch.cuda.is_available():raise ValueError('CUDA GPU required')
    props=torch.cuda.get_device_properties(0);free,total=torch.cuda.mem_get_info(0)
    hardware={'name':props.name,'total_bytes':props.total_memory,'free_bytes':free,
              'capability':list(torch.cuda.get_device_capability(0))}
    # Keep the CUDA driver occupancy check as well as the device-property check.
    if total-free>MAX_OCCUPIED and free>=MIN_FREE:
        raise ValueError('GPU memory already in use; ensure competing jobs have finished')
    validate_gpu(hardware)
    return hardware

def backend_identity(lock):
    import torch,platform
    props=torch.cuda.get_device_properties(0)
    return {'model':MODEL,'revision':REVISION,'model_lock':identity(lock),'torch':torch.__version__,
            'transformers':version('transformers'),'python':platform.python_version(),'numpy':version('numpy'),
            'math_verify':version('math-verify'),'sympy':version('sympy'),
            'gpu_name':props.name,'gpu_uuid':str(getattr(props,'uuid','')),
            'gpu_total_bytes':props.total_memory,'gpu_capability':list(torch.cuda.get_device_capability(0)),
            'cuda':torch.version.cuda,
            'attention':'sdpa','dtype':'float16','device':'cuda:0','batch_size':1}

def generation_options(request,native):
    return {**request['sampling'],'max_new_tokens':request['max_new_tokens'],
            'min_new_tokens':request.get('min_new_tokens'),
            **{k:native[k] for k in ('eos_token_id','pad_token_id','bos_token_id')}}

class QwenBackend:
    def __init__(self,directory,lock):
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM,GenerationConfig
        self.torch=torch;self.GenerationConfig=GenerationConfig
        hardware=check_gpu();start=time.perf_counter()
        # Caller validates cryptographic hashes BEFORE this adapter is constructed.
        self.tokenizer=AutoTokenizer.from_pretrained(str(directory),local_files_only=True,trust_remote_code=False)
        if not self.tokenizer.chat_template:raise ValueError('Native chat template missing')
        self.model=AutoModelForCausalLM.from_pretrained(str(directory),local_files_only=True,
                    trust_remote_code=False,dtype=torch.float16,attn_implementation='sdpa').to('cuda:0').eval()
        self.model.requires_grad_(False)
        if getattr(self.model,'is_quantized',False) or any(p.dtype!=torch.float16 or p.device.type!='cuda' for p in self.model.parameters() if p.is_floating_point()):raise ValueError('Model is not entirely FP16 CUDA')
        native_text=(Path(directory)/'generation_config.json').read_bytes().decode('utf-8')
        self.native_generation=json.loads(native_text)
        self.setup={'loading_seconds':time.perf_counter()-start,'hardware':hardware,'backend':backend_identity(lock),
                    'chat_template_sha256':hashlib.sha256(self.tokenizer.chat_template.encode()).hexdigest(),
                    'native_generation_config':native_text,'weights_trainable':False,'quantized':False,'offloaded':False,
                    'padding_side':self.tokenizer.padding_side,'generation_api':'GenerationConfig + generate; no training'}

    def generate(self,request):
        torch=self.torch
        torch.cuda.synchronize();end_to_end_start=time.perf_counter()
        rendered=self.tokenizer.apply_chat_template(request['messages'],tokenize=False,add_generation_prompt=True)
        inputs=self.tokenizer([rendered],return_tensors='pt',add_special_tokens=False)
        n=inputs['input_ids'].shape[1]
        cap=request['max_new_tokens']
        check_input_length(n,cap,'min_new_tokens' in request)
        if n+request['max_new_tokens']>self.model.config.max_position_embeddings:raise ValueError('Prompt plus allowance exceeds native context; never truncate')
        inputs=inputs.to('cuda:0')
        torch.manual_seed(request['seed']);torch.cuda.manual_seed_all(request['seed'])
        config=self.GenerationConfig(**generation_options(request,self.native_generation))
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        with torch.inference_mode():output=self.model.generate(**inputs,generation_config=config)
        torch.cuda.synchronize();seconds=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated()
        reserved=torch.cuda.max_memory_reserved()
        ids=output[0,n:].tolist();input_ids=inputs['input_ids'][0].tolist()
        if output[0,:n].tolist()!=input_ids:raise ValueError('Decoder prompt prefix differs')
        text=self.tokenizer.decode(ids,skip_special_tokens=True)
        eos=config.eos_token_id if isinstance(config.eos_token_id,list) else [config.eos_token_id]
        end_to_end_seconds=time.perf_counter()-end_to_end_start
        result={'text':text,'token_ids':ids,'input_token_ids':input_ids,'input_tokens':n,'output_tokens':len(ids),
                'retained_token_ids':[t for t in ids if t not in eos],
                'hit_cap':len(ids)==request['max_new_tokens'],'ended_with_eos':bool(ids and ids[-1] in eos),
                'generation_seconds':end_to_end_seconds,'cuda_generation_seconds':seconds,
                'timing_scope':TIMING_SCOPE,
                'peak_allocated_bytes':peak,'peak_reserved_bytes':reserved,'rendered_prompt':rendered,
                'hardware':self.setup['hardware'],'backend':self.setup['backend'],
                'setup_id':identity(self.setup),
                'effective_generation_config':config.to_dict(),'padding_side':self.tokenizer.padding_side,
                'attention_implementation':self.model.config._attn_implementation}
        if 'min_new_tokens' in request:
            result['free_after_bytes']=torch.cuda.mem_get_info(0)[0]
        return result

    def close(self):
        del self.model;gc.collect();self.torch.cuda.empty_cache()

def stress_requests():
    problem='Synthetic memory probe. Continue with arithmetic.'+' x'*4096
    prior='Synthetic prior response.'+' x'*4096
    return {
        'longer_memory':{'messages':messages(problem),'max_new_tokens':3072,
                         'min_new_tokens':3072,'seed':6304,'sampling':SAMPLING},
        'revision_memory':{'messages':messages(problem,prior),'max_new_tokens':2048,
                           'min_new_tokens':2048,'seed':6305,'sampling':SAMPLING},
    }

def smoke_request():
    return {'messages':[{'role':'system','content':'You are a helpful assistant.'},
        {'role':'user','content':r'What is 1 + 1? Give the answer inside \boxed{}.'}],
        'max_new_tokens':64,'seed':6304,'sampling':SAMPLING}

def validate_stress(record,request):
    cap=request['max_new_tokens'];minimum=INPUT_LIMITS[cap]
    if not minimum<=record['input_tokens']<=minimum+256 or record['output_tokens']!=cap:
        raise ValueError('Memory probe did not cover the full input envelope and generation cap')
    if record['effective_generation_config']['min_new_tokens']!=cap:
        raise ValueError('Memory probe was not configured for full-length generation')
    if record['free_after_bytes']<MIN_HEADROOM:
        raise ValueError('Memory probe left less than 2 GiB GPU headroom')
    if not 0<record['peak_allocated_bytes']<=record['peak_reserved_bytes']:
        raise ValueError('Invalid GPU memory probe measurements')

def preflight(assets,model_path,output,execute=False):
    from atml_pa3.task3.io import atomic_json
    from .live_spec import load_assets
    from .live_store import CallStore,RunLock
    load_assets(assets)
    from .live_run import check_scoring_dependencies
    check_scoring_dependencies(assets)
    if not execute:return {'status':'PREFLIGHT NOT EXECUTED; requires --execute-preflight'}
    output=Path(output)
    if output.exists():raise FileExistsError('Preflight needs a fresh isolated directory')
    lock=read_json(Path(assets)/'model_files.json');validate_model_files(model_path,lock)
    import torch
    backend=None
    try:
        backend=QwenBackend(model_path,lock)
        with RunLock(output):
            store=CallStore(output,{'mode':'non-official bounded preflight','backend':backend_identity(lock)})
            request=smoke_request()
            from .live_spec import last_boxed
            result=backend.generate(request)
            store.write('PREFLIGHT_NON_OFFICIAL','smoke',{**result,'request':request,'extracted_answer':last_boxed(result['text'])})
            restored=store.read('PREFLIGHT_NON_OFFICIAL','smoke',request)
            if restored['extracted_answer'] is None:raise ValueError('Preflight generated no boxed answer; inspect before official execution')
            from atml_pa3.task3.cache import EquivalenceWorker
            worker=EquivalenceWorker(output/'preflight_scorer.log')
            try:grade=worker.compute(dataset='math',answers=[restored['extracted_answer']],gold='2',metadata=None)
            finally:worker.close()
            if not grade['correct'][0]:raise ValueError('Preflight arithmetic/scoring failed')
            stress_hashes={}
            for call,request in stress_requests().items():
                result=backend.generate(request)
                validate_stress(result,request)
                record=store.write('PREFLIGHT_NON_OFFICIAL',call,{**result,'request':request,
                    'extracted_answer':last_boxed(result['text'])})
                stress_hashes[call]=identity(record)
            atomic_json(output/'setup.json',backend.setup)
            atomic_json(output/'PASSED.json',{'non_official':True,'call_sha256':identity(restored),'checks':[
                'model_file_hashes','native_template','FP16_CUDA','sampling','boxed_extraction','atomic_roundtrip','CUDA_timing_memory','scoring','full_length_longer_and_revision'],
                'stress_call_hashes':stress_hashes,
                'backend':backend_identity(lock),'assets':identity(load_assets(assets)[2]),'backend_code_sha256':sha256(__file__)})
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError('CUDA out of memory during bounded Task 3 preflight. No successful proof was written.') from exc
    finally:
        if backend is not None:backend.close()
    return {'status':'PREFLIGHT PASSED','official_questions_generated':0}
