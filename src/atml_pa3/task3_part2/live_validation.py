"""Fail-closed validation of committed live evidence, without loading the model."""
import hashlib
import json
import math
from pathlib import Path

from atml_pa3.task3.io import identity,sha256
from .live_analysis import grade_source,strategy_results
from .live_backend import TIMING_SCOPE,check_input_length,generation_options,smoke_request,stress_requests,validate_gpu,validate_stress
from .live_spec import LIMITS,MODEL,REVISION
from .live_store import read_json,validate_record


def positive_number(value):
    return type(value) in (int,float) and math.isfinite(value) and value>0


def validate_setup(setup,backend,assets):
    lock=read_json(Path(assets)/'model_files.json')
    expected={'model':MODEL,'revision':REVISION,'model_lock':identity(lock),
              'device':'cuda:0','dtype':'float16','attention':'sdpa','batch_size':1}
    if (any(identity(backend.get(k))!=identity(v) for k,v in expected.items())
            or any(not isinstance(backend.get(k),str) or not backend[k]
                   for k in ('torch','transformers','python','numpy','math_verify','sympy','cuda'))
            or not isinstance(backend.get('gpu_uuid'),str)
            or setup.get('backend')!=backend):
        raise ValueError('Invalid CUDA backend provenance')
    hardware=setup['hardware'];validate_gpu(hardware)
    if (hardware['name'],hardware['total_bytes'],hardware['capability'])!=(
            backend['gpu_name'],backend['gpu_total_bytes'],backend['gpu_capability']):
        raise ValueError('GPU setup differs from backend identity')
    if (not positive_number(setup.get('loading_seconds'))
            or any(setup.get(k) is not False for k in ('weights_trainable','quantized','offloaded'))
            or setup.get('padding_side') not in ('left','right')
            or setup.get('generation_api')!='GenerationConfig + generate; no training'):
        raise ValueError('Invalid model setup evidence')
    info=read_json(Path(assets)/'qwen25_model_info.json')
    template=info['config']['tokenizer_config']['chat_template']
    if setup.get('chat_template_sha256')!=hashlib.sha256(template.encode()).hexdigest():
        raise ValueError('Native chat template differs')
    raw=setup['native_generation_config'].encode('utf-8')
    pinned=lock['files']['generation_config.json']
    digest=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    if pinned['algorithm']!='git-sha1' or len(raw)!=pinned['size'] or digest!=pinned['digest']:
        raise ValueError('Pinned native generation config differs')
    return json.loads(raw)


def validate_call_evidence(record,request,setup,native):
    validate_record(record)
    if (identity(record['request'])!=identity(request)
            or record.get('setup_id')!=identity(setup)
            or record.get('backend')!=setup['backend'] or record.get('hardware')!=setup['hardware']
            or record.get('padding_side')!=setup['padding_side']
            or record.get('attention_implementation')!='sdpa'
            or record.get('timing_scope')!=TIMING_SCOPE):
        raise ValueError('Call setup, prompt or timing provenance differs')
    # Exact no-tools path of the hash-pinned Qwen template.
    prompt=''.join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in request['messages'])
    if record['rendered_prompt']!=prompt+'<|im_start|>assistant\n':
        raise ValueError('Rendered native prompt differs')
    cap=request['max_new_tokens']
    check_input_length(record['input_tokens'],cap,'min_new_tokens' in request)
    from transformers import GenerationConfig
    expected=GenerationConfig(**generation_options(request,native)).to_dict()
    if (identity(record.get('effective_generation_config'))!=identity(expected)
            or expected['transformers_version']!=setup['backend']['transformers']):
        raise ValueError('Effective generation configuration differs')
    eos=native['eos_token_id']
    eos=eos if isinstance(eos,list) else [eos]
    ids=record['token_ids'];ended=ids[-1] in eos
    if (any(type(t) is not int for t in ids+record['input_token_ids'])
            or type(record['input_tokens']) is not int or type(record['output_tokens']) is not int
            or type(record['hit_cap']) is not bool or record.get('ended_with_eos') is not ended
            or (not record['hit_cap'] and not ended) or any(t in eos for t in ids[:-1])
            or record.get('retained_token_ids')!=[t for t in ids if t not in eos]):
        raise ValueError('Invalid EOS or token accounting')
    seconds=record.get('cuda_generation_seconds');allocated=record['peak_allocated_bytes']
    reserved=record.get('peak_reserved_bytes')
    if (not positive_number(seconds) or not positive_number(record['generation_seconds'])
            or seconds>record['generation_seconds'] or type(allocated) is not int or type(reserved) is not int
            or not 0<allocated<=reserved<=setup['hardware']['total_bytes']):
        raise ValueError('Invalid CUDA resource measurements')


def validate_preflight_snapshot(snapshot,config,assets):
    setup=snapshot['setup'];proof=snapshot['proof'];records=snapshot['records']
    backend=config['backend'];native=validate_setup(setup,backend,assets)
    if (proof.get('non_official') is not True or proof.get('backend')!=backend
            or proof.get('assets')!=config['asset_manifest']
            or proof.get('backend_code_sha256')!=config['code']['live_backend.py']):
        raise ValueError('Preflight provenance differs')
    stress=stress_requests();requests={'smoke':smoke_request(),**stress}
    if set(records)!=set(requests) or set(proof['stress_call_hashes'])!=set(stress):
        raise ValueError('Preflight call coverage differs')
    hashes={'smoke':proof['call_sha256'],**proof['stress_call_hashes']}
    preflight_id=identity({'mode':'non-official bounded preflight','backend':backend})
    for call,req in requests.items():
        record=records[call]
        if ((record['sample_id'],record['call'],record['config_id'])!=('PREFLIGHT_NON_OFFICIAL',call,preflight_id)
                or identity(record)!=hashes[call]):
            raise ValueError('Preflight call identity differs')
        validate_call_evidence(record,req,setup,native)
        if call in stress:
            validate_stress(record,req)
            free=record['free_after_bytes']
            if type(free) is not int or free>setup['hardware']['total_bytes']:
                raise ValueError('Invalid preflight memory headroom evidence')
    if records['smoke']['extracted_answer'] is None:
        raise ValueError('Preflight smoke extraction missing')


def load_setups(store,assets):
    setups={}
    for path in (store.root/'setups').glob('*.json'):
        setup=read_json(path)
        if path.stem!=identity(setup):raise ValueError('Setup identity differs')
        setups[path.stem]=(setup,validate_setup(setup,store.config['backend'],assets))
    return setups


def validate_saved_call(record,setups):
    if record.get('setup_id') not in setups:
        raise ValueError('Missing call setup evidence')
    setup,native=setups[record['setup_id']]
    validate_call_evidence(record,record['request'],setup,native)


def validate_semantics(sem,records):
    n=len(LIMITS)
    for field in ('missing','parsed','correct'):
        if len(sem[field])!=n or any(type(v) is not bool for v in sem[field]):
            raise ValueError('Invalid live grade vector')
    relation=sem['equivalent']
    if (len(relation)!=n or any(len(row)!=n or any(type(v) is not bool for v in row) for row in relation)
            or any(not relation[i][i] or any(relation[i][j]!=relation[j][i] for j in range(n)) for i in range(n))):
        raise ValueError('Invalid live grade equivalence relation')
    answers=[records[c]['extracted_answer'] for c in LIMITS]
    if (sem['missing']!=[a is None for a in answers]
            or sem['normalized']!=['NO_ANSWER' if a is None else a for a in answers]
            or any(not p and c or m and p for m,p,c in zip(sem['missing'],sem['parsed'],sem['correct']))
            or any(relation[i][j]!=(sem['missing'][i] and sem['missing'][j])
                   for i in range(n) for j in range(i) if not sem['parsed'][i] or not sem['parsed'][j])
            or not isinstance(sem['operation_timeouts'],list)):
        raise ValueError('Grade semantics disagree with saved extractions')


def completion_evidence(assets,store,questions,records):
    """Validate raw evidence and recompute every strategy field before sealing."""
    root=store.root
    expected={(q['sample_id'],call) for q in questions for call in LIMITS}
    if len(questions)!=50 or len(expected)!=250 or set(records)!=expected:
        raise ValueError('Official call coverage differs')
    setups=load_setups(store,assets)
    for record in records.values():validate_saved_call(record,setups)
    path=root/'strategy_results.json';result=read_json(path)
    grade_names={f"{identity(q['sample_id'])}.json" for q in questions}
    if (result['config_id']!=store.config_id or set(result['grading_files'])!=grade_names
            or {p.name for p in (root/'grading').glob('*.json')}!=grade_names):
        raise ValueError('Official grade index coverage differs')
    rebuilt=[]
    for q in questions:
        sid=q['sample_id'];rr={call:records[sid,call] for call in LIMITS}
        grade=root/'grading'/f'{identity(sid)}.json'
        if sha256(grade)!=result['grading_files'][grade.name]:raise ValueError('Live grade corrupt')
        saved=read_json(grade);sem=saved['semantics']
        if identity(saved['source'])!=identity(grade_source(store.config_id,q,rr)) or saved['sha256']!=identity(sem):
            raise ValueError('Live grade provenance differs')
        validate_semantics(sem,rr)
        rebuilt.extend(strategy_results(q,rr,sem))
    if identity(result['rows'])!=identity(rebuilt):
        raise ValueError('Strategy results differ from authoritative calls and grades')
    return {'config_id':store.config_id,'questions':50,'unique_calls':250,
            'calls':{store.path(*key).name:sha256(store.path(*key)) for key in expected},
            'setups':{f'{sid}.json':sha256(root/'setups'/f'{sid}.json') for sid in setups},
            'grades':result['grading_files'],'preflight_sha256':sha256(root/'preflight.json'),
            'strategy_results_sha256':sha256(path),'generation_is_real':True},rebuilt
