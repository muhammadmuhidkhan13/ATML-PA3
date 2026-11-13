"""Locked live protocol, exact prompts, stable seeds and instructor extraction."""
from pathlib import Path
from atml_pa3.task3.io import identity, read_json, sha256, atomic_json

MODEL='Qwen/Qwen2.5-3B-Instruct'
REVISION='aa8e72537993ba99e69dfaafa59ed015b17504d1'
SYSTEM='You are a helpful assistant.'
INSTRUCTION=r'Think step by step inside <think> </think> tags, then give your final answer inside \boxed{}.'
REVIEW=r'Review the solution above. Identify and correct any errors; if it is correct, retain its answer. Give your final answer inside \boxed{}.'
LIMITS={'longer':3072,'parallel0':1024,'parallel1':1024,'parallel2':1024,'revision':2048}
STRATEGIES={'longer':['longer'],'parallel':['parallel0','parallel1','parallel2'],'refinement':['parallel0','revision']}
SAMPLING={'do_sample':True,'temperature':.7,'top_p':.8,'top_k':20,'repetition_penalty':1.,'num_beams':1,'use_cache':True}

def seed_for(sid,call):
    if call not in LIMITS:raise ValueError('Unknown call')
    return int(identity({'seed':6304,'question_id':sid,'unique_call':call})[:8],16)

def messages(problem,initial=None):
    result=[{'role':'system','content':SYSTEM},{'role':'user','content':problem+'\n\n'+INSTRUCTION}]
    if initial is not None:result.extend([{'role':'assistant','content':initial},{'role':'user','content':REVIEW}])
    return result

def last_boxed(text):
    # README: LAST LITERAL box; an invalid last box does not fall back to an earlier one.
    start=text.rfind(r'\boxed{')
    if start<0:return None
    start+=len(r'\boxed{');depth=1
    for i in range(start,len(text)):
        if text[i]=='{':depth+=1
        elif text[i]=='}':
            depth-=1
            if depth==0:return text[start:i].strip() or None
    return None

def load_assets(directory):
    directory=Path(directory);manifest=read_json(directory/'manifest.json')
    for name,digest in manifest['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Unsafe asset path')
        if sha256(directory/name)!=digest:raise ValueError('Corrupt live asset: '+name)
    spec=read_json(directory/'specification.json')
    expected={'model':MODEL,'revision':REVISION,'dtype':'float16','quantization':False,'batch_size':1,
              'sampling':SAMPLING,'limits':LIMITS,'system':SYSTEM,'instruction':INSTRUCTION,'review':REVIEW,'seed':6304}
    if spec['protocol']!=expected:raise ValueError('Live protocol differs from locked specification')
    questions=read_json(directory/'questions.json');view=read_json(directory/'live_math_50.json');split=read_json(directory/'math_split.json')
    ids=[q['sample_id'] for q in questions]
    if ids!=view['question_ids'] or ids!=split['evaluation'][:50] or ids!=split['live_math_50'] or len(set(ids))!=50:raise ValueError('Frozen 50 question IDs differ')
    if sha256(directory/'math_split.json')!=view['source_split_sha256']:raise ValueError('Live subset split hash differs')
    info=read_json(directory/'qwen25_model_info.json')
    if info['sha']!=REVISION or info['id']!=MODEL:raise ValueError('Model metadata differs')
    return spec,questions,manifest

def prepare_assets(project,output,model_lock):
    from atml_pa3.task3.data import DatasetSource
    project,output=Path(project),Path(output)
    if output.exists():raise FileExistsError('Use new live asset directory')
    a=project/'artifacts/task3'; output.mkdir(parents=True)
    import shutil
    for source,name in [(a/'prepared-v1/math/live_math_50.json','live_math_50.json'),
        (a/'prepared-v1/math/split.json','math_split.json'),(a/'reference-v1/qwen25_model_info.json','qwen25_model_info.json'),
        (a/'semantics-v2/scorer_config.json','scorer_config.json'),(Path(model_lock),'model_files.json')]:shutil.copyfile(source,output/name)
    source=DatasetSource(project/'ATML-PA3-Data','math');qs=[]
    for sid in read_json(output/'live_math_50.json')['question_ids']:
        row=source.question(sid)[0]
        qs.append({'sample_id':sid,'problem':row['question_text'],'gold_answer':row['gold_answer']})
    atomic_json(output/'questions.json',qs)
    protocol={'model':MODEL,'revision':REVISION,'dtype':'float16','quantization':False,'batch_size':1,
              'sampling':SAMPLING,'limits':LIMITS,'system':SYSTEM,'instruction':INSTRUCTION,'review':REVIEW,'seed':6304}
    atomic_json(output/'specification.json',{'version':'live-v1','protocol':protocol,'unique_calls':250,
        'seed_derivation':'first 8 hex digits of task3.io.identity({seed:6304,question_id,unique_call}) as uint32',
        'scorer_sha256':sha256(output/'scorer_config.json'),'question_source_manifest':sha256(a/'prepared-v1/source_manifest.json')})
    atomic_json(output/'manifest.json',{'files':{p.name:sha256(p) for p in output.iterdir() if p.is_file()},'complete':True})
    load_assets(output)
