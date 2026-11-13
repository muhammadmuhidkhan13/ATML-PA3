"""Versioned answer/equivalence cache with an external per-question watchdog."""
from concurrent.futures import ThreadPoolExecutor,TimeoutError
from pathlib import Path
import json,subprocess,sys
import numpy as np
from .data import DatasetSource,assert_source_identity,validate_mathvista_metadata
from .equivalence import SCORER,normalize_mathvista,NO_ANSWER
from .io import atomic_json,read_json,identity,sha256,save_npz

def cache_path(output,dataset,sid):return Path(output)/dataset/(identity(sid)+'.npz')

def metadata_reference(metadata:dict)->str:
    choices=metadata['choices']
    if metadata['question_type']=='multi_choice' and metadata['answer'] in choices:
        return chr(65+choices.index(metadata['answer']))
    result=normalize_mathvista(metadata['answer'],metadata)
    if result==NO_ANSWER:raise ValueError('Released gold metadata cannot be normalized')
    return result

class EquivalenceWorker:
    """A real subprocess avoids broken Windows multiprocessing timeout handles.

    math-verify runs in its main thread; a parent thread watches its stdout with
    a120s inactivity limit. Math operations emit progress heartbeats and retain
    their independent5s wall deadline. A stalled worker aborts, without fake labels.
    """
    def __init__(self,log):
        self.log=Path(log).open('a',encoding='utf-8')
        self.process=subprocess.Popen([sys.executable,'-X','utf8',str(Path(__file__).with_name('equivalence_worker.py'))],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,encoding='utf-8')
        self.reader=ThreadPoolExecutor(max_workers=1)
    def compute(self,**request):
        self.process.stdin.write(json.dumps(request)+'\n');self.process.stdin.flush()
        while True:
            future=self.reader.submit(self.process.stdout.readline)
            try:line=future.result(timeout=SCORER['external_question_timeout_seconds'])
            except TimeoutError:
                if sys.platform=='win32':subprocess.run(['taskkill','/PID',str(self.process.pid),'/T','/F'],capture_output=True)
                else:self.process.kill()
                raise RuntimeError('Equivalence worker stalled; cache remains resumable, no fabricated labels')
            if not line:raise RuntimeError('Equivalence worker exited unexpectedly')
            result=json.loads(line)
            if not result.get('heartbeat'):break
        if 'error' in result:raise ValueError(result['error'])
        return result['result']
    def close(self):
        if self.process.poll() is None:self.process.stdin.close();self.process.wait(timeout=30)
        self.reader.shutdown(wait=True);self.log.close()

def build_cache(root,prepared,output,metadata_path):
    assert_source_identity(root,prepared)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'worker_supervision.json',{'operation_wall_seconds':5,'question_worker_inactivity_seconds':120,
                                                'protocol':'progress heartbeat after each operation; no total question wall cutoff'})
    config={**SCORER,'equivalence_code_sha256':sha256(Path(__file__).with_name('equivalence.py')),
            'source_manifest_sha256':sha256(Path(prepared)/'source_manifest.json'),
            'mathvista_metadata_sha256':sha256(metadata_path)}
    path=output/'scorer_config.json'
    if path.exists() and read_json(path)!=config:raise ValueError('Frozen scorer configuration changed')
    if not path.exists():atomic_json(path,config)
    config_id=identity(config);metadata=read_json(metadata_path);validate_mathvista_metadata(metadata)
    manifest_path=output/'manifest.json'
    manifest=read_json(manifest_path) if manifest_path.exists() else {'config_id':config_id,'files':{},'complete':False}
    if manifest['config_id']!=config_id:raise ValueError('Cached scorer/source identity differs')
    worker=EquivalenceWorker(output/'equivalence_worker.log')
    try:
        for dataset in ('math','mathvista'):
            source=DatasetSource(root,dataset)
            if dataset=='mathvista' and set(source.ids)!=set(metadata):raise ValueError('MathVista metadata coverage differs')
            for index,sid in enumerate(source.ids):
                target=cache_path(output,dataset,sid);name=target.relative_to(output).as_posix()
                if name in manifest['files']:
                    if not target.exists() or sha256(target)!=manifest['files'][name]:raise ValueError('Corrupt semantic cache')
                    continue
                table=source.tables['generations']
                rows=[r for group in source.groups['generations'][sid]
                      for r in table.read_row_group(group,columns=['sample_id','rollout_id','extracted_answer','gold_answer']).to_pylist() if r['sample_id']==sid]
                rows.sort(key=lambda r:r['rollout_id']);gold=rows[0]['gold_answer'];meta=metadata[sid] if dataset=='mathvista' else None
                if meta is not None:
                    # Released metadata stores the exact CHOICE TEXT. In two
                    # questions that text is '(b)' but its option is E/D, so it
                    # must not be parsed as a raw prediction containing letterB.
                    reference=metadata_reference(meta)
                    if normalize_mathvista(gold,meta)!=reference:raise ValueError(f'Gold and released metadata disagree:{sid}')
                result=worker.compute(dataset=dataset,answers=[r['extracted_answer'] for r in rows],gold=gold,metadata=meta)
                if result.get('operation_timeouts'):
                    atomic_json(output/dataset/(sid.replace(':','_')+'.timeouts.json'),result['operation_timeouts'])
                save_npz(target,normalized=np.array(result['normalized'],dtype=str),missing=np.array(result['missing'],bool),
                         parsed=np.array(result['parsed'],bool),equivalent=np.array(result['equivalent'],bool),
                         correct=np.array(result['correct'],bool),sample_id=np.array(sid),config_id=np.array(config_id))
                manifest['files'][name]=sha256(target);atomic_json(manifest_path,manifest)
                if index%25==0:print(f'{dataset} semantics {index+1}/{len(source.ids)}',flush=True)
        manifest['complete']=True;atomic_json(manifest_path,manifest)
    finally:worker.close()
    return manifest

def load_semantics(output,dataset,sid):
    with np.load(cache_path(output,dataset,sid),allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
