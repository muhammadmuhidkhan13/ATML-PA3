"""Atomic hash-checked call records; a run lock prevents concurrent writers."""
from pathlib import Path
import json,os
from atml_pa3.task3.io import atomic_json,identity

def read_json(path):
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError(f'Duplicate JSON key {key!r}: {path}')
            result[key]=value
        return result
    def reject_constant(value):
        raise ValueError(f'Invalid JSON constant {value}: {path}')
    return json.loads(Path(path).read_text(encoding='utf-8'),object_pairs_hook=unique,
                      parse_constant=reject_constant)

class RunLock:
    def __init__(self,directory):self.path=Path(directory)/'RUNNING.lock'
    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        try:self.fd=os.open(self.path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        except FileExistsError:raise ValueError('Run locked; verify previous process stopped before removing stale lock')
        os.write(self.fd,str(os.getpid()).encode());return self
    def __exit__(self,*args):os.close(self.fd);self.path.unlink()

class CallStore:
    def __init__(self,directory,config):
        self.root=Path(directory);self.root.mkdir(parents=True,exist_ok=True)
        self.config=config
        path=self.root/'run_config.json'
        if path.exists():
            if identity(read_json(path))!=identity(config):raise ValueError('Incompatible live resume')
        elif any(self.root.iterdir()):
            # RunLock is allowed before the first immutable config is installed.
            if any(p.name not in ('RUNNING.lock','run_config.json.partial') for p in self.root.iterdir()):raise ValueError('Output directory belongs to another run')
            atomic_json(path,config)
        else:atomic_json(path,config)
        self.config_id=identity(config)

    def path(self,sid,call):return self.root/'calls'/f'{identity([sid,call])}.json'
    def read(self,sid,call,expected=None):
        p=self.path(sid,call)
        if not p.exists():return None
        envelope=read_json(p);record=envelope['record']
        if identity(record)!=envelope['sha256']:raise ValueError('Corrupt call record hash')
        if record['config_id']!=self.config_id or record['sample_id']!=sid or record['call']!=call:raise ValueError('Wrong call identity')
        if expected is not None and identity(record['request'])!=identity(expected):raise ValueError('Call prompt/config changed')
        validate_record(record)
        return record

    def write(self,sid,call,record):
        if self.path(sid,call).exists():raise FileExistsError('Never overwrite a completed call')
        record={**record,'config_id':self.config_id,'sample_id':sid,'call':call}
        validate_record(record)
        atomic_json(self.path(sid,call),{'record':record,'sha256':identity(record)})
        return self.read(sid,call,record['request'])

def validate_record(r):
    import math
    from .live_spec import last_boxed
    if not isinstance(r['text'],str) or r['output_tokens']!=len(r['token_ids']) or r['input_tokens']!=len(r['input_token_ids']):raise ValueError('Token record lengths differ')
    if any(not isinstance(i,int) or i<0 for i in r['token_ids']+r['input_token_ids']):raise ValueError('Invalid token IDs')
    if not 0<r['output_tokens']<=r['request']['max_new_tokens']:raise ValueError('Output outside allowance')
    if r['hit_cap']!=(r['output_tokens']==r['request']['max_new_tokens']):raise ValueError('Cap indicator differs')
    if r['extracted_answer']!=last_boxed(r['text']):raise ValueError('Extraction record differs')
    if not math.isfinite(r['generation_seconds']) or r['generation_seconds']<0 or not isinstance(r['peak_allocated_bytes'],int) or r['peak_allocated_bytes']<0:raise ValueError('Invalid resource measurement')
    if not r['rendered_prompt'] or not r['input_tokens']:raise ValueError('Missing prompt evidence')
