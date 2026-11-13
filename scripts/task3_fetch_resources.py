"""Fetch small authoritative metadata, never model weights or images."""
import hashlib,json
from pathlib import Path
import requests

if __name__=='__main__':
    root=Path('artifacts/task3/reference-v1');root.mkdir(parents=True,exist_ok=True)
    revision=json.loads((root/'manifest.json').read_text())['revision']
    urls={'mathvista_testmini.json':f'https://raw.githubusercontent.com/lupantech/MathVista/{revision}/data/testmini.json',
          'qwen25_model_info.json':'https://huggingface.co/api/models/Qwen/Qwen2.5-3B-Instruct/revision/aa8e72537993ba99e69dfaafa59ed015b17504d1'}
    manifest={}
    for name,url in urls.items():
        target=root/name
        if not target.exists():
            response=requests.get(url,timeout=60);response.raise_for_status();target.write_bytes(response.content)
        data=target.read_bytes();json.loads(data)
        manifest[name]={'url':url,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
        print(name,len(data),flush=True)
    (root/'metadata_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
