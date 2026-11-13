"""Download small pinned model metadata (NO weights), then package frozen questions."""
import sys,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from atml_pa3.task3.io import atomic_json
from atml_pa3.task3_part2.live_spec import MODEL,REVISION,prepare_assets

def fetch_lock(path):
    import requests
    url=f'https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true'
    response=requests.get(url,timeout=60);response.raise_for_status();info=response.json()
    if info['sha']!=REVISION:raise ValueError('Pinned API revision mismatch')
    names={'config.json','generation_config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt',
           'model.safetensors.index.json'}
    files={}
    for row in info['siblings']:
        name=row['rfilename']
        if name in names or name.endswith('.safetensors'):
            lfs=row.get('lfs')
            files[name]={'size':row['size'],'algorithm':'sha256' if lfs else 'git-sha1',
                         'digest':lfs['sha256'] if lfs else row['blobId']}
    if not names<=files.keys() or not any(n.endswith('.safetensors') for n in files):raise ValueError('Incomplete model file metadata')
    atomic_json(path,{'model':MODEL,'revision':REVISION,'source_url':url,'files':files})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,default=Path('.'))
    p.add_argument('--output',type=Path,default=Path('artifacts/task3/live-assets-v1'))
    p.add_argument('--lock',type=Path,default=Path('artifacts/task3/model-files-v1.json'))
    a=p.parse_args()
    if not a.lock.exists():fetch_lock(a.lock)
    prepare_assets(a.project,a.output,a.lock)
