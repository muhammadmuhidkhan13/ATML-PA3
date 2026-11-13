"""Read-only verification of a transported bundle, without loading model weights."""
import sys,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from atml_pa3.task3.io import read_json,sha256
from atml_pa3.task3_part2.live_spec import load_assets
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,default=Path('.'))
    a=p.parse_args();manifest=read_json(a.bundle/'BUNDLE_MANIFEST.json')
    for name,digest in manifest['files'].items():
        if sha256(a.bundle/name)!=digest:raise ValueError('Bundle file corrupt: '+name)
    load_assets(a.bundle/'assets');print({'status':'BUNDLE VERIFIED','files':len(manifest['files']),'model_loaded':False})
