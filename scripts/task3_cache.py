"""Freeze and compute answer equivalence/grade caches; no model inference."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from atml_pa3.task3.cache import build_cache
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('ATML-PA3-Data'))
    p.add_argument('--prepared',type=Path,default=Path('artifacts/task3/prepared-v1'))
    p.add_argument('--output',type=Path,default=Path('artifacts/task3/semantics-v2'))
    p.add_argument('--metadata',type=Path,default=Path('artifacts/task3/reference-v1/mathvista_testmini.json'))
    a=p.parse_args();build_cache(a.root,a.prepared,a.output,a.metadata)
