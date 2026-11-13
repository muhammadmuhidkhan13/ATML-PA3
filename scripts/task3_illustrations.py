"""Export deterministic retrospective trajectory evidence from completed outputs."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from atml_pa3.task3.illustrations import export_illustrations
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('ATML-PA3-Data'))
    p.add_argument('--prepared',type=Path,default=Path('artifacts/task3/prepared-v1'))
    p.add_argument('--semantics',type=Path,default=Path('artifacts/task3/semantics-v2'))
    p.add_argument('--output',type=Path,default=Path('artifacts/task3/offline-v1'))
    p.add_argument('--split',choices=('development','evaluation'),default='evaluation')
    a=p.parse_args();print(export_illustrations(a.root,a.prepared,a.semantics,a.output,split=a.split))
