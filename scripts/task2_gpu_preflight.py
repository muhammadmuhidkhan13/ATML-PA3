"""Separate bounded backward/generation check, not an official training run."""
import argparse
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from atml_pa3.task2.gpu_preflight import preflight
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("artifacts/task2"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--batch-size",type=int,choices=(1,2,8),default=2)
    p.add_argument("--device",default="cuda:0")
    a=p.parse_args()
    print(json.dumps(preflight(a.root,a.output,batch_size=a.batch_size,device=a.device),indent=2))
