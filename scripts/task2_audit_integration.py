"""Explicitly bounded real-model audit; never starts the official experiment."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from atml_pa3.task2.audit_integration import optimizer_smoke,evaluation_equivalence

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('artifacts/task2'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=('optimizer','evaluation'),required=True)
    p.add_argument('--microbatch',type=int,choices=(1,2,8),default=1)
    p.add_argument('--batch-sizes',type=int,nargs='+',default=[1,2])
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--execute-smoke',action='store_true')
    a=p.parse_args()
    if not a.execute_smoke:
        print(json.dumps({'mode':'validation_only','smoke_started':False,'official_training_started':False}))
    else:
        result=(optimizer_smoke(a.root,a.output,microbatch=a.microbatch,device=a.device) if a.mode=='optimizer'
                else evaluation_equivalence(a.root,a.output,batch_sizes=tuple(a.batch_sizes),device=a.device))
        print(json.dumps(result,indent=2))
