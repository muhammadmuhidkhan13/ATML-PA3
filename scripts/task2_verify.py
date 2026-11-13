"""Non-training real-data/cache/calibration and toy-system verifier."""
import argparse
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from atml_pa3.task2.verification import verify
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("artifacts/task2"))
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result=verify(a.root,a.output)
    print(json.dumps({k:v for k,v in result.items() if k!="spec"},indent=2))
