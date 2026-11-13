"""Development selection or held-out execution; never generates model responses."""
import sys,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from atml_pa3.task3_part2.adaptive_run import run
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=Path('.'))
    p.add_argument('--output',type=Path,default=Path('artifacts/task3/adaptive-v1'))
    p.add_argument('--stage',choices=['development','evaluation'],required=True)
    a=p.parse_args();print(run(a.project,a.output,a.stage))
