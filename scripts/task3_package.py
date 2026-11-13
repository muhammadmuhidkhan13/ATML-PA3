"""Build a local Task 3 live deployment package without accessing any server."""
import sys,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from atml_pa3.task3_part2.package import build
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,default=Path('.'))
    p.add_argument('--output',type=Path,default=Path('artifacts/task3/deployment-v6'))
    a=p.parse_args();print(build(a.project,a.output))
