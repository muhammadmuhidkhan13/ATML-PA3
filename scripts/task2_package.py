"""Create a local Task 2 transfer package; never upload or contact a server."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from atml_pa3.task2.deployment import build_package
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("artifacts/task2"))
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    print(build_package(Path(__file__).resolve().parents[1],a.root,a.output))
