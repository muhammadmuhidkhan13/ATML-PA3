"""Plot/export saved Task 2 results; no model loading or invented values."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from atml_pa3.task2.plotting import export_artifacts
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",type=Path)
    p.add_argument("--evaluation-dir",type=Path)
    p.add_argument("--root",type=Path)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if not a.run_dir and not a.evaluation_dir:p.error("Supply saved training or evaluation output")
    print(export_artifacts(a.output,run_dir=a.run_dir,evaluation_dir=a.evaluation_dir,root=a.root))
