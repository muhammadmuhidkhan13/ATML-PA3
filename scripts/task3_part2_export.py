"""Export saved adaptive results and integrated fixed-N comparisons."""
import sys,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from atml_pa3.task3_part2.adaptive_export import export
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',type=Path,default=Path('.'))
    p.add_argument('--output',type=Path,default=Path('artifacts/task3/adaptive-v1'))
    a=p.parse_args();print(export(a.project,a.output))
