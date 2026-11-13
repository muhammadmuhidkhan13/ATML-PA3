"""Validate by default; official generation requires explicit --execute-generation."""
import sys,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from atml_pa3.task3_part2.live_run import launch
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,default=Path('assets'))
    p.add_argument('--model-path',type=Path,default=Path('model'))
    p.add_argument('--output',type=Path,default=Path('runs/live-v1'))
    p.add_argument('--execute-generation',action='store_true')
    p.add_argument('--preflight',type=Path)
    a=p.parse_args();print(launch(a.assets,a.model_path,a.output,execute=a.execute_generation,preflight_path=a.preflight))
