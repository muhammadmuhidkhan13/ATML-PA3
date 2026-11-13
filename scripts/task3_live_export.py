"""Verify complete real live results and export numeric comparisons and figures."""
import sys,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from atml_pa3.task3_part2.live_export import export
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,default=Path('assets'));p.add_argument('--run',type=Path,default=Path('runs/live-v1'))
    p.add_argument('--output',type=Path,default=Path('exports/live-v1'))
    a=p.parse_args();print(export(a.assets,a.run,a.output))
