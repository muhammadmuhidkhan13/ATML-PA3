"""Validate actual source Parquets and freeze Task3 splits/orders; no model use."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from atml_pa3.task3.preparation import prepare
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('ATML-PA3-Data'))
    p.add_argument('--output',type=Path,default=Path('artifacts/task3/prepared-v1'))
    a=p.parse_args();prepare(a.root,a.output)
