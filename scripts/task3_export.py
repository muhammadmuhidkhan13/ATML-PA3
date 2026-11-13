"""Validate saved offline outputs and export reproducible tables/figures."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from atml_pa3.task3.validation import validate_run
from atml_pa3.task3.plots import export
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('artifacts/task3/offline-v1'))
    p.add_argument('--split',choices=('development','evaluation'),default='evaluation');p.add_argument('--expected-questions',type=int,default=800)
    p.add_argument('--validate-only',action='store_true',help='Read-only verification; preserve existing figures')
    a=p.parse_args();print(validate_run(a.output,split=a.split,expected_questions=a.expected_questions))
    if not a.validate_only:print(export(a.output,split=a.split))
