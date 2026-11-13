"""Default: non-training asset/configuration verification. Explicit opt-in required."""
import argparse
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from atml_pa3.task2.runtime import launch
from atml_pa3.task2.batching import BatchOptions

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("artifacts/task2"))
    p.add_argument("--run-dir",type=Path,required=True)
    p.add_argument("--resume-from",type=Path)
    p.add_argument("--stop-after-stage",type=int,choices=(1,2,3),default=3)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--execute-training",action="store_true")
    p.add_argument('--batch-size',default='8',help='Official microbatch8; auto/other sizes only with --batch-preflight')
    p.add_argument('--preferred-batch-size',type=int,default=32)
    p.add_argument('--effective-batch-size',type=int,default=16,help='PDF requires16;32 would change the experiment')
    p.add_argument('--max-batch-size',type=int,default=64)
    p.add_argument('--memory-headroom-fraction',type=float,default=.10)
    p.add_argument('--batch-preflight',action='store_true',help='Restored real optimizer-step sizing only; no official training')
    p.add_argument('--preflight-report',type=Path)
    a=p.parse_args(argv)
    print(json.dumps(launch(a.root,a.run_dir,execute=a.execute_training,resume_from=a.resume_from,
                            stop_after_stage=a.stop_after_stage,device=a.device,
                            batch_options=BatchOptions(a.batch_size,a.preferred_batch_size,a.effective_batch_size,
                                                       a.max_batch_size,a.memory_headroom_fraction),
                            batch_preflight=a.batch_preflight,preflight_report=a.preflight_report),indent=2))
if __name__=="__main__":
    main()
