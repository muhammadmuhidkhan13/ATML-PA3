"""Read-only completed-final-output validation; no model loading or generation."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from atml_pa3.task2.result_validation import verify_final
from atml_pa3.task2.io import read_jsonl,sha256
from atml_pa3.task2.state import read_payload,identity

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    a=p.parse_args()
    suites={'grounding':read_jsonl(a.data_dir/'grounding_test.jsonl'),'natural':read_jsonl(a.data_dir/'vqa_test.jsonl')}
    if len(suites['grounding'])!=64 or len(suites['natural'])!=96:raise ValueError('Official final pair counts differ')
    payload=read_payload(a.checkpoint)
    if payload['progress']['global_update']!=4500:raise ValueError('Final checkpoint must be update4500')
    model_id={'spec':identity(payload['spec']),'checkpoint':a.checkpoint.name,
              'checkpoint_sha256':sha256(a.checkpoint/'state.pt'),'progress':payload['progress']}
    print(json.dumps(verify_final(a.output,expected_suites=suites,model_identity=model_id,
                                 dataset_identity=sha256(a.data_dir/'manifest.json')),indent=2))
