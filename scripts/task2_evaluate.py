"""Default validation-only; full development/final generation requires explicit opt-in."""
import argparse
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from atml_pa3.task2.runtime import validate_assets,load_evaluation_model,asset_paths
from atml_pa3.task2.features import FeatureStore
from atml_pa3.task2.evaluation import ModelEvaluator,evaluate_development,evaluate_final
from atml_pa3.task2.io import read_jsonl,sha256
from atml_pa3.task2.state import preserve_rng,read_payload,identity

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("artifacts/task2"))
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--mode",choices=("development","final"),required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--execute-evaluation",action="store_true")
    a=p.parse_args(argv)
    assets=validate_assets(a.root)
    if not a.execute_evaluation:
        from atml_pa3.task2.runtime import current_spec
        saved=read_payload(a.checkpoint)
        payload=read_payload(a.checkpoint,current_spec(a.root,assets,saved['spec']))
        from atml_pa3.task2.runtime import recipe_from_spec
        recipe_from_spec(payload['spec']).validate_progress(payload['progress'])
        if a.mode=="final" and payload["progress"]["global_update"]!=4500:
            raise ValueError("Final evaluation requires update 4500")
        print(json.dumps({"mode":"validation_only","evaluation_started":False,"assets":assets},indent=2))
        return
    model,tokenizer,assets,payload=load_evaluation_model(a.root,a.checkpoint,a.device,a.mode=="final")
    store=FeatureStore(asset_paths(a.root)["features"])
    model_id={"spec":identity(payload["spec"]),"checkpoint":str(a.checkpoint.name),
              "checkpoint_sha256":sha256(a.checkpoint/"state.pt"),
              "progress":payload["progress"]}
    with preserve_rng(model):
        model.eval()
        if a.mode=="development":
            result=evaluate_development(model,tokenizer,store,asset_paths(a.root)["data"],a.output,model_identity=model_id)
        else:
            data=asset_paths(a.root)["data"]
            suites={"grounding":read_jsonl(data/"grounding_test.jsonl"),"natural":read_jsonl(data/"vqa_test.jsonl")}
            result=evaluate_final(ModelEvaluator(model,tokenizer,store),suites,a.output,
                                  model_identity=model_id,dataset_identity=assets["identities"]["dataset_manifest"])
    print(json.dumps(result,indent=2))
if __name__=="__main__":
    main()
