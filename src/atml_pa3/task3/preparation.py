"""One authoritative split/order implementation shared by all offline methods."""
from importlib.metadata import version
import numpy as np
from .io import atomic_json,sha256,read_json,identity
from .data import DatasetSource

SEED=6304
BUDGETS=(1,2,4,8,16,32)

def split_ids(ids:list[str],*,development:int=200,seed:int=SEED)->dict:
    ordered=sorted(ids)
    if len(set(ordered))!=len(ordered):raise ValueError('Duplicate question IDs')
    permutation=np.random.default_rng(seed).permutation(len(ordered)).tolist()
    shuffled=[ordered[i] for i in permutation]
    return {'ordered_ids':ordered,'permutation_indices':permutation,'development':shuffled[:development],
            'evaluation':shuffled[development:],'live_math_50':shuffled[development:development+50],
            'seed':seed,'numpy':np.__version__}

def rollout_orders(ids:list[str],*,rollouts:int=32,orders:int=10,seed:int=SEED)->dict[str,list[list[int]]]:
    rng=np.random.default_rng(seed)
    # Reset per DATASET, then sorted question IDs, then order0..9. No hashing of
    # Python strings and no dependence on Parquet storage order.
    return {sid:[rng.permutation(rollouts).tolist() for _ in range(orders)] for sid in sorted(ids)}

def prepare(root,output):
    from pathlib import Path
    output=Path(output)
    if (output/'source_manifest.json').exists():raise FileExistsError('Completed split/order preparation is immutable; use a fresh version')
    output.mkdir(parents=True,exist_ok=True)
    summary={'readme_sha256':sha256(Path(root)/'README.MD'),'datasets':{}}
    for dataset in ('math','mathvista'):
        source=DatasetSource(root,dataset);audit=source.audit();split=split_ids(source.ids)
        split['source_hashes']={k:v['sha256'] for k,v in audit['files'].items()}
        if dataset!='math':split.pop('live_math_50')
        atomic_json(output/dataset/'split.json',split)
        atomic_json(output/dataset/'orders.json',{'seed':SEED,'procedure':'per-dataset RNG; sorted question; ten sequential permutations',
                                                 'numpy':np.__version__,'orders':rollout_orders(source.ids)})
        atomic_json(output/dataset/'source_audit.json',audit);summary['datasets'][dataset]=audit
        print(f'{dataset}: {len(source.ids)} questions, all exact-key joins and token alignments validated',flush=True)
    atomic_json(output/'source_manifest.json',summary)
    return summary
