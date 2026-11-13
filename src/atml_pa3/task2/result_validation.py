"""Read-only verification of completed evaluation outputs, without inference."""
from pathlib import Path
import math
from .io import read_json,sha256
from .state import identity
from .metrics import score_record,summarize_suite


def payload_hashes(output):
    output=Path(output)
    return {p.relative_to(output).as_posix():sha256(p) for p in sorted(output.rglob('*'))
            if p.is_file() and p!=output/'manifest.json' and p.suffix not in ('.partial',)}


def verify_batches(directory,*,expected=None,generate=False,model_identity=None,expected_spec=None):
    directory=Path(directory);header=read_json(directory/'manifest.json')
    if not header['complete'] or identity(header['spec'])!=header['spec_id']:
        raise ValueError('Incomplete or incompatible batch manifest')
    if expected_spec is not None and header['spec']!=expected_spec:raise ValueError('Batch output configuration differs')
    count=header['expected_examples'];size=header.get('batch_size',8)
    names={f'batch_{i:05d}.json' for i in range((count+size-1)//size)}
    if set(header['batch_hashes'])!=names or {p.name for p in directory.glob('batch_*.json')}!=names:
        raise ValueError('Missing or unexpected evaluation batches')
    inputs=[];rows=[]
    for name in sorted(names):
        path=directory/name
        if sha256(path)!=header['batch_hashes'][name] or path.with_suffix('.sha256').read_text().strip()!=sha256(path):
            raise ValueError('Evaluation batch integrity failure')
        batch=read_json(path)
        if batch['spec_id']!=header['spec_id'] or len(batch['inputs'])!=len(batch['records']):
            raise ValueError('Batch configuration or record count differs')
        inputs.extend(batch['inputs']);rows.extend(batch['records'])
    if len(rows)!=count or identity(inputs)!=header['inputs_id'] or (expected is not None and inputs!=expected):
        raise ValueError('Expected evaluation inputs/counts differ')
    for row,source in zip(rows,inputs):
        if any(row.get(k)!=v for k,v in source.items()) or row['model_identity']!=header['spec']['model_identity']:
            raise ValueError('Raw output input/checkpoint identity differs')
        if model_identity is not None and row['model_identity']!=model_identity:
            raise ValueError('Expected model/checkpoint identity differs')
        for key in ('reference_nll','reference_seconds'):
            if not math.isfinite(row[key]) or row[key]<0:raise ValueError('Invalid reference value')
        if generate:
            if (row['generated_tokens']!=len(row['generated_token_ids']) or not 0<row['generated_tokens']<=16
                    or not isinstance(row['reached_token_cap'],bool)
                    or row['reached_token_cap'] and row['generated_tokens']!=16
                    or not math.isfinite(row['generation_seconds']) or row['generation_seconds']<0):
                raise ValueError('Invalid generated token counts/timing')
            if any(row.get(k)!=v for k,v in score_record(source,row['prediction_text']).items()):
                raise ValueError('Raw scores differ from independent recomputation')
    return rows


def _header(output,kind,model_identity=None,dataset_identity=None,*,allow_incomplete=False):
    output=Path(output);manifest=read_json(output/'manifest.json');spec=manifest['spec']
    if ((not manifest['complete'] and not allow_incomplete) or manifest['identity']!=identity(spec) or spec['kind']!=kind
            or spec.get('schema_version')!=2):
        raise ValueError('Evaluation incomplete or configuration identity differs')
    if model_identity is not None and spec['model_identity']!=model_identity:raise ValueError('Checkpoint identity differs')
    if dataset_identity is not None and spec['dataset']!=dataset_identity:raise ValueError('Dataset identity differs')
    if manifest.get('files')!=payload_hashes(output):raise ValueError('Evaluation payload integrity failure')
    if kind=='final' and manifest.get('summary_sha256')!=sha256(output/'summary.json'):
        raise ValueError('Combined summary fingerprint differs')
    return spec


def verify_final(output,*,expected_suites=None,model_identity=None,dataset_identity=None,_allow_incomplete=False):
    from .evaluation import CONDITIONS,condition_records,mismatched_mapping
    output=Path(output);spec=_header(output,'final',model_identity,dataset_identity,allow_incomplete=_allow_incomplete)
    suites=read_json(output/'expected_pairs.json')
    if identity(suites)!=spec['suite_pairs_id'] or (expected_suites is not None and suites!=expected_suites):
        raise ValueError('Expected suite/pair identities differ')
    mapping=read_json(output/'mismatched_images.json')
    if sha256(output/'mismatched_images.json')!=spec['mapping_sha256'] or set(mapping)!=set(suites):
        raise ValueError('Mismatched image plan differs')
    recomputed={}
    for suite,pairs in suites.items():
        ids=[p['pair_id'] for p in pairs]
        if len(ids)!=len(set(ids)):raise ValueError('Duplicate expected pair')
        if mapping[suite]!=mismatched_mapping(pairs,spec['bootstrap_seed']):raise ValueError('Image assignments differ')
        records=[]
        for condition in CONDITIONS:
            expected=condition_records(pairs,suite,condition,mapping[suite])
            current=verify_batches(output/suite/condition,expected=expected,generate=True,model_identity=spec['model_identity'],
                                   expected_spec={**spec,'suite':suite,'condition':condition})
            for pair_id in ids:
                members=[r for r in current if r['pair_id']==pair_id]
                if len(members)!=2 or {r['member_index'] for r in members}!={0,1}:raise ValueError('Incomplete pair')
                correct=all(r['canonical_correct'] for r in members)
                for row in members:row['pair_correct']=correct
            records.extend(current)
        if records!=read_json(output/suite/'records.json'):raise ValueError('Derived per-example records differ')
        recomputed[suite]=summarize_suite(records,seed=spec['bootstrap_seed'],resamples=spec['bootstrap_resamples'])
        if recomputed[suite]!=read_json(output/suite/'summary.json'):raise ValueError('Suite summary recomputation differs')
    if recomputed!=read_json(output/'summary.json'):raise ValueError('Combined summary recomputation differs')
    return {'status':'passed','kind':'final','pairs':{s:len(p) for s,p in suites.items()},'inference_performed':False}


def verify_development(output,*,_allow_incomplete=False):
    output=Path(output);spec=_header(output,'development',allow_incomplete=_allow_incomplete)
    captions=verify_batches(output/'captions',model_identity=spec['model_identity'],expected_spec={**spec,'suite':'captions'})
    summary={'model_identity':spec['model_identity'],'complete':True,'suites':{},
             'caption_reference_loss':sum(r['reference_nll'] for r in captions)/len(captions)}
    for suite in ('grounding','natural'):
        means={}
        for condition in ('correct','complementary','no_image'):
            rows=verify_batches(output/suite/condition,model_identity=spec['model_identity'],
                                expected_spec={**spec,'suite':suite,'condition':condition})
            means[condition]=sum(r['reference_nll'] for r in rows)/len(rows)
        summary['suites'][suite]={'reference_nll':means,'swap_penalty':means['complementary']-means['correct']}
    if summary!=read_json(output/'summary.json'):raise ValueError('Development summary recomputation differs')
    return {'status':'passed','kind':'development','inference_performed':False}
