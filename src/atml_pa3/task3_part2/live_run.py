"""Model-independent resumable five-call schedule; grading happens after generation."""
from pathlib import Path
from atml_pa3.task3.io import identity,atomic_json,sha256
from .live_spec import LIMITS,SAMPLING,seed_for,messages,last_boxed,STRATEGIES,load_assets
from .live_store import CallStore,RunLock,read_json

def request_for(question,call,initial=None):
    if call=='revision' and initial is None:raise ValueError('Revision requires immutable shared initial call')
    return {'messages':messages(question['problem'],initial['text'] if call=='revision' else None),
            'max_new_tokens':LIMITS[call],'seed':seed_for(question['sample_id'],call),'sampling':SAMPLING,
            'shared_initial_sha256':identity(initial) if call=='revision' else None}

def execute_calls(store,questions,backend,validate=None):
    """backend.generate receives no gold, correctness or original offline responses."""
    for q in questions:
        initial=None
        for call in LIMITS:
            req=request_for(q,call,initial)
            record=store.read(q['sample_id'],call,req)
            if record is None:
                result=backend.generate(req)
                result={**result,'request':req,'extracted_answer':last_boxed(result['text'])}
                if validate is not None:validate(result)
                record=store.write(q['sample_id'],call,result)
            if call=='parallel0':initial=record

def verify_calls(store,questions):
    expected=set();records={}
    for q in questions:
        initial=None
        for call in LIMITS:
            req=request_for(q,call,initial);r=store.read(q['sample_id'],call,req)
            if r is None:raise ValueError('Incomplete unique calls')
            records[q['sample_id'],call]=r;expected.add(store.path(q['sample_id'],call).name)
            if call=='parallel0':initial=r
    if {p.name for p in (store.root/'calls').glob('*.json')}!=expected:raise ValueError('Unexpected call records')
    return records

def official_config(assets,backend_identity):
    spec,questions,manifest=load_assets(assets)
    source=Path(__file__).parent
    names=('live_spec.py','live_store.py','live_run.py','live_backend.py','live_analysis.py','live_validation.py','live_export.py')
    scorer_names=('equivalence.py','equivalence_worker.py','safe_math.py','math_rpc.py','selection.py','statistics.py','cache.py','io.py','experiment.py')
    return {'protocol':spec,'asset_manifest':identity(manifest),'question_ids':[q['sample_id'] for q in questions],
            'backend':backend_identity,'code':{n:sha256(source/n) for n in names},
            'scoring_code':{n:sha256(source.parent/'task3'/n) for n in scorer_names}}

def check_scoring_dependencies(assets):
    from importlib.metadata import version
    from atml_pa3.task3.equivalence import SCORER
    scorer=read_json(Path(assets)/'scorer_config.json')
    if scorer['math_verify']!=version('math-verify') or SCORER['math_verify_options']!=scorer['math_verify_options']:
        raise ValueError('Frozen MATH scorer dependency/options differ')
    if sha256(Path(__file__).parent.parent/'task3/equivalence.py')!=scorer['equivalence_code_sha256']:
        raise ValueError('Frozen scorer implementation differs')
    for name,expected in {'sympy':'1.14.0','latex2sympy2_extended':'1.11.0','antlr4-python3-runtime':'4.11.0'}.items():
        if version(name)!=expected:raise ValueError('Scorer dependency differs: '+name)

def verify_preflight(path,config):
    from .live_backend import smoke_request,stress_requests,validate_stress
    path=Path(path);proof=read_json(path/'PASSED.json')
    if (proof.get('backend')!=config['backend'] or proof.get('assets')!=config['asset_manifest']
            or proof.get('backend_code_sha256')!=config['code']['live_backend.py']):
        raise ValueError('Missing/incompatible preflight evidence')
    store=CallStore(path,read_json(path/'run_config.json'))
    smoke=store.read('PREFLIGHT_NON_OFFICIAL','smoke',smoke_request())
    if smoke is None or identity(smoke)!=proof.get('call_sha256'):
        raise ValueError('Preflight call evidence corrupt')
    requests=stress_requests();hashes=proof.get('stress_call_hashes',{})
    if set(hashes)!=set(requests):raise ValueError('Full-length memory probe evidence is required')
    expected={store.path('PREFLIGHT_NON_OFFICIAL',c).name for c in ('smoke',*requests)}
    if {p.name for p in (path/'calls').glob('*.json')}!=expected:raise ValueError('Preflight call coverage differs')
    records={'smoke':smoke}
    for call,request in requests.items():
        record=store.read('PREFLIGHT_NON_OFFICIAL',call,request)
        if record is None or identity(record)!=hashes[call]:
            raise ValueError('Memory probe call evidence corrupt')
        validate_stress(record,request)
        records[call]=record
    return {'proof':proof,'setup':read_json(path/'setup.json'),'records':records}

def validate_completed_run(assets,output):
    from .live_validation import completion_evidence,validate_preflight_snapshot
    output=Path(output)
    complete=read_json(output/'COMPLETE.json')
    _,qs,_=load_assets(assets)
    config=read_json(output/'run_config.json')
    store=CallStore(output,config)
    expected_names={store.path(q['sample_id'],call).name for q in qs for call in LIMITS}
    if (complete.get('generation_is_real') is not True or complete.get('config_id')!=store.config_id
            or set(complete.get('calls',{}))!=expected_names):
        raise ValueError('Official completion or call hash index coverage differs')
    if not isinstance(config.get('backend'),dict):
        raise ValueError('Missing official backend provenance')
    snapshot=read_json(output/'preflight.json')
    expected_config={**official_config(assets,config['backend']),'preflight_id':identity(snapshot)}
    if identity(config)!=identity(expected_config):raise ValueError('Official run configuration differs')
    validate_preflight_snapshot(snapshot,config,assets)
    records=verify_calls(store,qs)
    expected,rows=completion_evidence(assets,store,qs,records)
    if identity(complete)!=identity(expected):raise ValueError('Official completion hashes or counts differ')
    return qs,records,rows

def launch(assets,model_path,output,*,execute=False,preflight_path=None):
    spec,qs,manifest=load_assets(assets)
    check_scoring_dependencies(assets)
    if not execute:
        # Default mode imports no torch backend and loads no tokenizer/model.
        return {'status':'VALIDATED; GENERATION NOT EXECUTED','questions':len(qs),'unique_calls_planned':250,
                'model_assets_checked':False,'requires':'explicit --execute-generation after bounded GPU preflight'}
    from .live_backend import QwenBackend,validate_model_files,backend_identity,check_gpu
    from .live_validation import completion_evidence,load_setups,validate_preflight_snapshot,validate_saved_call,validate_setup
    lock=read_json(Path(assets)/'model_files.json');validate_model_files(model_path,lock);check_gpu()
    config=official_config(assets,backend_identity(lock))
    if preflight_path is None:raise ValueError('A successful compatible bounded preflight is required')
    snapshot=verify_preflight(preflight_path,config)
    validate_preflight_snapshot(snapshot,config,assets)
    config['preflight_id']=identity(snapshot)
    with RunLock(output):
        store=CallStore(output,config)
        if (store.root/'COMPLETE.json').exists():
            validate_completed_run(assets,output)
            return {'status':'COMPLETE','unique_calls':250}
        snapshot_path=store.root/'preflight.json'
        if snapshot_path.exists():
            if identity(read_json(snapshot_path))!=config['preflight_id']:raise ValueError('Saved preflight differs')
        else:
            atomic_json(snapshot_path,snapshot)
        backend=None
        setups=load_setups(store,assets)
        # Verify every already-saved request before allocating model weights.
        pending=False
        for q in qs:
            initial=store.read(q['sample_id'],'parallel0')
            for call in LIMITS:
                if call=='revision' and initial is None:
                    if store.path(q['sample_id'],call).exists():raise ValueError('Revision without shared initial')
                    pending=True;continue
                record=store.read(q['sample_id'],call,request_for(q,call,initial))
                if record is None:pending=True
                else:validate_saved_call(record,setups)
        try:
            if pending:
                backend=QwenBackend(model_path,lock)
                setup_id=identity(backend.setup)
                setups[setup_id]=(backend.setup,validate_setup(backend.setup,config['backend'],assets))
                atomic_json(store.root/'setups'/f'{setup_id}.json',backend.setup)
                execute_calls(store,qs,backend,validate=lambda record:validate_saved_call(record,setups))
            records=verify_calls(store,qs)
        finally:
            if backend is not None:backend.close()
        from .live_analysis import grade_calls,export_live_tables,select_live_examples,csv_rows
        grade_calls(store,qs,records)
        complete,rows=completion_evidence(assets,store,qs,records)
        csv_rows(store.root/'strategy_results.csv',rows)
        export_live_tables(store.root,qs,rows,records)
        select_live_examples(store.root,qs,rows,records)
        atomic_json(store.root/'COMPLETE.json',complete)
    return {'status':'COMPLETE','unique_calls':250}
