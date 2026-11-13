"""Small, hash-sealed local deployment bundle; no weights or rollout data."""
from pathlib import Path
import shutil,zipfile
from atml_pa3.task3.io import sha256,atomic_json,read_json
from .live_spec import load_assets

def build(project,output):
    project,output=Path(project),Path(output)
    if output.exists() or output.with_suffix('.zip').exists():raise FileExistsError('Never replace a deployment version')
    output.mkdir(parents=True)
    def copy(src,dest):
        target=output/dest;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(project/src,target)
    copy(Path('src/atml_pa3/__init__.py'),Path('src/atml_pa3/__init__.py'))
    for p in (project/'src/atml_pa3/task3').glob('*.py'):copy(p.relative_to(project),p.relative_to(project))
    for name in ('__init__.py','live_spec.py','live_backend.py','live_run.py','live_store.py','live_analysis.py','live_validation.py','live_export.py'):
        p=Path('src/atml_pa3/task3_part2')/name;copy(p,p)
    for name in ('task3_live.py','task3_gpu_preflight.py','task3_live_export.py','task3_bundle_verify.py'):
        p=Path('scripts')/name;copy(p,p)
    for name in ('__init__.py','conftest.py','test_live.py','test_live_safety.py','test_live_completion.py'):
        p=Path('tests/task3')/name;copy(p,p)
    for name in ('TASK3_DEPLOYMENT.md','TASK3_LIVE_INFERENCE.md'):
        copy(Path(name),Path(name))
    copy(Path('environment/task3-live-requirements.txt'),Path('requirements.txt'))
    shutil.copytree(project/'artifacts/task3/live-assets-v1',output/'assets')
    load_assets(output/'assets')
    files={p.relative_to(output).as_posix():sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}
    atomic_json(output/'BUNDLE_MANIFEST.json',{'files':files,'questions':50,'planned_unique_calls':250,
        'contains_weights':False,'contains_offline_rollouts':False,'uploaded':False})
    verify(output)
    with zipfile.ZipFile(output.with_suffix('.zip'),'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(output).as_posix())
    return {'folder':str(output.resolve()),'zip_sha256':sha256(output.with_suffix('.zip')),
            'zip_bytes':output.with_suffix('.zip').stat().st_size,'files':len(files)}

def verify(output):
    output=Path(output);manifest=read_json(output/'BUNDLE_MANIFEST.json')
    for name,digest in manifest['files'].items():
        if sha256(output/name)!=digest:raise ValueError('Bundle file corrupt: '+name)
    load_assets(output/'assets')
    return {'status':'BUNDLE VERIFIED','files':len(manifest['files']),'questions':50,'model_loaded':False}
