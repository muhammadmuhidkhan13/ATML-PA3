"""Verify preserved assets and seal the completed local Part2 implementation."""
import sys,re,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from atml_pa3.task3.io import read_json,atomic_json,sha256
from atml_pa3.task3_part2.package import verify
from atml_pa3.task3_part2.adaptive_run import specification

def seal(project):
    project=Path(project).resolve();a=project/'artifacts/task3';out=a/'part2-audit-v1';out.mkdir(exist_ok=True)
    baseline=read_json(a/'resource-audit-v1/task12_baseline.json');changed=[]
    for name,digest in baseline.items():
        if sha256(project/name)!=digest:changed.append(name)
    old=read_json(a/'resource-audit-v1/preservation.json')
    for name,digest in old['immutable_asset_hashes'].items():
        if sha256(project/name)!=digest:changed.append(name)
    p1=read_json(a/'resource-audit-v1/source_inventory.json')
    for name,digest in p1.items():
        if Path(name).name!='TASK3_STATUS.md' and sha256(name)!=digest:changed.append(name)
    artifacts=read_json(a/'resource-audit-v1/offline_artifact_manifest.json')['files']
    for name,record in artifacts.items():
        if sha256(a/'offline-v1'/name)!=record['sha256']:changed.append('offline-v1/'+name)
    if changed:raise ValueError('Protected assets changed: '+str(changed))
    atomic_json(out/'preservation.json',{'status':'PASSED','changed_files':changed,
        'task12_baseline_files':len(baseline),'part1_source_files_checked':len(p1)-1,
        'part1_output_files_checked':len(artifacts),'intentional_existing_file_update':'TASK3_STATUS.md',
        'immutable_task12_assets':old['immutable_asset_hashes'],'remote_access':False,'training':False,'official_live_calls':0})
    adaptive=a/'adaptive-v1';validation=read_json(adaptive/'validation.json')
    if validation['status']!='PASSED' or validation['total_outcomes']!=216000:raise ValueError('Adaptive validation incomplete')
    if read_json(adaptive/'frozen_config.json')['specification']!=specification(project):raise ValueError('Frozen controller code differs')
    tests={}
    for key,name,expected in [('task3','part2-task3-tests-final.log',83),('combined','part2-regression-final.log',272),('bundle','part2-bundle-v2-tests.log',20)]:
        text=(a/name).read_text(encoding='utf-8-sig');match=re.search(r'(\d+) passed in ([\d.]+)s',text)
        if not match or int(match[1])!=expected or re.search(r'\d+ (failed|error|skipped)',text):raise ValueError('Fresh test evidence differs: '+name)
        tests[key]={'passed':int(match[1]),'seconds':float(match[2]),'log':str((a/name).resolve()),'sha256':sha256(a/name)}
    bundle=verify(a/'deployment-v2');zip_path=a/'deployment-v2.zip'
    with zipfile.ZipFile(zip_path) as z:
        if z.testzip() is not None:raise ValueError('Corrupt ZIP')
        for name in z.namelist():
            if z.read(name)!=(a/'deployment-v2'/name).read_bytes():raise ValueError('ZIP/stage differs')
    files=list((project/'src/atml_pa3/task3_part2').glob('*.py'))
    names=['task3_adaptive.py','task3_live_assets.py','task3_live.py','task3_gpu_preflight.py','task3_live_export.py',
        'task3_part2_export.py','task3_part2_validate.py','task3_compute_analysis.py','task3_package.py','task3_bundle_verify.py','task3_part2_seal.py']
    files += [project/'scripts'/n for n in names]
    files += [project/'tests/task3'/n for n in ('test_adaptive.py','test_live.py','test_live_safety.py','test_part2_integration.py')]
    files += [project/n for n in ('TASK3_STATUS.md','TASK3_PART2_VALIDATION.md','TASK3_ADAPTIVE_METHODS.md','TASK3_LIVE_INFERENCE.md','TASK3_DEPLOYMENT.md','environment/task3-live-requirements.txt')]
    inventory={str(p.resolve()):sha256(p) for p in sorted(files)}
    atomic_json(out/'source_inventory.json',inventory)
    (out/'source_inventory.txt').write_text('\n'.join(inventory)+'\n',encoding='utf-8')
    manifest={}
    for root in (adaptive,a/'live-assets-v1'):
        for p in sorted(root.rglob('*')):
            if p.is_file():manifest[p.relative_to(a).as_posix()]={'bytes':p.stat().st_size,'sha256':sha256(p)}
    atomic_json(out/'artifact_manifest.json',{'files':manifest})
    config=read_json(adaptive/'frozen_config.json')
    atomic_json(out/'completion.json',{'classification':'READY FOR A100 LIVE-INFERENCE PREFLIGHT',
        'implementation_stage_complete':True,'real_live_experiment_complete':False,'adaptive_outcomes':216000,
        'development_outcomes':120000,'heldout_outcomes':96000,'adaptive_figures':26,
        'selected':{ds:{family:{k:r[k] for k in ('policy_id','tau','gamma')} for family,r in ss.items()} for ds,ss in config['selected'].items()},
        'frozen_config_sha256':sha256(adaptive/'frozen_config.json'),'tests':tests,'bundle':bundle,
        'bundle_zip':str(zip_path),'bundle_zip_bytes':zip_path.stat().st_size,'bundle_zip_sha256':sha256(zip_path),
        'source_files':len(files),'source_inventory_sha256':sha256(out/'source_inventory.json'),
        'artifact_manifest_sha256':sha256(out/'artifact_manifest.json'),
        'exact_frozen_live_questions':50,'unique_calls_planned':250,'unique_official_calls_completed':0,
        'full_qwen_loaded':False,'tiny_random_cpu_model_tested':True,'mock_schedule_tested':True,
        'task12_and_part1_preserved':True,'remote_access':False,'training':False,'committed_pushed_uploaded':False,
        'remaining':['separately authorized A100 preflight','250 real live calls after authorization','live statistical/figure exports after actual calls'],
        'limitations':['frozen Part1 conservative mathematical grading','no full Qwen/FP16 CUDA validation locally','cross-stack random sampling not bitwise guaranteed']})
    return read_json(out/'completion.json')

if __name__=='__main__':print(seal(Path(__file__).resolve().parents[1]))
