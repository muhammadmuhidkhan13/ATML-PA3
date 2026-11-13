from pathlib import Path
import numpy as np
import pytest
from atml_pa3.task3.io import read_json,sha256
from atml_pa3.task3.data import DatasetSource,assert_source_identity,validate_mathvista_metadata
from atml_pa3.task3.scoring import base_scores
from atml_pa3.task3.visconf import calibrate

ROOT=Path(__file__).resolve().parents[2]

def ready():
    if not (ROOT/'ATML-PA3-Data/README.MD').exists():pytest.skip('Instructor rollout data absent; no synthetic substitution')

def test_real_source_hashes_counts_and_saved_splits():
    ready();prepared=ROOT/'artifacts/task3/prepared-v1'
    if not (prepared/'source_manifest.json').exists():pytest.skip('Run task3_prepare.py first')
    assert_source_identity(ROOT/'ATML-PA3-Data',prepared)
    for dataset,tokens,capped,missing in [('math',20742094,523,521),('mathvista',8961703,487,3032)]:
        source=DatasetSource(ROOT/'ATML-PA3-Data',dataset)
        audit=read_json(prepared/dataset/'source_audit.json');split=read_json(prepared/dataset/'split.json')
        assert len(source.ids)==1000 and audit['counts']=={'rows':32000,'tokens':tokens,'capped':capped,'missing_extracted':missing}
        assert len(split['development'])==200 and len(split['evaluation'])==800
        assert set(split['development'])|set(split['evaluation'])==set(source.ids)
        if dataset=='math':assert split['live_math_50']==split['evaluation'][:50]

def test_real_mathvista_metadata_and_full_prefix_covariance():
    ready();metadata=ROOT/'artifacts/task3/reference-v1/mathvista_testmini.json'
    if not metadata.exists():pytest.skip('Released metadata absent')
    validate_mathvista_metadata(read_json(metadata));source=DatasetSource(ROOT/'ATML-PA3-Data','mathvista')
    assert set(source.ids)==set(read_json(metadata))
    sid=read_json(ROOT/'artifacts/task3/prepared-v1/mathvista/split.json')['development'][0]
    base=[base_scores(row) for row in source.question(sid)]
    scores=calibrate([b['kl'] for b in base],[b['v'] for b in base])
    for c,b,z,zt in zip(scores['calibrated'],base,scores['visconf'],scores['tokenwise']):
        expected=np.mean((c-np.mean(c))*(b['v']-np.mean(b['v'])))
        assert np.isclose(zt-z,expected,atol=1e-12)
