from pathlib import Path
import numpy as np
import pytest
from atml_pa3.task3.io import read_json,identity,sha256
from atml_pa3.task3.data import DatasetSource
from atml_pa3.task3.scoring import base_scores
from atml_pa3.task3.visconf import calibrate

ROOT=Path(__file__).resolve().parents[2]

def test_real_prefix_cache_is_exact_current_candidate_calibration():
    if not (ROOT/'ATML-PA3-Data').is_dir():
        pytest.skip('Instructor rollout data is required for the real calibration check')
    a=ROOT/'artifacts/task3';ds='mathvista';split=read_json(a/'prepared-v1'/ds/'split.json');sid=split['development'][0]
    source=DatasetSource(ROOT/'ATML-PA3-Data',ds);base=[base_scores(r) for r in source.question(sid)]
    orders=read_json(a/'prepared-v1'/ds/'orders.json')['orders'][sid]
    with np.load(a/'offline-v1'/ds/'development/questions'/f'{identity(sid)}.scores.npz',allow_pickle=False) as f:
        for oi in (0,1):
            for n in (4,8,16):
                prefix=orders[oi][:n];expected=calibrate([base[i]['kl'] for i in prefix],[base[i]['v'] for i in prefix]);j=(1,2,4,8,16,32).index(n)
                assert np.array_equal(f['visconf'][oi,j,:n],expected['visconf'])
                assert np.isnan(f['visconf'][oi,j,n:]).all()
                assert np.array_equal(f['quantiles'][oi,j],[expected['q05'],expected['q95']])

def test_real_adaptive_freeze_and_complete_heldout_coverage():
    d=ROOT/'artifacts/task3/adaptive-v1';config=read_json(d/'frozen_config.json')
    assert sha256(d/'frozen_config.json')==read_json(d/'FROZEN.json')['sha256']
    for ds in ('math','mathvista'):
        manifest=read_json(d/ds/'evaluation/manifest.json')
        assert manifest['complete'] and len(manifest['question_ids'])==800 and len(manifest['files'])==800
        assert manifest['config_id']==identity(config)
        means=np.load(d/ds/'evaluation/question_means.npz',allow_pickle=False)
        assert means['values'].shape==(800,6,15)
        assert np.isfinite(means['values']).all()
        for policy in ('agreement','quality'):
            chosen=config['selected'][ds][policy]
            assert chosen['accuracy']>=config['fixed32_development_accuracy'][ds]-.01-1e-12
