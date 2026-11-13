import numpy as np
import pytest
from atml_pa3.task3.selection import Candidate
from atml_pa3.task3.equivalence import NO_ANSWER
from atml_pa3.task3_part2.adaptive import Policy,evidence,should_stop,decide,choose_policy,policies

def snapshot(groups,quality=None):
    return evidence([Candidate(i,g,i+1) for i,g in enumerate(groups)],quality if quality is not None else list(range(len(groups))))

def test_agreement_boundaries_and_unique_missing():
    s=snapshot(['a','a','a','b'])
    assert should_stop(Policy('agreement',.75),s)
    assert not should_stop(Policy('agreement',.875),s)
    assert not should_stop(Policy('agreement',.5),snapshot(['a','a','b','b']))
    assert not should_stop(Policy('agreement',.5),snapshot([NO_ANSWER]*3+['a']))
    assert s['selected_rollout_id']==0 and s['token_cost']==10

def test_rwc_hand_computed_margin_and_disagreement():
    # Ranks1,2,3,4 give weights1,2/3,1/3,0. A gets5/3, B1/3.
    s=snapshot(['a','a','b','b'],[4,3,2,1])
    assert s['rwc_margin']==pytest.approx(2/3)
    assert s['rwc_vote_numerators']=={'a':10,'b':2}
    s=snapshot(['a','a','a','b'],[1,1,1,10])
    assert s['rwc_margin']==0 and s['rwc_agrees'] # tied votes, earliest support
    assert should_stop(Policy('quality',.75,0),s)
    assert not should_stop(Policy('quality',.75,.1),s)
    s=snapshot(['a']*5+['b']*3,[1]*5+[10]*3)
    assert not s['rwc_agrees'] and not should_stop(Policy('quality',.625,0),s)

def test_single_group_and_all_quality_ties():
    s=snapshot(['a']*4,[1]*4)
    assert s['rwc_margin']==1 and should_stop(Policy('quality',1,.5),s)

@pytest.mark.parametrize('stop',[4,8,16,32])
def test_checkpoints_and_exact_cost(stop):
    ss={n:snapshot(['a']*n if n>=stop else ['a','b']*(n//2)) for n in (4,8,16,32)}
    d=decide(Policy('agreement',1),ss)
    assert d['n']==stop and d['token_cost']==stop*(stop+1)//2

def test_no_future_or_gold_interface():
    ss={n:snapshot(['a']*n) for n in (4,8,16,32)}
    before=decide(Policy('quality',1,.5),ss)
    ss[32]=snapshot([NO_ANSWER]*32)
    assert before==decide(Policy('quality',1,.5),ss)
    assert set(Candidate.__dataclass_fields__)=={'rollout_id','answer_group','token_count'}

def test_dev_selection_tolerance_cost_accuracy_and_ties():
    grid=[dict(family='always32',accuracy=.8,token_cost=100,tau=None,gamma=None),
          dict(family='agreement',accuracy=.789,token_cost=1,tau=.5,gamma=None),
          dict(family='agreement',accuracy=.79,token_cost=50,tau=.875,gamma=None),
          dict(family='agreement',accuracy=.79,token_cost=50,tau=.75,gamma=None),
          dict(family='agreement',accuracy=.81,token_cost=51,tau=.5,gamma=None)]
    assert choose_policy(grid,'agreement',.8)['tau']==.75
    assert choose_policy(grid[:2],'agreement',.8)['family']=='always32'
    grid=[dict(family='quality',accuracy=.8,token_cost=50,tau=t,gamma=g) for t,g in ((.75,.1),(.625,.25),(.625,.1))]
    assert choose_policy(grid,'quality',.8)['gamma']==.1
    assert len(policies())==26
