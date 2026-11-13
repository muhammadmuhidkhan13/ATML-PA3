import dataclasses
import numpy as np
import pytest
from atml_pa3.task3.preparation import split_ids,rollout_orders,BUDGETS
from atml_pa3.task3.equivalence import math_equivalent,semantics,normalize_mathvista,NO_ANSWER,prefix_groups
from atml_pa3.task3.selection import Candidate,select,descending_average_ranks,methods
from atml_pa3.task3.scoring import base_scores,visual_engagement
from atml_pa3.task3.visconf import calibrate

def test_exact_split_and_live_subset():
    ids=[str(i) for i in range(1000)];actual=split_ids(ids)
    expected=np.random.default_rng(6304).permutation(1000)
    assert actual['permutation_indices']==expected.tolist()
    assert len(actual['development'])==200 and len(actual['evaluation'])==800
    assert actual['live_math_50']==actual['evaluation'][:50]
    assert not set(actual['development'])&set(actual['evaluation'])
    assert actual['ordered_ids']==sorted(ids) and actual==split_ids(list(reversed(ids)))

def test_ten_shared_nested_orders():
    orders=rollout_orders(['b','a'])
    assert orders==rollout_orders(['a','b'])
    for values in orders.values():
        assert len(values)==10
        for order in values:
            assert sorted(order)==list(range(32))
            for n in BUDGETS:assert order[:n]==order[:32][:n]

@pytest.mark.parametrize('a,b,expected',[(r'\frac{1}{2}','0.5',True),('x+x','2x',True),('-2','2',False),
                                      (r'\$40','40',True),('1/3','2/3',False),(None,'0',False)])
def test_math_equivalence(a,b,expected):assert math_equivalent(a,b)==expected

def test_gold_cannot_change_equivalence():
    a=semantics('math',['1/2','0.5',None,'0.2'],'1/2')
    b=semantics('math',['1/2','0.5',None,'0.2'],'0.2')
    assert a['equivalent']==b['equivalent'] and a['correct']!=b['correct']
    assert a['correct'][2] is False
    assert prefix_groups([0,1,2],a['equivalent'],a['missing'])==['group:0','group:0',NO_ANSWER]

def test_complete_link_prefix_groups_do_not_read_future():
    relation=[[True,True,False],[True,True,True],[False,True,True]]
    assert prefix_groups([0,1],relation,[False]*3)==['group:0','group:0']
    assert prefix_groups([0,1,2],relation,[False]*3)==['group:0','group:0','group:2']

@pytest.mark.parametrize('answer,expected',[('B','B'),('(b) text','B'),('blue','B'),('',NO_ANSWER),(None,NO_ANSWER)])
def test_choice_normalization(answer,expected):
    m={'question_type':'multi_choice','answer_type':'text','choices':['red','blue'],'precision':None}
    assert normalize_mathvista(answer,m)==expected

@pytest.mark.parametrize('kind,precision,answer,expected',[('float',2,'1.234','1.23'),('float',1,'1.25','1.2'),
    ('integer',None,'3.9','3'),('integer',None,'x',NO_ANSWER),('list',None,'[1, 2]','[1, 2]'),('text',None,' Cat ','Cat')])
def test_metadata_specific_types(kind,precision,answer,expected):
    assert normalize_mathvista(answer,{'question_type':'free_form','answer_type':kind,'choices':None,'precision':precision})==expected

def test_missing_metadata_refused():
    with pytest.raises(ValueError,match='metadata'):normalize_mathvista('3',{})

def candidates(groups=('a','b','b')):return [Candidate(i,g,10+i) for i,g in enumerate(groups)]

def test_structural_gold_separation_and_costs():
    assert {f.name for f in dataclasses.fields(Candidate)}=={'rollout_id','answer_group','token_count'}
    c=candidates();first=select(c,'first');sc=select(c,'consistency')
    assert first.index==0 and first.token_cost==10 and first.consumed==1
    assert sc.index==1 and sc.token_cost==33 and sc.consumed==3
    assert select(candidates(('a','b')),'consistency').index==0
    assert select(c,'highest',[1.,3.,3.]).index==1

@pytest.mark.parametrize('vision',[False,True])
def test_all_methods_n1(vision):
    for method,score in methods(vision).values():
        result=select(candidates()[:1],method,[0.])
        assert result.index==0 and result.token_cost==10

def test_rwc_average_rank_and_votes():
    np.testing.assert_array_equal(descending_average_ranks([4,4,2,0]),[1.5,1.5,3,4])
    c=candidates(('a','b','b'))
    assert select(c,'rwc',[3.,2.,1.]).index==0 # weights1,.5,0
    assert select(c,'rwc',[1.,1.,1.]).index==1 # all weights.5 => majority
    assert select(candidates(('b','a')),'rwc',[2.,2.]).index==0

def test_rwc_mathematical_vote_tie_is_not_broken_by_float_summation():
    c=candidates(('b','a','c','b','a','d','e','f'))
    # b:6/7+4/7; a:7/7+3/7. Float addition differs by2e-16.
    assert select(c,'rwc',[7,8,6,5,4,3,2,1]).index==0

def test_token_scores_and_invalid_arrays():
    row={'generated_token_ids':[1,2],'logp':[-1,-3],'kl_u_p':[2,6]}
    s=base_scores(row);assert s['mean_logp']==-2 and s['self_certainty']==4
    row['logp']=[];assert base_scores(row)['mean_logp']==-float('inf')
    assert base_scores({'generated_token_ids':[],'logp':[],'kl_u_p':[]})['self_certainty']==-float('inf')

def test_normalized_visual_engagement_zero_and_nonfinite():
    v,valid=visual_engagement([2,0,float('nan')],[2,0,1],[4,0,1],3)
    np.testing.assert_array_equal(v,[.25,0,0]);assert valid.tolist()==[True,False,False]

def test_visconf_hand_calculation_and_prefix_only_calibration():
    one=calibrate([[0.,10.]],[[1.,0.]])
    assert one['q05']==.5 and one['q95']==9.5
    assert one['visconf']==[.25] and one['tokenwise']==[0.] and one['covariance']==[-.25]
    both=calibrate([[0.,10.],[100.,200.]],[[1.,0.],[.2,.2]])
    assert both['visconf'][0]!=one['visconf'][0]
    empty=calibrate([[]],[[]]);assert empty['visconf']==[0.]

def test_all_equal_calibration_and_missing_values():
    c=calibrate([[2.,2.],[float('nan')]],[[.2,.4],[0.]])
    assert c['visconf']==[0.,0.] and c['q05']==c['q95']==2
