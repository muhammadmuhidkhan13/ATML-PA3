import numpy as np
import pytest
from atml_pa3.task3.statistics import aggregate_orders,paired_bootstrap,failure_decomposition
from atml_pa3.task3.diagnostics import correlation,quartiles

def test_orders_are_averaged_within_questions_first():
    values=np.array([[1]*10,[0]*10,[1,0]*5])
    question,overall=aggregate_orders(values)
    np.testing.assert_array_equal(question,[1,0,.5]);assert overall==.5

def test_paired_question_bootstrap_independent_quantiles():
    a=np.array([1.,0.,.5]);b=np.array([0.,0.,.5])
    indices=np.random.default_rng(6304).integers(3,size=(1000,3))
    expected=np.quantile((a-b)[indices].mean(1),[.025,.975])
    actual=paired_bootstrap(a,b)
    assert actual['mean']==pytest.approx(1/3)
    np.testing.assert_array_equal(actual['ci95'],expected)
    with pytest.raises(ValueError):paired_bootstrap(np.ones((3,10)),np.zeros((3,10)))

@pytest.mark.parametrize('correct,selected,expected',[([False,False],1,(1,0,0)),([True,False],1,(0,1,0)),([True,False],0,(0,0,1))])
def test_failure_partition(correct,selected,expected):
    result=failure_decomposition(correct,selected)
    assert tuple(result[k] for k in ('sampling_failure','selection_failure','selection_success'))==expected

def test_quartiles_and_correlations_known_values():
    q=quartiles(range(32),[False]*24+[True]*8)
    assert q['top_accuracy']==1 and q['bottom_accuracy']==0 and q['gap']==1
    assert not set(q['top_rollouts'])&set(q['bottom_rollouts'])
    assert quartiles([1]*32,[True]*32)['gap'] is None
    assert correlation([1,2,3],[3,2,1],rank=True)==-1
    assert correlation([1,1],[2,3]) is None
