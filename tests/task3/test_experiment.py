import copy
import numpy as np
from atml_pa3.task3.equivalence import semantics,NO_ANSWER
from atml_pa3.task3.experiment import evaluate_question
from atml_pa3.task3.preparation import rollout_orders
from atml_pa3.task3.selection import Candidate,select

def fixture():
    rows=[]
    for i in range(32):
        n=i+1
        rows.append({'generated_token_ids':list(range(n)),'logp':[-1-i/32]*n,'kl_u_p':[1+i/32]*n,
                     'img_attn_total':[1.]*n,'prompt_text_attn_total':[1+i/32]*n,'generated_text_attn_total':[0.]*n,
                     'extracted_answer':'A' if i%3==0 else 'B'})
    meta={'question_type':'multi_choice','answer_type':'text','choices':['red','blue'],'precision':None}
    sem=semantics('mathvista',[r['extracted_answer'] for r in rows],'A',meta)
    return rows,sem,rollout_orders(['q'])['q']

def test_full_fixed_method_matrix_and_oracle_monotonicity():
    rows,sem,orders=fixture();result,*_=evaluate_question(rows,sem,orders)
    assert len(result)==10*6*12
    for order in range(10):
        one=[r for r in result if r['order']==order and r['budget']==1]
        assert len({r['selected_rollout_id'] for r in one})==1
        first=sorted([r for r in result if r['order']==order and r['method']=='FirstAns'],key=lambda r:r['budget'])
        assert len({r['token_cost'] for r in first})==1
        assert all(a['oracle']<=b['oracle'] for a,b in zip(first,first[1:]))
    for row in result:
        assert row['sampling_failure']+row['selection_failure']+row['selection_success']==1
        if row['method']!='FirstAns':assert row['token_cost']==sum(len(rows[i]['generated_token_ids']) for i in orders[row['order']][:row['budget']])

def test_unrevealed_quality_cannot_change_current_prefix():
    rows,sem,orders=fixture();before,*_=evaluate_question(rows,sem,orders)
    changed=copy.deepcopy(rows)
    for i in set(range(32))-set(orders[0][:4]):changed[i]['kl_u_p']=[10000.]*len(changed[i]['generated_token_ids'])
    after,*_=evaluate_question(changed,sem,orders)
    a=[r for r in before if r['order']==0 and r['budget']==4]
    b=[r for r in after if r['order']==0 and r['budget']==4]
    assert a==b

def test_missing_answers_can_win_and_remain_incorrect():
    meta={'question_type':'multi_choice','answer_type':'text','choices':['red','blue'],'precision':None}
    sem=semantics('mathvista',[None,'','A'],'A',meta)
    c=[Candidate(0,NO_ANSWER,9),Candidate(1,NO_ANSWER,8),Candidate(2,'A',7)]
    selected=select(c,'consistency')
    assert selected.answer_group==NO_ANSWER and selected.token_cost==24 and not sem['correct'][selected.index]
