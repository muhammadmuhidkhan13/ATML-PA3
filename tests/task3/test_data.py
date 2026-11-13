from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from atml_pa3.task3.data import DatasetSource
from atml_pa3.task3.data import validate_mathvista_metadata

def write_source(root,*,duplicate=False,unmatched=False,shifted=False,wrong_length=False):
    folder=root/'qwen25vl-mathvista';folder.mkdir(parents=True)
    g=[];p=[];a=[]
    for sid in ('1','2'):
        for rid in range(32):
            k={'dataset':'mathvista','sample_id':sid,'rollout_id':rid}
            g.append({**k,'generated_token_ids':[1,2],'n_generated_text_tokens':3 if wrong_length else 2,
                      'gold_answer':'A','extracted_answer':None,'question_text':'Q','hit_max_new_tokens':False,'generated_text':'x'})
            p.append({**k,'logp':[-1.,-2.],'kl_u_p':[2.,3.]})
            a.append({**k,'n_generated_text_tokens':[1,2] if shifted else [0,1],
                      'n_image_tokens':4,'n_prompt_text_tokens':2,'img_attn_total':[.1,.2],
                      'prompt_text_attn_total':[.2,.3],'generated_text_attn_total':[0.,.1]})
    if duplicate:g.append(g[0])
    if unmatched:p.pop()
    for name,rows in (('generations',g),('probability_metrics',p),('attention_metrics',a)):
        pq.write_table(pa.Table.from_pylist(list(reversed(rows)) if name=='probability_metrics' else rows),folder/f'{name}.parquet',row_group_size=32)

def test_exact_three_key_join_independent_of_row_storage_order(tmp_path):
    write_source(tmp_path);source=DatasetSource(tmp_path,'mathvista',expected_questions=2)
    assert len(source.question('1'))==32
    assert source.question('1')[0]['rollout_id']==0
    report=source.audit();assert report['counts']['rows']==64 and report['counts']['tokens']==128
    assert source.question('1')[0]['n_generated_text_tokens']==2
    assert source.question('1')[0]['preceding_generated_counts']==[0,1]

@pytest.mark.parametrize('problem',['duplicate','unmatched','shifted','wrong_length'])
def test_invalid_source_is_detected(tmp_path,problem):
    write_source(tmp_path,**{problem:True})
    with pytest.raises(ValueError):DatasetSource(tmp_path,'mathvista',expected_questions=2).audit()

def test_missing_numeric_metadata_blocks_real_scoring():
    m={'1':{'question_type':'free_form','answer_type':'float','choices':None,'precision':None,'answer':'1.2'}}
    with pytest.raises(ValueError,match='precision'):validate_mathvista_metadata(m)

@pytest.mark.parametrize('choices,expected',[(['(c)','(a)','(e)','(d)','(b)'],'E'),(['(c)','(d)','(a)','(b)','(e)'],'D')])
def test_released_gold_choice_text_is_not_reinterpreted_as_predicted_letter(choices,expected):
    from atml_pa3.task3.cache import metadata_reference
    m={'question_type':'multi_choice','answer_type':'text','choices':choices,'precision':None,'answer':'(b)'}
    assert metadata_reference(m)==expected
