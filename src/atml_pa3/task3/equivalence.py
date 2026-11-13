"""Gold-free equivalence matrices; grading is a separate returned field."""
from functools import lru_cache
from importlib.metadata import version
import math,re
from math_verify import parse,verify,LatexExtractionConfig

NO_ANSWER='NO_ANSWER'
MATH_ENGINE=None
SCORER={'version':'task3-scorer-v2','math_verify':version('math-verify'),
        'math_parse':'boxed supplied extraction; LatexExtractionConfig; first_match; no_fallback',
        'math_verify_options':{'strict':True,'float_rounding':6,'numeric_precision':15},
        'library_timeouts':None,'external_operation_timeout_seconds':5,'external_question_timeout_seconds':120,
        'grouping':'prefix order complete-link, mutual equivalence, earliest compatible group',
        'mathvista_source_revision':'c5acd37d75a8c8f9756cb66c78fe2e54fa2c31fb',
        'mathvista_missing':'NO_ANSWER before official normalization; always incorrect',
        'mathvista_choices':'official edit-distance fallback; first tied choice; output option letter'}

def missing(answer):return answer is None or not str(answer).strip()

@lru_cache(maxsize=20000)
def parse_math(answer):
    if missing(answer):return ()
    text=str(answer).strip()
    # Inputs already ARE supplied extractions; wrapping makes a bare algebraic
    # expression explicit to the parser, not a re-extraction from response text.
    if text.startswith('$') and text.endswith('$'):text=text[1:-1]
    if text.startswith('\\(') and text.endswith('\\)'):text=text[2:-2]
    return tuple(parse('\\boxed{'+text+'}',extraction_config=[LatexExtractionConfig()],
                       fallback_mode='no_fallback',extraction_mode='first_match',parsing_timeout=None))

@lru_cache(maxsize=100000)
def math_equivalent(a,b):
    if MATH_ENGINE is not None:return MATH_ENGINE.equivalent(a,b)
    left,right=parse_math(a),parse_math(b)
    if not left or not right:return False
    kw={'strict':True,'float_rounding':6,'numeric_precision':15,'timeout_seconds':None}
    return bool(verify(list(left),list(right),**kw) and verify(list(right),list(left),**kw))

def edit_distance(a,b):
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]

def normalize_mathvista(answer,metadata):
    if missing(answer):return NO_ANSWER
    required={'question_type','answer_type','choices','precision'}
    if not required<=metadata.keys():raise ValueError('Missing MathVista question metadata')
    text=str(answer).strip();kind=metadata['answer_type']
    if metadata['question_type']=='multi_choice':
        choices=metadata['choices']
        if not choices:raise ValueError('Missing choice options')
        letters=[chr(65+i) for i in range(len(choices))]
        found=re.findall(r'\(([a-zA-Z])\)',text)
        if found:text=found[0].upper()
        if text in letters:return text
        return letters[min(range(len(choices)),key=lambda i:edit_distance(text,choices[i]))]
    try:
        if kind=='integer':
            number=float(text)
            if not math.isfinite(number):return NO_ANSWER
            return str(int(number))
        if kind=='float':
            if metadata['precision'] is None:raise ValueError('Float precision is missing')
            number=float(text)
            if not math.isfinite(number):return NO_ANSWER
            return str(round(number,int(metadata['precision'])))
        if kind=='list':return str(answer)
        # Only enable a text type if released metadata actually declares it.
        if kind in ('text','string'):return text
    except (ValueError,TypeError,OverflowError):return NO_ANSWER
    raise ValueError(f'Unsupported released answer type:{kind}')

def semantics(dataset,answers,gold,metadata=None):
    """Pairwise relation uses candidate answers only; gold cannot alter grouping."""
    n=len(answers);is_missing=[missing(a) for a in answers]
    if dataset=='math':
        normalized=[NO_ANSWER if m else str(a) for a,m in zip(answers,is_missing)]
        parsed=[(MATH_ENGINE.parsed(a) if MATH_ENGINE is not None else bool(parse_math(a))) if not m else False for a,m in zip(answers,is_missing)]
        relation=[[bool(i==j or (is_missing[i] and is_missing[j]) or
                          (not is_missing[i] and not is_missing[j] and math_equivalent(answers[i],answers[j])))
                   for j in range(n)] for i in range(n)]
        correct=[bool(not m and math_equivalent(a,gold)) for a,m in zip(answers,is_missing)]
    elif dataset=='mathvista':
        normalized=[normalize_mathvista(a,metadata) for a in answers]
        reference=normalize_mathvista(gold,metadata)
        relation=[[a==b for b in normalized] for a in normalized]
        correct=[a!=NO_ANSWER and a==reference for a in normalized]
        parsed=[a!=NO_ANSWER for a in normalized]
        is_missing=[a==NO_ANSWER for a in normalized]
    else:raise ValueError('Unknown dataset')
    return {'normalized':normalized,'missing':is_missing,'parsed':parsed,'equivalent':relation,'correct':correct}

def prefix_groups(order,relation,missing_flags):
    """Adding unseen candidates never changes groups of already revealed ones."""
    groups=[];labels=[]
    for i in order:
        match=next((g for g in groups if all(relation[i][j] and relation[j][i] for j in g)),None)
        if match is None:match=[];groups.append(match)
        match.append(i)
        labels.append(NO_ANSWER if missing_flags[i] else 'group:'+str(match[0]))
    return labels
