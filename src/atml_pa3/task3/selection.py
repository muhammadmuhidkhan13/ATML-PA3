"""Selection has no gold answers, correctness labels, or unrevealed candidates."""
from dataclasses import dataclass
from typing import Sequence
import numpy as np

@dataclass(frozen=True)
class Candidate:
    rollout_id:int
    answer_group:str
    token_count:int

@dataclass(frozen=True)
class Selection:
    index:int
    answer_group:str
    consumed:int
    token_cost:int
    score:float|None

def descending_average_ranks(scores:Sequence[float])->np.ndarray:
    scores=np.asarray(scores,dtype=float)
    if np.isnan(scores).any():raise ValueError('NaN scores must have explicit missing-score policy')
    order=np.argsort(-scores,kind='stable');ranks=np.empty(len(scores),float);start=0
    while start<len(order):
        end=start+1
        while end<len(order) and scores[order[end]]==scores[order[start]]:end+=1
        ranks[order[start:end]]=(start+1+end)/2.;start=end
    return ranks

def select(candidates:Sequence[Candidate],method:str,quality:Sequence[float]|None=None)->Selection:
    n=len(candidates)
    if not n:raise ValueError('Empty candidate prefix')
    if method=='first':index=0;consumed=1;score=None
    elif method=='shortest':index=min(range(n),key=lambda i:candidates[i].token_count);consumed=n;score=-candidates[index].token_count
    elif method=='highest':
        if quality is None or len(quality)!=n:raise ValueError('One score per revealed candidate required')
        index=max(range(n),key=lambda i:quality[i]);consumed=n;score=float(quality[index])
    elif method in ('consistency','rwc'):
        denominator=1
        if method=='consistency':weights=np.ones(n,dtype=int)
        elif n==1:weights=np.ones(1,dtype=int)
        else:
            if quality is None or len(quality)!=n:raise ValueError('Missing RWC quality')
            # Ranks are integers/half-integers. Sum exact integer numerators
            # with a shared denominator, preserving mathematical vote ties.
            weights=np.rint(2*(n-descending_average_ranks(quality))).astype(int)
            denominator=2*(n-1)
        votes={}
        for candidate,weight in zip(candidates,weights):votes[candidate.answer_group]=votes.get(candidate.answer_group,0)+int(weight)
        winner=max(votes,key=votes.get)  # insertion order is earliest support
        index=next(i for i,c in enumerate(candidates) if c.answer_group==winner);consumed=n;score=votes[winner]/denominator
    else:raise ValueError('Unknown selector')
    return Selection(index,candidates[index].answer_group,consumed,sum(c.token_count for c in candidates[:consumed]),score)

def methods(vision:bool=False)->dict[str,tuple[str,str|None]]:
    result={'FirstAns':('first',None),'SC':('consistency',None),'HighestLogP':('highest','mean_logp'),
            'HighestK':('highest','self_certainty'),'K_RWC':('rwc','self_certainty'),'Shortest':('shortest',None)}
    if vision:result.update({'HighestV':('highest','vision'),'V_RWC':('rwc','vision'),
                            'HighestVisConf':('highest','visconf'),'VisConf_RWC':('rwc','visconf'),
                            'HighestTokenFusion':('highest','tokenwise'),'TokenFusion_RWC':('rwc','tokenwise')})
    return result
