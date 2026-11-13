"""Retrospective signal diagnostics; never called to choose experimental answers."""
from itertools import combinations
import numpy as np
from .selection import descending_average_ranks
from .scoring import finite_mean

def correlation(x,y,*,rank=False):
    x,y=np.asarray(x,float),np.asarray(y,float);mask=np.isfinite(x)&np.isfinite(y)
    x,y=x[mask],y[mask]
    if len(x)<2 or np.ptp(x)==0 or np.ptp(y)==0:return None
    if rank:x,y=descending_average_ranks(x),descending_average_ranks(y)
    return float(np.corrcoef(x,y)[0,1])

def quartiles(values,correct):
    values=np.asarray(values,float);correct=np.asarray(correct,float)
    if len(values)<4 or not np.isfinite(values).all() or np.ptp(values)==0:
        return {'top_accuracy':None,'bottom_accuracy':None,'gap':None,'degenerate':True}
    order=np.lexsort((np.arange(len(values)),values));size=len(values)//4
    bottom,top=order[:size],order[-size:]
    return {'top_accuracy':float(correct[top].mean()),'bottom_accuracy':float(correct[bottom].mean()),
            'gap':float(correct[top].mean()-correct[bottom].mean()),'degenerate':False,
            'top_rollouts':top.tolist(),'bottom_rollouts':bottom.tolist(),
            'boundary_ties':bool(values[order[size-1]]==values[order[size]] or values[order[-size-1]]==values[order[-size]])}

def question_diagnostics(base,correct,fusion=None):
    signals={name:[r[name] for r in base] for name in ('length','mean_logp','self_certainty')}
    if fusion is not None:
        signals.update(vision=[r['vision'] for r in base],visconf=fusion['visconf'],tokenwise=fusion['tokenwise'])
    quartile={name:quartiles(values,correct) for name,values in signals.items()}
    association={**signals,'correctness':list(map(float,correct))}
    correlations={a+'__'+b:correlation(association[a],association[b],rank=True) for a,b in combinations(association,2)}
    result={'quartiles':quartile,'spearman':correlations,'signals':signals,'correctness':list(map(bool,correct))}
    if fusion is not None:
        rho=[correlation(c,b['v']) for c,b in zip(fusion['calibrated'],base)]
        valid=[v for v in rho if v is not None]
        result.update(token_confidence_vision_pearson=rho,question_mean_token_pearson=float(np.mean(valid)) if valid else None,
                      undefined_rollout_correlations=len(rho)-len(valid),population_covariance=fusion['covariance'])
    return result

def illustration_candidates(diag,base,correct):
    k=np.asarray([r['self_certainty'] for r in base]);v=np.asarray([r['vision'] for r in base]);correct=np.asarray(correct,bool)
    highk=k>=np.quantile(k,.75);lowk=k<=np.quantile(k,.25)
    return {'high_confidence_incorrect':np.flatnonzero(highk&~correct).tolist(),
            'low_confidence_correct':np.flatnonzero(lowk&correct).tolist(),
            'high_vision_low_confidence':np.flatnonzero((v>=np.quantile(v,.75))&lowk).tolist(),
            'high_confidence_low_vision':np.flatnonzero(highk&(v<=np.quantile(v,.25))).tolist()}
