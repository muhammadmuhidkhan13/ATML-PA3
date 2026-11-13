"""Order-within-question aggregation and paired question bootstrap."""
import numpy as np

METRICS=('accuracy','oracle','distinct_groups','token_cost','sampling_failure','selection_failure','selection_success')

def failure_decomposition(correct,selected):
    oracle=bool(any(correct));success=bool(correct[selected])
    return {'oracle':float(oracle),'accuracy':float(success),'sampling_failure':float(not oracle),
            'selection_failure':float(oracle and not success),'selection_success':float(success)}

def aggregate_orders(values):
    values=np.asarray(values,dtype=float)
    if values.ndim<2:raise ValueError('Question and order axes required')
    return values.mean(axis=1),values.mean(axis=1).mean(axis=0)

def paired_bootstrap(left,right,*,seed=6304,resamples=1000):
    left,right=np.asarray(left,float),np.asarray(right,float)
    if left.ndim!=1 or left.shape!=right.shape or not len(left):raise ValueError('One averaged outcome per question required')
    indices=np.random.default_rng(seed).integers(len(left),size=(resamples,len(left)))
    difference=left-right
    return {'mean':float(difference.mean()),'ci95':np.quantile(difference[indices].mean(1),[.025,.975],method='linear').tolist(),
            'questions':len(left),'resamples':resamples,'seed':seed,'unit':'question_after_order_average'}
