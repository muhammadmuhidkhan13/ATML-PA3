"""Candidate-prefix-only percentile calibration and both fusion definitions."""
import numpy as np

def calibrate(kl_arrays:list,vision_arrays:list)->dict:
    pool=np.concatenate([np.asarray(a,dtype=float) for a in kl_arrays])
    pool=pool[np.isfinite(pool)]
    low,high=np.quantile(pool,[.05,.95],method='linear') if pool.size else (0.,0.)
    calibrated=[];trajectory=[];tokenwise=[];covariance=[]
    for kl,vision in zip(kl_arrays,vision_arrays):
        kl=np.asarray(kl,dtype=float);vision=np.asarray(vision,dtype=float)
        if len(kl)!=len(vision):raise ValueError('Confidence and attention are not aligned')
        c=np.where(np.isfinite(kl),np.clip((kl-low)/max(high-low,1e-8),0.,1.),0.)
        z=float(c.mean()*vision.mean()) if len(c) else 0.
        zt=float((c*vision).mean()) if len(c) else 0.
        cov=float(((c-c.mean())*(vision-vision.mean())).mean()) if len(c) else 0.
        if not np.isclose(zt-z,cov,atol=1e-12,rtol=1e-10):raise AssertionError('Fusion covariance identity failed')
        calibrated.append(c);trajectory.append(z);tokenwise.append(zt);covariance.append(cov)
    return {'q05':float(low),'q95':float(high),'calibrated':calibrated,'visconf':trajectory,
            'tokenwise':tokenwise,'covariance':covariance}
