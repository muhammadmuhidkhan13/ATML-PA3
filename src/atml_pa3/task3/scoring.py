"""Scores use aligned token measurements only, never correctness labels."""
import numpy as np

def aligned(values,n):
    if values is None or len(values)!=n:return np.full(n,np.nan)
    return np.array(values,dtype=float)

def finite_mean(values,*,default=-float('inf')):
    values=np.asarray(values,dtype=float);valid=values[np.isfinite(values)]
    return float(valid.mean()) if valid.size else default

def visual_engagement(image,prompt,generated,n):
    image,prompt,generated=(aligned(v,n) for v in (image,prompt,generated))
    denominator=image+prompt+generated
    valid=np.isfinite(image)&np.isfinite(prompt)&np.isfinite(generated)&(image>=0)&(prompt>=0)&(generated>=0)&(denominator>0)
    ratio=np.zeros(n);np.divide(image,denominator,out=ratio,where=valid)
    return ratio,valid

def base_scores(row):
    n=len(row['generated_token_ids']);logp=aligned(row.get('logp'),n);kl=aligned(row.get('kl_u_p'),n)
    vision=None;valid=None
    if 'img_attn_total' in row:
        vision,valid=visual_engagement(row['img_attn_total'],row['prompt_text_attn_total'],row['generated_text_attn_total'],n)
    return {'length':n,'mean_logp':finite_mean(logp),'self_certainty':finite_mean(kl),
            'vision':float(vision.mean()) if n and vision is not None else 0.,
            'kl':kl,'v':vision,'vision_valid':valid,
            'invalid_logp':int((~np.isfinite(logp)).sum()),'invalid_kl':int((~np.isfinite(kl)).sum())}
