"""GPU replay of the official TDA cache equations on pre-extracted features."""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np
import torch
from run_streams import stream_order, expected_calibration_error


def ent(p):
    return float((-(p * torch.log(p.clamp_min(1e-12))).sum() / np.log(p.numel())).item())


def update(cache, cls, feat, loss, cap, prob=None):
    cache.setdefault(int(cls), []).append((feat.detach(), loss, None if prob is None else prob.detach()))
    cache[int(cls)] = sorted(cache[int(cls)], key=lambda x: x[1])[:cap]


def logits(feat, cache, C, alpha, beta, negative=False):
    if not cache:
        return torch.zeros(C, device=feat.device)
    keys=[]; values=[]
    for cls in sorted(cache):
        for key, _loss, prob in cache[cls]:
            keys.append(key)
            if negative:
                values.append(((prob > .03) & (prob < 1.0)).to(torch.float32))
            else:
                one=torch.zeros(C,device=feat.device); one[cls]=1.; values.append(one)
    K=torch.stack(keys); V=torch.stack(values)
    weights=torch.exp(-beta*(1.-K@feat))
    return alpha*(weights[:,None]*V).sum(0)


@torch.inference_mode()
def run_one(cache, stream, seed):
    x=torch.as_tensor(cache['features'],device='cuda',dtype=torch.float32)
    text=torch.as_tensor(cache['text_features'],device='cuda',dtype=torch.float32)
    labels=np.asarray(cache['labels']); domains=np.asarray(cache['domains']); order=stream_order(labels,domains,stream,seed)
    C=text.shape[0]; pos={}; neg={}; corr=[]; zcorr=[]; conf=[]
    for i in order:
        feat=x[int(i)]; base=100.*(feat@text.T); p=torch.softmax(base,0); pred=int(base.argmax())
        loss=ent(p); update(pos,pred,feat,loss,3)
        if .2 < loss < .5: update(neg,pred,feat,loss,2,p)
        adapted=base+logits(feat,pos,C,2.,5.)-logits(feat,neg,C,.117,1.,True)
        final=int(adapted.argmax()); true=int(labels[int(i)])
        corr.append(float(final==true)); zcorr.append(float(pred==true)); conf.append(float(torch.softmax(adapted,0).max()))
    c=np.asarray(corr); z=np.asarray(zcorr); cf=np.asarray(conf); w=min(500,len(c)); roll=np.convolve(c,np.ones(w)/w,mode='valid')
    return {'strategy':'TDA','stream':stream,'seed':seed,'samples':len(c),'budget_percent':0.,'query_count':0,
            'online_accuracy':float(c.mean()),'zero_shot_accuracy':float(z.mean()),'gain_over_zero_shot':float(c.mean()-z.mean()),
            'worst_window_accuracy':float(roll.min()),'negative_adaptation_rate':float(((z==1)&(c==0)).mean()),
            'expected_calibration_error':expected_calibration_error(cf,c),'positive_cache_size':sum(len(v) for v in pos.values()),
            'negative_cache_size':sum(len(v) for v in neg.values())}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cache-root',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--seeds',nargs='+',type=int,default=[10]); ap.add_argument('--streams',nargs='+',default=['iid_random'])
    a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True); rows=[]
    assert torch.cuda.is_available(), 'CUDA required'
    for path in sorted(a.cache_root.glob('*.npz')):
        loaded=np.load(path,allow_pickle=False); cache={k:loaded[k] for k in loaded.files}
        for stream in a.streams:
            for seed in a.seeds:
                row=run_one(cache,stream,seed); row['cache']=path.stem; rows.append(row); print(path.stem,stream,seed,f"{row['online_accuracy']:.6f}",flush=True)
    with (a.output_dir/'tda_cached_gpu_runs.csv').open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=sorted(rows[0]));w.writeheader();w.writerows(rows)
    print('wrote',len(rows),'GPU TDA rows')
if __name__=='__main__': main()
