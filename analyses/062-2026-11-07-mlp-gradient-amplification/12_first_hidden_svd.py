"""Trace hidden oscillation backward: H1 -> second linear Z2 -> tanh H2."""
import csv
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
RUN=ROOT/'runs/241-2026-11-07-mlp-first-hidden-oscillation'
RAW=RUN/'artifacts/spark/payload/worker'
OLD=ROOT/'runs/240-2026-11-07-mlp-hidden-change-svd/artifacts/spark/payload/worker'
ART=HERE/'artifacts/first-hidden-svd'
spec=importlib.util.spec_from_file_location('previous_svd',HERE/'09_three_space_svd.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
# Reuse the tested blockwise Gram/SVD reduction; no previous file is modified.
m.ART=ART;m.DIMENSIONS={'hidden1':1000000,'preactivation2':1000000}


class Data:
    def __init__(self):
        self.h1=np.load(RAW/'hidden1.npy',mmap_mode='r')
        self.z2=np.load(RAW/'preactivation2.npy',mmap_mode='r')
        self.h2=np.load(OLD/'hidden.npy',mmap_mode='r')
        for a in [self.h1,self.z2,self.h2]:assert a.shape==(355,5000,200)
        self.layer=torch.load(RAW/'second_layer.pt',weights_only=False)
        assert self.layer['states']==list(range(447,802))
        self.w=self.layer['weights'].double();self.b=self.layer['biases'].double()

    def blocks(self,name):
        raw=self.h1 if name=='hidden1' else self.z2
        for start in range(0,5000,125):
            level=torch.from_numpy(np.array(raw[:,start:start+125],dtype=np.float64))
            yield start*200,(level[1:]-level[:-1]).flatten(1)


def secant(z,z_next):
    dz=z_next-z
    a=torch.tanh(z);b=torch.tanh(z_next)
    slope=torch.empty_like(z);moving=dz!=0
    slope[moving]=(b[moving]-a[moving])/dz[moving]
    slope[~moving]=1-a[~moving].square()
    return slope


def tests():
    z=torch.tensor([0.,1.,-1.,3.,-3.],dtype=torch.float64)
    zn=torch.tensor([0.,1.1,-.8,3.5,-3.1],dtype=torch.float64)
    s=secant(z,zn)
    torch.testing.assert_close(s*(zn-z),torch.tanh(zn)-torch.tanh(z),rtol=1e-13,atol=1e-14)
    assert s[0]==1 and bool(((s>=0)&(s<=1)).all())
    h=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.float64)
    dh=torch.tensor([[.2,-.1],[.3,.2]],dtype=torch.float64)
    w=torch.tensor([[1.,2.],[-1.,.5]],dtype=torch.float64);dw=w*.03
    b=torch.tensor([.1,-.2],dtype=torch.float64);db=b*.02
    z=h@w.T+b;zn=(h+dh)@(w+dw).T+b+db;s=secant(z,zn)
    terms=[dh@w.T,h@dw.T,dh@dw.T,db.expand_as(z)]
    torch.testing.assert_close(sum(terms),zn-z,rtol=1e-13,atol=1e-14)
    torch.testing.assert_close(sum(s*t for t in terms),torch.tanh(zn)-torch.tanh(z),rtol=1e-12,atol=1e-14)
    return ['tanh secant including exactly zero displacement','finite linear decomposition','secant-transmitted additive decomposition']


def main():
    torch.set_num_threads(4);begin=time.monotonic();checks=tests();ART.mkdir(parents=True,exist_ok=True)
    assert json.loads((RUN/'retrieval.json').read_text())['verified']
    for item in json.loads((RAW.parent/'transfer_inventory.json').read_text())['files']:
        p=RAW.parent/item['path'];assert p.stat().st_size==item['bytes'] and m.sha(p)==item['sha256']
    previous=json.loads((HERE/'three_space_svd_results.json').read_text())
    assert m.sha(OLD/'hidden.npy')==previous['sources']['hidden.npy']
    previous_grams=np.load(HERE/'artifacts/three-space-svd/grams.npz')
    with (HERE/'three_space_svd_values.csv').open() as f:previous_rows=list(csv.DictReader(f))
    data=Data();grams={};coefficients={};streams={}
    for name in ['hidden1','preactivation2']:
        gram=torch.zeros((354,354),dtype=torch.float64)
        for _,block in data.blocks(name):gram+=block@block.T
        grams[name]=gram.numpy();coefficients[name],streams[name]=m.fit(data,name,grams[name])
        print(json.dumps(dict(phase='svd',space=name,top1=streams[name]['top1'],top10=streams[name]['top10'],elapsed=time.monotonic()-begin)),flush=True)
    for name,old in [('hidden2','hidden'),('output','output')]:
        grams[name]=previous_grams[old];streams[name]=previous['streams'][old]
        coefficients[name]=np.array([[float(r[old+f'_alpha_{k:02d}']) for k in range(1,11)] for r in previous_rows])
    np.savez(ART/'grams.npz',**grams)
    cosine={name:m.cosine_from_gram(g) for name,g in grams.items()}
    # Temporal mode comparison is invariant to changing spatial basis/signs.
    temporal={}
    left={name:torch.linalg.eigh(torch.from_numpy(g[1:,1:].copy()))[1][:,-10:].flip(1).numpy() for name,g in grams.items()}
    for a,b in [('hidden1','preactivation2'),('hidden1','hidden2'),('hidden1','output'),('preactivation2','hidden2')]:
        singular=np.linalg.svd(left[a].T@left[b],compute_uv=False)
        temporal[a+'_'+b]=dict(principal_cosines=singular.tolist(),mean_squared_principal_cosine=float(np.mean(singular**2)),
            fraction_a_energy_in_b_temporal_basis=float(np.trace(left[b].T@grams[a][1:,1:]@left[b])/np.trace(grams[a][1:,1:])),
            fraction_b_energy_in_a_temporal_basis=float(np.trace(left[a].T@grams[b][1:,1:]@left[a])/np.trace(grams[b][1:,1:])))
    rows=[];roundoff_z=roundoff_h=forward_h=forward_z=0.;s_min=1.;s_max=0.
    for j,t in enumerate(range(448,801)):
        i=j+1;h=torch.from_numpy(np.array(data.h1[i],dtype=np.float64))
        hn=torch.from_numpy(np.array(data.h1[i+1],dtype=np.float64));dh=hn-h
        z=torch.from_numpy(np.array(data.z2[i],dtype=np.float64));zn=torch.from_numpy(np.array(data.z2[i+1],dtype=np.float64));dz=zn-z
        h2=torch.from_numpy(np.array(data.h2[i],dtype=np.float64));h2n=torch.from_numpy(np.array(data.h2[i+1],dtype=np.float64));dh2=h2n-h2
        w=data.w[i];dw=data.w[i+1]-w;db=data.b[i+1]-data.b[i]
        terms=[dh@w.T,h@dw.T,dh@dw.T,db.expand_as(z)];s=secant(z,zn)
        s_min=min(s_min,float(s.min()));s_max=max(s_max,float(s.max()))
        transmitted=[s*term for term in terms]
        rz=dz-sum(terms);rh=dh2-sum(transmitted)
        roundoff_z=max(roundoff_z,float(rz.abs().max()));roundoff_h=max(roundoff_h,float(rh.abs().max()))
        forward_z=max(forward_z,float((z-(h@w.T+data.b[i])).abs().max()))
        forward_h=max(forward_h,float((h2-torch.tanh(z)).abs().max()),float((h2n-torch.tanh(zn)).abs().max()))
        h1n=float(dh.norm());znorm=float(dz.norm());h2norm=float(dh2.norm())
        row=dict(state=t,loss=float(previous_rows[j]['loss']),h1_activation_rms=float(h.square().mean().sqrt()),
                 z2_activation_rms=float(z.square().mean().sqrt()),h2_activation_rms=float(h2.square().mean().sqrt()),
                 linear_transmission_gain=float(terms[0].norm())/h1n if h1n else None,
                 actual_linear_change_ratio=znorm/h1n if h1n else None,
                 tanh_displacement_gain=h2norm/znorm if znorm else None,
                 second_weight_relative_drift=float((w-data.w[1]).norm()/data.w[1].norm()),
                 z_roundoff_dot=float((rz*dz).sum()),h_roundoff_dot=float((rh*dh2).sum()))
        for name in grams:
            row[name+'_cosine']=float(cosine[name][j]);row[name+'_norm']=float(np.sqrt(grams[name][i,i]))
            for k in range(10):row[name+f'_alpha_{k+1:02d}']=float(coefficients[name][j,k])
        for label,term,trans in zip(['upstream','weights','interaction','bias'],terms,transmitted):
            row['z_'+label+'_dot']=float((term*dz).sum());row['z_'+label+'_squared_norm']=float(term.square().sum())
            row['h_'+label+'_dot']=float((trans*dh2).sum());row['h_'+label+'_squared_norm']=float(trans.square().sum())
        rows.append(row)
        if j%64==0:print(json.dumps(dict(phase='transmission',state=t,elapsed=time.monotonic()-begin)),flush=True)
    assert s_min>=-1e-10 and s_max<=1+1e-8
    csvpath=HERE/'first_hidden_svd_values.csv'
    with csvpath.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    attribution={}
    for prefix,name in [('z','preactivation2'),('h','hidden2')]:
        energy=sum(r[name+'_norm']**2 for r in rows)
        attribution[name]={label:dict(signed_projection_share=sum(r[prefix+'_'+label+'_dot'] for r in rows)/energy,
            relative_rms=(sum(r[prefix+'_'+label+'_squared_norm'] for r in rows)/energy)**.5)
            for label in ['upstream','weights','interaction','bias']}
        attribution[name]['roundoff_signed_projection_share']=sum(r[prefix+'_roundoff_dot'] for r in rows)/energy
    correlations={a+'_'+b:float(np.corrcoef(cosine[a],cosine[b])[0,1]) for a,b in
                  [('hidden1','preactivation2'),('hidden1','hidden2'),('hidden1','output'),('preactivation2','hidden2'),('hidden2','output')]}
    ranges={k:[min(r[k] for r in rows),max(r[k] for r in rows)] for k in
            ['h1_activation_rms','z2_activation_rms','h2_activation_rms','linear_transmission_gain','tanh_displacement_gain','second_weight_relative_drift']}
    result=dict(status='numeric_checks_passed',source_run=241,states=[448,800],tests=checks,streams=streams,
        correlations=correlations,negative_alignment_counts={n:int((v<0).sum()) for n,v in cosine.items()},
        temporal_subspaces=temporal,attribution=attribution,ranges=ranges,secant_slope_range=[s_min,s_max],
        preactivation_identity_max_roundoff=roundoff_z,hidden_identity_max_roundoff=roundoff_h,
        tanh_forward_max_roundoff=forward_h,linear_forward_max_roundoff=forward_z,
        checkpoints={str(t):rows[t-448] for t in [465,473,474,485,512,521,522,529,683]},
        sources={str(p.relative_to(ROOT).as_posix()):m.sha(p) for p in
                 [RAW/'hidden1.npy',RAW/'preactivation2.npy',RAW/'second_layer.pt',OLD/'hidden.npy',HERE/'three_space_svd_values.csv',HERE/'three_space_svd_results.json',HERE/'artifacts/three-space-svd/grams.npz',HERE/'09_three_space_svd.py']},
        generator_sha256=m.sha(Path(__file__)),csv_sha256=m.sha(csvpath),elapsed_seconds=time.monotonic()-begin)
    m.dump(HERE/'first_hidden_svd_results.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['streams','checkpoints','sources']},indent=2),flush=True)


if __name__=='__main__':main()
