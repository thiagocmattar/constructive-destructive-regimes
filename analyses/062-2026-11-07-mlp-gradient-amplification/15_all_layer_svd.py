"""Complete fixed-input forward path and first-layer actual-increment analysis."""
import csv
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
RUN=ROOT/'runs/242-2026-11-07-mlp-all-layer-oscillation';RAW=RUN/'artifacts/spark/payload/worker'
PREVIOUS=ROOT/'runs/241-2026-11-07-mlp-first-hidden-oscillation/artifacts/spark/payload/worker'
ART=HERE/'artifacts/all-layer-svd'
spec=importlib.util.spec_from_file_location('previous_hidden',HERE/'12_first_hidden_svd.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
m=e.m;m.ART=ART;m.DIMENSIONS={'preactivation1':1000000,'parameter_step':614600}


class Data:
    def __init__(self):
        self.x=np.load(RAW/'inputs.npy',mmap_mode='r')
        self.z=np.load(RAW/'preactivation1.npy',mmap_mode='r')
        self.w=np.load(RAW/'first_weights.npy',mmap_mode='r')
        self.b=np.load(RAW/'first_biases.npy',mmap_mode='r')
        self.h=np.load(PREVIOUS/'hidden1.npy',mmap_mode='r')
        assert self.x.shape==(5000,3072) and self.z.shape==self.h.shape==(355,5000,200)
        assert self.w.shape==(355,200,3072) and self.b.shape==(355,200)

    def blocks(self,name):
        if name=='preactivation1':
            sources=[(0,self.z.reshape(355,-1))]
        else:
            sources=[(0,self.w.reshape(355,-1)),(614400,self.b)]
        for offset,raw in sources:
            for start in range(0,raw.shape[1],25000):
                level=torch.from_numpy(np.array(raw[:,start:start+25000],dtype=np.float64))
                yield offset+start,level[1:]-level[:-1]


def tests():
    x=torch.tensor([[1.,2.],[3.,-1.]],dtype=torch.float64)
    w=torch.tensor([[.3,.2],[-.1,.5]],dtype=torch.float64);dw=w*.01
    b=torch.tensor([.2,-.3],dtype=torch.float64);db=b*.02
    z=x@w.T+b;zn=x@(w+dw).T+b+db
    torch.testing.assert_close(zn-z,x@dw.T+db,rtol=1e-13,atol=1e-14)
    zero=np.zeros((354,354));assert np.isnan(m.cosine_from_gram(zero)).all()
    assert np.linalg.matrix_rank(zero)==0
    return ['fixed-input finite first-linear identity','zero input-change cosine undefined and rank zero',*e.tests()]


def main():
    torch.set_num_threads(4);begin=time.monotonic();checks=tests();ART.mkdir(parents=True,exist_ok=True)
    assert json.loads((RUN/'retrieval.json').read_text())['verified']
    for item in json.loads((RAW.parent/'transfer_inventory.json').read_text())['files']:
        p=RAW.parent/item['path'];assert p.stat().st_size==item['bytes'] and m.sha(p)==item['sha256']
    input_audit=json.loads((RAW/'input_verification.json').read_text())
    assert input_audit['status']=='passed' and input_audit['states_checked']==355 and input_audit['max_input_change']==0
    data=Data();prior=json.loads((HERE/'first_hidden_svd_results.json').read_text())
    assert m.sha(PREVIOUS/'hidden1.npy')==prior['sources'][str((PREVIOUS/'hidden1.npy').relative_to(ROOT).as_posix())]
    with (HERE/'first_hidden_svd_values.csv').open() as f:old=list(csv.DictReader(f))
    prior_grams=np.load(HERE/'artifacts/first-hidden-svd/grams.npz')
    grams={};coefficients={};streams={}
    for name in ['preactivation1','parameter_step']:
        gram=torch.zeros((354,354),dtype=torch.float64)
        for _,block in data.blocks(name):gram+=block@block.T
        grams[name]=gram.numpy();coefficients[name],streams[name]=m.fit(data,name,grams[name])
        print(json.dumps(dict(phase='svd',space=name,top1=streams[name]['top1'],top10=streams[name]['top10'],elapsed=time.monotonic()-begin)),flush=True)
    for name in ['hidden1','preactivation2','hidden2','output']:
        grams[name]=prior_grams[name];streams[name]=prior['streams'][name]
        coefficients[name]=np.array([[float(r[name+f'_alpha_{k:02d}']) for k in range(1,11)] for r in old])
    np.savez(ART/'grams.npz',**grams)
    cosine={name:m.cosine_from_gram(g) for name,g in grams.items()}
    u={name:torch.linalg.eigh(torch.from_numpy(g[1:,1:].copy()))[1][:,-10:].numpy() for name,g in grams.items()}
    temporal={}
    for a,b in [('parameter_step','preactivation1'),('preactivation1','hidden1'),('preactivation1','output')]:
        s=np.linalg.svd(u[a].T@u[b],compute_uv=False)
        temporal[a+'_'+b]=dict(principal_cosines=s.tolist(),overlap=float(np.mean(s*s)),
            fraction_a_energy_in_b_temporal_basis=float(np.trace(u[b].T@grams[a][1:,1:]@u[b])/np.trace(grams[a][1:,1:])),
            fraction_b_energy_in_a_temporal_basis=float(np.trace(u[a].T@grams[b][1:,1:]@u[a])/np.trace(grams[b][1:,1:])))
    x=torch.from_numpy(np.array(data.x,dtype=np.float64));rows=[];max_z_roundoff=max_h_roundoff=0.;smin=1.;smax=0.
    wref=torch.from_numpy(np.array(data.w[1],dtype=np.float64))
    for j,t in enumerate(range(448,801)):
        i=j+1;w=torch.from_numpy(np.array(data.w[i],dtype=np.float64));dw=torch.from_numpy(np.array(data.w[i+1],dtype=np.float64))-w
        b=torch.from_numpy(np.array(data.b[i],dtype=np.float64));db=torch.from_numpy(np.array(data.b[i+1],dtype=np.float64))-b
        z=torch.from_numpy(np.array(data.z[i],dtype=np.float64));zn=torch.from_numpy(np.array(data.z[i+1],dtype=np.float64));dz=zn-z
        h=torch.from_numpy(np.array(data.h[i],dtype=np.float64));hn=torch.from_numpy(np.array(data.h[i+1],dtype=np.float64));dh=hn-h
        weight_term=x@dw.T;bias_term=db.expand_as(z);s=e.secant(z,zn)
        rz=dz-weight_term-bias_term;rh=dh-s*weight_term-s*bias_term
        max_z_roundoff=max(max_z_roundoff,float(rz.abs().max()));max_h_roundoff=max(max_h_roundoff,float(rh.abs().max()))
        smin=min(smin,float(s.min()));smax=max(smax,float(s.max()))
        dz_energy=float(dz.square().sum());dh_energy=float(dh.square().sum())
        high=s>.5;low=s<.01
        row=dict(state=t,loss=float(old[j]['loss']),input_change_norm=0.,input_change_cosine=None,
                 z1_activation_rms=float(z.square().mean().sqrt()),h1_activation_rms=float(h.square().mean().sqrt()),
                 first_tanh_displacement_gain=(dh_energy/dz_energy)**.5,
                 secant_gt_half_coordinate_fraction=float(high.double().mean()),
                 secant_lt_one_percent_coordinate_fraction=float(low.double().mean()),
                 h1_change_energy_on_secant_gt_half=float(dh[high].square().sum())/dh_energy,
                 z1_change_energy_on_secant_gt_half=float(dz[high].square().sum())/dz_energy,
                 first_weight_norm=float(w.norm()),first_bias_norm=float(b.norm()),
                 first_weight_increment_norm=float(dw.norm()),first_bias_increment_norm=float(db.norm()),
                 first_weight_relative_drift=float((w-wref).norm()/wref.norm()),
                 z_roundoff_dot=float((rz*dz).sum()),h_roundoff_dot=float((rh*dh).sum()))
        for name in grams:
            row[name+'_cosine']=float(cosine[name][j]);row[name+'_norm']=float(np.sqrt(grams[name][i,i]))
            for k in range(10):row[name+f'_alpha_{k+1:02d}']=float(coefficients[name][j,k])
        for label,term in [('weights',weight_term),('bias',bias_term)]:
            for prefix,delta,part in [('z',dz,term),('h',dh,s*term)]:
                row[prefix+'_'+label+'_dot']=float((delta*part).sum())
                row[prefix+'_'+label+'_squared_norm']=float(part.square().sum())
        rows.append(row)
        if j%32==0:print(json.dumps(dict(phase='first_linear_and_tanh',state=t,elapsed=time.monotonic()-begin)),flush=True)
    assert smin>=-1e-10 and smax<=1+1e-8
    csvpath=HERE/'all_layer_svd_values.csv'
    with csvpath.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    attribution={}
    for prefix,name in [('z','preactivation1'),('h','hidden1')]:
        energy=sum(r[name+'_norm']**2 for r in rows)
        attribution[name]={label:dict(signed_projection_share=sum(r[prefix+'_'+label+'_dot'] for r in rows)/energy,
            relative_rms=(sum(r[prefix+'_'+label+'_squared_norm'] for r in rows)/energy)**.5) for label in ['weights','bias']}
        attribution[name]['roundoff_signed_projection_share']=sum(r[prefix+'_roundoff_dot'] for r in rows)/energy
    correlations={a+'_'+b:float(np.corrcoef(cosine[a],cosine[b])[0,1]) for a,b in
        [('parameter_step','preactivation1'),('parameter_step','hidden1'),('preactivation1','hidden1'),('preactivation1','output'),('hidden1','output')]}
    ranges={k:[min(r[k] for r in rows),max(r[k] for r in rows)] for k in
        ['z1_activation_rms','h1_activation_rms','first_tanh_displacement_gain','secant_gt_half_coordinate_fraction',
         'secant_lt_one_percent_coordinate_fraction','h1_change_energy_on_secant_gt_half',
         'first_weight_norm','first_bias_norm','first_weight_relative_drift']}
    result=dict(status='numeric_checks_passed',source_run=242,states=[448,800],tests=checks,
        input=dict(shape=[5000,3072],states_checked=355,exactly_constant=True,change_energy=0.,change_rank=0,
                   change_cosine=None,normalized_singular_energy_fractions=None,mode_directions_identifiable=False,
                   reason='Every observed first-linear input equals the same fixed matrix; no nonzero temporal displacement.'),
        streams=streams,correlations=correlations,negative_alignment_counts={n:int((v<0).sum()) for n,v in cosine.items()},
        temporal_subspaces=temporal,attribution=attribution,ranges=ranges,secant_slope_range=[smin,smax],
        preactivation_identity_max_roundoff=max_z_roundoff,hidden_identity_max_roundoff=max_h_roundoff,
        checkpoints={str(t):rows[t-448] for t in [465,473,474,485,512,521,522,529,683]},
        sources={str(p.relative_to(ROOT).as_posix()):m.sha(p) for p in
          [RAW/'inputs.npy',RAW/'preactivation1.npy',RAW/'first_weights.npy',RAW/'first_biases.npy',RAW/'input_verification.json',
           PREVIOUS/'hidden1.npy',HERE/'first_hidden_svd_values.csv',HERE/'first_hidden_svd_results.json',HERE/'artifacts/first-hidden-svd/grams.npz',
           HERE/'09_three_space_svd.py',HERE/'12_first_hidden_svd.py']},
        generator_sha256=m.sha(Path(__file__)),csv_sha256=m.sha(csvpath),elapsed_seconds=time.monotonic()-begin)
    m.dump(HERE/'all_layer_svd_results.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['streams','checkpoints','sources']},indent=2),flush=True)


if __name__=='__main__':main()
