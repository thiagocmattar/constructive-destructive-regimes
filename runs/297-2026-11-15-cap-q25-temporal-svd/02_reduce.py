"""Exact temporal/cross Grams and fixed-baseline spatial-subspace comparisons."""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np

def spectrum(K,P):
    K=(K+K.T)/2;lam,U=np.linalg.eigh(K);lam=lam[::-1];U=U[:,::-1]
    assert lam.min()>=-max(np.trace(K),1e-30)*1e-9
    lam=np.maximum(lam,0);energy=float(lam.sum());assert np.isclose(energy,np.trace(K),rtol=1e-9,atol=1e-30)
    if energy==0:return dict(energy=0,rank95=None,rank99=None,participation_rank=None,entropy_rank=None),lam,U
    share=lam/energy;cum=share.cumsum();positive=share[share>0]
    norm=np.sqrt(np.maximum(np.diag(K),0));den=norm[:-1]*norm[1:]
    cos=np.divide(np.diag(K,1),den,out=np.full(len(den),np.nan),where=den>0)
    for j in range(U.shape[1]):
        if U[np.argmax(np.abs(U[:,j])),j]<0:U[:,j]*=-1
    result=dict(energy=energy,coordinate_rms=float(np.sqrt(energy/(len(K)*P))),
        rank95=int(np.searchsorted(cum,.95)+1),rank99=int(np.searchsorted(cum,.99)+1),
        participation_rank=float(1/(share@share)),entropy_rank=float(np.exp(-(positive*np.log(positive)).sum())),
        stable_rank=float(energy/lam[0]),top1=float(share[0]),top5=float(share[:5].sum()),top10=float(share[:10].sum()),
        reversal_fraction=float(np.mean(cos[np.isfinite(cos)]<0)) if np.isfinite(cos).any() else None,
        retention=float(np.sqrt(max(float(K.sum()),0))/norm.sum()) if norm.sum() else None,
        eigenvalues=lam.tolist(),energy_shares=share.tolist(),cumulative=cum.tolist())
    return result,lam,U

def compare(Kb,Kc,C,P):
    b,lb,ub=spectrum(Kb,P);c,lc,uc=spectrum(Kc,P)
    if not b['energy'] or not c['energy']:return dict(control=b,cap_q25=c,comparison=None)
    k=b['rank95'];assert lb[k-1]>lb[0]*1e-12
    cap_per_base_mode=((ub[:,:k].T@C)**2).sum(1)/lb[:k]
    inside=float(cap_per_base_mode.sum());outside=max(c['energy']-inside,0)
    assert inside<=c['energy']*(1+1e-8)
    common=min(k,c['rank95']);M=ub[:,:common].T@C@uc[:,:common]
    overlap=float(np.sum(M*M/(lb[:common,None]*lc[None,:common]))/common)
    comp=dict(baseline_basis_rank95=k,cap_energy_inside_baseline95=inside,
        cap_energy_outside_baseline95=outside,cap_fraction_outside_baseline95=outside/c['energy'],
        baseline_energy_outside_own95=float(lb[k:].sum()),energy_ratio=c['energy']/b['energy'],
        rank95_change=c['rank95']-b['rank95'],participation_rank_ratio=c['participation_rank']/b['participation_rank'],
        common_spatial_subspace_rank=common,spatial_subspace_overlap=overlap,
        cap_energy_per_baseline_mode=cap_per_base_mode.tolist())
    return dict(control=b,cap_q25=c,comparison=comp)

def gram_pair(A,B,block):
    import torch
    T=A.shape[0]-1;P=A.shape[1];assert A.shape==B.shape
    K=torch.zeros((T,T),dtype=torch.float64,device='cuda');J=K.clone();C=K.clone()
    diag=np.zeros(T);adj=np.zeros(T-1);diag2=diag.copy();adj2=adj.copy()
    for lo in range(0,P,block):
        da=np.diff(np.array(A[:,lo:lo+block],dtype=np.float64),axis=0)
        db=np.diff(np.array(B[:,lo:lo+block],dtype=np.float64),axis=0)
        diag+=np.einsum('ij,ij->i',da,da);diag2+=np.einsum('ij,ij->i',db,db)
        adj+=np.einsum('ij,ij->i',da[:-1],da[1:]);adj2+=np.einsum('ij,ij->i',db[:-1],db[1:])
        x=torch.from_numpy(da).cuda();y=torch.from_numpy(db).cuda();K+=x@x.T;J+=y@y.T;C+=x@y.T
    K,J,C=[v.cpu().numpy() for v in [K,J,C]]
    for matrix,d,e in [(K,diag,adj),(J,diag2,adj2)]:
        assert np.allclose(np.diag(matrix),d,rtol=1e-9,atol=1e-25)
        assert np.allclose(np.diag(matrix,1),e,rtol=1e-8,atol=max(d.max(),1e-30)*1e-11)
    return K,J,C

def main():
    p=argparse.ArgumentParser();p.add_argument('--case-root',type=Path,required=True);a=p.parse_args()
    root=a.case_root;out=root/'reduction';out.mkdir(exist_ok=False);begin=time.monotonic()
    cfg=json.loads(Path(__file__).with_name('config.json').read_text())
    layouts={arm:json.loads((root/arm/'layout.json').read_text()) for arm in cfg['arms']}
    rows={arm:[json.loads(t) for t in (root/arm/'boundaries.jsonl').read_text().splitlines()] for arm in cfg['arms']}
    assert layouts['control']==layouts['cap_q25'];layout=layouts['control'];results=[];grams={}
    for j,(name,item) in enumerate(layout['sites'].items()):
        A=np.load(root/'control'/item['file'],mmap_mode='r');B=np.load(root/'cap_q25'/item['file'],mmap_mode='r')
        assert np.array_equal(A[0],B[0]),'Common fork representations differ'
        K,J,C=gram_pair(A,B,cfg['gram_block_coordinates']);P=item['coordinates']
        for arm,matrix in [('control',K),('cap_q25',J)]:
            expected=np.array([row['representations'][name]['change_rms']**2*P for row in rows[arm][1:]])
            assert np.allclose(np.diag(matrix),expected,rtol=1e-8,atol=1e-24)
        grams[f'{j}_control']=K;grams[f'{j}_cap_q25']=J;grams[f'{j}_cross']=C
        for lo,hi in cfg['windows']:
            result=compare(K[lo:hi,lo:hi],J[lo:hi,lo:hi],C[lo:hi,lo:hi],P)
            result.update(site=name,coordinates=P,updates=hi-lo,first_change_state=layout['first_state']+lo+1,last_change_state=layout['first_state']+hi,window=f'{lo}:{hi}')
            results.append(result)
        print(json.dumps(dict(event='site_reduced',case=root.name,site=name,seconds=time.monotonic()-begin,rank95_control=results[-1]['control']['rank95'],rank95_cap=results[-1]['cap_q25']['rank95'])),flush=True)
    np.savez_compressed(out/'grams.npz',**grams)
    (out/'results.json').write_text(json.dumps(dict(case=root.name,layout=layout,results=results,seconds=time.monotonic()-begin),indent=2,allow_nan=False)+'\n')
    inventory={}
    for f in sorted(root.rglob('*')):
        if f.is_file():
            with f.open('rb') as h:d=hashlib.file_digest(h,'sha256').hexdigest()
            inventory[f.relative_to(root).as_posix()]=dict(bytes=f.stat().st_size,sha256=d)
    (out/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps(dict(event='reduction_complete',case=root.name,seconds=time.monotonic()-begin,bytes=sum(x['bytes'] for x in inventory.values()))),flush=True)
if __name__=='__main__':main()
