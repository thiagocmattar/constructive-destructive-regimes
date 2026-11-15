"""Independent direct SVD of full manageable readouts, after capture completes."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--case-root',type=Path,required=True);a=p.parse_args()
root=a.case_root;data=json.loads((root/'reduction/results.json').read_text());begin=time.monotonic()
site='correct_label_log_probability' if root.name=='transformer' else 'output'
item=data['layout']['sites'][site];Ds={}
for arm in ['control','cap_q25']:
    A=np.load(root/arm/item['file'],mmap_mode='r');Ds[arm]=np.diff(np.array(A,dtype=np.float64),axis=0)
checks=[]
for row in data['results']:
    if row['site']!=site:continue
    lo,hi=map(int,row['window'].split(':'));A=Ds['control'][lo:hi];B=Ds['cap_q25'][lo:hi]
    u,s,v=np.linalg.svd(A,full_matrices=False);k=row['control']['rank95']
    assert np.allclose(s*s,row['control']['eigenvalues'],rtol=1e-7,atol=s[0]**2*1e-11)
    sc=np.linalg.svd(B,compute_uv=False);assert np.allclose(sc*sc,row['cap_q25']['eigenvalues'],rtol=1e-7,atol=sc[0]**2*1e-11)
    inside=float(np.linalg.norm(B@v[:k].T)**2);expected=row['comparison']['cap_energy_inside_baseline95']
    assert np.isclose(inside,expected,rtol=1e-7,atol=1e-20)
    outside=float(np.linalg.norm(B-(B@v[:k].T)@v[:k])**2)
    assert np.isclose(outside,row['comparison']['cap_energy_outside_baseline95'],rtol=1e-7,atol=1e-20)
    checks.append(dict(window=row['window'],site=site,direct_svd_matches=True,explicit_right_projection_matches=True,inside_relative_error=abs(inside-expected)/max(abs(expected),1e-30)))
result=dict(status='passed',case=root.name,checks=checks,seconds=time.monotonic()-begin)
path=root/'reduction/dense_verification.json';path.write_text(json.dumps(result,indent=2)+'\n')
invpath=root/'reduction/inventory.json';inv=json.loads(invpath.read_text())
inv['reduction/dense_verification.json']=dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest());invpath.write_text(json.dumps(inv,indent=2)+'\n')
print(json.dumps(result))
