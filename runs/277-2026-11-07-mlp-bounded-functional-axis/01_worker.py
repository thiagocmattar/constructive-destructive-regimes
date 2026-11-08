"""Run277: past functional-axis modification with a native parameter-norm ceiling."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.eos_tasks import representation_continuation as c
from src.eos_tasks import representation_policy as policy


def components(x,dw,db,pw,pb,kind,tau):
    z=F.linear(x.double().reshape(-1,x.shape[-1]),dw,db);v=F.linear(x.double().reshape(-1,x.shape[-1]),pw,pb)
    energy=float(v.square().sum());beta=float((z*v).sum())/energy if energy else 0.;kept=beta
    rms=float(v.square().mean().sqrt())
    if beta<0:
        kept=0. if kind=='remove' else beta/2 if kind=='half' else max(beta,-tau/rms)
    removed=beta-kept
    return dw-removed*pw,db-removed*pb,dict(coefficient=beta,retained_coefficient=kept,previous_response_rms=rms,
        proposal_response_rms=float(z.square().mean().sqrt()),response_cosine=policy.cosine(z,v),
        affected_fraction=float(removed!=0.),threshold=tau,removed_coefficient=removed)


def test():
    gen=torch.Generator().manual_seed(9);x=torch.randn(7,3,generator=gen,dtype=torch.float64)
    pw=torch.randn(4,3,generator=gen,dtype=torch.float64);pb=torch.randn(4,generator=gen,dtype=torch.float64);dw=-2*pw;db=-2*pb
    for kind in ['remove','half','cap']:
        nw,nb,r=components(x,dw,db,pw,pb,kind,.2);z=x.numpy()@dw.numpy().T+db.numpy();v=x.numpy()@pw.numpy().T+pb.numpy()
        beta=float(np.sum(z*v)/np.sum(v*v));kept=0 if kind=='remove' else beta/2 if kind=='half' else max(beta,-.2/np.sqrt(np.mean(v*v)))
        np.testing.assert_allclose(nw.numpy(),dw.numpy()-(beta-kept)*pw.numpy(),atol=1e-12)
        np.testing.assert_allclose(nb.numpy(),db.numpy()-(beta-kept)*pb.numpy(),atol=1e-12)
    native=np.r_[dw.numpy().ravel(),db.numpy()];candidate=np.r_[nw.numpy().ravel(),nb.numpy()]
    scale=min(1.,np.linalg.norm(native)/np.linalg.norm(candidate)) if np.linalg.norm(candidate) else 1.
    assert np.linalg.norm(scale*candidate)<=np.linalg.norm(native)+1e-12
    print(json.dumps(dict(status='passed',checks=['Independent NumPy functional projection','weight/bias-space exact affine identity','remove half and cap rules'])))


def main():
    if '--selftest' in sys.argv:test();return
    original_apply=policy.apply_policy;original_arm=c.run_arm;memory={}
    def run_arm(context,fork,cfg,arm,folder,thresholds,ema,groups,entry,trunk,probe_history,parameter_history,mechanics=False):
        memory.clear();offset=0
        for name,p in context['task'].model.named_parameters():
            if name in ['1.weight','1.bias']:memory[name]=parameter_history[-1][offset:offset+p.numel()].reshape_as(p).to(p.device).clone()
            offset+=p.numel()
        return original_arm(context,fork,cfg,arm,folder,thresholds,ema,groups,entry,trunk,probe_history,parameter_history,mechanics)
    @torch.no_grad()
    def apply(model,before,arm,groups,metrics,inputs,thresholds,targets):
        if 'projection_kind' not in arm:return original_apply(model,before,arm,groups,metrics,inputs,thresholds,targets)
        module=model[1];ow=before['1.weight'].double();ob=before['1.bias'].double()
        dw=module.weight.double()-ow;db=module.bias.double()-ob;pw=ow-memory['1.weight'].double();pb=ob-memory['1.bias'].double()
        tau=thresholds['1']['functional'][str(arm['quantile'])]
        nw,nb,record=components(inputs['1'],dw,db,pw,pb,arm['projection_kind'],tau)
        if arm.get('norm_cap_native'):
            native_norm=float(torch.cat([dw.flatten(),db]).norm());candidate_norm=float(torch.cat([nw.flatten(),nb]).norm())
            scale=min(1.,native_norm/candidate_norm) if candidate_norm else 1.
            nw=nw*scale;nb=nb*scale
            record.update(normalization_factor=scale,native_parameter_norm=native_norm,candidate_parameter_norm=candidate_norm)
        module.weight.copy_((ow+nw).float());module.bias.copy_((ob+nb).float())
        actual=torch.cat([(module.weight.double()-ow).flatten(),module.bias.double()-ob]);native=torch.cat([dw.flatten(),db])
        if arm.get('norm_cap_native'):
            rounding=float((actual-torch.cat([nw.flatten(),nb])).norm())
            assert float(actual.norm())<=float(native.norm())+rounding+1e-10
            record['rounding_norm_bound']=rounding
        record.update(proposed_norm=float(native.norm()),actual_norm=float(actual.norm()),projection_kind=arm['projection_kind'])
        memory['1.weight']=before['1.weight'].clone();memory['1.bias']=before['1.bias'].clone()
        return {'1':record}
    c.run_arm=run_arm;policy.apply_policy=apply;c.main()


if __name__=='__main__':main()
