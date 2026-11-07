"""Full-objective MLP internals and exact static parameter decompositions."""
import itertools
import math
import torch

GROUPS=['1.weight','1.bias','3.weight','3.bias','5.weight','5.bias']


def cosine(a,b):
    a,b=a.flatten().double(),b.flatten().double()
    den=float(a.norm()*b.norm())
    return float(a.dot(b))/den if den else None


def stats(x, detailed=False):
    x=x.detach().double()
    result=dict(mean=float(x.mean()),rms=float(x.square().mean().sqrt()),
                std=float(x.std(unbiased=False)),max_abs=float(x.abs().max()))
    if detailed:
        result['abs_quantiles']=torch.quantile(x.abs().flatten(),torch.tensor([.5,.9,.99],device=x.device,dtype=x.dtype)).tolist()
    return result


@torch.no_grad()
def forward_arrays(model,x,physical=1000):
    arrays={k:[] for k in ['z1','h1','z2','h2','output']}
    for batch in x.split(physical):
        z1=model[1](model[0](batch));h1=model[2](z1)
        z2=model[3](h1);h2=model[4](z2);output=model[5](h2)
        for key,value in zip(arrays,[z1,h1,z2,h2,output]):arrays[key].append(value)
    return {key:torch.cat(value) for key,value in arrays.items()}


def residual_summary(output,y):
    f=output.double();target=torch.nn.functional.one_hot(y,10).double();r=f-target
    unit_loss=.5*r.square().sum(1);mean=r.mean(0);center=r-mean
    common=r.mean(1,keepdim=True)
    total=float(unit_loss.mean())
    result=dict(loss_fp64=total,accuracy=float((f.argmax(1)==y).double().mean()),
        mean_residual_loss=float(.5*mean.square().sum()),centered_residual_loss=float(.5*center.square().sum(1).mean()),
        common_class_loss=float(.5*10*common.square().mean()),
        true_class_loss=float(.5*r.gather(1,y[:,None]).square().mean()),
        output_mean=f.mean(0).tolist(),residual_mean=mean.tolist(),output_std=f.std(0,unbiased=False).tolist(),
        sample_loss_quantiles=torch.quantile(unit_loss,torch.tensor([.5,.9,.99,1.],device=f.device,dtype=f.dtype)).tolist(),
        top1pct_loss_fraction=float(unit_loss.topk(max(1,len(y)//100)).values.sum()/unit_loss.sum()))
    assert abs(total-result['mean_residual_loss']-result['centered_residual_loss'])<1e-12
    result['wrong_class_loss']=total-result['true_class_loss']
    return result


@torch.no_grad()
def observe(model,optimizer,x,y,physical=1000,detailed=False):
    arrays=forward_arrays(model,x,physical)
    result=dict(parameters={},activations={},output=residual_summary(arrays['output'],y))
    for name,p in model.named_parameters():
        v=stats(p,detailed);v['norm']=float(p.double().norm())
        v['gradient_norm']=float(p.grad.double().norm())
        state=optimizer.state[p]
        if state:
            v['incoming_momentum_norm']=float(state['exp_avg'].double().norm())
            v['gradient_incoming_momentum_cosine']=cosine(p.grad,state['exp_avg'])
        if detailed and p.ndim==2:
            sv=torch.linalg.svdvals(p.double());v['spectral_norm']=float(sv[0]);v['singular_values']=sv.tolist()
        result['parameters'][name]=v
    for key,a in arrays.items():
        v=stats(a,detailed);v['feature_mean_norm']=float(a.double().mean(0).norm())
        if key.startswith('h'):
            v.update(saturated95=float((a.abs()>.95).double().mean()),
                saturated99=float((a.abs()>.99).double().mean()),mean_tanh_slope=float((1-a.double().square()).mean()))
        result['activations'][key]=v
    result['output']['head_coordinate_bound_max']=float((model[5].weight.double().abs().sum(1)+model[5].bias.double().abs()).max())
    return result,arrays


@torch.no_grad()
def native_loss(model,x,y,physical=1000):
    loss=torch.zeros((),device=x.device,dtype=torch.float64)
    for xb,yb in zip(x.split(physical),y.split(physical)):
        output=model(xb);target=torch.nn.functional.one_hot(yb,10).to(output.dtype)
        loss+=(.5*(output-target).square().sum()/len(x)).double()
    return float(loss)


def shapley(values,n):
    result=[]
    for i in range(n):
        total=0.
        for mask in range(1<<n):
            if mask & (1<<i):continue
            k=mask.bit_count();weight=math.factorial(k)*math.factorial(n-k-1)/math.factorial(n)
            total+=weight*(values[mask|(1<<i)]-values[mask])
        result.append(total)
    assert abs(sum(result)-(values[-1]-values[0]))<1e-10
    return result


@torch.no_grad()
def hybrids(model,a,b,x,y,physical=1000):
    values=[]
    for mask in range(64):
        model.load_state_dict({key:(b if mask&(1<<i) else a)[key] for i,key in enumerate(GROUPS)})
        values.append(native_loss(model,x,y,physical))
    shares=shapley(values,6)
    return dict(loss_by_mask=values,groups=GROUPS,shapley=dict(zip(GROUPS,shares)),
        total_loss_change=values[-1]-values[0],
        only_new={k:values[1<<i] for i,k in enumerate(GROUPS)},
        revert_one={k:values[63^(1<<i)] for i,k in enumerate(GROUPS)})


def output_decomposition(a,b,aa,bb,y):
    h0,h1=aa['h2'].double(),bb['h2'].double()
    w0,w1=a['5.weight'].double(),b['5.weight'].double()
    b0,b1=a['5.bias'].double(),b['5.bias'].double()
    terms=dict(hidden_features=(h1-h0)@w0.T,head_weights=h0@(w1-w0).T,
               interaction=(h1-h0)@(w1-w0).T,head_bias=(b1-b0).expand(len(y),-1))
    # FP64 evaluation of the affine head makes the algebra exact. Native FP32
    # outputs differ by rounding, measured separately rather than hidden.
    f0=h0@w0.T+b0;f1=h1@w1.T+b1;delta=f1-f0
    target=torch.nn.functional.one_hot(y,10).double();residual=f0-target
    exact_delta=float(.5*((f1-target).square().sum(1)-(f0-target).square().sum(1)).mean())
    result=dict(loss_change_fp64_head=exact_delta,terms={})
    for name,v in terms.items():
        result['terms'][name]=dict(rms=float(v.square().mean().sqrt()),
            loss_allocation=float((residual*v+.5*delta*v).sum(1).mean()),
            linear_term=float((residual*v).sum(1).mean()),
            mean_vector=v.mean(0).tolist())
    assert torch.allclose(sum(terms.values()),delta,atol=1e-12,rtol=1e-12)
    assert abs(sum(r['loss_allocation'] for r in result['terms'].values())-exact_delta)<1e-12
    result['native_output_max_rounding']=max(float((aa['output'].double()-f0).abs().max()),float((bb['output'].double()-f1).abs().max()))
    den=float(delta.square().sum())
    result['sample_coherent_fraction']=float(len(y)*delta.mean(0).square().sum())/den if den else None
    result['class_common_fraction']=float(10*delta.mean(1).square().sum())/den if den else None
    eig=torch.linalg.eigvalsh(delta.T@delta)
    result['first_output_mode_fraction']=float(eig[-1]/eig.sum()) if den else None
    result['native_output_change_rms']=float((bb['output'].double()-aa['output'].double()).square().mean().sqrt())
    return result


def changes(a,b,aa,bb):
    result=dict(parameters={},activations={})
    for key in GROUPS:
        x,z=a[key].double(),b[key].double();delta=z-x
        result['parameters'][key]=dict(relative_displacement=float(delta.norm()/x.norm()),
            relative_norm_change=float(z.norm()/x.norm()-1),cosine=cosine(x,z),delta_norm=float(delta.norm()))
    for key in aa:
        x,z=aa[key].double(),bb[key].double();delta=z-x
        result['activations'][key]=dict(change_rms=float(delta.square().mean().sqrt()),
            relative_change_norm=float(delta.norm()/x.norm()),cosine=cosine(x,z),
            sign_flip_fraction=float(((x>0)!=(z>0)).double().mean()))
    return result
