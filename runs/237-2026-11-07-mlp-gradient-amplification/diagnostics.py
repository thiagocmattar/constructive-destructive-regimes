"""Analytic full-example backpropagation, Jacobian traces and step response."""
import torch

KEYS=['1.weight','1.bias','3.weight','3.bias','5.weight','5.bias']


def summary(x):
    v=x.detach().double().flatten()
    return dict(min=float(v.min()),p01=float(torch.quantile(v,.01)),median=float(v.median()),
                p99=float(torch.quantile(v,.99)),max=float(v.max()),rms=float(v.square().mean().sqrt()))


def forward(model,x):
    x=x.flatten(1);z1=model[1](x);h1=z1.tanh();z2=model[3](h1);h2=z2.tanh();f=model[5](h2)
    return dict(x=x,z1=z1,h1=h1,z2=z2,h2=h2,output=f,s1=1-h1.square(),s2=1-h2.square())


def unit_backward(model,a,y):
    r=a['output']-torch.nn.functional.one_hot(y,10).to(a['output'].dtype)
    dh2=r@model[5].weight;dz2=dh2*a['s2'];dh1=dz2@model[3].weight;dz1=dh1*a['s1']
    dx=dz1@model[1].weight
    return dict(residual=r,dh2=dh2,dz2=dz2,dh1=dh1,dz1=dz1,dx=dx)


def jacobian_traces(model,a):
    # C2 and C1 are derivatives of the ten outputs with respect to preactivations.
    w1,w2,w3=model[1].weight,model[3].weight,model[5].weight
    c2=w3[None,:,:]*a['s2'][:,None,:]
    c1=(c2@w2)*a['s1'][:,None,:]
    n1=c1.double().square().sum((1,2));n2=c2.double().square().sum((1,2))
    parameter=n1*(a['x'].double().square().sum(1)+1)+n2*(a['h1'].double().square().sum(1)+1)+10*(a['h2'].double().square().sum(1)+1)
    # ||C1 W1||_F^2 = tr(C1 (W1 W1^T) C1^T), avoiding N*10*3072 storage.
    gram=w1@w1.T
    input_trace=((c1@gram).double()*c1.double()).sum((1,2))
    return parameter,input_trace


def directional(model,a,delta):
    dz1=a['x']@delta['1.weight'].T+delta['1.bias'];dh1=a['s1']*dz1
    ddh1=-2*a['h1']*a['s1']*dz1.square()
    dz2=dh1@model[3].weight.T+a['h1']@delta['3.weight'].T+delta['3.bias']
    dh2=a['s2']*dz2
    ddz2=ddh1@model[3].weight.T+2*dh1@delta['3.weight'].T
    ddh2=a['s2']*ddz2-2*a['h2']*a['s2']*dz2.square()
    df=dh2@model[5].weight.T+a['h2']@delta['5.weight'].T+delta['5.bias']
    ddf=ddh2@model[5].weight.T+2*dh2@delta['5.weight'].T
    return df,ddf


@torch.no_grad()
def observe(model,x,y,physical=1000):
    accum={k:[] for k in ['residual','dh2','dz2','dh1','dz1']};statistics={k:[] for k in ['dx','parameter_jacobian_trace','input_jacobian_trace','h1_rms','h2_rms','h1_slope','h2_slope']}
    per_example={k:[] for k in KEYS};aggregate={k:torch.zeros_like(p) for k,p in model.named_parameters()}
    for xb,yb in zip(x.split(physical),y.split(physical)):
        a=forward(model,xb);b=unit_backward(model,a,yb);jp,jx=jacobian_traces(model,a)
        for k in accum:accum[k].append(b[k])
        statistics['dx'].append(b['dx'].double().square().sum(1))
        statistics['parameter_jacobian_trace'].append(jp);statistics['input_jacobian_trace'].append(jx)
        for key in ['h1','h2']:
            statistics[key+'_rms'].append(a[key].double().square().mean(1).sqrt())
            statistics[key+'_slope'].append(a['s'+key[-1]].double().mean(1))
        for prefix,adjoint,features in [('1',b['dz1'],a['x']),('3',b['dz2'],a['h1']),('5',b['residual'],a['h2'])]:
            e=adjoint.double().square().sum(1)
            per_example[prefix+'.weight'].append(e*features.double().square().sum(1))
            per_example[prefix+'.bias'].append(e)
            aggregate[prefix+'.weight']+=adjoint.T@features/len(x)
            aggregate[prefix+'.bias']+=adjoint.sum(0)/len(x)
    arrays={k:torch.cat(v) for k,v in accum.items()};statistics={k:torch.cat(v) for k,v in statistics.items()}
    energies={k:torch.cat(v) for k,v in per_example.items()}
    record=dict(backward={},gradient_groups={},jacobian={},activations={},loss={})
    for key,value in arrays.items():
        square=value.double().square();record['backward'][key]=dict(norm=float(square.sum().sqrt()),
            mean_example_squared_norm=float(square.sum(1).mean()),example_norms=summary(square.sum(1).sqrt()),
            neuron_squared_energy=square.sum(0).tolist())
    record['backward']['dx']=dict(norm=float(statistics['dx'].sum().sqrt()),mean_example_squared_norm=float(statistics['dx'].mean()))
    record['jacobian']={k:dict(mean=float(statistics[k].mean()),distribution=summary(statistics[k].sqrt())) for k in ['parameter_jacobian_trace','input_jacobian_trace']}
    record['activations']={k:summary(statistics[k]) for k in ['h1_rms','h2_rms','h1_slope','h2_slope']}
    gradient_error=0.;total_gradient_energy=0.;total_unit_energy=0.
    for key,p in model.named_parameters():
        actual=p.grad.double();unit_energy=float(energies[key].mean());energy=float(actual.square().sum())
        error=float((aggregate[key].double()-actual).norm()/max(float(actual.norm()),1e-30));gradient_error=max(gradient_error,error)
        record['gradient_groups'][key]=dict(norm=energy**.5,unit_mean_squared_norm=unit_energy,
            coherence=energy/unit_energy if unit_energy else None,analytic_relative_gradient_error=error,
            neuron_norms=actual.square().sum(1).sqrt().tolist() if p.ndim==2 else actual.abs().tolist())
        total_gradient_energy+=energy;total_unit_energy+=unit_energy
    record.update(gradient_norm=total_gradient_energy**.5,unit_gradient_mean_squared_norm=total_unit_energy,
        gradient_coherence=total_gradient_energy/total_unit_energy,analytic_relative_gradient_error=gradient_error)
    loss=.5*arrays['residual'].double().square().sum(1)
    record['loss']=dict(mean=float(loss.mean()),example_distribution=summary(loss),
        top1pct_energy_fraction=float(loss.topk(max(1,len(x)//100)).values.sum()/loss.sum()),
        class_mean={str(k):float(loss[y==k].mean()) for k in range(10)})
    assert gradient_error<5e-5,gradient_error
    return record,arrays


def adam_vectors(optimizer):
    result={}
    for group in optimizer.param_groups:
        b1,b2=group['betas']
        for p in group['params']:
            state=optimizer.state[p];step=int(state['step'])
            m=state['exp_avg'].double()/(1-b1**step)
            den=state['exp_avg_sq'].double().sqrt()/(1-b2**step)**.5+group['eps']
            result[p]=(m,den)
    return result


@torch.no_grad()
def adam_observe(model,optimizer,before,old_vectors,lr=.001,eps=1e-8):
    new=adam_vectors(optimizer);record={};total_actual=0.;old_d=0.;old_m=0.;minimum_ratio=1e100
    gdot=0.;actual_delta={}
    for name,p in model.named_parameters():
        m,den=new[p];om,od=old_vectors[p];delta=p.detach().double()-before[name].double();actual_delta[name]=delta.to(p.dtype)
        weight=delta.square();flat_den=den.flatten();thresholds=torch.quantile(flat_den,torch.tensor([.001,.01,.1],device=den.device,dtype=den.dtype))
        low_energy=[float(weight[den<=q].sum()/weight.sum()) for q in thresholds]
        ratio=den/od;minimum_ratio=min(minimum_ratio,float(ratio.min()))
        prediction=-lr*m/den;dot=float((p.grad.double()*delta).sum());gdot+=dot
        record[name]=dict(step_norm=float(delta.norm()),moment_norm=float(m.norm()),denominator=summary(den),inverse_denominator=summary(1/den),
            denominator_ratio=summary(ratio),step_weighted_denominator_relative_change=float((weight*(ratio-1).square()).sum()/weight.sum())**.5,
            low_denominator_quantiles=[.001,.01,.1],low_denominator_step_energy_fraction=low_energy,
            epsilon_near_fraction=float((den<=10*eps).double().mean()),epsilon_near_step_energy=float(weight[den<=10*eps].sum()/weight.sum()),
            predicted_step_residual_norm=float((delta-prediction).norm()),gradient_dot_step=dot,
            cf_old_denominator_norm=float((lr*m/od).norm()),cf_old_momentum_norm=float((lr*om/den).norm()))
        total_actual+=float(weight.sum());old_d+=float((lr*m/od).square().sum());old_m+=float((lr*om/den).square().sum())
    return dict(groups=record,actual_step_norm=total_actual**.5,cf_old_denominator_norm=old_d**.5,
        cf_old_momentum_norm=old_m**.5,minimum_denominator_ratio=minimum_ratio,gradient_dot_step=gdot),actual_delta
