"""Run269-271 policies: local affine displacement caps and reversal-aware damping."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def cosine(a, b):
    if b is None: return None
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    den = float(a.norm() * b.norm())
    return float(a @ b) / den if den else None


def group_layout(model):
    """One group per direct parameter owner; preserve named-parameter order."""
    groups = {}; offset = 0
    for name, p in model.named_parameters():
        owner = name.rsplit('.', 1)[0] if '.' in name else ''
        g = groups.setdefault(owner, dict(start=offset, end=offset, names=[]))
        assert g['end'] == offset, 'Noncontiguous or tied parameter owner'
        offset += p.numel(); g['end'] = offset; g['names'].append(name)
    return groups


def affine_targets(model, case):
    modules = dict(model.named_modules())
    if case == 'mlp': trunk = ['1', '3']; entry = ['1']
    elif case == 'resnet':
        trunk = [n for n,m in modules.items() if isinstance(m, nn.Conv2d)]; entry = ['conv1']
    else:
        trunk = [n for n,m in modules.items() if isinstance(m, nn.MultiheadAttention)
                 or (isinstance(m, nn.Linear) and n.startswith('transformer_encoder') and not n.endswith('out_proj'))]
        entry = ['transformer_encoder.layers.0.self_attn']
    return entry, trunk


def affine_parameters(model, name):
    m = dict(model.named_modules())[name]
    if isinstance(m, nn.MultiheadAttention):
        return m, m.in_proj_weight, m.in_proj_bias
    return m, m.weight, m.bias


@torch.no_grad()
def capture(task, case, probe, target_names):
    """Fixed original normalization group; hook tensors cloned before in-place ops."""
    model = task.model; x = task.data['x_train'][:probe]; y = task.data['y_train'][:probe]
    values, inputs, hooks = {}, {}, []
    modules = dict(model.named_modules())
    names = {'mlp':['1','2','3','4'], 'resnet':['conv1','layer1','layer2','layer3','avgpool'],
             'transformer':['encoder','transformer_encoder.layers.0.self_attn','transformer_encoder.layers.0',
                            'transformer_encoder.layers.1.self_attn','transformer_encoder.layers.1']}[case]
    def output_hook(name):
        def hook(module, args, output):
            a = output[0] if isinstance(output, tuple) else output
            values.setdefault(name,[]).append(a.detach().clone())
        return hook
    def input_hook(name):
        def hook(module, args): inputs.setdefault(name,[]).append(args[0].detach().clone())
        return hook
    for n in names: hooks.append(modules[n].register_forward_hook(output_hook(n)))
    for n in target_names: hooks.append(modules[n].register_forward_pre_hook(input_hook(n)))
    # MLP has no batch-dependent normalization; native physical chunks agree.
    outputs=[];loss=torch.zeros((),device=x.device,dtype=torch.float64)
    try:
        chunks=zip(x.split(task.physical),y.split(task.physical)) if case=='mlp' else [(x,y)]
        for xb,yb in chunks:
            value,output=task.loss(xb,yb);loss+=value.double();outputs.append(output.detach().clone())
    finally:
        for h in hooks: h.remove()
    values={n:torch.cat(v,dim=0) for n,v in values.items()}
    inputs={n:torch.cat(v,dim=0) for n,v in inputs.items()}
    output=torch.cat(outputs,dim=0)
    values['output'] = (output if case in ['mlp','transformer'] else output.log_softmax(-1)).detach().clone()
    if case == 'transformer':
        mask = y != -100
        for n,a in values.items():
            values[n] = (a if n=='output' else a.transpose(0,1))[mask].contiguous()
    denominator = int((y != -100).sum()) if case=='transformer' else len(y)
    return values, inputs, float(loss.double()/denominator)


@torch.no_grad()
def affine_scores(module, x, dw, db):
    """FP64 RMS of frozen-input local affine parameter displacement per output."""
    x, dw = x.double(), dw.double()
    db = None if db is None else db.double()
    if isinstance(module, nn.Conv2d):
        # Chunking changes no normalization: x is already the captured native input.
        square = torch.zeros(dw.shape[0], device=x.device, dtype=torch.float64); count = 0
        for xb in x.split(128):
            z = F.conv2d(xb, dw, db, module.stride, module.padding, module.dilation, module.groups)
            square += z.square().sum((0,2,3)); count += z.shape[0]*z.shape[2]*z.shape[3]
        return (square/count).sqrt()
    z = F.linear(x.reshape(-1,x.shape[-1]), dw, db)
    return z.square().mean(0).sqrt()


def block_diagnostics(g, previous, groups, ema):
    result = {}
    for name, span in dict(groups, **{'__global__':dict(start=0,end=g.numel())}).items():
        v = g[span['start']:span['end']].double()
        old = None if previous is None else previous[span['start']:span['end']].double()
        norm = float(v.norm()); pn = float(old.norm()) if old is not None else 0.
        dot = float(v @ old) if old is not None else None
        result[name] = dict(norm=norm, previous_norm=pn, dot_previous=dot,
            rho=dot/(norm*pn) if norm*pn else None,
            eta_kappa_hat=1-dot/(pn*pn) if pn else None,
            previous_ema=ema.get(name,norm))
    return result


def scalar_factor(metric, kind, target=2.):
    if kind=='shock':
        if metric['rho'] is not None and metric['rho']<0 and metric['norm']>metric['previous_ema']:
            return metric['previous_ema']/metric['norm']
        return 1.
    if kind=='secant':
        k=metric['eta_kappa_hat']
        return min(1.,target/k) if k is not None and k>target else 1.
    raise ValueError(kind)


def advance_ema(ema, metrics, beta=.9):
    for name,m in metrics.items(): ema[name]=beta*m['previous_ema']+(1-beta)*m['norm']


@torch.no_grad()
def apply_policy(model, before, arm, groups, metrics, inputs, calibration, targets):
    """Called after native Adam. Return actual deltas and explicit policy checks."""
    params=dict(model.named_parameters()); kind=arm['kind']; records={}
    if kind in ['shock','secant','fixed','norm_match']:
        active = list(groups) if arm.get('scope') in ['all','global'] else targets
        for n in active:
            span=groups[n]; pp=[params[k] for k in span['names']]
            u=torch.cat([(p.double()-before[k].double()).flatten() for k,p in zip(span['names'],pp)])
            if kind=='fixed': factor=arm['factor']
            elif kind=='norm_match':
                factor=arm['target_norm']/float(u.norm()) if float(u.norm()) else 0.
            else: factor=scalar_factor(metrics['__global__' if arm.get('scope')=='global' else n],kind,arm.get('target',2.))
            for k,p in zip(span['names'],pp): p.copy_((before[k].double()+factor*(p.double()-before[k].double())).to(p.dtype))
            records[n]=dict(factor=factor,affected_fraction=float(factor!=1.),proposed_norm=float(u.norm()))
    elif kind in ['discard','cap','functional']:
        for name in targets:
            module,w,b=affine_parameters(model,name)
            wn=name+('.in_proj_weight' if isinstance(module,nn.MultiheadAttention) else '.weight')
            bn=name+('.in_proj_bias' if isinstance(module,nn.MultiheadAttention) else '.bias')
            ow=before[wn]; ob=None if b is None else before[bn]
            dw=w.double()-ow.double(); db=None if b is None else b.double()-ob.double()
            u=torch.cat([dw.flatten()]+([] if db is None else [db.flatten()]))
            key=str(arm['quantile']); threshold=calibration[name]['functional' if kind=='functional' else 'coordinate'][key]
            if threshold==0:
                records[name]=dict(threshold=0.,affected_fraction=0.,proposed_norm=float(u.norm()),actual_norm=float(u.norm()),disabled_zero_reference=True)
                continue
            if kind=='functional':
                scores=affine_scores(module,inputs[name],dw,db)
                factors=torch.where(scores>0,(threshold/scores).clamp(max=1),torch.ones_like(scores))
                nw=dw*factors.reshape(-1,*([1]*(dw.ndim-1)));nb=None if db is None else db*factors
                affected=float((scores>threshold).double().mean())
            else:
                transformed=torch.where(u.abs()<=threshold,u,0.) if kind=='discard' else u.clamp(-threshold,threshold)
                nw=transformed[:dw.numel()].reshape_as(dw);nb=None if db is None else transformed[dw.numel():]
                affected=float((u.abs()>threshold).double().mean());factors=None;scores=None
            w.copy_((ow.double()+nw).to(w.dtype))
            if b is not None:b.copy_((ob.double()+nb).to(b.dtype))
            aw=w.double()-ow.double();ab=None if b is None else b.double()-ob.double()
            record=dict(threshold=threshold,affected_fraction=affected,proposed_norm=float(u.norm()),
                        actual_norm=float(torch.cat([aw.flatten()]+([] if ab is None else [ab.flatten()])).norm()))
            if kind=='functional':
                actual_scores=affine_scores(module,inputs[name],aw,ab)
                # Cauchy-Schwarz bound for storage rounding, per neuron/channel.
                width=dw[0].numel(); bound=math.sqrt(width)*float(inputs[name].abs().max())*(aw-nw).flatten(1).norm(dim=1)
                if ab is not None:bound+=(ab-nb).abs()
                assert bool((actual_scores<=threshold+bound+1e-10).all())
                record.update(proposed_scores=scores.cpu().tolist(),actual_scores=actual_scores.cpu().tolist(),
                    factors=factors.cpu().tolist(),rounding_bound=bound.cpu().tolist(),cap_excess=max(0.,float(actual_scores.max())-threshold))
            if kind=='discard':
                assert bool((aw[dw.abs()>threshold]==0).all())
                if db is not None:assert bool((ab[db.abs()>threshold]==0).all())
            records[name]=record
    else: assert kind=='control'
    return records
