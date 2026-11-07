"""Post-Adam first-layer selection with intact raw gradients and Adam state."""
import torch


def neuron_rms(x, dw, db):
    """N samples by D input coordinates; one RMS per output neuron."""
    z = x.double().flatten(1) @ dw.double().T + db.double()
    return z.square().mean(0).sqrt()


def transform(dw, db, kind, threshold=None, x=None, target_norm=None):
    dw, db = dw.double(), db.double()
    flat = torch.cat([dw.flatten(), db.flatten()])
    if kind in ['discard', 'cap']:
        assert threshold is not None and threshold > 0
        if kind == 'discard':
            result = torch.where(flat.abs() <= threshold, flat, 0.)
        else:
            result = flat.clamp(-threshold, threshold)
        factors = None
        affected = float((flat.abs() > threshold).double().mean())
        nw, nb = result[:dw.numel()].reshape_as(dw), result[dw.numel():].reshape_as(db)
    elif kind == 'neuron':
        assert threshold is not None and threshold > 0 and x is not None
        scores = neuron_rms(x, dw, db)
        factors = torch.where(scores > 0, (threshold / scores).clamp(max=1), torch.ones_like(scores))
        nw, nb = dw * factors[:, None], db * factors
        affected = float((scores > threshold).double().mean())
    elif kind == 'uniform':
        assert target_norm is not None and target_norm >= 0
        norm = float(flat.norm())
        assert norm > 0 or target_norm == 0
        factor = target_norm/norm if norm else 0.
        nw, nb = dw*factor, db*factor
        factors = torch.full_like(db, factor)
        affected = float(factor != 1.)
    else:
        raise ValueError(kind)
    result = torch.cat([nw.flatten(), nb.flatten()])
    return nw, nb, dict(affected_fraction=affected, proposed_norm=float(flat.norm()),
        intended_norm=float(result.norm()), removed_norm=float((flat-result).norm()),
        factors=factors)


@torch.no_grad()
def apply_after_adam(model, before, kind, threshold, x, target_norm=None, capture_audit=False):
    """Called after native optimizer.step; changes only first-layer parameters."""
    oldw, oldb = before['1.weight'], before['1.bias']
    dw = model[1].weight.double()-oldw.double()
    db = model[1].bias.double()-oldb.double()
    nw, nb, details = transform(dw, db, kind, threshold, x, target_norm)
    model[1].weight.copy_((oldw.double()+nw).to(oldw.dtype))
    model[1].bias.copy_((oldb.double()+nb).to(oldb.dtype))
    aw, ab = model[1].weight.double()-oldw.double(), model[1].bias.double()-oldb.double()
    proposed_scores=neuron_rms(x,dw,db)
    actual_scores=neuron_rms(x,aw,ab)
    intended_scores=neuron_rms(x,nw,nb)
    rounding_scores=neuron_rms(x,aw-nw,ab-nb)
    # Minkowski's inequality gives a data-dependent allowance for FP32 storage.
    if kind=='neuron':
        assert bool((intended_scores <= threshold + 1e-12).all())
        assert bool((actual_scores <= threshold + rounding_scores + 1e-12).all())
    if kind=='discard':
        assert torch.equal(aw[dw.abs()>threshold],torch.zeros_like(aw[dw.abs()>threshold]))
        assert torch.equal(ab[db.abs()>threshold],torch.zeros_like(ab[db.abs()>threshold]))
    actual=torch.cat([aw.flatten(),ab.flatten()])
    intended=torch.cat([nw.flatten(),nb.flatten()])
    qerror=float((actual-intended).norm())
    if kind=='uniform':assert abs(float(actual.norm())-target_norm)<=qerror+1e-12
    factors=details.pop('factors')
    stats=dict(details,kind=kind,threshold=threshold,target_norm=target_norm,
        actual_norm=float(actual.norm()),rounding_error_norm=qerror,
        retained_norm_fraction=float(actual.norm())/details['proposed_norm'] if details['proposed_norm'] else None,
        proposed_neuron_rms_max=float(proposed_scores.max()),actual_neuron_rms_max=float(actual_scores.max()),
        actual_neuron_rms_mean=float(actual_scores.mean()),rounding_neuron_rms_max=float(rounding_scores.max()),
        actual_neuron_cap_excess=max(0.,float(actual_scores.max())-threshold) if kind=='neuron' else None,
        factor_min=float(factors.min()) if factors is not None else None,
        factor_mean=float(factors.mean()) if factors is not None else None,
        factor_max=float(factors.max()) if factors is not None else None)
    audit=dict(before_weight=oldw.cpu(),before_bias=oldb.cpu(),proposed_weight=dw.cpu(),proposed_bias=db.cpu(),
        actual_weight=aw.cpu(),actual_bias=ab.cpu(),proposed_neuron_rms=proposed_scores.cpu(),
        actual_neuron_rms=actual_scores.cpu(),rounding_neuron_rms=rounding_scores.cpu(),
        factors=None if factors is None else factors.cpu(),stats=stats) if capture_audit else None
    return stats,audit
