"""Run213: geometry of recorded Adam streams, without changing the optimizer."""
import torch

from src.cifar_eos.core import Geometry, cosine, flat


def ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else None


def projected_change(previous, current):
    if previous is None:
        return None
    previous, current = previous.double(), current.double()
    denominator = float(previous.square().sum())
    return None if denominator == 0 else float(previous.dot(previous-current)) / denominator


@torch.no_grad()
def adam_streams(optimizer, step):
    """Read actual FP32 stored moments; reconstruct the LR-free direction in FP64."""
    moments, corrected, directions = [], [], []
    for group in optimizer.param_groups:
        beta1, beta2 = group['betas']
        if group['weight_decay'] != 0 or group['amsgrad']:
            raise ValueError('This contract requires plain Adam without decay or AMSGrad.')
        for parameter in group['params']:
            state = optimizer.state[parameter]
            if int(state['step']) != step:
                raise ValueError('Adam clocks differ from accepted-update count.')
            moment = state['exp_avg'].detach().double()
            unbiased = moment / (1-beta1**step)
            denominator = (state['exp_avg_sq'].detach().double() / (1-beta2**step)).sqrt() + group['eps']
            moments.append(moment)
            corrected.append(unbiased)
            directions.append(unbiased / denominator)
    return dict(moment=flat(moments), unbiased_moment=flat(corrected), adaptive=flat(directions))


class Diagnostics:
    def __init__(self, count, device):
        self.streams = {name: Geometry(count, device) for name in ('raw', 'moment', 'adaptive', 'actual')}

    @torch.no_grad()
    def consume(self, gradient, optimizer, step, before, after):
        gradient = gradient.detach().double()
        vectors = adam_streams(optimizer, step)
        delta = after.detach().double() - before.detach().double()
        lr = optimizer.param_groups[0]['lr']
        if any(group['lr'] != lr for group in optimizer.param_groups):
            raise ValueError('This contract requires one learning rate.')
        q = projected_change(self.streams['raw'].previous, gradient)
        previous_moment = self.streams['moment'].previous
        incoming = None if previous_moment is None else cosine(gradient, previous_moment)
        values = dict(raw=gradient, moment=vectors['moment'], adaptive=vectors['adaptive'], actual=delta)
        stats = {name: self.streams[name].consume(vector) for name, vector in values.items()}
        unbiased_norm = float(vectors['unbiased_moment'].norm())
        expected = -lr * vectors['adaptive']
        return dict(streams=stats, projected_gradient_change=q,
            incoming_gradient_prior_moment_cosine=incoming,
            gradient_current_moment_cosine=cosine(gradient, vectors['moment']),
            gradient_adaptive_cosine=cosine(gradient, vectors['adaptive']),
            gradient_actual_step_cosine=cosine(gradient, delta),
            unbiased_moment_norm=unbiased_norm,
            moment_over_gradient_norm=ratio(stats['moment']['norm'], stats['raw']['norm']),
            unbiased_moment_over_gradient_norm=ratio(unbiased_norm, stats['raw']['norm']),
            gradient_dot_actual_step=float(gradient.dot(delta)),
            adam_step_rounding_residual_norm=float((delta-expected).norm()),
            adam_step_rounding_relative_residual=ratio(float((delta-expected).norm()), float(expected.norm())),
            adaptive_actual_step_cosine=cosine(-vectors['adaptive'], delta),
            cumulative_step_rounding_residual_norm=float((self.streams['actual'].total+lr*self.streams['adaptive'].total).norm()))

    def state_dict(self):
        return {name: stream.state_dict() for name, stream in self.streams.items()}

    def load_state_dict(self, state):
        for name, stream in self.streams.items():
            stream.load_state_dict(state[name])
