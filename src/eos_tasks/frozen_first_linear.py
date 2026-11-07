"""First-linear parameter suppression, shared after its qualified Run244 use."""
import torch


@torch.no_grad()
def freeze_first_step(model, optimizer, fixed):
    optimizer.step()
    model[1].weight.copy_(fixed['weight'])
    model[1].bias.copy_(fixed['bias'])


def cosine(a,b):
    a,b=a.double().reshape(-1),b.double().reshape(-1)
    den=float(a.norm()*b.norm())
    return float(a.dot(b))/den if den else None
