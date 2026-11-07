"""Whole-example raw-gradient energies for the pinned three-linear-layer MLP."""
import torch


@torch.no_grad()
def example_energies(model,x,y,physical=1000):
    energies=[]
    for xb,yb in zip(x.split(physical),y.split(physical)):
        a=xb.flatten(1)
        h1=model[1](a).tanh();h2=model[3](h1).tanh();f=model[5](h2)
        r=f-torch.nn.functional.one_hot(yb,f.shape[1]).to(f.dtype)
        d2=(r@model[5].weight)*(1-h2.square())
        d1=(d2@model[3].weight)*(1-h1.square())
        # Each +1 includes this layer's bias gradient; sum before square root.
        e=(d1.double().square().sum(1)*(a.double().square().sum(1)+1)
           +d2.double().square().sum(1)*(h1.double().square().sum(1)+1)
           +r.double().square().sum(1)*(h2.double().square().sum(1)+1))
        energies.append(e)
    return torch.cat(energies)


def coherence(gradient_norm,energies):
    mean_norm=float(energies.sqrt().mean());mean_energy=float(energies.mean())
    return dict(mean_example_gradient_norm=mean_norm,mean_example_gradient_energy=mean_energy,
        intrabatch_coherence=gradient_norm/mean_norm if mean_norm else None,
        energy_coherence=gradient_norm**2/mean_energy if mean_energy else None)
