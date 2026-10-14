"""Independent objective-weighting, HVP, noise-variance and secant checks."""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from eos import Task, configure
from src.cifar_eos.core import flat, write_json


def main():
    configure()
    torch.manual_seed(17)
    model = torch.nn.Sequential(torch.nn.Linear(3,7),torch.nn.Tanh(),torch.nn.Linear(7,10)).double()
    data = {"x_train":torch.randn(11,3,dtype=torch.float64),"y_train":torch.arange(11)%10}
    task = Task("mlp",model,data,4)
    loss, gradient = task.gradient()
    independent = .5*(model(data["x_train"])-torch.nn.functional.one_hot(data["y_train"],10)).square().sum()/11
    reference = flat(torch.autograd.grad(independent, model.parameters()))
    assert torch.allclose(reference, gradient, atol=1e-12,rtol=1e-12)
    assert abs(float(independent.detach())-loss)<1e-12
    pieces = torch.stack([task.gradient(torch.tensor([i]))[1] for i in range(11)])
    assert torch.allclose(pieces.mean(0),gradient,atol=1e-12,rtol=1e-12)
    vector = torch.randn_like(gradient)
    hv = task.hvp(vector)
    theta = flat(model.parameters()).detach().clone()
    def set_vector(value):
        offset=0
        with torch.no_grad():
            for p in model.parameters():
                p.copy_(value[offset:offset+p.numel()].reshape_as(p));offset+=p.numel()
    delta=1e-5
    set_vector(theta+delta*vector);gp=task.gradient()[1]
    set_vector(theta-delta*vector);gm=task.gradient()[1]
    set_vector(theta)
    assert torch.allclose(hv,(gp-gm)/(2*delta),atol=3e-8,rtol=3e-7)
    # Exact finite-population variance for every possible 3-of-11 batch.
    empirical=np.mean([float((pieces[list(ids)].mean(0)-gradient).square().sum())
                       for ids in itertools.combinations(range(11),3)])
    population=float((pieces-gradient).square().sum()/10)
    predicted=(1/3-1/11)*population
    assert abs(empirical-predicted)<1e-12
    # Fresh independent batches at a frozen state: expected unnormalized
    # cross-step dot equals signal energy; energy has the variance contribution.
    averages=torch.stack([pieces[list(ids)].mean(0) for ids in itertools.combinations(range(11),3)])
    mean=averages.mean(0)
    expected_dot=float(mean.dot(mean))
    assert abs(expected_dot-float(gradient.square().sum()))<1e-12
    assert abs(float(averages.square().sum(1).mean())-expected_dot-predicted)<1e-12
    previous=pieces[:3].mean(0); current=pieces[3:6].mean(0)
    xi_previous=previous-gradient; xi_current=current-(gradient+.01*vector)
    q=1-float(previous.dot(current)/previous.square().sum())
    epsilon=float(previous.dot(xi_current-xi_previous)/previous.square().sum())
    corrected=float(previous.dot(gradient-(gradient+.01*vector))/previous.square().sum())
    assert abs(q+epsilon-corrected)<1e-12
    # BN group objectives: only whole groups can be sampled without changing
    # normalization context. No running-statistics mutation is permitted.
    bn=torch.nn.Sequential(torch.nn.BatchNorm1d(3,track_running_stats=False),torch.nn.Linear(3,10)).double()
    bn_data={"x_train":torch.randn(2000,3,dtype=torch.float64),"y_train":torch.arange(2000)%10}
    ghost=Task("resnet",bn,bn_data,1000)
    full=ghost.gradient()[1]
    one=ghost.gradient(torch.tensor([0]))[1];two=ghost.gradient(torch.tensor([1]))[1]
    assert torch.allclose(full,(one+two)/2,atol=1e-12,rtol=1e-12)
    write_json(HERE/"mathematical_verification.json",{"status":"passed","physical_accumulation":True,
        "unbiased_objective_unit_sampling":True,"hvp_central_difference":True,
        "exact_finite_population_gradient_variance":{"measured":empirical,"formula":predicted},
        "independent_batch_dot_and_energy":True,"noise_corrected_secant_identity":True,
        "fixed_ghost_bn_gradient_parity":True,"not_scientific_evidence":True})
    print("Mathematical and objective-placement verification passed.")


if __name__=="__main__":main()
