"""Vector-gradient population variance; never infer it from scalar loss noise."""
import torch

from src.cifar_eos.core import flat


@torch.no_grad()
def mlp_unit_energy(task):
    # Exact per-image gradient norm squared for Linear-tanh-Linear-tanh-Linear.
    # Each weight gradient is an outer product; its squared Frobenius norm is
    # ||backpropagated residual||^2 ||layer input||^2. Bias adds residual energy.
    _, l1, _, l2, _, l3 = task.model
    total = torch.zeros((),dtype=torch.float64,device=l1.weight.device)
    for x,y in task.chunks():
        x=x.flatten(1)
        h1=torch.tanh(l1(x));h2=torch.tanh(l2(h1))
        residual=l3(h2)-torch.nn.functional.one_hot(y,10).to(x.dtype)
        d2=(residual@l3.weight)*(1-h2.square())
        d1=(d2@l2.weight)*(1-h1.square())
        energy=(d1.double().square().sum(1)*(x.double().square().sum(1)+1)
            +d2.double().square().sum(1)*(h1.double().square().sum(1)+1)
            +residual.double().square().sum(1)*(h2.double().square().sum(1)+1))
        total+=energy.sum()
    return float(total)


def variance_probe(task, generator, reference, batch_units, sample_units=32):
    device=reference.device
    if task.name=="mlp":
        n=task.n
        energy=mlp_unit_energy(task)
        variance=(energy-n*float(reference.double().square().sum()))/(n-1)
        coverage={"population_units":n,"measured_units":n,"method":"exact all-image MLP gradient energy"}
    else:
        # Exact all-group population for ResNet; unbiased sample covariance
        # trace for the Transformer. Sampling is without replacement.
        r=task.n if task.name=="resnet" else min(sample_units,task.n)
        ids=torch.arange(r,device=device) if r==task.n else torch.randperm(task.n,generator=generator,device=device)[:r]
        mean=torch.zeros_like(reference,dtype=torch.float64)
        m2=torch.zeros((),dtype=torch.float64,device=device)
        for count,i in enumerate(ids,1):
            _,g=task.gradient(i.reshape(1))
            g=g.double()
            delta=g-mean
            mean+=delta/count
            m2+=delta.dot(g-mean)
        variance=float(m2)/(r-1)
        coverage={"population_units":task.n,"measured_units":r,
            "method":"exact fixed-ghost-group gradient covariance" if r==task.n else "unbiased sampled sequence-gradient covariance trace",
            "probe_mean_difference_norm":float((mean-reference.double()).norm())}
    # Only floating-point cancellation within an explicitly recorded tolerance
    # may produce a tiny negative energy estimate.
    tolerance=1e-5*max(1,float(reference.double().square().sum()))
    if variance < -tolerance:raise RuntimeError(f"Negative variance: {variance}")
    variance=max(0,variance)
    return {"trace_population_sample_covariance":variance,
        "estimated_batch_gradient_noise_energy":(1/batch_units-1/task.n)*variance,
        "effective_batch_units":batch_units,"reference_gradient_norm":float(reference.double().norm()),
        "coverage":coverage,"finite_population_formula":"(1/b-1/N)*S^2; S^2=trace of unit covariance with N-1 denominator"}
