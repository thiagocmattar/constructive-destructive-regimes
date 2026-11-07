"""Observe the final hidden activations without modifying model state."""
import torch


@torch.no_grad()
def capture(model, x, physical):
    hidden = []
    handle = model[-1].register_forward_pre_hook(lambda _layer, inputs: hidden.append(inputs[0].detach()))
    try:
        outputs = torch.cat([model(xb) for xb in x.split(physical)])
    finally:
        handle.remove()
    return torch.cat(hidden), outputs
