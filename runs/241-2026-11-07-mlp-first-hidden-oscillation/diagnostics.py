"""Observe H1, Z2 and H2 with forward hooks; do not modify training state."""
import torch


@torch.no_grad()
def capture(model,x,physical):
    h1=[];z2=[];h2=[]
    handles=[model[3].register_forward_pre_hook(lambda _,inputs:h1.append(inputs[0].detach())),
             model[3].register_forward_hook(lambda _,inputs,out:z2.append(out.detach())),
             model[-1].register_forward_pre_hook(lambda _,inputs:h2.append(inputs[0].detach()))]
    try:
        outputs=torch.cat([model(xb) for xb in x.split(physical)])
    finally:
        for handle in handles:handle.remove()
    return torch.cat(h1),torch.cat(z2),torch.cat(h2),outputs
