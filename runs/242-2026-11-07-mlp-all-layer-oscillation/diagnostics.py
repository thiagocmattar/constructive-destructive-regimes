"""Capture all forward stages and verify the exact input of the first linear."""
import torch


@torch.no_grad()
def capture(model,x,physical,reference):
    z1=[];h1=[];z2=[];h2=[];seen=[0]
    def check_input(_,inputs):
        value=inputs[0];start=seen[0];end=start+len(value)
        assert torch.equal(value,reference.flatten(1)[start:end]),('changed input',start,end)
        seen[0]=end
    handles=[model[1].register_forward_pre_hook(check_input),
             model[1].register_forward_hook(lambda _,inputs,out:z1.append(out.detach())),
             model[3].register_forward_pre_hook(lambda _,inputs:h1.append(inputs[0].detach())),
             model[3].register_forward_hook(lambda _,inputs,out:z2.append(out.detach())),
             model[-1].register_forward_pre_hook(lambda _,inputs:h2.append(inputs[0].detach()))]
    try:
        outputs=torch.cat([model(xb) for xb in x.split(physical)])
    finally:
        for handle in handles:handle.remove()
    return torch.cat(z1),torch.cat(h1),torch.cat(z2),torch.cat(h2),outputs,seen[0]
