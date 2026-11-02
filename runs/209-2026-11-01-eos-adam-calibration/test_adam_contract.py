import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.eos_tasks.adam_profile import select_rate
from src.eos_tasks.lag_profile import cpu_products, gpu_products


def test_products_and_zero_denominator():
    a=np.array([1.,-2.,4.],dtype=np.float32)
    b=np.array([-2.,3.,1.],dtype=np.float32)
    cpu=cpu_products(a,b);gpu=gpu_products(torch.from_numpy(a).double(),torch.from_numpy(b))
    assert abs(cpu['cosine']-gpu['cosine'])<1e-14
    assert cpu_products(a,np.zeros_like(a))['cosine'] is None
    assert abs(cpu_products(a,-a)['cosine']+1)<1e-14


def test_cold_adam_and_unapplied_terminal_boundary():
    x=torch.nn.Parameter(torch.tensor([2.,-3.],dtype=torch.float64))
    optimizer=torch.optim.Adam([x],lr=.003,betas=(.9,.999),eps=1e-8,foreach=False,fused=False)
    g=torch.tensor([.4,-.7],dtype=torch.float64)
    assert not optimizer.state
    x.grad=g.clone();before=x.detach().clone();optimizer.step()
    expected=before-.003*g/(g.abs()+1e-8)
    assert torch.allclose(x,expected,atol=1e-14,rtol=0)
    state=optimizer.state[x]
    assert state['step']==1
    assert torch.allclose(state['exp_avg'],.1*g)
    assert torch.allclose(state['exp_avg_sq'],.001*g.square())
    # Evaluating the terminal gradient does not advance the optimizer clock.
    x.grad=None;(.5*x.square().sum()).backward()
    assert state['step']==1


def test_selection_ignores_flagged_states_and_cosines():
    def arm(rate,loss,peak=1.):
        return dict(lr=rate,status='complete',rows=[dict(loss=peak)]+[dict(loss=loss,adjacent_cosine=-1) for _ in range(64)])
    results=[dict(primary_eligible=True,arms=[arm(.001,.8),arm(.01,.7)]),
        dict(primary_eligible=False,arms=[arm(.001,.1),arm(.01,10.)])]
    decision=select_rate(results,[.001,.01])
    assert decision['selected_lr']==.01
    results[0]['arms'][1]['rows'][1]['loss']=5.
    assert select_rate(results,[.001,.01])['selected_lr']==.001
