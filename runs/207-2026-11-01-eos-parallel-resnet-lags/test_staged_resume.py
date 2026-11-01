"""Mechanics: pause/reload/recompute an unapplied gradient without a skipped step."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.eos_tasks import lag_profile
spec=importlib.util.spec_from_file_location('original',ROOT/'runs/206-2026-11-01-eos-long-lags/01_worker.py')
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)


class StagedResume(unittest.TestCase):
    def test_unapplied_boundary_at_all_recovery_scales(self):
        def sample(p,opt):
            opt.zero_grad();(.5*p.square().sum()).backward();return p.grad.clone()
        p=torch.nn.Parameter(torch.tensor([1.,-2.,.5]));opt=torch.optim.SGD([p],lr=1.9,foreach=False)
        gradients=[];states={}
        for step in range(401):
            gradients.append(sample(p,opt))
            if step in [10,128,256]:states[step]=p.detach().clone()
            if step<400:opt.step()
        for boundary,state in states.items():
            q=torch.nn.Parameter(state);other=torch.optim.SGD([q],lr=1.9,foreach=False)
            for step in range(boundary,401):
                torch.testing.assert_close(sample(q,other),gradients[step],rtol=0,atol=0)
                if step<400:other.step()
            torch.testing.assert_close(q,p,rtol=0,atol=0)

    def test_shared_exact_reductions(self):
        rng=np.random.default_rng(207)
        a,b=[rng.normal(size=150001).astype('f') for _ in range(2)]
        self.assertEqual(original.cpu_products(a,b),lag_profile.cpu_products(a,b))
        self.assertEqual(original.gpu_products(torch.from_numpy(a).double(),torch.from_numpy(b)),lag_profile.gpu_products(torch.from_numpy(a).double(),torch.from_numpy(b)))
        self.assertIsNone(lag_profile.cpu_products(a,np.zeros_like(a))['cosine'])


if __name__=='__main__':unittest.main()
