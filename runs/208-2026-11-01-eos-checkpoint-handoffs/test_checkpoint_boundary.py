"""Mechanics: long saved states retain the unapplied gradient and exact prefix."""
import unittest
import numpy as np
import torch
from src.eos_tasks.lag_profile import cpu_products,envelope


class CheckpointBoundaryTests(unittest.TestCase):
    def test_long_resumes_have_no_duplicate_or_skipped_update(self):
        parameter=torch.nn.Parameter(torch.tensor([1.,2.]))
        optimizer=torch.optim.SGD([parameter],lr=.99,foreach=False)
        gradients=[];saved={};rows=[]
        for lag in range(601):
            optimizer.zero_grad();(.5*parameter[0].square()+parameter[1].square()).backward()
            gradient=parameter.grad.detach().numpy().copy();gradients.append(gradient)
            rows.append(dict(lag=lag,**cpu_products(gradients[0],gradient)))
            if lag in [256,512]:saved[lag]=parameter.detach().clone()
            if lag<600:optimizer.step()
        for start in saved:
            resumed=torch.nn.Parameter(saved[start]);other=torch.optim.SGD([resumed],lr=.99,foreach=False)
            merged=list(rows[:start+1])
            for lag in range(start,601):
                other.zero_grad();(.5*resumed[0].square()+resumed[1].square()).backward()
                gradient=resumed.grad.detach().numpy().copy()
                np.testing.assert_array_equal(gradient,gradients[lag])
                if lag>start:merged.append(dict(lag=lag,**cpu_products(gradients[0],gradient)))
                if lag<600:other.step()
            self.assertEqual(merged,rows)
            self.assertEqual(envelope(merged),envelope(rows))
            torch.testing.assert_close(resumed,parameter,rtol=0,atol=0)

if __name__=='__main__':unittest.main()
