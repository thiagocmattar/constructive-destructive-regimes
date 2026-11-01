"""Mechanics only: exact reductions, parity stop and unapplied boundary."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

spec = importlib.util.spec_from_file_location('worker', Path(__file__).with_name('01_worker.py'))
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class StreamingTests(unittest.TestCase):
    def test_chunked_products_and_undefined(self):
        rng = np.random.default_rng(12)
        a, b = [rng.normal(size=150007).astype(np.float32) for _ in range(2)]
        expected = float(np.dot(a.astype('d'),b.astype('d'))/np.linalg.norm(a.astype('d'))/np.linalg.norm(b.astype('d')))
        self.assertAlmostEqual(worker.cpu_products(a,b)['cosine'], expected, places=14)
        self.assertAlmostEqual(worker.gpu_products(torch.from_numpy(a).double(), torch.from_numpy(b))['cosine'], expected, places=14)
        self.assertIsNone(worker.cpu_products(a,np.zeros_like(a))['cosine'])

    def test_persistent_and_damped_branches(self):
        rows = [dict(lag=k,cosine=(-1.)**k) for k in range(257)]
        self.assertEqual(worker.envelope(rows), 1)
        for r in rows[-64:]: r['cosine'] *= .05
        self.assertAlmostEqual(worker.envelope(rows), .05)
        self.assertIsNone(worker.envelope(rows[:10]))

    def test_resume_unapplied_gradient_and_retention(self):
        theta = torch.nn.Parameter(torch.tensor([1.,-2.]))
        opt = torch.optim.SGD([theta],lr=.1,foreach=False)
        gradients = []
        for lag in range(21):
            opt.zero_grad(); (.5*theta.square().sum()).backward()
            gradients.append(theta.grad.clone())
            if lag == 10:
                saved = theta.detach().clone()
            if lag < 20: opt.step()
        resumed = torch.nn.Parameter(saved)
        other = torch.optim.SGD([resumed],lr=.1,foreach=False)
        for lag in range(10,21):
            other.zero_grad(); (.5*resumed.square().sum()).backward()
            torch.testing.assert_close(resumed.grad,gradients[lag],rtol=0,atol=0)
            if lag < 20: other.step()
        torch.testing.assert_close(theta,resumed,rtol=0,atol=0)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'g.npy';worker.atomic_array(path,gradients[-1].numpy())
            np.testing.assert_array_equal(np.load(path),gradients[-1].numpy())


if __name__ == '__main__': unittest.main()
