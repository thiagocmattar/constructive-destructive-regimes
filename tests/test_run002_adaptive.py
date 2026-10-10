"""Check Adam capture semantics against independent moments and actual updates."""

import importlib.util
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from baseline.gradient_geometry import GradientGeometry

spec = importlib.util.spec_from_file_location("run002_adam_diagnostics",
    ROOT / "runs/002-2026-10-03-pythia14m-adamw-adaptive-geometry/adam_diagnostics.py")
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


class AdaptiveCaptureTests(unittest.TestCase):
    def test_matches_independent_moments_and_actual_update_without_decay(self):
        p = torch.nn.Parameter(torch.tensor([.2, -.3, .7]))
        q = torch.nn.Parameter(torch.tensor([-.4, .1]))
        named = [("p", p), ("q", q)]
        opt = torch.optim.AdamW([{"params": [q], "weight_decay": 0},
                               {"params": [p], "weight_decay": .1}],
                               lr=.02, betas=(.9, .95), eps=1e-4)
        m = [torch.zeros_like(x, dtype=torch.float64) for _, x in named]
        v = [torch.zeros_like(x, dtype=torch.float64) for _, x in named]
        for t, gradients in enumerate([
            ([2., -1., .5], [-.7, .4]),
            ([-.2, .5, .8], [.3, -.6]),
            ([0., .2, -.4], [.1, .8]),
        ], 1):
            before = [x.detach().double().clone() for _, x in named]
            for (_, x), g in zip(named, gradients):
                x.grad = torch.tensor(g)
            opt.step()
            states = [opt.state[x]["exp_avg"].clone() for _, x in named]
            directions = diagnostics.adaptive_directions(opt, named, t)
            for i, ((_, x), g, d) in enumerate(zip(named, gradients, directions)):
                g = torch.tensor(g, dtype=torch.float64)
                m[i] = .9 * m[i] + .1 * g
                v[i] = .95 * v[i] + .05 * g.square()
                expected = (m[i] / (1 - .9 ** t)) / ((v[i] / (1 - .95 ** t)).sqrt() + 1e-4)
                # Actual fused FP32 moments differ by a few ULPs from ideal FP64 recurrence.
                torch.testing.assert_close(d, expected, rtol=1e-6, atol=1e-8)
                decay = .1 if i == 0 else 0
                observed = (before[i] - x.detach().double()) / .02 - decay * before[i]
                torch.testing.assert_close(d, observed, rtol=3e-5, atol=3e-6)
                torch.testing.assert_close(states[i], opt.state[x]["exp_avg"], rtol=0, atol=0)
                self.assertEqual(d.dtype, torch.float64)

    def test_zero_learning_rate_still_updates_moments_and_direction(self):
        p = torch.nn.Parameter(torch.tensor([1., 2.]))
        opt = torch.optim.AdamW([p], lr=0, betas=(.9, .95))
        before = p.detach().clone()
        p.grad = torch.tensor([3., -4.])
        opt.step()
        d = diagnostics.adaptive_directions(opt, [("p", p)], 1)[0]
        torch.testing.assert_close(p, before, rtol=0, atol=0)
        torch.testing.assert_close(d, torch.tensor([1., -1.], dtype=torch.float64), rtol=1e-6, atol=1e-8)

    def test_moments_use_clipped_gradient_and_epsilon_outside_root(self):
        p = torch.nn.Parameter(torch.tensor([.1, .2]))
        p.grad = torch.tensor([3., 4.])
        torch.nn.utils.clip_grad_norm_([p], 1.)
        clipped = p.grad.double().clone()
        opt = torch.optim.AdamW([p], lr=.02, betas=(.9, .95), eps=.2)
        opt.step()
        d = diagnostics.adaptive_directions(opt, [("p", p)], 1)[0]
        torch.testing.assert_close(d, clipped / (clipped.abs() + .2), rtol=1e-6, atol=1e-8)
        self.assertGreater(float(torch.linalg.vector_norm(clipped)), .99999)
        with self.assertRaises(ValueError):
            diagnostics.adaptive_directions(opt, [("p", p)], 2)

    def test_streaming_geometry_matches_explicit_prefix_with_zero_vector(self):
        p = torch.nn.Parameter(torch.zeros(3))
        geometry = GradientGeometry([("p", p)], previous_dtype=torch.float64)
        vectors = [torch.tensor(x, dtype=torch.float64) for x in
                   ([1., 2., 0.], [-2., .5, 1.], [0., 0., 0.], [1., -1., 2.])]
        total = torch.zeros(3, dtype=torch.float64)
        length = 0.
        for i, vector in enumerate(vectors):
            measured = geometry.consume([vector])
            total += vector
            length += float(torch.linalg.vector_norm(vector))
            self.assertAlmostEqual(measured["cumulative_norm_sum"], length, places=12)
            self.assertAlmostEqual(measured["cumulative_gradient_norm"], float(torch.linalg.vector_norm(total)), places=12)
            if i == 0 or not bool(vector.any()) or not bool(vectors[i-1].any()):
                self.assertIsNone(measured["adjacent_cosine"])
            else:
                expected = float(torch.dot(vector, vectors[i-1]) /
                    (torch.linalg.vector_norm(vector) * torch.linalg.vector_norm(vectors[i-1])))
                self.assertAlmostEqual(measured["adjacent_cosine"], expected, places=12)
        restored = GradientGeometry([("p", p)], previous_dtype=torch.float64)
        restored.load_state_dict(geometry.state_dict())
        torch.testing.assert_close(restored.cumulative[0], total, rtol=0, atol=0)

    def test_zero_direction_and_zero_norm_denominator(self):
        p = torch.nn.Parameter(torch.zeros(2))
        p.grad = torch.zeros(2)
        opt = torch.optim.AdamW([p], lr=.01, betas=(.9, .95))
        opt.step()
        d = diagnostics.adaptive_directions(opt, [("p", p)], 1)[0]
        self.assertFalse(bool(d.any()))
        geometry = GradientGeometry([("p", p)], previous_dtype=torch.float64)
        self.assertIsNone(geometry.consume([d])["cumulative_ratio"])
        self.assertIsNone(diagnostics.norm_comparison(0., 0.)["adaptive_to_raw_norm_ratio"])


if __name__ == "__main__":
    unittest.main()
