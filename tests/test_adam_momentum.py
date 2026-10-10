"""Independent first-moment recurrence, capture invariance, schedule and recovery."""
from pathlib import Path
import copy
import sys
import tempfile
import unittest
import numpy as np
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from baseline.gradient_geometry import GradientGeometry
from baseline.momentum_diagnostics import momentum_vectors, momentum_norm_comparison
from baseline.recipe import DynamicLossScaler, recipe_learning_rate, warmup_constant_learning_rate
from baseline.training import optimizer_boundary, checkpoint


class MomentumTests(unittest.TestCase):
    def test_stored_clipped_moment_order_prefix_and_resume(self):
        p = torch.nn.Parameter(torch.tensor([1., 2.]))
        q = torch.nn.Parameter(torch.tensor([3.]))
        named = [("p", p), ("q", q)]
        opt = torch.optim.AdamW([q, p], lr=0, betas=(.9, .95))
        geometry = GradientGeometry(named)
        ideal = torch.zeros(3, dtype=torch.float64)
        history = []
        for t, g in enumerate(([3., 4., 0.], [-1., .2, .8], [0., 0., 0.]), 1):
            p.grad, q.grad = torch.tensor(g[:2]), torch.tensor(g[2:])
            torch.nn.utils.clip_grad_norm_([p, q], 1.)
            clipped = torch.cat([p.grad, q.grad]).double()
            ideal = .9 * ideal + .1 * clipped
            opt.step()
            before = copy.deepcopy(opt.state_dict())
            vectors = momentum_vectors(opt, named, t)
            actual = torch.cat(vectors).double()
            torch.testing.assert_close(actual, ideal, rtol=1e-6, atol=1e-8)
            if t == 1:
                self.assertLess(float(actual.norm()), .101)  # uncorrected m, not m-hat
            history.append(actual.clone())
            metrics = geometry.consume(vectors)
            direct = torch.stack(history)
            self.assertAlmostEqual(metrics["cumulative_norm_sum"], float(direct.norm(dim=1).sum()), places=12)
            self.assertAlmostEqual(metrics["cumulative_gradient_norm"], float(direct.sum(0).norm()), places=12)
            if t > 1:
                self.assertAlmostEqual(metrics["adjacent_cosine"], float(torch.dot(direct[-1], direct[-2]) /
                    (direct[-1].norm() * direct[-2].norm())), places=12)
            for key, state in before["state"].items():
                for field in ("exp_avg", "exp_avg_sq", "step"):
                    torch.testing.assert_close(state[field], opt.state_dict()["state"][key][field], rtol=0, atol=0)
            restored = GradientGeometry(named)
            restored.load_state_dict(geometry.state_dict())
            torch.testing.assert_close(restored.cumulative[0], geometry.cumulative[0], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            momentum_vectors(opt, named, 4)
        self.assertIsNone(momentum_norm_comparison(0, 0, 0)["momentum_to_raw_norm_ratio"])
        self.assertIsNone(momentum_norm_comparison(1, 0, 1)["adaptive_to_momentum_norm_ratio"])

    def test_warmup_clock_and_exact_tenfold_plateaus(self):
        for peak in (1e-4, 1e-3, 1e-2):
            rates = [warmup_constant_learning_rate(t, peak=peak, max_steps=712, warmup_fraction=.01)
                     for t in range(1, 713)]
            self.assertEqual(rates[0], 0.)
            self.assertEqual(rates[8:], [peak] * 704)
            for t in range(1, 9):
                self.assertEqual(rates[t-1], recipe_learning_rate(t, peak=peak, max_steps=712,
                                                                 warmup_fraction=.01, minimum=0))

    def test_extra_capture_does_not_change_update_and_checkpoint_roundtrip(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cfg = GPTNeoXConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=8)
        original = GPTNeoXForCausalLM(cfg).float()
        a, b = copy.deepcopy(original), copy.deepcopy(original)
        config = {"data": {"sequence_length": 8}, "training": {"maximum_overflow_retries": 2,
                  "micro_batch_size": 2, "gradient_clip_norm": 1.}}
        tokens = np.arange(32, dtype=np.int32) % 32
        states = []
        for model, capture in ((a, False), (b, True)):
            opt = torch.optim.AdamW(model.parameters(), lr=.01, betas=(.9, .95))
            raw = GradientGeometry(model.named_parameters())
            adaptive = GradientGeometry(model.named_parameters(), previous_dtype=torch.float64)
            momentum = GradientGeometry(model.named_parameters()) if capture else None
            scaler = DynamicLossScaler(scale=1)
            metrics = optimizer_boundary(model, opt, scaler, raw, adaptive, tokens, np.arange(4),
                                          config, torch.device("cpu"), momentum_geometry=momentum)
            states.append((opt, raw, adaptive, momentum, scaler, metrics))
        for p, q in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
        self.assertEqual(states[0][-1]["loss"], states[1][-1]["loss"])
        opt, raw, adaptive, momentum, scaler, metrics = states[1]
        self.assertIn("momentum", metrics)
        with tempfile.TemporaryDirectory(dir=ROOT / ".pytest_tmp") as tmp:
            checkpoint(Path(tmp), "test", b, opt, scaler, raw, adaptive, 4, "order", "config", momentum)
            recovery = torch.load(Path(tmp) / "checkpoints/test/recovery.pt", weights_only=False)
            restored = GradientGeometry(b.named_parameters())
            restored.load_state_dict(recovery["momentum_geometry"])
            self.assertEqual(restored.updates, 1)
            for p, q in zip(restored.cumulative, momentum.cumulative):
                torch.testing.assert_close(p, q, rtol=0, atol=0)
            self.assertEqual(recovery["data_position"], 4)


if __name__ == "__main__":
    unittest.main()
