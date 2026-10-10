"""Run001 mathematical, accumulation-boundary, and checkpoint-continuity checks."""

import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

RUN = Path(__file__).resolve().parents[1] / "runs/001-2026-10-03-pythia14m-adamw-gradient-baseline"
sys.path.insert(0, str(RUN))
from data_protocol import block_order, target_count, update_microbatches
from gradient_geometry import GradientGeometry
from recipe import DynamicLossScaler, recipe_learning_rate
from training import optimizer_boundary


class GeometryTests(unittest.TestCase):
    def parameters(self):
        return [("first", torch.nn.Parameter(torch.zeros(2))),
                ("second", torch.nn.Parameter(torch.zeros(1)))]

    def assign(self, parameters, vector):
        parameters[0][1].grad = torch.tensor(vector[:2], dtype=torch.float32)
        parameters[1][1].grad = torch.tensor(vector[2:], dtype=torch.float32)

    def test_every_prefix_matches_direct_sum_and_global_cosine(self):
        parameters = self.parameters()
        geometry = GradientGeometry(parameters)
        history = torch.tensor([[1., 2., 3.], [-2., 0., 1.], [0., 0., 0.], [4., 1., -2.]], dtype=torch.float32)
        for i, gradient in enumerate(history):
            self.assign(parameters, gradient.tolist())
            result = geometry.consume()
            reference = history[:i+1].double()
            self.assertAlmostEqual(result["cumulative_gradient_norm"], float(reference.sum(0).norm()), places=12)
            self.assertAlmostEqual(result["cumulative_norm_sum"], float(reference.norm(dim=1).sum()), places=12)
            if i and gradient.norm() and history[i-1].norm():
                expected = float(torch.dot(gradient.double(), history[i-1].double()) /
                                 (gradient.double().norm()*history[i-1].double().norm()))
                self.assertAlmostEqual(result["adjacent_cosine"], expected, places=12)
            else:
                self.assertIsNone(result["adjacent_cosine"])

    def test_cancellation_and_zero_denominators(self):
        parameters = self.parameters()
        geometry = GradientGeometry(parameters)
        self.assign(parameters, [0, 0, 0])
        self.assertIsNone(geometry.consume()["cumulative_ratio"])
        self.assign(parameters, [1, 0, 0]); geometry.consume()
        self.assign(parameters, [-1, 0, 0])
        result = geometry.consume()
        self.assertEqual(result["adjacent_cosine"], -1)
        self.assertEqual(result["cumulative_ratio"], 0)

    def test_clipping_and_clearing_do_not_mutate_saved_previous(self):
        parameters = self.parameters()
        geometry = GradientGeometry(parameters)
        self.assign(parameters, [3, 4, 0]); geometry.consume()
        torch.nn.utils.clip_grad_norm_([p for _, p in parameters], .01)
        for _, p in parameters: p.grad = None
        self.assign(parameters, [6, 8, 0])
        result = geometry.consume()
        self.assertAlmostEqual(result["adjacent_cosine"], 1)
        self.assertEqual(result["cumulative_norm_sum"], 15)

    def test_resume_and_reject_nonfinite_without_state_change(self):
        parameters = self.parameters()
        first = GradientGeometry(parameters)
        self.assign(parameters, [1, 2, 3]); first.consume()
        restored = GradientGeometry(parameters)
        restored.load_state_dict(first.state_dict())
        self.assign(parameters, [4, 5, 6])
        self.assertEqual(first.consume(), restored.consume())
        self.assign(parameters, [float("nan"), 0, 0])
        before = restored.state_dict()
        with self.assertRaises(RuntimeError): restored.consume()
        self.assertEqual(restored.updates, before["updates"])
        for a, b in zip(restored.cumulative, before["cumulative"]):
            self.assertTrue(torch.equal(a, b))


class DataAndBoundaryTests(unittest.TestCase):
    def test_nonrepeating_order_and_short_tail_coverage(self):
        tokens = np.arange(35, dtype=np.int32)
        order = block_order(len(tokens), 8, 1234)
        self.assertEqual(sorted(order.tolist()), list(range(5)))
        self.assertEqual(order[-1], 4)
        batches = list(update_microbatches(tokens, order, 8, 3, "cpu"))
        self.assertEqual(sum(batch.numel() for batch, _ in batches), len(tokens))
        self.assertEqual(sum(n for _, n in batches), len(tokens)-len(order))
        self.assertEqual(target_count(len(tokens), order, 8), 30)

    def test_token_weighted_accumulation_before_clipping(self):
        torch.manual_seed(1)
        model = GPTNeoXForCausalLM(GPTNeoXConfig(vocab_size=32, hidden_size=8,
                    intermediate_size=16, num_hidden_layers=1, num_attention_heads=2,
                    max_position_embeddings=16, hidden_dropout=0, attention_dropout=0))
        model.config.use_cache = False
        reference = copy.deepcopy(model)
        tokens = (np.arange(19) % 32).astype(np.int32)
        indices = np.arange(3)  # Two complete sequences plus a three-token tail.
        denominator = target_count(len(tokens), indices, 8)
        for batch, count in update_microbatches(tokens, indices, 8, 99, "cpu"):
            (reference(input_ids=batch, labels=batch).loss * count / denominator).backward()
        config = {"data": {"sequence_length": 8}, "training": {"micro_batch_size": 1,
                  "maximum_overflow_retries": 0, "gradient_clip_norm": .0001}}
        optimizer = torch.optim.AdamW(model.parameters(), lr=0, weight_decay=0)
        geometry = GradientGeometry(model.named_parameters())
        result = optimizer_boundary(model, optimizer, DynamicLossScaler(scale=4), geometry,
                                    tokens, indices, config, torch.device("cpu"))
        expected = float(torch.stack([p.grad.double().square().sum() for p in reference.parameters()]).sum().sqrt())
        self.assertAlmostEqual(result["gradient_norm"], expected, places=6)
        for total, p in zip(geometry.cumulative, reference.parameters()):
            torch.testing.assert_close(total.float(), p.grad, rtol=3e-5, atol=2e-7)
        self.assertLess(float(torch.stack([p.grad.double().square().sum() for p in model.parameters()]).sum().sqrt()), .00011)

    def test_reference_schedule_and_scaler_state(self):
        options = dict(peak=.001, minimum=.0001, max_steps=712, warmup_fraction=.01)
        self.assertEqual(recipe_learning_rate(1, **options), 0)
        self.assertAlmostEqual(recipe_learning_rate(2, **options), .001/7.12)
        self.assertEqual(recipe_learning_rate(712, **options), .0001)
        scaler = DynamicLossScaler(scale=8, hysteresis=2)
        scaler.update(finite=False)
        self.assertEqual(scaler.scale, 8)
        scaler.update(finite=False)
        self.assertEqual(scaler.scale, 4)
        restored = DynamicLossScaler()
        restored.load_state_dict(scaler.state_dict())
        self.assertEqual(restored.state_dict(), scaler.state_dict())


if __name__ == "__main__":
    unittest.main()
