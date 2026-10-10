"""Resume equivalence with stochastic forwards and global Adam/diagnostic clocks."""
import copy
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np
import torch
from transformers import GPTNeoXConfig, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baseline.continuation import learning_rate, load_parent, protected_evaluate, rng_state
from baseline.data_protocol import sha256
from baseline.gradient_geometry import GradientGeometry
from baseline.recipe import DynamicLossScaler, build_recipe_adamw
from baseline.training import checkpoint, optimizer_boundary

TRAINING = {"optimizer": "adamw", "peak_learning_rate": 1e-3, "minimum_learning_rate": 1e-4,
    "warmup_fraction": .01, "adamw_betas": [.9, .95], "adamw_eps": 1e-8, "weight_decay": .1,
    "effective_batch_sequences": 4, "micro_batch_size": 2, "maximum_overflow_retries": 2,
    "gradient_clip_norm": 1.0}


class Resume(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        np.random.seed(17)
        random.seed(17)

    def test_schedule_junction_and_floor_indexing(self):
        resume = {"junction_update": 356, "full_updates": 712, "schedule": "original"}
        original = [learning_rate(u, TRAINING, resume) for u in range(1, 713)]
        self.assertEqual(next(u for u, lr in enumerate(original, 1) if u > 356 and lr == 1e-4), 575)
        resume["schedule"] = "accelerated"
        fast = [learning_rate(u, TRAINING, resume) for u in range(1, 713)]
        self.assertEqual(original[:356], fast[:356])
        self.assertEqual(next(u for u, lr in enumerate(fast, 1) if u > 356 and lr == 1e-4), 466)
        resume["schedule"] = "hold"
        held = [learning_rate(u, TRAINING, resume) for u in range(1, 713)]
        self.assertEqual(original[:356], held[:356])
        self.assertTrue(all(lr == original[355] for lr in held[356:]))

    def test_uninterrupted_equals_resume_with_dropout_and_rng(self):
        device = torch.device("cpu")
        architecture = GPTNeoXConfig(vocab_size=32, hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=16,
            hidden_dropout=.1, attention_dropout=.1, rotary_pct=.5, use_cache=False)
        architecture._attn_implementation = "sdpa"
        model = AutoModelForCausalLM.from_config(architecture).float().train()
        optimizer, _ = build_recipe_adamw(model, TRAINING, torch=torch)
        geometry = GradientGeometry(model.named_parameters())
        adaptive = GradientGeometry(model.named_parameters(), previous_dtype=torch.float64)
        scaler = DynamicLossScaler(growth_interval=3)
        config = {"training": TRAINING, "data": {"sequence_length": 8}, "validation": {"batch_size": 2},
            "model": {"parameter_count": sum(p.numel() for p in model.parameters())}}

        def step(loaded):
            m, o, s, g, a = loaded
            tokens = torch.randint(0, 32, (32,)).numpy().astype(np.int32)
            result = optimizer_boundary(m, o, s, g, a, tokens, np.arange(4), config, device)
            return tokens, result

        loaded = model, optimizer, scaler, geometry, adaptive
        for _ in range(5):
            step(loaded)
        with tempfile.TemporaryDirectory() as temp:
            checkpoint(Path(temp), "junction", *loaded, 20, "order", "parent-config")
            parent = Path(temp) / "checkpoints/junction"
            config["resume"] = {"junction_update": 5, "parent_config_sha256": "parent-config",
                "order_sha256": "order", "checkpoint_sha256": {
                    p.relative_to(parent).as_posix(): sha256(p) for p in parent.rglob("*") if p.is_file()}}
            tokens, expected = step(loaded)
            tensors = {name: p.detach().clone() for name, p in model.named_parameters()}
            rng = torch.get_rng_state().clone()
            np_draw, python_draw = np.random.rand(), random.random()
            resumed = load_parent(config, parent, device)
            actual_tokens, actual = step(resumed[:5])
            self.assertTrue(np.array_equal(tokens, actual_tokens))
            for name, parameter in resumed[0].named_parameters():
                self.assertTrue(torch.equal(parameter, tensors[name]), name)
            self.assertEqual(actual["loss"], expected["loss"])
            self.assertEqual(actual["adjacent_cosine"], expected["adjacent_cosine"])
            self.assertEqual(actual["cumulative_ratio"], expected["cumulative_ratio"])
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            self.assertEqual(np.random.rand(), np_draw)
            self.assertEqual(random.random(), python_draw)
            self.assertEqual(resumed[3].updates, 6)
            self.assertEqual(resumed[4].updates, 6)
            self.assertEqual(resumed[2].state_dict(), scaler.state_dict())
            self.assertTrue(all(int(v["step"]) == 6 for v in resumed[1].state.values()))
            invalid = copy.deepcopy(config)
            invalid["resume"]["order_sha256"] = "different"
            with self.assertRaisesRegex(ValueError, "data order"):
                load_parent(invalid, parent, device)

    def test_validation_preserves_rng_and_training_mode(self):
        model = AutoModelForCausalLM.from_config(GPTNeoXConfig(vocab_size=32, hidden_size=8,
            intermediate_size=16, num_hidden_layers=1, num_attention_heads=2,
            max_position_embeddings=16, rotary_pct=.5)).float().train()
        config = {"data": {"sequence_length": 8}, "validation": {"batch_size": 2}}
        state = rng_state()
        validation = protected_evaluate(model, np.arange(26, dtype=np.int32) % 32, config, torch.device("cpu"))
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(state["torch_rng"], torch.get_rng_state()))
        self.assertEqual(state["python_rng"], random.getstate())
        self.assertTrue(np.array_equal(state["numpy_rng"][1], np.random.get_state()[1]))
        self.assertEqual(validation["supervised_tokens"], 22)
        self.assertTrue(validation["short_tail_included"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
