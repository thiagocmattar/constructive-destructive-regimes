"""Descent identities, isolated interventions, probe placement and exact recovery."""
import copy
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, GPTNeoXConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baseline.causal_training import restore_state, save_state
from baseline.continuation import rng_state, restore_rng
from baseline.descent_diagnostics import (DisplacementMetrics, adam_components, apply_direction,
    descent_accounting, direction_trials, fixed_probe, vector_norm)
from baseline.gradient_geometry import GradientGeometry
from baseline.recipe import DynamicLossScaler, build_recipe_adamw
from baseline.training import optimizer_boundary


@pytest.fixture
def tiny():
    torch.set_num_threads(2)
    torch.manual_seed(43)
    arch = GPTNeoXConfig(vocab_size=32, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=16,
        hidden_dropout=.1, attention_dropout=.1, rotary_pct=.5, use_cache=False)
    arch._attn_implementation = "sdpa"
    model = AutoModelForCausalLM.from_config(arch).float().train()
    training = dict(optimizer="adamw", peak_learning_rate=.01, adamw_betas=[.9, .95],
        adamw_eps=1e-8, weight_decay=.1, gradient_clip_norm=1., micro_batch_size=2,
        maximum_overflow_retries=2)
    config = dict(training=training, data=dict(sequence_length=8), causal=dict(probe_microbatch=1))
    optimizer, _ = build_recipe_adamw(model, training, torch=torch)
    geometries = (GradientGeometry(model.named_parameters()),
        GradientGeometry(model.named_parameters(), previous_dtype=torch.float64),
        GradientGeometry(model.named_parameters()))
    tokens = np.random.default_rng(19).integers(0, 32, 64, dtype=np.int32)
    return model, optimizer, geometries, DynamicLossScaler(), config, tokens


def boundary(tiny, displacement):
    model, optimizer, geometries, scaler, config, tokens = tiny
    before = [p.detach().clone() for p in model.parameters()]
    result = optimizer_boundary(model, optimizer, scaler, geometries[0], geometries[1],
        tokens, np.arange(4), config, torch.device("cpu"), momentum_geometry=geometries[2])
    displacement.consume(before, list(model.parameters()))
    return before, result


def test_quadratic_accounting_uses_full_displacement():
    theta = torch.tensor([.7, -.3], dtype=torch.float64)
    h = torch.tensor([[3., .5], [.5, 2.]], dtype=torch.float64)
    step = torch.tensor([-.04, .02], dtype=torch.float64)
    objective = lambda x: float(.5 * x @ h @ x)
    r = descent_accounting(objective(theta), objective(theta + step), [h @ theta], [theta], [theta + step])
    assert r["first_order_decrease"] == pytest.approx(float(-(h @ theta) @ step))
    assert r["nonlinear_residual"] == pytest.approx(float(.5 * step @ h @ step), abs=1e-14)
    assert r["actual_decrease"] == pytest.approx(r["first_order_decrease"] - r["nonlinear_residual"])


def test_direction_components_and_matching_preserve_decay_groups():
    p = torch.nn.Parameter(torch.tensor([.2, -.3]))
    q = torch.nn.Parameter(torch.tensor([.7]))
    optimizer = torch.optim.AdamW([dict(params=[p], weight_decay=.1), dict(params=[q], weight_decay=0)],
                                 lr=.02, betas=(.9, .95), eps=.01)
    named = [("p", p), ("q", q)]
    p.grad, q.grad = torch.tensor([.4, -.2]), torch.tensor([.1])
    before = [x.detach().clone() for _, x in named]
    optimizer.step()
    states = copy.deepcopy(optimizer.state_dict())
    target = .02 * vector_norm(adam_components(optimizer, named, 1, "adam"))
    for kind in ("adam", "momentum", "current_preconditioned", "gradient"):
        vectors = adam_components(optimizer, named, 1, kind)
        if kind in {"momentum", "gradient"}:
            for (_, parameter), vector in zip(named, vectors):
                torch.testing.assert_close(vector, parameter.grad.double(), rtol=1e-6, atol=1e-8)
        record = apply_direction(optimizer, named, before, vectors, target)
        assert record["nominal_adaptive_step_norm"] == pytest.approx(target, rel=1e-12)
        for i, ((_, parameter), vector) in enumerate(zip(named, vectors)):
            decay = .1 if i == 0 else 0
            expected = (before[i].double() * (1 - .02 * decay) - record["coefficient"] * vector).float()
            torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
    for key, state in optimizer.state_dict()["state"].items():
        for field, value in state.items():
            torch.testing.assert_close(value, states["state"][key][field], rtol=0, atol=0)
    with pytest.raises(ValueError, match="zero direction"):
        apply_direction(optimizer, named, before, [torch.zeros_like(p), torch.zeros_like(q)], target)


def test_probe_token_weights_tail_and_protection(tiny):
    model, _, _, _, config, tokens = tiny
    tokens = tokens[:19]
    indices = np.arange(3)
    for p in model.parameters():
        p.grad = torch.full_like(p, .03125)
    old_gradients = [p.grad for p in model.parameters()]
    rng = torch.get_rng_state().clone()
    loss, gradients = fixed_probe(model, tokens, indices, 8, 2, torch.device("cpu"), gradient=True)
    assert model.training and torch.equal(torch.get_rng_state(), rng)
    assert all(p.grad is g for p, g in zip(model.parameters(), old_gradients))
    model.eval()
    pieces = [tokens[:8], tokens[8:16], tokens[16:]]
    with torch.no_grad():
        expected = sum(float(model(input_ids=torch.tensor(p).long()[None], labels=torch.tensor(p).long()[None]).loss)
                       * (len(p) - 1) for p in pieces) / 16
    assert loss == pytest.approx(expected, rel=2e-7)
    assert all(torch.isfinite(g).all() for g in gradients)


def test_trials_preserve_accepted_step_rng_gradients_and_moments(tiny):
    model, optimizer, _, _, config, tokens = tiny
    before_loss, probe_gradient = fixed_probe(model, tokens, np.arange(4, 8), 8, 1,
                                              torch.device("cpu"), gradient=True)
    before, _ = boundary(tiny, DisplacementMetrics())
    accepted = [p.detach().clone() for p in model.parameters()]
    gradients = [p.grad for p in model.parameters()]
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    rng = torch.get_rng_state().clone()
    rows = direction_trials(model, optimizer, before, probe_gradient, before_loss,
        tokens, np.arange(4, 8), config, torch.device("cpu"), 1)
    assert len(rows) == 7
    assert torch.equal(torch.get_rng_state(), rng) and model.training
    for p, value, g in zip(model.parameters(), accepted, gradients):
        torch.testing.assert_close(p, value, rtol=0, atol=0)
        assert p.grad is g
    for key, state in optimizer.state_dict()["state"].items():
        for field, value in state.items():
            torch.testing.assert_close(value, optimizer_before["state"][key][field], rtol=0, atol=0)
    assert abs(next(r for r in rows if r["candidate"] == "adam")["native_reconstruction_loss_gap"]) < 1e-5


def test_recovery_restores_all_clocks_and_next_stochastic_update(tiny):
    model, optimizer, geometries, scaler, config, tokens = tiny
    displacement = DisplacementMetrics()
    boundary(tiny, displacement)
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "recovery.pt"
        save_state(path, model, optimizer, scaler, geometries, displacement, 1, "order", "config", Path(temp), 4)
        _, expected = boundary(tiny, displacement)
        expected_weights = [p.detach().clone() for p in model.parameters()]
        update, _ = restore_state(path, model, optimizer, scaler, geometries, displacement,
                                  torch.device("cpu"), "order", "config", 4)
        assert update == 1
        _, actual = boundary(tiny, displacement)
        assert actual["loss"] == expected["loss"]
        for p, value in zip(model.parameters(), expected_weights):
            torch.testing.assert_close(p, value, rtol=0, atol=0)
        assert all(g.updates == 2 for g in geometries)
        with pytest.raises(ValueError, match="identity"):
            restore_state(path, model, optimizer, scaler, geometries, displacement,
                          torch.device("cpu"), "other-order", "config")
        with pytest.raises(ValueError, match="data position"):
            restore_state(path, model, optimizer, scaler, geometries, displacement,
                          torch.device("cpu"), "order", "config", 3)


def test_complete_runner_baseline_and_fork_verify(tiny, tmp_path, monkeypatch):
    from baseline import causal_training as runner
    from baseline.data_protocol import block_order, sha256
    template, _, _, _, config, tokens = tiny
    template.config.hidden_dropout = template.config.attention_dropout = 0
    # Keep the small stochastic modules: this smoke also exercises RNG isolation.
    monkeypatch.setattr(runner, "build_model", lambda *args: (copy.deepcopy(template), {"smoke": True}))
    config = copy.deepcopy(config)
    config["run_id"] = "toy"
    config["model"] = {"initial_parameter_sha256": runner.parameter_sha(template)}
    config["training"].update(effective_batch_sequences=2, minimum_learning_rate=.001,
        warmup_fraction=.1, initial_loss_scale=4096., loss_scale_growth_interval=1000,
        loss_scale_hysteresis=2, minimum_loss_scale=1.)
    cache = tmp_path / "input"
    for name, values in (("train", tokens), ("validation", tokens[:19])):
        directory = cache / name
        directory.mkdir(parents=True)
        values.tofile(directory / "tokens.int32.bin")
        metadata = dict(dataset_revision="test-data", tokenizer_revision="test-tokenizer",
            block_size=8, dtype="int32", append_eos=True, tokens=len(values))
        (directory / "metadata.json").write_text(json.dumps(metadata))
        config["data"][name + "_sha256"] = sha256(directory / "tokens.int32.bin")
    config["data"].update(revision="test-data", tokenizer_revision="test-tokenizer")
    config["validation"] = {"batch_size": 2}
    order = block_order(len(tokens), 8, 1234)
    probe = tmp_path / "probe.json"
    probe.write_text(json.dumps(dict(indices=order[4:8].tolist())))
    config["seeds"] = {"model": 1234, "data_order": 1234}
    config["causal"].update(condition="baseline", baseline_updates=2, schedule_updates=5,
        probe_path=str(probe), probe_sha256=sha256(probe),
        order_sha256=hashlib.sha256(order.astype("<i8").tobytes()).hexdigest(),
        measurement_updates=[1, 2], junctions=[1], branch_updates=1, recovery_every_updates=1)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    def run(path, output):
        runner.execute(argparse.Namespace(config=str(path), cache=str(cache), output=str(output),
            device="cpu", calibrate_updates=0, recover_from=None, deadline_epoch=None))
    baseline = tmp_path / "baseline"
    run(path, baseline)
    verify_path = Path(__file__).resolve().parents[1] / "analyses/025-2026-10-17-pythia-descent-mechanism/03_verify.py"
    spec = importlib.util.spec_from_file_location("verify_descent", verify_path)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    accepted = verifier.verify(baseline)
    (baseline / "verification.json").write_text(json.dumps(accepted))
    branch_config = copy.deepcopy(config)
    branch_config["run_id"] = "toy-branch"
    branch_config["causal"].update(condition="norm_matched_no_preconditioning", junction_update=1,
        parent_config=str(path), parent_config_sha256=sha256(path), parent_output=str(baseline))
    branch_path = tmp_path / "branch.json"
    branch_path.write_text(json.dumps(branch_config))
    branch = tmp_path / "branch"
    run(branch_path, branch)
    checked = verifier.verify(branch)
    assert checked["matched_parent_adaptive_step_schedule"] and checked["accepted_updates"] == 1
