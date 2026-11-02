"""Independent arithmetic, optimizer placement, accumulation, and resume checks."""
import importlib.util
import io
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from diagnostics import Diagnostics, projected_change
from src.cifar_eos.core import flat, make_model
from src.eos_tasks.production import ProductionTask

spec = importlib.util.spec_from_file_location('screening_classifier', HERE/'02_classify.py')
classifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(classifier)


def optimizer_for(parameters):
    return torch.optim.Adam(parameters, lr=.003, betas=(.9, .999), eps=1e-8, foreach=False, fused=False)


def test_projected_change_is_mixed_curvature_on_quadratic():
    hessian = torch.tensor([[4., 1.], [1., 2.]], dtype=torch.float64)
    point = torch.tensor([2., -1.], dtype=torch.float64)
    gradient = hessian@point
    delta = torch.tensor([-.003, .002], dtype=torch.float64)
    future = hessian@(point+delta)
    observed = projected_change(gradient, future)
    independent = -float(gradient@hessian@delta)/float(gradient@gradient)
    assert observed == pytest.approx(independent, abs=2e-16)
    # A response along a non-gradient step is not the raw-gradient Rayleigh quotient.
    assert observed/.003 != pytest.approx(float(gradient@hessian@gradient)/float(gradient@gradient))
    assert projected_change(None, future) is None
    assert projected_change(torch.zeros(2), future) is None


def test_all_stream_prefixes_match_numpy_and_actual_motion():
    parameters = [torch.nn.Parameter(torch.tensor([2., -3.])), torch.nn.Parameter(torch.tensor([.1]))]
    optimizer = optimizer_for(parameters)
    diagnostics = Diagnostics(3, 'cpu')
    histories = {name: [] for name in diagnostics.streams}
    initial = flat(parameters).detach().double().numpy().copy()
    previous_moment = None
    for step, values in enumerate([[1., -2., .3], [-2., 1., 0.], [0., 0., 0.], [3., -1., -.5]], 1):
        gradient = torch.tensor(values)
        parameters[0].grad = gradient[:2].clone(); parameters[1].grad = gradient[2:].clone()
        before = flat(parameters).detach().double()
        optimizer.step()
        after = flat(parameters).detach().double()
        row = diagnostics.consume(gradient, optimizer, step, before, after)
        moment = np.concatenate([optimizer.state[p]['exp_avg'].numpy() for p in parameters]).astype(np.float64)
        second = np.concatenate([optimizer.state[p]['exp_avg_sq'].numpy() for p in parameters]).astype(np.float64)
        direction = (moment/(1-.9**step))/(np.sqrt(second/(1-.999**step))+1e-8)
        vectors = dict(raw=gradient.numpy().astype(np.float64), moment=moment, adaptive=direction,
            actual=after.numpy()-before.numpy())
        for name, vector in vectors.items():
            histories[name].append(vector.copy())
            array = np.asarray(histories[name])
            norms = np.linalg.norm(array, axis=1)
            stats = row['streams'][name]
            assert stats['A'] == pytest.approx(norms.sum(), abs=1e-13)
            assert stats['B'] == pytest.approx(np.linalg.norm(array.sum(axis=0)), abs=1e-13)
            assert stats['energy'] == pytest.approx(np.sum(array**2), abs=1e-12)
            if step > 1 and norms[-1]*norms[-2] > 0:
                expected_cosine = np.dot(array[-1], array[-2])/(norms[-1]*norms[-2])
                assert stats['cosine'] == pytest.approx(expected_cosine, abs=2e-15)
            elif step == 1 or norms[-1]*norms[-2] == 0:
                assert stats['cosine'] is None
        assert np.allclose(diagnostics.streams['actual'].total.numpy(), after.numpy()-initial, atol=1e-16)
        assert row['unbiased_moment_norm'] == pytest.approx(np.linalg.norm(moment/(1-.9**step)))
        if previous_moment is None:
            assert row['incoming_gradient_prior_moment_cosine'] is None
        previous_moment = moment.copy()
        gradient.zero_()
        for p in parameters:
            p.grad.zero_()
        assert np.array_equal(diagnostics.streams['raw'].previous.numpy(), histories['raw'][-1])


def test_zero_first_step_is_undefined_and_bias_correction_is_explicit():
    parameter = torch.nn.Parameter(torch.tensor([1., -1.]))
    optimizer = optimizer_for([parameter]); diagnostics = Diagnostics(2, 'cpu')
    parameter.grad = torch.zeros(2); before = parameter.detach().double().clone()
    optimizer.step(); row = diagnostics.consume(torch.zeros(2), optimizer, 1, before, parameter.detach().double())
    assert row['moment_over_gradient_norm'] is None
    assert row['projected_gradient_change'] is None
    assert all(stream['B_over_A'] is None for stream in row['streams'].values())
    parameter.grad = torch.ones(2); before = parameter.detach().double().clone()
    optimizer.step(); row = diagnostics.consume(torch.ones(2), optimizer, 2, before, parameter.detach().double())
    assert row['moment_over_gradient_norm'] == pytest.approx(.1, rel=1e-6)
    assert row['unbiased_moment_over_gradient_norm'] == pytest.approx(.1/(1-.9**2), rel=1e-6)
    assert row['incoming_gradient_prior_moment_cosine'] is None
    assert row['projected_gradient_change'] is None


def test_optimizer_and_diagnostic_resume_matches_uninterrupted():
    gradients = [torch.tensor([1., -2.]), torch.tensor([-.4, .3]), torch.zeros(2), torch.tensor([.8, .2])]
    def execute(parameter, optimizer, diagnostics, start, end):
        rows = []
        for index in range(start, end):
            parameter.grad = gradients[index].clone(); before = parameter.detach().double().clone()
            optimizer.step()
            rows.append(diagnostics.consume(gradients[index], optimizer, index+1, before, parameter.detach().double()))
        return rows
    parameter = torch.nn.Parameter(torch.tensor([2., -3.]))
    optimizer = optimizer_for([parameter]); diagnostics = Diagnostics(2, 'cpu')
    execute(parameter, optimizer, diagnostics, 0, 2)
    buffer = io.BytesIO()
    torch.save(dict(parameter=parameter.detach().clone(), optimizer=optimizer.state_dict(), diagnostics=diagnostics.state_dict()), buffer)
    expected = execute(parameter, optimizer, diagnostics, 2, 4)
    buffer.seek(0); saved = torch.load(buffer, weights_only=False)
    resumed = torch.nn.Parameter(saved['parameter']); other = optimizer_for([resumed]); other.load_state_dict(saved['optimizer'])
    restored = Diagnostics(2, 'cpu'); restored.load_state_dict(saved['diagnostics'])
    actual = execute(resumed, other, restored, 2, 4)
    assert torch.equal(parameter, resumed)
    assert expected == actual


def test_full_objective_accumulation_uses_all_examples_and_preserves_gradient():
    torch.manual_seed(17)
    model = make_model(input_size=3, widths=(4,), classes=10)
    x = torch.randn(11, 3); y = torch.arange(11) % 10
    task = ProductionTask('mlp', model, dict(x_train=x, y_train=y), 4)
    loss, accumulated = task.gradient()
    target = torch.nn.functional.one_hot(y, 10).float()
    model.zero_grad(); direct = .5*(model(x)-target).square().sum()/11
    direct.backward(); reference = flat([p.grad for p in model.parameters()]).detach()
    assert loss == pytest.approx(float(direct.detach()), rel=2e-7)
    assert torch.allclose(accumulated, reference, atol=6e-8, rtol=2e-5)
    before = [p.grad.clone() for p in model.parameters()]
    task.evaluate('train')
    assert all(torch.equal(p.grad, old) for p, old in zip(model.parameters(), before))


@pytest.mark.parametrize('tail,label', [([.5]*64, 'smooth_progress'),
    ([.49, .51]*32, 'oscillatory_progress'), ([1.2]*64, 'unclassified')])
def test_loss_regimes_ignore_startup_transient(tail, label):
    assert classifier.classify([1., 4., 2.]+tail)['label'] == label
    assert classifier.classify([1.]+tail, status='smoke_complete')['label'] == 'incomplete_or_failed'


def test_regime_selection_reports_missing_regimes():
    assert not classifier.select_regimes([])['spans_requested_regimes']
    rows = [dict(lr=.0001, label='smooth_progress', status='complete'),
        dict(lr=.0003, label='unclassified', status='complete'),
        dict(lr=.001, label='oscillatory_progress', status='complete')]
    assert classifier.select_regimes(rows) == dict(smooth_lr=.0001, intermediate_lr=.0003,
        oscillatory_lr=.001, spans_requested_regimes=True)


def test_classifier_keeps_failed_arm_without_boundary_file(tmp_path, monkeypatch):
    import json
    arm = tmp_path/'failed'; arm.mkdir()
    (arm/'manifest.json').write_text(json.dumps(dict(status='failed', task='mlp', lr=.001)))
    output = tmp_path/'classification.json'
    monkeypatch.setattr(sys, 'argv', ['02_classify.py', str(arm), '--output', str(output)])
    classifier.main()
    result = json.loads(output.read_text())
    assert result['arms'][0]['label'] == 'incomplete_or_failed'
    assert not result['cohort_complete']
    assert result['candidates'] == {}


def test_classifier_rejects_tampered_boundary_artifact(tmp_path, monkeypatch):
    import json
    arm = tmp_path/'arm'; arm.mkdir()
    (arm/'boundaries.jsonl').write_text('{"state":0,"loss":1.0}\n')
    (arm/'manifest.json').write_text(json.dumps(dict(status='complete', task='mlp', lr=.001,
        effective_updates=256, files={'boundaries.jsonl': dict(bytes=1, sha256='wrong')})))
    monkeypatch.setattr(sys, 'argv', ['02_classify.py', str(arm), '--output', str(tmp_path/'result.json')])
    with pytest.raises(ValueError, match='integrity'):
        classifier.main()
