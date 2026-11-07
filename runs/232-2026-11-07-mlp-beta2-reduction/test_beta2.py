"""CPU mathematics/placement/resume checks for native warm beta2 switching."""
import copy
import importlib.util
import json
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('beta_worker', HERE/'01_worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def main():
    cases = 0
    for beta in [.95, .9, .5]:
        p = torch.nn.Parameter(torch.tensor([1., -2., .5], dtype=torch.float64))
        opt = torch.optim.Adam([p], lr=.001, betas=(.9,.999), eps=1e-8, foreach=False, fused=False)
        opt.state[p] = dict(step=torch.tensor(512.), exp_avg=torch.tensor([.02,-.1,.04], dtype=p.dtype),
                            exp_avg_sq=torch.tensor([.001,.2,.003], dtype=p.dtype))
        old = copy.deepcopy(opt.state_dict())
        before = p.detach().clone()
        worker.change_beta2(opt, beta)
        worker.w.equal(old['state'], opt.state_dict()['state'])
        assert torch.equal(before, p)
        gradient = torch.tensor([.3,-.4,.1], dtype=p.dtype)
        p.grad = gradient.clone()
        expected_m = .9*old['state'][0]['exp_avg'] + .1*gradient
        expected_v = beta*old['state'][0]['exp_avg_sq'] + (1-beta)*gradient.square()
        expected_d = expected_m/(1-.9**513)/((expected_v/(1-beta**513)).sqrt()+1e-8)
        opt.step()
        torch.testing.assert_close(p, before-.001*expected_d, rtol=1e-13, atol=1e-14)
        m,v,d = worker.vectors(opt, 513)
        for actual, expected in [(m,expected_m),(v,expected_v),(d,expected_d)]:
            torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-14)
        assert int(opt.state[p]['step']) == 513
        assert abs((1-beta**513)-(1-.999**513)) > .5
        # Resume must retain changed betas, moments and the continuing global clock.
        q = torch.nn.Parameter(p.detach().clone())
        resumed = torch.optim.Adam([q], lr=.001, foreach=False, fused=False)
        resumed.load_state_dict(copy.deepcopy(opt.state_dict()))
        p.grad = gradient*.7
        q.grad = p.grad.clone()
        opt.step()
        resumed.step()
        assert torch.equal(p,q)
        worker.w.equal(opt.state_dict(),resumed.state_dict())
        cases += 1
    assert worker.cosine(torch.zeros(3),torch.ones(3)) is None
    print(json.dumps(dict(status='passed', beta_cases=cases,
        checks=['switch preserves tensors/clock', 'new-beta second-moment recurrence',
                'native global-clock bias correction', 'direction/parameter update formula',
                'exact optimizer resume', 'zero cosine denominator'])))


if __name__ == '__main__':
    main()
