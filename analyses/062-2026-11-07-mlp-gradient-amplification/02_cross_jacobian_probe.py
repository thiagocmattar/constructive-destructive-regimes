"""Post hoc frozen-Jacobian / actual-direction crossing; no training trajectory."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT/'runs/237-2026-11-07-mlp-gradient-amplification'
PREVIOUS = ROOT/'runs/236-2026-11-07-mlp-adam-spike-internals'
sys.path.insert(0, str(RUN)); sys.path.insert(0, str(ROOT))
import diagnostics as d
from src.cifar_eos.core import make_model


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


@torch.no_grad()
def main(device):
    start = time.monotonic()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    sources = {}
    for item in json.loads((RUN/'input_manifest.json').read_text())['files']:
        path = ROOT/(item.get('local_path', item['path']) if device == 'cpu' else item['path'])
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
    worker = PREVIOUS/('artifacts/spark/payload/worker' if device == 'cpu' else 'artifacts/attempt-01/worker')
    cache = ROOT/'cache/cifar10-eos/data.pt'
    sources['cache/cifar10-eos/data.pt'] = sha(cache)
    x = torch.load(cache, map_location=device, weights_only=True)['x_train']
    model = make_model().to(device)
    states = {}; directions = {}
    for state in [465, 473, 492, 502, 512, 521]:
        path = worker/f'snapshots/state{state}.pt'
        sources[f'state{state}.pt'] = sha(path)
        saved = torch.load(path, map_location=device, weights_only=False)
        states[state] = saved['model']
        model.load_state_dict(saved['model'])
        opt = torch.optim.Adam(model.parameters(), lr=.001, betas=(.9,.999), eps=1e-8, foreach=False, fused=False)
        opt.load_state_dict(saved['optimizer'])
        offset = 0
        for p in model.parameters():
            p.grad = saved['current_gradient'][offset:offset+p.numel()].reshape_as(p).clone()
            offset += p.numel()
        assert offset == saved['current_gradient'].numel()
        opt.step()
        directions[state] = {k:(p.detach().double()-states[state][k].double()).float() for k,p in model.named_parameters()}
    records = []
    for baseline, incoming, peak in [(465,473,474), (492,492,493), (502,502,503), (512,521,522)]:
        crosses = []
        for jstate in sorted(set([baseline,incoming])):
            model.load_state_dict(states[jstate])
            for dstate in sorted(set([baseline,incoming])):
                delta = directions[dstate]
                energy = 0.
                for left in range(0,len(x),1000):
                    a = d.forward(model,x[left:left+1000])
                    df,_ = d.directional(model,a,delta)
                    energy += float(df.double().square().sum())/len(x)
                step_energy = sum(float(v.double().square().sum()) for v in delta.values())
                crosses.append(dict(jacobian_state=jstate,direction_state=dstate,
                    step_norm=step_energy**.5,mean_tangent_energy=energy,directional_gain=(energy/step_energy)**.5))
        records.append(dict(baseline=baseline,incoming=incoming,peak=peak,crosses=crosses))
    result = dict(status='completed',device=device,examples=len(x),events=records,
        source_sha256=sources,generator_sha256=sha(Path(__file__)),diagnostics_sha256=sha(RUN/'diagnostics.py'),
        additional_training_updates=0,elapsed_seconds=time.monotonic()-start)
    path = HERE/('cross_jacobian_cpu.json' if device == 'cpu' else 'cross_jacobian_probe.json')
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',default='cuda:0',choices=['cuda:0','cpu'])
    main(parser.parse_args().device)
