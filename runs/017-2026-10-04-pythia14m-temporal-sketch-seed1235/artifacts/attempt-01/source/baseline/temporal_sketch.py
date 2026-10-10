"""Fixed independent CountSketch maps for temporal raw-gradient inner products.

One signed bucket per input coordinate; each map has E[R.T @ R] = I.
This calibrated sketch is not claimed to meet a worst-case sparse-JL bound.
Projection RNG is isolated from every training RNG. No gradient is modified.
"""
from pathlib import Path
import json
import time
import numpy as np
import torch


class FixedSketch:
    def __init__(self, shapes, device, dimension, seeds):
        self.dimension = dimension
        self.maps = []
        for seed in seeds:
            rng = np.random.Generator(np.random.PCG64(seed))
            row = []
            for shape in shapes:
                n = int(np.prod(shape))
                buckets = torch.from_numpy(rng.integers(0, dimension, n, dtype=np.int64)).to(device)
                signs = torch.from_numpy(rng.integers(0, 2, n, dtype=np.int8) * 2 - 1).to(device)
                row.append((buckets, signs))
            self.maps.append(row)

    @torch.no_grad()
    def project(self, vectors):
        vectors = list(vectors)
        rows = []
        for mapping in self.maps:
            out = torch.zeros(self.dimension, dtype=torch.float32, device=vectors[0].device)
            for vector, (buckets, signs) in zip(vectors, mapping):
                out.scatter_add_(0, buckets, vector.detach().reshape(-1).float() * signs)
            rows.append(out)
        return torch.stack(rows)


class TemporalRecorder:
    def __init__(self, parameters, config, output):
        self.parameters = list(parameters)
        self.config = config
        self.output = Path(output)
        self.output.mkdir(exist_ok=True)
        self.spec = config['temporal_sketch']
        self.count = 0
        self.norms = []
        self.projection_seconds = []
        self.anchors = {}
        self.window_cumulative = None
        self.window_energy = self.window_lag_one = 0.
        self.completed_windows = []
        self.projector = FixedSketch([p.shape for _, p in self.parameters], self.parameters[0][1].device,
                                    self.spec['dimension'], self.spec['seeds'])
        self.history = np.lib.format.open_memmap(self.output / 'raw_sketches.npy', mode='w+',
            dtype='<f4', shape=(712, len(self.spec['seeds']), self.spec['dimension']))
        rng = np.random.Generator(np.random.PCG64(self.spec['anchor_seed']))
        cadence = self.spec['anchor_block_updates']
        self.anchor_updates = set(1 + start + int(rng.integers(0, min(cadence, 712-start)))
                                  for start in range(0, 712, cadence))
        self.meta = {'family': 'CountSketch', 'nonzeros_per_column': 1,
            'map_normalization': 'signed unit columns; independent maps averaged after dot products',
            'fixed_across_updates': True, 'training_rng_untouched': True,
            'dimension': self.spec['dimension'], 'seeds': self.spec['seeds'],
            'scope': [{'name': n, 'shape': list(p.shape)} for n,p in self.parameters],
            'anchor_updates': sorted(self.anchor_updates), 'exact_lags': self.spec['exact_lags'],
            'history_dtype': 'float32', 'offline_reduction_dtype': 'float64',
            'capture': 'unscaled accumulated raw gradient before clipping and AdamW',
            'planned_updates': 712}
        (self.output/'sketch_metadata.json').write_text(json.dumps(self.meta, indent=2)+'\n')

    def pair(self, source, target, lag, dot, source_norm, target_norm):
        denominator = source_norm * target_norm
        left = np.asarray(self.history[source-1], dtype=np.float64)
        right = np.asarray(self.history[target-1], dtype=np.float64)
        projected = np.sum(left * right, axis=1)
        event = {'source_update': source, 'target_update': target, 'lag': lag,
            'exact_dot': float(dot), 'source_norm': source_norm, 'target_norm': target_norm,
            'exact_cosine': float(dot/denominator) if denominator else None,
            'projected_dots': projected.tolist(),
            'projected_cosines': (projected/denominator).tolist() if denominator else None}
        with (self.output/'exact_pairs.jsonl').open('a') as stream:
            stream.write(json.dumps(event, allow_nan=False)+'\n')

    @torch.no_grad()
    def consume(self, metrics):
        start = time.perf_counter()
        vectors = [p.grad.detach() for _,p in self.parameters]
        t = self.count + 1
        sketch = self.projector.project(vectors).cpu().numpy()
        self.history[t-1] = sketch
        self.history.flush()
        norm = metrics['gradient_norm']
        for name,(first,last) in self.spec['windows'].items():
            if first <= t <= last:
                if t == first:
                    self.window_cumulative = [torch.zeros_like(v,dtype=torch.float64) for v in vectors]
                    self.window_energy = self.window_lag_one = 0.
                for total,vector in zip(self.window_cumulative,vectors): total.add_(vector)
                self.window_energy += norm**2
                if t > first:
                    self.window_lag_one += (metrics['adjacent_cosine'] or 0.) * norm * self.norms[-1]
                if t == last:
                    b2 = float(torch.stack([v.square().sum() for v in self.window_cumulative]).sum().item())
                    record = {'window':name,'first_update':first,'last_update':last,
                        'energy':self.window_energy,'cumulative_squared_norm':b2,
                        'all_pair_interaction':(b2-self.window_energy)/2,
                        'lag_one_interaction':self.window_lag_one,
                        'beyond_lag_one_interaction':(b2-self.window_energy)/2-self.window_lag_one}
                    self.completed_windows.append(record)
                    with (self.output/'exact_windows.jsonl').open('a') as stream:
                        stream.write(json.dumps(record,allow_nan=False)+'\n')
                    self.window_cumulative = None
        if t > 1:
            dot = (metrics['adjacent_cosine'] or 0.) * norm * self.norms[-1]
            self.pair(t-1, t, 1, dot, self.norms[-1], norm)
        for s,(old, old_norm) in list(self.anchors.items()):
            lag = t-s
            if lag in self.spec['exact_lags']:
                dot = float(torch.stack([torch.sum(a.double()*b.double())
                    for a,b in zip(old,vectors)]).sum().item())
                self.pair(s,t,lag,dot,old_norm,norm)
            if lag >= max(self.spec['exact_lags']):
                del self.anchors[s]
        if t in self.anchor_updates:
            self.anchors[t] = ([v.clone() for v in vectors], norm)
        self.norms.append(norm)
        self.count = t
        elapsed = time.perf_counter()-start
        self.projection_seconds.append(elapsed)
        progress = {'completed_updates': t, 'planned_updates': 712,
            'resident_exact_anchors': len(self.anchors), 'capture_seconds': elapsed}
        temp = self.output/'sketch_progress.json.tmp'
        temp.write_text(json.dumps(progress)+'\n')
        temp.replace(self.output/'sketch_progress.json')
        return {'temporal_capture_seconds': elapsed, 'temporal_completed_updates': t}

    def state_dict(self):
        return {'completed_updates': self.count, 'norms': self.norms,
            'metadata': self.meta, 'history_file': 'temporal/raw_sketches.npy',
            'window_cumulative': None if self.window_cumulative is None else
                [v.detach().cpu().clone() for v in self.window_cumulative],
            'window_energy': self.window_energy, 'window_lag_one': self.window_lag_one,
            'completed_windows': self.completed_windows,
            'anchors': {s: ([v.detach().cpu().clone() for v in vectors], norm)
                        for s,(vectors,norm) in self.anchors.items()},
            'anchor_rng': 'fixed preselected positions recorded in metadata'}


def calibration_report(output, completed):
    output = Path(output)
    pairs = [json.loads(x) for x in (output/'exact_pairs.jsonl').read_text().splitlines()]
    defined = [p for p in pairs if p['exact_cosine'] is not None]
    errors = np.array([np.mean(p['projected_cosines'])-p['exact_cosine'] for p in defined])
    return {'completed_updates': completed, 'exact_pairs': len(pairs),
        'mean_sketch_cosine_rmse': float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
        'mean_sketch_cosine_max_abs_error': float(np.max(np.abs(errors))) if len(errors) else None,
        'qualified': bool(len(errors) and np.sqrt(np.mean(errors**2)) < .02 and np.max(np.abs(errors)) < .06)}
