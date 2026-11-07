"""Capture native module boundaries without altering forward computations."""
import numpy as np
import torch


class Capture:
    def __init__(self, model, case, emit, mask=None):
        self.model, self.case, self.emit, self.mask = model, case, emit, mask
        self.handles = []
        self.count = {}
        self.order = []

    def save(self, name, value):
        if isinstance(value, tuple): value = value[0]
        if self.case == 'transformer' and value.ndim == 3:
            value = value.transpose(0, 1)[self.mask]
        # Snapshot before the next in-place module can mutate this tensor.
        a = value.detach().float().cpu().numpy().copy()
        self.order.append(name)
        self.emit(name, a)

    def pre(self, name):
        def hook(module, inputs):
            k = self.count.get(name, 0); self.count[name] = k + 1
            if isinstance(module, (torch.nn.ReLU, torch.nn.LayerNorm)) or name.endswith('linear2'):
                self.save(name + f'.call{k}.input', inputs[0])
        return hook

    def post(self, name):
        def hook(module, inputs, output):
            k = self.count[name] - 1
            self.save(name + f'.call{k}.output', output)
        return hook

    def __enter__(self):
        for name, m in self.model.named_modules():
            if not name: continue
            if self.case == 'resnet':
                selected = isinstance(m, (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU,
                                          torch.nn.AvgPool2d, torch.nn.AdaptiveAvgPool2d, torch.nn.Linear))
            else:
                selected = name in ['encoder', 'pos_encoder', 'decoder'] or isinstance(m, (torch.nn.MultiheadAttention, torch.nn.LayerNorm)) or name.endswith(('linear1', 'linear2'))
            if selected:
                self.handles += [m.register_forward_pre_hook(self.pre(name)), m.register_forward_hook(self.post(name))]
        return self

    def __exit__(self, *args):
        for h in self.handles: h.remove()


def gram_reduce(path, raw_gradient=False):
    """All coordinates, float64, no projection or subsampling; direct audits."""
    a = np.load(path, mmap_mode='r')
    flat = a.reshape(len(a), -1)
    n = len(a) - (0 if raw_gradient else 1)
    gram = np.zeros((n, n)); levels = np.zeros(len(a)); audit = np.zeros(n)
    adjacent = np.zeros(n-1)
    for j in range(0, flat.shape[1], 65536):
        block = np.array(flat[:, j:j+65536], dtype=np.float64)
        levels += np.einsum('ij,ij->i', block, block)
        delta = block if raw_gradient else np.diff(block, axis=0)
        gram += delta @ delta.T
        audit += np.einsum('ij,ij->i', delta, delta)
        adjacent += np.einsum('ij,ij->i', delta[1:], delta[:-1])
    np.testing.assert_allclose(np.diag(gram), audit, rtol=1e-11, atol=1e-8)
    np.testing.assert_allclose(np.diag(gram, 1), adjacent, rtol=1e-10, atol=1e-8)
    return gram, levels
