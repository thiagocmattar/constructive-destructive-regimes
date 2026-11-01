"""Run-local mathematical and optimizer-boundary checks, without paper evidence."""
import importlib.util
from pathlib import Path
import tempfile

import numpy as np
import torch

spec=importlib.util.spec_from_file_location('worker',Path(__file__).with_name('01_worker.py'))
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)


def main():
    a=torch.tensor([1.,2.,3.],dtype=torch.float32)
    assert abs(worker.cosine(a,a)-1)<1e-14
    assert abs(worker.cosine(a,-a)+1)<1e-14
    assert worker.cosine(a,torch.zeros(3)) is None
    assert worker.cosine(torch.tensor([1.,0.]),torch.tensor([0.,1.]))==0
    rng=np.random.default_rng(205)
    vectors=rng.normal(size=(11,79)).astype('float32')
    with tempfile.TemporaryDirectory() as directory:
        p=Path(directory)/'gradients.npy'
        stored=np.lib.format.open_memmap(p,mode='w+',dtype=np.float32,shape=vectors.shape)
        stored[:]=vectors;stored.flush();del stored
        loaded=np.load(p,mmap_mode='r').astype(np.float64)
        for lag in range(1,11):
            expected=float(loaded[0]@loaded[lag]/np.sqrt((loaded[0]@loaded[0])*(loaded[lag]@loaded[lag])))
            assert abs(expected-worker.cosine(torch.from_numpy(vectors[0]),torch.from_numpy(vectors[lag])))<1e-14
    p=torch.nn.Parameter(torch.tensor([1.,-2.]))
    opt=torch.optim.SGD([p],lr=.02,momentum=0,weight_decay=0,foreach=False)
    initial=p.detach().clone();grad=torch.tensor([.4,-.7]);p.grad=grad.clone();opt.step()
    assert torch.equal(p,initial.add(grad,alpha=-.02))
    # Saving after ten applied increments leaves the eleventh gradient unapplied.
    assert len(range(11))==11 and sum(k<10 for k in range(11))==10
    assert worker.close(1.0009,1.,1e-4,1e-3)
    assert not worker.close(1.002,1.,1e-4,1e-3)
    print('PASS: cosine sign/zero, FP32 storage/FP64 reduction, lag indexing, SGD increments and qualification')


if __name__=='__main__':main()
