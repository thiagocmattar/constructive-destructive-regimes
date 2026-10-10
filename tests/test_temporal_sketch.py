import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from baseline.temporal_sketch import FixedSketch


def test_fixed_linear_map_and_rng_isolation():
    torch.manual_seed(19)
    np.random.seed(29)
    torch_state = torch.get_rng_state().clone()
    numpy_state = np.random.get_state()
    sketch = FixedSketch([(5,), (3,)], 'cpu', 128, [101,102])
    assert torch.equal(torch_state, torch.get_rng_state())
    assert np.array_equal(numpy_state[1], np.random.get_state()[1])
    a = [torch.arange(5,dtype=torch.float32), torch.tensor([3.,-2.,1.])]
    b = [torch.tensor([1.,-1.,2.,0.,3.]), torch.tensor([1.,2.,-3.])]
    torch.testing.assert_close(sketch.project(a),sketch.project(a),rtol=0,atol=0)
    torch.testing.assert_close(sketch.project([x+y for x,y in zip(a,b)]),
                               sketch.project(a)+sketch.project(b),rtol=0,atol=0)
    before = [x.clone() for x in a]
    sketch.project(a)
    assert all(torch.equal(x,y) for x,y in zip(a,before))


def test_signed_gram_and_collision_free_reference():
    sketch = FixedSketch([(4,)], 'cpu', 4, [1])
    sketch.maps = [[(torch.arange(4), torch.tensor([1,-1,1,-1],dtype=torch.int8))]]
    a = torch.tensor([1.,2.,-3.,4.])
    z = sketch.project([a])[0].double()
    torch.testing.assert_close(z@z, a.double()@a.double())
    torch.testing.assert_close(z@sketch.project([-a])[0].double(), -(a.double()@a.double()))
    gradients = torch.stack([a,-a,a,-a])
    history = torch.stack([sketch.project([x])[0] for x in gradients]).double()
    for lag in [1,2,3]:
        cos = (history[:-lag]*history[lag:]).sum(1)/(a.double().square().sum())
        torch.testing.assert_close(cos,torch.full_like(cos,(-1.)**lag))
    interaction = sum((history[:-lag]*history[lag:]).sum() for lag in [1,2,3])
    torch.testing.assert_close(2*interaction, history.sum(0).square().sum()-history.square().sum())
