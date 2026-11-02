"""Independent mathematical, sampling and recovery checks; CPU only."""
import importlib.util
import itertools
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('measurement', HERE/'01_measure.py')
measurement = importlib.util.module_from_spec(spec); spec.loader.exec_module(measurement)
from src.cifar_eos.core import Geometry, flat
from src.eos_tasks.task import Task


class TinyLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([[.2,-.1,.4],[.1,.3,-.2]], dtype=torch.float64))
    def forward(self, x):
        return self.logits[x].log_softmax(-1)


class MeasurementTests(unittest.TestCase):
    def test_geometry_against_direct_pair_matrix(self):
        v = np.random.default_rng(4).normal(size=(200,19))
        stream = Geometry(19, 'cpu')
        for row in v: result = stream.consume(torch.from_numpy(row))
        norm = np.linalg.norm(v, axis=1); gram=v@v.T
        self.assertAlmostEqual(result['B_over_A'], np.linalg.norm(v.sum(0))/norm.sum(), places=14)
        self.assertAlmostEqual(result['all_pair'], gram.sum()-np.trace(gram), places=10)
        self.assertAlmostEqual(result['lag_one'], 2*np.trace(gram, offset=1), places=11)

    def test_zero_and_cancellation(self):
        stream=Geometry(2,'cpu')
        self.assertIsNone(stream.consume(torch.zeros(2))['B_over_A'])
        self.assertIsNone(stream.consume(torch.ones(2))['cosine'])
        result=stream.consume(-torch.ones(2))
        self.assertAlmostEqual(result['cosine'], -1.)
        self.assertEqual(result['B_over_A'], 0.)

    def test_variable_token_unbiased_sampling(self):
        data={'x_train':torch.tensor([[0,1],[1,0],[0,0]]),
              'y_train':torch.tensor([[0,1],[2,-100],[1,2]])}
        task=Task('transformer',TinyLanguageModel(),data,2)
        full_loss, full_g=task.gradient()
        sampled=[task.gradient(torch.tensor(ids)) for ids in itertools.combinations(range(3),2)]
        np.testing.assert_allclose(np.mean([x[0] for x in sampled]),full_loss,atol=1e-14)
        torch.testing.assert_close(torch.stack([x[1] for x in sampled]).mean(0),full_g,rtol=1e-12,atol=1e-14)
        # Prove the random-token-denominator substitute changes the expectation.
        wrong=[]
        for ids in itertools.combinations(range(3),2):
            x=data['x_train'][list(ids)]; y=data['y_train'][list(ids)]
            loss,_=task.loss(x,y); wrong.append(float(loss.detach())/int((y!=-100).sum()))
        self.assertGreater(abs(np.mean(wrong)-full_loss),1e-4)

    def test_ghost_group_boundary_and_scale(self):
        data={'x_train':torch.arange(3000).reshape(3000,1),'y_train':torch.zeros(3000,dtype=torch.long)}
        task=Task('resnet',torch.nn.Identity(),data,17)
        chunks=list(task.chunks(torch.tensor([2,0])))
        self.assertEqual([len(x) for x,y in chunks],[1000,1000])
        self.assertEqual([int(x[0]) for x,y in chunks],[2000,0])
        self.assertEqual(task.scale(torch.tensor([2,0])),1/2000)

    def test_batch_prefix_and_repetition_independence(self):
        a=measurement.sampling_rng('resnet',0,2000,2,22)
        b=measurement.sampling_rng('resnet',0,2000,2,22)
        c=measurement.sampling_rng('resnet',0,2000,3,22)
        for _ in range(8):
            pa=a.permutation(50); pb=b.permutation(50); pc=c.permutation(50)
            np.testing.assert_array_equal(pa[:5],pb[:20][:5])
            self.assertEqual(len(set(pa[:20])),20)
            self.assertFalse(np.array_equal(pa,pc))

    def test_resume_preserves_rng_and_geometry(self):
        rng=measurement.sampling_rng('transformer',1,2000,4,2026)
        stream=Geometry(5,'cpu')
        for _ in range(13): stream.consume(torch.from_numpy(rng.normal(size=5)))
        state=stream.state_dict(); rng_state=rng.bit_generator.state
        expected=[]
        for _ in range(7): expected.append(stream.consume(torch.from_numpy(rng.normal(size=5))))
        restarted=Geometry(5,'cpu'); restarted.load_state_dict(state)
        other=np.random.default_rng(0); other.bit_generator.state=rng_state
        observed=[restarted.consume(torch.from_numpy(other.normal(size=5))) for _ in range(7)]
        self.assertEqual(expected,observed)
        torch.testing.assert_close(stream.total,restarted.total,rtol=0,atol=0)

    def test_actual_vs_nominal_updates(self):
        model=torch.nn.Sequential(torch.nn.Linear(2,3),torch.nn.Linear(3,1))
        origin=flat(model.parameters()).detach().double(); raw=Geometry(len(origin),'cpu'); actual=Geometry(len(origin),'cpu')
        rng=np.random.default_rng(2)
        for _ in range(200):
            g=torch.from_numpy(rng.normal(size=len(origin)).astype('float32'))
            before=flat(model.parameters()).detach().double(); raw.consume(g)
            measurement.step(model,g,.03)
            after=flat(model.parameters()).detach().double(); actual.consume(after-before)
        torch.testing.assert_close(actual.total,after-origin,rtol=0,atol=0)
        np.testing.assert_allclose((after-origin).numpy(),(-.03*raw.total).numpy(),atol=1e-6,rtol=1e-5)


if __name__=='__main__':
    unittest.main(verbosity=2)
