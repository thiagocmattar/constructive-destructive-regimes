"""Validate the scalar reduction against explicit vector-pair products."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

path=Path(__file__).resolve().parents[1]/'runs/001-2026-10-03-pythia14m-adamw-gradient-baseline/05_temporal_reduction.py'
spec=importlib.util.spec_from_file_location('run001_temporal',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class TemporalReductionTests(unittest.TestCase):
    def check_vectors(self,vectors):
        vectors=[np.asarray(v,dtype=float) for v in vectors]
        rows=[]
        for i,g in enumerate(vectors):
            norm=np.linalg.norm(g)
            denominator=norm*np.linalg.norm(vectors[i-1]) if i else 0
            cosine=float(np.dot(g,vectors[i-1])/denominator) if denominator else None
            rows.append({'update':i+1,'gradient_norm':float(norm),'adjacent_cosine':cosine,
                         'cumulative_gradient_norm':float(np.linalg.norm(np.sum(vectors[:i+1],axis=0)))})
        reduced=module.reduce_prefixes(rows)
        for i,row in enumerate(reduced):
            pair_sum=2*sum(np.dot(vectors[a],vectors[b]) for a in range(i+1) for b in range(a))
            adjacent=2*sum(np.dot(vectors[a],vectors[a-1]) for a in range(1,i+1))
            self.assertAlmostEqual(row['all_cross_step_interactions'],pair_sum,places=10)
            self.assertAlmostEqual(row['adjacent_interactions'],adjacent,places=10)
            self.assertAlmostEqual(row['nonadjacent_interactions'],pair_sum-adjacent,places=10)
        return reduced

    def test_nonadjacent_reinforcement_with_zero_adjacent_alignment(self):
        result=self.check_vectors([(1,0),(0,1),(1,0)])
        self.assertEqual(result[-1]['adjacent_only_sum_norm_squared'],3.)
        self.assertAlmostEqual(result[-1]['exact_sum_norm_squared'],5.)

    def test_signed_negative_adjacent_only_proxy_is_not_clamped(self):
        result=self.check_vectors([(1,0),(-1,0),(1,0)])
        self.assertEqual(result[-1]['adjacent_only_sum_norm_squared'],-1.)
        self.assertEqual(result[-1]['exact_sum_norm_squared'],1.)
        self.assertIsNone(result[1]['relative_squared_norm_error'])

    def test_zero_gradient_and_mixed_vector_scope(self):
        self.check_vectors([(2,-3,1),(0,0,0),(1,2,-2),(-4,1,3)])


if __name__=='__main__':
    unittest.main()
