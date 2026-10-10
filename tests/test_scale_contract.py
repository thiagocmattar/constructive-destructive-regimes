"""Guard matched initialization and promoted measurement code for scale runs."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import unittest
import torch
from transformers import AutoModelForCausalLM, GPTNeoXConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from baseline.initialization import apply_pythia_initialization, apply_pythia_14m_initialization

class ScaleContractTests(unittest.TestCase):
    def test_initial_identity_and_architecture_guards(self):
        torch.set_num_threads(2)
        for folder in ['002-2026-10-03-pythia14m-adamw-adaptive-geometry',
                       '003-2026-10-03-pythia31m-adamw-scale',
                       '004-2026-10-03-pythia70m-adamw-scale']:
            run = ROOT / 'runs' / folder
            config = json.loads((run / 'config.json').read_text())
            torch.manual_seed(config['seeds']['model'])
            c = GPTNeoXConfig.from_json_file(run / 'model_config.json')
            c.use_cache = False
            c._attn_implementation = 'sdpa'
            m = AutoModelForCausalLM.from_config(c).float()
            if config['run_id'] == '002':
                apply_pythia_14m_initialization(m, torch=torch)
            else:
                apply_pythia_initialization(m, torch=torch,
                    expected_architecture=config['model']['expected_architecture'])
                self.assertEqual(sum(p.numel() for p in m.parameters()), config['model']['parameter_count'])
                with self.assertRaises(ValueError):
                    apply_pythia_initialization(m, torch=torch, expected_architecture={'hidden_size': -1})
            digest = hashlib.sha256()
            for name, p in m.named_parameters():
                digest.update(name.encode() + b'\0')
                digest.update(p.detach().numpy().tobytes())
            if config['run_id'] == '002':
                # Torch normal draws can differ between Windows and Linux. Compare
                # the promoted function to the frozen implementation on this host.
                frozen = run / 'artifacts/attempt-02/source/baseline/initialization.py'
                spec = importlib.util.spec_from_file_location('frozen_init', frozen)
                old = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(old)
                torch.manual_seed(config['seeds']['model'])
                other = AutoModelForCausalLM.from_config(c).float()
                old.apply_pythia_14m_initialization(other, torch=torch)
                for p, q in zip(m.parameters(), other.parameters()):
                    self.assertTrue(torch.equal(p, q))
            else:
                provenance = json.loads((run / 'prelaunch/model-provenance.json').read_text())
                self.assertEqual(digest.hexdigest(), provenance['initial_parameter_sha256'])

    def test_adaptive_code_promotion_is_exact(self):
        old = ROOT / 'runs/002-2026-10-03-pythia14m-adamw-adaptive-geometry/adam_diagnostics.py'
        self.assertEqual(old.read_bytes(), (ROOT / 'src/baseline/adam_diagnostics.py').read_bytes())

if __name__ == '__main__':
    unittest.main()
