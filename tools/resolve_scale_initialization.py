"""Identify CPU starting tensors on the execution platform before optimizer steps."""
from pathlib import Path
import hashlib
import json
import sys
import torch
from transformers import GPTNeoXConfig, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from baseline.initialization import apply_pythia_initialization

run = ROOT / 'runs' / sys.argv[1]
config = json.loads((run / 'config.json').read_text())
torch.manual_seed(config['seeds']['model'])
c = GPTNeoXConfig.from_json_file(run / 'model_config.json')
c._attn_implementation = 'sdpa'
c.use_cache = False
m = AutoModelForCausalLM.from_config(c).float()
init = apply_pythia_initialization(m, torch=torch,
    expected_architecture=config['model']['expected_architecture'])
h = hashlib.sha256()
for n, p in m.named_parameters():
    h.update(n.encode()+b'\0')
    h.update(p.detach().numpy().tobytes())
print(json.dumps({'initial_parameter_sha256':h.hexdigest(), 'parameters':sum(p.numel() for p in m.parameters()),
    'platform':sys.platform,'torch':torch.__version__, 'transformers':__import__('transformers').__version__,
    'initialization':init,'optimizer_steps':0}))
