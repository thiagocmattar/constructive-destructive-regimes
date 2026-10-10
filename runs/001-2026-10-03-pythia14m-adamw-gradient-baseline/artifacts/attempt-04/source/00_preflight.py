"""Check exact Linux library/backend availability and save hardware identity."""
import json
from pathlib import Path
import platform
import subprocess
import torch
import transformers
import numpy
import datasets
import torchvision
import torchaudio
from transformers import GPTNeoXForCausalLM

expected = {'torch':'2.11.0','torchvision':'0.26.0','torchaudio':'2.11.0','transformers':'5.12.1','numpy':'2.5.0','datasets':'5.0.0'}
actual = {name:module.__version__ for name,module in
          [('torch',torch),('torchvision',torchvision),('torchaudio',torchaudio),('transformers',transformers),('numpy',numpy),('datasets',datasets)]}
for name, version in expected.items():
    if actual[name].split('+')[0] != version:
        raise RuntimeError(f'{name} version differs: {actual[name]}')
assert torch.cuda.is_available() and torch.backends.cuda.is_flash_attention_available()
device = torch.cuda.get_device_properties(0)
value = {'python':platform.python_version(), 'platform':platform.platform(),
         'libraries':actual,'cuda':torch.version.cuda,'flash_attention_compiled':True,
         'gpu':device.name,'vram_bytes':device.total_memory,
         'nvidia_smi':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total',
                                             '--format=csv'],text=True)}
Path('/workspace/constructive/hardware.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value,indent=2))
