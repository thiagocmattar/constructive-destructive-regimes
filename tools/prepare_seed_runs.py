"""Allocate the six authorized seed replications without changing executed inputs."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import torch
from transformers import GPTNeoXConfig, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from baseline.initialization import EXPECTED_ARCHITECTURE, apply_pythia_initialization
from baseline.data_protocol import block_order

REFERENCES = {
    '14M': '002-2026-10-03-pythia14m-adamw-adaptive-geometry',
    '31M': '003-2026-10-03-pythia31m-adamw-scale',
    '70M': '004-2026-10-03-pythia70m-adamw-scale',
}
TEMPLATE = ROOT/'runs'/REFERENCES['70M']
COHORT = [(f'{5+i:03d}', size, seed) for i,(seed,size) in enumerate(
    (s,m) for s in (1235,1236) for m in REFERENCES)]

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')

def main():
    names = [f'{rid}-2026-10-04-pythia{size.lower()}-seed{seed}' for rid,size,seed in COHORT]
    if any((ROOT/'runs'/name).exists() and any(p.is_file() for p in
               (ROOT/'runs'/name).rglob('*')) for name in names):
        raise RuntimeError('Run IDs already allocated; existing inputs were not rewritten.')
    torch.set_num_threads(2)
    records = []
    for (rid,size,seed),name in zip(COHORT,names):
        run = ROOT/'runs'/name
        (run/'prelaunch').mkdir(parents=True,exist_ok=True)
        reference = ROOT/'runs'/REFERENCES[size]
        config = json.loads((reference/'config.json').read_text())
        config.update(run_id=rid,name=f'pythia{size.lower()}-seed{seed}',hypothesis='H002',
                      matched_reference_run=reference.name[:3])
        config['seeds'] = {'model':seed,'data_order':seed}
        config['model'].setdefault('expected_architecture',EXPECTED_ARCHITECTURE)
        config['execution'].update(maximum_hours=4,maximum_total_usd=4.1,
                                  cohort_maximum_total_usd=25,cohort_workers=6)
        torch.manual_seed(seed)
        arch = GPTNeoXConfig.from_json_file(reference/'model_config.json')
        arch._attn_implementation='sdpa'; arch.use_cache=False
        model = AutoModelForCausalLM.from_config(arch).float()
        init=apply_pythia_initialization(model,torch=torch,
                    expected_architecture=config['model']['expected_architecture'])
        digest=hashlib.sha256()
        for n,p in model.named_parameters():
            digest.update(n.encode()+b'\0'); digest.update(p.detach().numpy().tobytes())
        config['model']['initial_parameter_sha256']=digest.hexdigest()
        config['model']['parameter_count']=sum(p.numel() for p in model.parameters())
        # This identity must be resolved on Linux and pinned before calibration.
        save(run/'config.json',config)
        shutil.copy2(reference/'model_config.json',run/'model_config.json')
        for filename in ('02_train.py','03_verify.py'):
            shutil.copy2(TEMPLATE/filename,run/filename)
        setup=(TEMPLATE/'00_setup_remote.sh').read_text().replace(TEMPLATE.name,name)
        (run/'00_setup_remote.sh').write_text(setup,encoding='utf-8',newline='\n')
        for filename in ('accept_local_cache.py','audit_gpu_adam.py','deadline-guard.ps1','remote_status.py'):
            shutil.copy2(TEMPLATE/'prelaunch'/filename,run/'prelaunch'/filename)
        order=block_order(1491711416,2048,seed)
        save(run/'prelaunch/model-provenance.json',{
            'model_revision':config['model']['revision'],'reference_run':reference.name,
            'platform':sys.platform,'initialization':init,'initial_parameter_sha256':digest.hexdigest(),
            'parameters':config['model']['parameter_count'],'pretrained_weights_loaded':False,
            'execution_platform_identity_pending':True,
            'order_sha256':hashlib.sha256(order.tobytes()).hexdigest()})
        (run/'README.md').write_text(f'''# Run {rid}: Pythia-{size}, seed {seed}

Status: preparing; the user explicitly authorized two additional seeds per size,
parallel GPU execution, and completion through analysis.

This H002 replication changes model initialization and data permutation seeds
to {seed}. The same seed and exact batch order are paired across 14M/31M/70M;
the order differs between seed blocks. Both sources of training variability
change together; their effects are not separately identified. The reference is
Run {reference.name[:3]}, seed 1234. No pretrained weights are loaded.

Match pinned model architecture within size, token/cache bytes, full MiniPile
coverage, 712 updates, 1024 sequences per full update, microbatch 4, AdamW/LR,
clipping, loss scaling, validation, and diagnostic placement. Model shapes differ
across sizes; the initialization recipe is shared, not the tensor values.

Measure each update for raw gradients g (unscaled accumulated batch mean before
clipping) and adaptive Adam directions d (bias-corrected actual FP32 moments,
evaluated in FP64 after the optimizer step). d excludes learning rate and weight
decay; moments use clipped gradients. For each stream record adjacent cosine,
A=sum of vector norms, B=norm of cumulative vector sum, B/A, individual norm,
and the paired adaptive/raw norm ratio. Initial cosine is undefined with a reason.
All unique trainable parameters; FP64 cumulative sums/reductions.

Consume all 1,491,711,416 input tokens once, including the short tail. Validate
all 500 documents with token weighting and tail inclusion. Retain every update;
main evolution plots omit final partial update 712 as previously requested.
Preserve initial/midpoint/final models and recovery state (optimizer, scaler,
streaming diagnostic vectors, RNG, data position).

Execution: one RTX 5090 at live $0.99/hour, or RTX 4090 capacity fallback at
$0.74/hour, Secure Cloud. At most four hours per worker, $25 for all six including
storage and bounded retries. 60GB container and 40GB persistent storage per Pod;
no network volume. Detached training, timeout with 30-minute retrieval margin,
local deadline guard, and bounded read-only monitoring. Calibrate actual hardware
before the full pass, retaining environment/hardware and exact input identities.
Retrieve and verify every file size/SHA and scientific artifacts before deletion.
No task cloud storage is intentionally retained. Hardware differs from the
original RTX4090 cohort; numeric kernels may affect trajectories and are recorded.

Analysis002 combines three seeds per size. Compare full-pass medians, terminal
A/B/B/A, norms, negative-cosine fractions, and per-seed trajectories. Report all
seed points, mean/sample SD and range; paired cross-size differences and ordering
per seed. Three seeds are a small reproducibility check, not a scaling-law
validation or proof of causal optimization benefit. Steps are not independent
replicates. Do not infer stability/generalization from alignment alone.
''',encoding='utf-8')
        records.append({'run':name,'size':size,'seed':seed,
            'parameters':config['model']['parameter_count'],'reference_run':reference.name})
        del model
    save(ROOT/'cache/runpod/seed-cohort.json',records)
    print(json.dumps(records,indent=2))

if __name__=='__main__': main()
