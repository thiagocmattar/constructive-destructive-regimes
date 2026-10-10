"""Prepare the explicitly authorized 31M/70M cohort from the frozen Run002 recipe."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import requests
import torch
from transformers import GPTNeoXConfig, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from baseline.initialization import EXPECTED_ARCHITECTURE, apply_pythia_initialization

OLD = ROOT / 'runs/002-2026-10-03-pythia14m-adamw-adaptive-geometry'

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')

def main():
    for folder in ['003-2026-10-03-pythia31m-adamw-scale','004-2026-10-03-pythia70m-adamw-scale']:
        if (ROOT/'runs'/folder).exists():
            raise RuntimeError('Historical run ID already exists; no preparation inputs were rewritten.')
    torch.set_num_threads(2)
    # Promote only code that the two newly requested runs now reuse.
    shutil.copy2(OLD / 'adam_diagnostics.py', ROOT / 'src/baseline/adam_diagnostics.py')
    trainer = (OLD / 'training.py').read_text(encoding='utf-8')
    trainer = trainer.replace('from baseline.initialization import apply_pythia_14m_initialization',
                              'from baseline.initialization import apply_pythia_initialization')
    trainer = trainer.replace('from adam_diagnostics import', 'from baseline.adam_diagnostics import')
    trainer = trainer.replace('apply_pythia_14m_initialization(model, torch=torch)',
            'apply_pythia_initialization(model, expected_architecture=config["model"]["expected_architecture"], torch=torch)')
    trainer = trainer.replace('directory = Path(__file__).resolve().parent\n    config_path = Path(args.config).resolve()',
            'config_path = Path(args.config).resolve()\n    directory = config_path.parent')
    trainer = trainer.replace('"run_id": "002"', '"run_id": config["run_id"]')
    trainer = trainer.replace('Initial parameters differ from the matched Run001 checkpoint.',
                               'Initial parameters differ from the frozen architecture/seed recipe.')
    trainer = trainer.replace('default=str(Path(__file__).with_name("config.json"))', 'required=True')
    (ROOT / 'src/baseline/training.py').write_text(trainer, encoding='utf-8')
    results = []
    for rid, size in [('003', '31'), ('004', '70')]:
        name = f'pythia{size}m-adamw-scale'
        run = ROOT / 'runs' / f'{rid}-2026-10-03-{name}'
        run.mkdir(exist_ok=False)
        for sub in ['prelaunch', 'observations', 'figures', 'artifacts']:
            (run / sub).mkdir()
        repo = f'EleutherAI/pythia-{size}m-deduped'
        info = requests.get(f'https://huggingface.co/api/models/{repo}', timeout=60)
        info.raise_for_status()
        revision = info.json()['sha']
        response = requests.get(f'https://huggingface.co/{repo}/resolve/{revision}/config.json', timeout=60)
        response.raise_for_status()
        (run / 'model_config.json').write_bytes(response.content)
        architecture = GPTNeoXConfig.from_json_file(run / 'model_config.json')
        expected = {**EXPECTED_ARCHITECTURE,
                    'hidden_size': architecture.hidden_size,
                    'intermediate_size': architecture.intermediate_size,
                    'num_attention_heads': architecture.num_attention_heads}
        torch.manual_seed(1234)
        architecture._attn_implementation = 'sdpa'
        architecture.use_cache = False
        model = AutoModelForCausalLM.from_config(architecture).float()
        init = apply_pythia_initialization(model, expected_architecture=expected, torch=torch)
        digest = hashlib.sha256()
        for n, p in model.named_parameters():
            digest.update(n.encode() + b'\0')
            digest.update(p.detach().numpy().tobytes())
        config = json.loads((OLD / 'config.json').read_text())
        config.update(run_id=rid, name=name, hypothesis='H002', matched_reference_run='002')
        config['model'].update(architecture=repo, revision=revision,
             initial_parameter_sha256=digest.hexdigest(), expected_architecture=expected,
             parameter_count=sum(p.numel() for p in model.parameters()))
        config['data']['tokenizer'] = 'EleutherAI/pythia-14m-deduped'
        config['execution'].update(maximum_hours=6, maximum_total_usd=5,
                                   cohort_maximum_total_usd=10, cohort_workers=2)
        write(run / 'config.json', config)
        write(run / 'prelaunch/model-provenance.json', {
            'repository': repo, 'revision': revision,
            'url': response.url, 'config_sha256': hashlib.sha256(response.content).hexdigest(),
            'initialization': init, 'initial_parameter_sha256': digest.hexdigest(),
            'parameter_count': config['model']['parameter_count'],
            'pretrained_weights_loaded': False})
        wrapper = '''"""Run-local entry point for the matched scale baseline."""\nfrom pathlib import Path\nimport sys\nsys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))\nfrom baseline.training import main\nif __name__ == "__main__":\n    main()\n'''
        (run / '02_train.py').write_text(wrapper, encoding='utf-8')
        verify = (OLD / '03_verify.py').read_text().replace('manifest["run_id"] == "002"',
                                                           'manifest["run_id"] == config["run_id"]')
        (run / '03_verify.py').write_text(verify, encoding='utf-8')
        for filename in ['00_setup_remote.sh']:
            text = (OLD / filename).read_text().replace(OLD.name, run.name)
            (run / filename).write_text(text, encoding='utf-8', newline='\n')
        for filename in ['accept_local_cache.py', 'audit_gpu_adam.py', 'deadline-guard.ps1', 'remote_status.py']:
            shutil.copy2(OLD / 'prelaunch' / filename, run / 'prelaunch' / filename)
        # The CUDA audit imports the promoted diagnostics and tests the same fused boundary.
        audit = (run / 'prelaunch/audit_gpu_adam.py').read_text()
        audit = audit.replace('import adam_diagnostics', 'import baseline.adam_diagnostics')
        audit = audit.replace('from adam_diagnostics import', 'from baseline.adam_diagnostics import')
        (run / 'prelaunch/audit_gpu_adam.py').write_text(audit, encoding='utf-8')
        results.append({'run': run.name, 'model': repo, 'revision': revision,
                        'parameters': config['model']['parameter_count'], 'initial_sha256': digest.hexdigest()})
        del model
    write(ROOT / 'cache/runpod/scale-cohort.json', results)
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
