"""Freeze six matched checkpoint blocks and eighteen constant-rate forks."""
from pathlib import Path
import copy
import hashlib
import json
import sys
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from baseline.recipe import recipe_learning_rate
from baseline.data_protocol import block_order

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def save(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2) + '\n', encoding='utf-8', newline='\n')

def main():
    if (HERE / 'cohort.json').exists():
        raise RuntimeError('Prepared packet exists; do not overwrite frozen inputs.')
    scratch = HERE / 'scratch'
    scratch.mkdir(exist_ok=True)
    for name in ['analyses/INDEX.md','research/INDEX.md','research/HYPOTHESES.md','runs/INDEX.md']:
        (scratch / name.replace('/', '_')).write_bytes((ROOT / name).read_bytes())
    cache = ROOT.parent / 'sparsity-spillover/data/tokenized/minipile-pythia-14m-full'
    n_tokens = (cache / 'train/tokens.int32.bin').stat().st_size // 4
    orders = {seed: block_order(n_tokens,2048,seed) for seed in [1234,1235,1236]}
    available = np.ones(n_tokens // 2048, dtype=bool)
    for order in orders.values():
        used = order[:484*1024]
        available[used[used < len(available)]] = False
    probe = np.random.default_rng(2430).choice(np.flatnonzero(available),128,replace=False)
    save(HERE / 'probe.json',dict(indices=sorted(probe.tolist()),selection_seed=2430,
        available_complete_sequences=int(available.sum()),supervised_tokens=128*2047,
        rule='intersection of complete train sequences unseen through484 in all three seed orders'))
    sources, cohort, blocks = [], [], []
    for split in ['train','validation']:
        for name in ['tokens.int32.bin','metadata.json']:
            p = cache / split / name
            sources.append(dict(source=str(p),path=f'inputs/cache/{split}/{name}',bytes=p.stat().st_size,sha256=sha(p)))
    for bi,rid in enumerate([2,4,5,7,8,10]):
        run = next((ROOT/'runs').glob(f'{rid:03}-*'))
        parent = next(run.glob('artifacts/**/checkpoints/midpoint'))
        config = json.loads((run/'config.json').read_text())
        summary = json.loads((parent.parents[1]/'summary.json').read_text())
        state = torch.load(parent/'recovery.pt',map_location='cpu',weights_only=False)
        assert state['data_position'] ==356*1024
        assert state['config_sha256']==sha(run/'config.json')
        seed=config['seeds']['model']
        order_hash=hashlib.sha256(orders[seed].astype('<i8').tobytes()).hexdigest()
        assert order_hash==state['order_sha256']
        assert all(int(v['step'])==356 for v in state['optimizer']['state'].values())
        for split in ['train','validation']:
            assert next(s['sha256'] for s in sources if s['path']==f'inputs/cache/{split}/tokens.int32.bin')==config['data'][split+'_sha256']
        hashes={p.relative_to(parent).as_posix():sha(p) for p in parent.rglob('*') if p.is_file()}
        # Match retained source inventory, not merely the current file against itself.
        invpaths=list(run.glob('artifacts/**/inventory.json'))
        matched=False
        for ip in invpaths:
            inv=json.loads(ip.read_text())
            files={v['path']:v for v in inv.get('files',[])}
            if all((parent / rel).is_relative_to(ip.parent) and
                   files.get((parent/rel).relative_to(ip.parent).as_posix(),{}).get('sha256')==dig
                   for rel,dig in hashes.items()): matched=True;break
        if not matched: raise ValueError(f'No original inventory matches {parent}')
        modelcfg=json.loads((parent/'model/config.json').read_text())
        from safetensors import safe_open
        with safe_open(parent/'model/model.safetensors',framework='pt') as f:
            count=sum(np.prod(f.get_slice(k).get_shape()) for k in f.keys())
        size='14M' if count<20000000 else '70M'
        key=f'{size.lower()}-{seed}'
        for rel,digest in hashes.items():
            p=parent/rel
            sources.append(dict(source=p.relative_to(ROOT).as_posix(),path=f'inputs/parents/{key}/{rel}',bytes=p.stat().st_size,sha256=digest))
        events=[json.loads(l) for l in (parent.parents[1]/'events.jsonl').read_text().splitlines()]
        expected=[e for e in events if e.get('event')=='update' and 357<=e['update']<=359]
        save(HERE/'references'/f'{key}.json',dict(first_updates=expected,parent_summary_sha256=sha(parent.parents[1]/'summary.json')))
        t=config['training']
        base=recipe_learning_rate(356,peak=t['peak_learning_rate'],max_steps=712,warmup_fraction=t['warmup_fraction'],minimum=t['minimum_learning_rate'])
        block=dict(key=key,size=size,seed=seed,parent=parent.relative_to(ROOT).as_posix(),base_lr=base,parameter_count=int(count),checkpoint_sha256=hashes,original_inventory=ip.relative_to(ROOT).as_posix())
        blocks.append(block)
        for ai,multiplier in enumerate([.5,1.,2.]):
            new_id=187+3*bi+ai
            name=f'{new_id:03}-2026-10-30-pythia{size.lower()}-seed{seed}-lr{str(multiplier).replace(".","p")}'
            dest=ROOT/'runs'/name
            c=copy.deepcopy(config)
            c.update(run_id=f'{new_id:03}',name=name,hypothesis='H024')
            c['model']['parameter_count']=int(count)
            c['resume']=dict(junction_update=356,full_updates=712,checkpoint_sha256=hashes,parent_config_sha256=state['config_sha256'],order_sha256=order_hash,remote_checkpoint=f'inputs/parents/{key}')
            c['decision']=dict(block=key,multiplier=multiplier,base_lr=base,updates=128,common_shrink=base*t['weight_decay'],probe_path='analyses/'+HERE.name+'/probe.json',probe_sha256=sha(HERE/'probe.json'),validation_local_updates=[0,16,128],probe_local_updates=[0,16,32,64,128],checkpoint_local_updates=[64,128])
            c['execution']=dict(provider='runpod',authorization='user: Go on. You are allowed to run experiments using runpod',status='prepared')
            save(dest/'config.json',c)
            (dest/'README.md').write_text(f'# Run{new_id}: {key}, LR multiplier{multiplier}\n\nH024, controlled128-update continuation from356. Frozen protocol and measurement\ncontract: [Analysis030](../../analyses/{HERE.name}/README.md).\nStatus: prepared. Starting state and original hashes are in config.json; no\nscientific result is implied by mechanics checks.\n',encoding='utf-8',newline='\n')
            cohort.append(dict(run=name,block=key,size=size,seed=seed,multiplier=multiplier,worker=bi%2))
        del state
        print('Qualified',key,flush=True)
    save(HERE/'blocks.json',blocks)
    save(HERE/'input_sources.json',sources)
    save(HERE/'cohort.json',cohort)
    print('Prepared',len(cohort),'forks; input GB',sum(s['bytes'] for s in sources)/1e9)

if __name__=='__main__':main()
