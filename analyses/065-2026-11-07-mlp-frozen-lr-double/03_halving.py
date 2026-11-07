"""Prepare one next halving trial or assess its full post-change loss window."""
import argparse
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
TEMPLATE=ROOT/'runs/247-2026-11-07-mlp-frozen-lr-0p01'


def write(p,text):p.write_text(text,encoding='utf-8',newline='\n')
def dump(p,value):write(p,json.dumps(value,indent=2)+'\n')


def prepare(run_id,lr):
    token=format(lr,'.8f').rstrip('0').replace('.','p')
    dest=ROOT/f'runs/{run_id}-2026-11-07-mlp-frozen-lr-{token}'
    assert not dest.exists();dest.mkdir()
    cfg=json.loads((TEMPLATE/'config.json').read_text())
    plan=json.loads((HERE/'halving_plan.json').read_text())
    cfg.update(run_id=str(run_id),intervention_lr=lr,
        intervention=f'I{run_id-243}: freeze first-linear weights/biases from448; Adam LR{lr} from500; same moments/clocks and parent fork500',
        authorization='User requests continued LR halving until no peak, within existing MLP analysis065',
        adaptive_stop_rule=plan['stop_rule'])
    dump(dest/'config.json',cfg);write(dest/'.gitignore','artifacts/\n')
    write(dest/'01_worker.py',(TEMPLATE/'01_worker.py').read_text().replace('I4:',f'I{run_id-243}:').replace('LR to0.01',f'LR to{lr}'))
    write(dest/'02_remote.py',(TEMPLATE/'02_remote.py').read_text().replace('247',str(run_id)))
    write(dest/'03_verify.py',(TEMPLATE/'03_verify.py').read_text().replace("b['lr']==.01",f"b['lr']=={lr}").replace('final_lr=.01',f'final_lr={lr}'))
    factor=lr/.001;tol=max(2e-7,factor*1e-7)
    test=(TEMPLATE/'test_intervention.py').read_text().replace("group['lr']=.01",f"group['lr']={lr}").replace('+10*(',f'+{factor}*(').replace('atol=1e-6',f'atol={tol}').replace('10x baseline LR',f'{factor}x baseline LR')
    write(dest/'test_intervention.py',test)
    write(dest/'README.md',f'''# Run {run_id}: I{run_id-243}, first layer frozen with LR {lr}

Same ongoing MLP Analysis064/065, H045. Complete intervention: freeze first-linear
weights/biases from448 and set Adam LR{lr} at500, half the previous tested rate.
Branch from the same qualified frozen parent500, first changed update500->501,
end801. Compare primarily with unchanged Adam; previous rates are supplementary.
All5000 examples/order, seed0, tanh3072-200-200-10, mean half squared error,
physical1000, FP32/TF32off, Adam(.9,.999), eps1e-8, no clipping/decay are matched.
Native moments/clocks continue; zero first-layer applied increments with raw
gradients retained. Exact reconstruction/qualification448-501; full states and
all five representation streams, dense499-551, rolling500-800, retained as before.

Adaptive no-peak rule, fixed before this run: all losses500-800 finite, no adjacent
increase >1e-8, and no loss above loss500+1e-8. If failed, halve again from the same
fork; retain unsuccessful runs. This is a conservative finite-window criterion,
not a universal stability or generalization claim. See analysis halving_plan.json.

Owned Spark, zero cloud cost; measured prior run8.08s. Qualified ETC under2min
for execution plus retrieval;600s worker limit,630s detached timeout+30s grace.
Below3GB retained per run including archive. Verify67 source identities, native
moment/increment parity and freeze constancy; persisted progress, finite checks,
full recovery states, SHA256/size-verified retrieval and task exit required.
Preserve unrelated work. Same diagnostic definitions as Run247 and analysis065.
''')
    print(json.dumps(dict(run_id=run_id,lr=lr,path=dest.relative_to(ROOT).as_posix())),flush=True)


def assess(relative):
    path=ROOT/relative;spec=importlib.util.spec_from_file_location('transport',path/'02_remote.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    rule=json.loads((HERE/'halving_plan.json').read_text())['stop_rule']
    code=f'''from pathlib import Path
import json,hashlib
run=Path({m.REMOTE!r})/{m.REL!r};raw=run/'artifacts/attempt-01/worker'
manifest=json.loads((raw/'manifest.json').read_text());assert manifest['status']=='completed',manifest
p=raw/'boundaries.jsonl';rows=[json.loads(s) for s in p.read_text().splitlines()]
rows=[r for r in rows if 500<=r['state']<=800];assert [r['state'] for r in rows]==list(range(500,801))
import math
loss=[r['loss'] for r in rows];changes=[b-a for a,b in zip(loss,loss[1:])];tol={rule['absolute_tolerance']!r}
peak=max(rows,key=lambda r:r['loss']);largest=max(changes)
print(json.dumps(dict(run={relative!r},lr=json.loads((run/'config.json').read_text())['intervention_lr'],
 status='passed' if all(math.isfinite(x) for x in loss) and largest<=tol and peak['loss']<=loss[0]+tol else 'peak_present',
 states=[500,800],absolute_tolerance=tol,loss_start=loss[0],loss_final=loss[-1],loss_mean=sum(loss)/len(loss),
 loss_max=peak['loss'],peak_state=peak['state'],maximum_adjacent_increase=largest,
 increases_above_tolerance=sum(x>tol for x in changes),positive_variation=sum(max(0,x) for x in changes),
 boundaries_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),manifest=manifest)))'''
    result=m.remote(code);dump(path/'halving_assessment.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    q=sub.add_parser('prepare');q.add_argument('run_id',type=int);q.add_argument('lr',type=float)
    q=sub.add_parser('assess');q.add_argument('relative')
    args=p.parse_args()
    if args.action=='prepare':prepare(args.run_id,args.lr)
    else:assess(args.relative)
