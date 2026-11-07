"""Prepare six prescribed layer/global LR comparisons in the existing study."""
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
ARMS=[(252,'first_half',.0005,.001),(253,'first_quarter',.00025,.001),
      (254,'global_half',.0005,.0005),(255,'global_quarter',.00025,.00025),
      (256,'later_half',.001,.0005),(257,'later_quarter',.001,.00025)]

TEST='''import copy, json, sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.eos_tasks.layer_lr_continuation import layer_step
torch.manual_seed(11)
m=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(6,5),torch.nn.Tanh(),torch.nn.Linear(5,5),torch.nn.Tanh(),torch.nn.Linear(5,2))
o=torch.optim.Adam(m.parameters(),lr=.001,foreach=False,fused=False)
x,y=torch.randn(9,6),torch.randn(9,2)
for _ in range(4):
 o.zero_grad();((m(x)-y).square().sum()/18).backward();o.step()
for a,b in [(1,1),(.5,1),(.25,1),(.5,.5),(.25,.25),(1,.5),(1,.25)]:
 test=copy.deepcopy(m);ref=copy.deepcopy(m)
 testopt=torch.optim.Adam(test.parameters(),lr=.001,foreach=False,fused=False)
 refopt=torch.optim.Adam(ref.parameters(),lr=.001,foreach=False,fused=False)
 frozen=copy.deepcopy(o.state_dict())
 for opt in [testopt,refopt]:opt.load_state_dict(copy.deepcopy(frozen))
 # Independent native Adam parameter groups with copied warm moments.
 old=refopt.param_groups[0]
 refopt.param_groups=[dict(old,params=list(ref[layer].parameters()),lr=.001*(a if layer==1 else b)) for layer in [1,3,5]]
 for _ in range(3):
  for opt,model in [(testopt,test),(refopt,ref)]:
   opt.zero_grad();((model(x)-y).square().sum()/18).backward()
  rng=torch.get_rng_state().clone()
  layer_step(test,testopt,.001*a,.001*b);refopt.step()
  assert torch.equal(rng,torch.get_rng_state()) and len(testopt.param_groups)==1
  for p,q in zip(test.parameters(),ref.parameters()):
   assert torch.equal(p,q) and torch.equal(p.grad,q.grad)
   for key in testopt.state[p]:assert torch.equal(testopt.state[p][key],refopt.state[q][key])
 assert all(int(v['step'])==4 for v in frozen['state'].values())
print(json.dumps(dict(status='passed',checks=['seven LR combinations including identity','three warm steps exactly match independently grouped native Adam','biases included','moments clocks raw gradients RNG preserved','checkpoint parameter ordering preserved'])))
'''


def main():
    records=[]
    for number,arm,first,later in ARMS:
        run=ROOT/f'runs/{number}-2026-11-07-mlp-layer-lr-{arm.replace("_","-")}'
        assert not run.exists();run.mkdir()
        cfg=dict(run_id=str(number),analysis_id='065',hypothesis='H046',arm=arm,intervention_label=f'I{number-243}',
            seed=0,fork=448,end=801,first_lr=first,later_lr=later,base_lr=.001,physical_batch=1000,
            runtime_limit_seconds=600,primary_window=[448,800],event_window=[450,500],tail_window=[769,800],
            target_loss=.02,target_consecutive_states=10,
            intervention='No freezing; specified layer LRs from448, first changed update448->449; all other baseline settings and Adam state retained',
            authorization='User says Go on after selective first-layer LR proposal and global/later-layer controls; Spark execution already authorized',
            coverage='All5000 fixed training examples, seed0 tanh3072-200-200-10, mean half squared error, FP32/TF32off, Adam(.9,.999), eps1e-8, no clipping/decay',
            retention='Full fork448, native449, intervention449, qualification512, recovery/final801; dense raw gradients/five representations447..501; rolling metrics through801',
            optimizer_state_contract='No reset. Native Adam with temporary LR-only groups; original group layout restored between steps; applied LRs retained in checkpoint metadata/config',
            control='Untreated replay448..512 must match every loss/norm and complete historical512 state before reusing original control',
            comparisons='Every arm against unchanged Adam; location/rate contrasts supplementary; equal LR multipliers do not match actual increment norms')
        (run/'config.json').write_text(json.dumps(cfg,indent=2)+'\n',newline='\n')
        (run/'.gitignore').write_text('artifacts/\n',newline='\n')
        (run/'01_worker.py').write_text('from pathlib import Path\nimport sys\nHERE=Path(__file__).resolve().parent\nsys.path.insert(0,str(HERE.parents[1]))\nfrom src.eos_tasks.layer_lr_continuation import main\nif __name__=="__main__":main(HERE)\n',newline='\n')
        (run/'test_intervention.py').write_text(TEST,newline='\n')
        transport=(ROOT/'runs/247-2026-11-07-mlp-frozen-lr-0p01/02_remote.py').read_text().replace('247',str(number))
        transport=transport.replace("['frozen_first_linear.py','frozen_lr_continuation.py']","['frozen_first_linear.py','layer_lr_continuation.py']")
        transport=transport.replace("continuation=root/'src/eos_tasks/frozen_lr_continuation.py'\nassert continuation.exists() and hashlib.sha256(continuation.read_bytes()).hexdigest()=={sha(ROOT/'src/eos_tasks/frozen_lr_continuation.py')!r}","continuation=root/'src/eos_tasks/layer_lr_continuation.py'\nassert not continuation.exists() or hashlib.sha256(continuation.read_bytes()).hexdigest()=={sha(ROOT/'src/eos_tasks/layer_lr_continuation.py')!r}")
        transport=transport.replace("'src/eos_tasks/frozen_lr_continuation.py'","'src/eos_tasks/layer_lr_continuation.py'")
        (run/'02_remote.py').write_text(transport,newline='\n')
        (run/'README.md').write_text(f'''# Run {number}: {arm.replace('_',' ')}

H046 / ongoing Analysis065, intervention I{number-243}. From untreated state448,
first-linear weights and biases use Adam LR{first}; later layers use LR{later}.
All layers remain trainable. There is no inherited freeze and no change at500.
First affected update448->449; continue through801 to measure outgoing changes800.
All data/model/optimizer/precision settings and full fork match unchanged Adam.
No reset; native Adam moments and clocks continue with per-layer learning rates.

Primary objective: improve full-window448-800 mean loss while reducing large
excursions. Also report450-500 event, mean last32 states769-800, endpoint800,
positive variation/max upward change, and first10 consecutive states below0.02
(censored if not reached). Threshold fixed before these runs, not a claim of
optimality. Every arm compares with unchanged baseline, with global/later-layer
controls supplementary. Same multipliers do not match total update norms.

Qualification: untreated448-512 replay and full state512 equality; observer
preserves RNG/gradients; first changed step has native moments and scaled applied
increments. All5000 rows retained; dense447-501 gradients/five representations,
rolling cosine/RMS, actual per-layer increments, first moments and denominator
RMS through800; fork, qualification, first-step, recovery and end checkpoints.
No new SVD or validation result is planned; no architecture change.

Owned Spark, no cloud charge. Prior similar runs8-15s each; allow a few minutes
for six workers plus retrieval and verification.600s per worker limit with630s
detached timeout and30s grace. Six-run maximum63min of compute, under18GB total
retained including archives (under3GB/run). Progress every64 updates, signal/
STOP/time/nonfinite guards; verify sizes/SHA256 after retrieval and confirm exit.
Single seed and fork; no global stability or generalization inference.
''',newline='\n')
        records.append(dict(run_id=number,arm=arm,first_lr=first,later_lr=later,path=run.relative_to(ROOT).as_posix()))
    (HERE/'layer_lr_plan.json').write_text(json.dumps(dict(status='prepared',hypothesis='H046',arms=records,
        analysis='065',primary_window=[448,800],event_window=[450,500],tail_window=[769,800],
        target_loss=.02,target_consecutive_states=10,cloud_cost=0,maximum_worker_seconds=600,
        maximum_retained_gb=18),indent=2)+'\n',newline='\n')
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
