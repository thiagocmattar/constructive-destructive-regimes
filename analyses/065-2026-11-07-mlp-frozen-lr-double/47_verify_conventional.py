"""Independent post-run recurrence and coverage verification beside raw audits."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def audit(path):
    a=torch.load(path,map_location='cpu',weights_only=False);before=a['before'];after=a['after']
    group=before['optimizer']['param_groups'][0];b1,b2=group['betas'];lr=group['lr'];eps=group['eps'];step=a['state']+1
    raw=a['raw_gradient'];factor=min(1.,a['arm']['clip']/float(raw.double().norm())) if a['arm'].get('clip') else 1.
    assert torch.equal(a['optimizer_input'],raw*factor)
    offset=0;errors=dict(moment=0.,second_moment=0.,parameter=0.)
    for name,pid in zip(before['param_group_names'][0],group['params']):
        p=before['model'][name].double();m=before['optimizer']['state'][pid]['exp_avg'].double();v=before['optimizer']['state'][pid]['exp_avg_sq'].double()
        g=a['optimizer_input'][offset:offset+p.numel()].reshape(p.shape).double();offset+=p.numel()
        em=b1*m+(1-b1)*g;ev=b2*v+(1-b2)*g*g;ep=p-lr*(em/(1-b1**step))/((ev/(1-b2**step)).sqrt()+eps)
        actual=[after['optimizer']['state'][pid]['exp_avg'].double(),after['optimizer']['state'][pid]['exp_avg_sq'].double(),after['model'][name].double()]
        for key,x,y in zip(errors,actual,[em,ev,ep]):
            torch.testing.assert_close(x,y,rtol=4e-5,atol=3e-6)
            errors[key]=max(errors[key],float((x-y).abs().max()))
        assert int(after['optimizer']['state'][pid]['step'])==step
    assert offset==raw.numel() and torch.equal(after['previous_gradient'],raw)
    return dict(path=path.name,status='passed',max_absolute_errors=errors)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);args=parser.parse_args();out=args.output
    manifest=json.loads((out/'manifest.json').read_text());assert manifest['status']=='completed'
    inventory=json.loads((out/'inventory.json').read_text())
    for f in inventory['files']:
        p=out/f['path'];assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256'],str(p)
    reports=[]
    for status in manifest['results']:
        p=out/status['arm']['id'];bounds=[json.loads(s) for s in (p/'boundaries.jsonl').read_text().splitlines()]
        updates=[json.loads(s) for s in (p/'metrics.jsonl').read_text().splitlines()]
        assert [r['state'] for r in bounds]==list(range(status['start_state'],status['end_state']+1))
        assert len(updates)==status['end_state']-status['start_state']==status['updates']
        for r in updates:
            if status['arm'].get('clip'):assert r['optimizer_input_norm']<=status['arm']['clip']*(1+2e-7)
            else:assert r['raw_gradient_norm']==r['optimizer_input_norm']
            assert r['bias_correction2']==1-status['arm'].get('beta2',.999)**r['update']
        end=torch.load(p/'end.pt',map_location='cpu',weights_only=False)
        assert end['state']==status['end_state']
        assert end['current_gradient'] is not None
        audits=[audit(f) for f in sorted(p.glob('audit-*.pt'))];assert len(audits)==2
        reports.append(dict(arm=status['arm']['id'],boundaries=len(bounds),updates=len(updates),audits=audits,evaluation=status['evaluation']))
    result=dict(status='passed',inventory_files=len(inventory['files']),arms=reports)
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
