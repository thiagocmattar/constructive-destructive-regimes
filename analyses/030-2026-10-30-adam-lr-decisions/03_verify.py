"""Independent scalar/coverage acceptance for H024 forks (no cloud actions)."""
import json
import math
from pathlib import Path

def verify(output):
    output=Path(output)
    c=json.loads((output/'resolved_config.json').read_text())
    m=json.loads((output/'manifest.json').read_text())
    s=json.loads((output/'summary.json').read_text())
    rows=[json.loads(l) for l in (output/'events.jsonl').read_text().splitlines()]
    updates=[r for r in rows if r['event']=='update']
    n=c['decision']['updates'];j=c['resume']['junction_update']
    assert m['status']=='completed' and m['scope']=='scientific'
    assert [r['update'] for r in updates]==list(range(j+1,j+n+1))
    assert s['actual_displacement_relative_residual']<1e-10
    assert s['local_updates']==n
    for r in updates:
        assert r['sequences']==1024 and r['input_tokens']==2097152 and r['supervised_tokens']==2096128
        assert math.isfinite(r['loss'])
        lr=c['decision']['base_lr']*c['decision']['multiplier']
        assert math.isclose(r['learning_rate'],lr,rel_tol=1e-14)
        for wd in r['weight_decay_groups']:
            assert wd==0 or math.isclose(lr*wd,c['decision']['common_shrink'],rel_tol=1e-14)
        assert math.isclose(r['clipped']['gradient_norm'],r['gradient_norm']*r['clip_coefficient'],rel_tol=3e-7,abs_tol=1e-10)
        for stream in [r,r['momentum'],r['adaptive'],r['clipped'],r['actual_parameter_update']]:
            assert stream['adjacent_cosine'] is None or abs(stream['adjacent_cosine'])<=1+1e-10
            assert stream['cumulative_ratio'] is None or 0<=stream['cumulative_ratio']<=1+1e-10
        assert r['incoming_previous_moment_cosine'] is None or abs(r['incoming_previous_moment_cosine'])<=1+1e-10
    for a,b in zip(updates,updates[1:]):
        assert math.isclose(b['projected_gradient_change'],1-b['adjacent_cosine']*b['gradient_norm']/a['gradient_norm'],abs_tol=1e-12)
    assert [r['local_update'] for r in s['validations']]==[0,16,128]
    assert all(r['supervised_tokens']==693329 and r['sequences']==339 and r['short_tail_included'] for r in s['validations'])
    assert [r['local_update'] for r in s['probes']]==[0,16,32,64,128]
    assert all(r['supervised_tokens']==128*2047 for r in s['probes'])
    for local in [64,128]:assert (output/f'checkpoints/update-{j+local}.pt').is_file()
    return dict(verified=True,updates=len(updates),input_tokens=sum(r['input_tokens'] for r in updates),actual_displacement_relative_residual=s['actual_displacement_relative_residual'])

if __name__=='__main__':
    import sys
    print(json.dumps(verify(sys.argv[1]),indent=2))
