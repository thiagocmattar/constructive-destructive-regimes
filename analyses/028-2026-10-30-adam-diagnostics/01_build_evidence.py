"""Recheck existing AdamW events and generate two compact manuscript tables."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def write(p, value):
    p.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8', newline='\n')


def main():
    sweep_path = ROOT / 'analyses/009-2026-10-09-intermediate-lr/complete-seven-lr/comparison_data.json'
    schedule_path = ROOT / 'analyses/004-2026-10-04-lr-schedule-intervention/results.json'
    assert digest(sweep_path) == 'a02a1ba7ace7a7011063767e6e7fb1c25548be0a12be06b6dad6826e5e7b2739'
    sweep, schedules = read(sweep_path), read(schedule_path)
    report = dict(script_sha256=digest(Path(__file__)), numpy=np.__version__,
                  source_summaries={p.relative_to(ROOT).as_posix(): digest(p) for p in [sweep_path,schedule_path]},
                  event_sources={}, streams=[], schedules=[], tables={})

    def events(run, expected):
        p = ROOT / 'runs' / run / 'artifacts/attempt-01/full-pass/events.jsonl'
        if not p.exists():
            p = ROOT / 'analyses/009-2026-10-09-intermediate-lr/scratch/completed-metrics' / run[:3] / 'events.jsonl'
        assert digest(p) == expected, run
        report['event_sources'][p.relative_to(ROOT).as_posix()] = expected
        return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]

    for rid, old in sweep['runs'].items():
        folder = next((ROOT / 'runs').glob(rid + '-*'))
        records = events(folder.name, old['sources']['events.jsonl'])
        updates = [e for e in records if e.get('event') == 'update']
        assert [e['update'] for e in updates] == list(range(1,713))
        assert updates == old['events']
        late = updates[611:711]
        val = [e for e in records if e.get('event') == 'final_validation'][-1]
        assert val['supervised_tokens'] == 693329 and val['short_tail_included']
        row = dict(run=rid, lr=old['lr'], hardware=old['hardware'], validation_loss=val['loss'],
                   raw=float(np.median([e['adjacent_cosine'] for e in late])),
                   momentum=float(np.median([e['momentum']['adjacent_cosine'] for e in late])),
                   adaptive=float(np.median([e['adaptive']['adjacent_cosine'] for e in late])),
                   late_raw_norm=float(np.median([e['gradient_norm'] for e in late])),
                   late_momentum_norm=float(np.median([e['momentum']['momentum_norm'] for e in late])),
                   clipped_updates=sum(e['gradient_clip_returned_norm'] > 1 for e in updates),
                   negative_momentum_pairs=sum(e['momentum']['adjacent_cosine'] < 0 for e in updates[1:]))
        assert row['validation_loss'] == old['performance']['final_validation_loss']
        for field,key in [('raw','raw'),('momentum','momentum'),('adaptive','adaptive')]:
            assert row[field] == old['geometry'][key]['last100_median_cosine']
        assert row['clipped_updates'] == old['performance']['clipping_threshold_exceeded_updates']
        assert row['negative_momentum_pairs'] == 0
        report['streams'].append(row)

    hashes = {r['run']:r['events_sha256'] for r in schedules['source']}
    for old in schedules['records']:
        records = events(old['run'],hashes[old['run']])
        affected = [e for e in records if e.get('event') == 'update' and 358 <= e['update'] <= 712]
        assert len(affected) == 355
        val = [e for e in records if e.get('event') == 'validation'][-1]
        assert val['update'] == 712 and val['supervised_tokens'] == 693329
        row = dict(run=old['run'],size=old['size'],schedule=old['variant'],
                   raw=float(np.median([e['adjacent_cosine'] for e in affected])),validation_loss=val['loss'],
                   positive_recovery=old['recovery']['31'])
        assert row['raw'] == old['primary']['median_raw_cosine']
        assert row['validation_loss'] == old['terminal_validation_loss']
        report['schedules'].append(row)

    target=ROOT/'manuscript/draft/tables';target.mkdir(exist_ok=True)
    lines=[r'\begin{tabular}{rrrrr}',r'\hline',r'Rate & Raw & Moment & Adaptive & Val. loss \\',r'\hline']
    for label,ids in [('DGX Spark (GB10)',['022','021','023']),('RTX 4090',['024','025','026','027'])]:
        lines.append(r'\multicolumn{5}{l}{'+label+r'} \\')
        for rid in ids:
            row=next(r for r in report['streams'] if r['run']==rid)
            power=int(np.floor(np.log10(row['lr'])))
            mantissa=row['lr']/10**power
            rate=(rf'$10^{{{power}}}$' if abs(mantissa-1)<1e-9 else rf'${mantissa:.2f}\!\times\!10^{{{power}}}$')
            lines.append(rate+' & '+' & '.join(f"{row[key]:.3f}" for key in ['raw','momentum','adaptive','validation_loss'])+r' \\')
        lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    p=target/'analysis028_streams.tex';p.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n');report['tables'][p.name]=digest(p)
    lines=[r'\begin{tabular}{lrrrr}',r'\hline',r'& \multicolumn{2}{c}{14M} & \multicolumn{2}{c}{70M} \\',
           r'Schedule & Raw cosine & Val. loss & Raw cosine & Val. loss \\',r'\hline']
    for variant,label in [('original','Original decay'),('hold','Held rate'),('accelerated','Faster decay')]:
        fields=[]
        for size in ['14M','70M']:
            row=next(r for r in report['schedules'] if (r['schedule'],r['size'])==(variant,size))
            fields.extend([f"{row['raw']:.3f}",f"{row['validation_loss']:.3f}"])
        lines.append(label+' & '+' & '.join(fields)+r' \\')
    lines.extend([r'\hline',r'\end{tabular}'])
    p=target/'analysis028_schedules.tex';p.write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n');report['tables'][p.name]=digest(p)
    assert len(report['event_sources'])==13
    report['status']='passed'
    write(HERE/'evidence.json',report)
    print(json.dumps(dict(status='passed',event_files=13,sweep_updates=7*712,schedule_pairs=6*355,table_rows=10)))


if __name__=='__main__':
    main()
