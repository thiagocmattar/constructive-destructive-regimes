"""Prepare the selected V-C assets after full-window verification and plotting."""
import csv
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
BASE=ROOT/'manuscript/consolidation';OUT=BASE/'evidence';SECTION='05-adam/c-intervention'


def dump(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    result=json.loads((HERE/'conventional_results.json').read_text(encoding='utf-8'));assert result['status']=='passed'
    curves=list(csv.DictReader((HERE/'conventional_curves.csv').open(encoding='utf-8',newline='')))
    checks=[]
    for row in result['summaries']:
        control=next(r for r in result['summaries'] if (r['family'],r['model'],r['arm'])==(row['family'],row['model'],'control'))
        series=[r for r in curves if (r['family'],r['model'],r['arm'])==(row['family'],row['model'],row['arm'])]
        assert [int(r['state']) for r in series]==list(range(row['fork'],row['end']+1))
        loss=[float(r['loss']) for r in series[1:]]
        assert math.isclose(math.fsum(loss)/len(loss),row['mean_loss'],rel_tol=1e-12)
        assert math.isclose(math.fsum(loss[-32:])/32,row['late32_loss'],rel_tol=1e-12)
        assert max(loss)==row['maximum_loss'] and loss[-1]==row['final_loss']
        for absolute,relative in [('mean_loss','mean_change_percent'),('late32_loss','late32_change_percent'),('maximum_loss','maximum_change_percent'),('test_loss','test_change_percent')]:
            assert math.isclose(100*(row[absolute]/control[absolute]-1),row[relative],abs_tol=1e-10)
        if row['family']=='clipping':
            active=sum(float(r['clip_factor'])<1 for r in series[:-1])/len(loss)
            assert math.isclose(active,row['clipping_fraction'],abs_tol=1e-12)
        checks.append(dict(family=row['family'],model=row['model'],arm=row['arm'],rows=len(series)))
    lines=['# Conventional intervention outcomes','',
        '- Changes are relative to each intervention\'s matched unchanged Adam control; negative loss changes are lower.',
        '- One initialization; MLP states 513-800, ResNet 257-576, Transformer 545-928 for post-fork summaries.',
        '- MLP uses half-squared error; ResNet and Transformer use cross entropy/NLL. Compare within each pair.',
        '- Conventional and distributed-cap forks differ. This table is not a ranking across all four families.','',
        '| Family | Model | Setting | Mean loss change | Last-32 change | Maximum loss change | Test loss change | Clipping active |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for r in result['summaries']:
        if r['arm']=='control':continue
        setting=('beta2=' if r['family']=='beta2' else 'cap=')+r['arm'].split('_',1)[1].replace('p','.')
        active='n/a' if r['clipping_fraction'] is None else f"{100*r['clipping_fraction']:.1f}%"
        lines.append(f"| {r['family']} | {r['model']} | {setting} | {r['mean_change_percent']:+.2f}% | {r['late32_change_percent']:+.2f}% | {r['maximum_change_percent']:+.2f}% | {r['test_change_percent']:+.2f}% | {active} |")
    lines += ['', '- ResNet clipping at 0.5 remains missing; the [TODO](../../manuscript/consolidation/evidence/05-adam/c-intervention/TODO_ABLATIONS.md) records the remaining grid.',
        '- [Per-arm CSV](conventional_summary.csv) includes absolute means, last 32 means, terminal losses and coverage; [results and checks](conventional_results.json) retain all source hashes.', '',
        '## Unchanged-control context', '',
        '- Loss-only excursions use the existing Analysis036 definition, including unrecovered events. Counts concern events with a post-fork peak, using the complete historical prefix for event boundaries.',
        '- This is a descriptive portability check; no run is excluded or retuned based on the result.', '',
        '| Family | Model | Excursions >10% | Excursions at least 2x | 2x peak within 32 states of historical target |',
        '|---|---|---:|---:|---|']
    for r in result['control_context']:
        near='Yes' if r['double_excursion_within_32_states_of_historical_target'] else 'No'
        lines.append(f"| {r['family']} | {r['model']} | {r['excursions_over_10_percent']} | {r['excursions_at_least_double']} | {near} (target {r['historical_target_state']}) |")
    (HERE/'conventional_outcomes.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    dump(HERE/'conventional_export_verification.json',dict(status='passed',boundary_rows=len(curves),series=checks,
        mean_recomputed_with='stdlib math.fsum on exported CSV',curves_sha256=sha(HERE/'conventional_curves.csv')))
    catalog_path=BASE/'work/evidence_catalog.json';catalog=json.loads(catalog_path.read_text(encoding='utf-8'))
    assets=catalog['assets'];analysis=HERE.relative_to(ROOT).as_posix();observation=analysis+'/observations/015-conventional-model-coverage.md'
    old=['01-beta2-reductions-cifar10-mlp.pdf','05-gradient-clipping-cifar10-mlp-resnet.pdf']
    for name in old:
        item=next(a for a in assets if a['source'].endswith('/'+name))
        previous=OUT/item['destination'];item['destination']=SECTION+'/background/figures/'+name
        item.update(role='background only',title='Historical comparator: '+name,
            note='Original evidence preserved; selected display now includes all three architectures.')
        item.pop('presentation_order',None);destination=OUT/item['destination'];destination.parent.mkdir(parents=True,exist_ok=True)
        if previous!=destination and previous.exists():
            assert previous.resolve().is_relative_to(OUT.resolve()) and destination.resolve().is_relative_to(OUT.resolve())
            if destination.exists():
                assert sha(previous)==sha(destination)
                previous.unlink()
            else:previous.rename(destination)
    additions=[
        ('figures/15-beta2-reductions-cifar10-mlp-resnet-wikitext2-transformer.pdf','figures/15-beta2-reductions-cifar10-mlp-resnet-wikitext2-transformer.pdf','Figure 8: beta2 reductions across three architectures','selected conventional intervention',2),
        ('figures/16-gradient-clipping-cifar10-mlp-resnet-wikitext2-transformer.pdf','figures/16-gradient-clipping-cifar10-mlp-resnet-wikitext2-transformer.pdf','Figure 9: gradient clipping across three architectures','selected conventional intervention',3),
        ('observations/015-conventional-model-coverage.md','observations/65-015-conventional-model-coverage.md','Analysis065: complete conventional model coverage','record',None),
        ('conventional_summary.csv','data/conventional_summary.csv','Conventional intervention per-arm outcomes','data',None),
        ('conventional_curves.csv','data/conventional_curves.csv','Conventional intervention full boundary curves','data',None),
        ('conventional_results.json','data/conventional_results.json','Conventional intervention source and verification record','data',None),
        ('conventional_outcomes.md','tables/conventional-outcomes.md','Conventional intervention outcomes','table',None)]
    for source,dest,title,role,order in additions:
        item=dict(section=SECTION,source=analysis+'/'+source,destination=SECTION+'/'+dest,title=title,role=role,
            observation=observation if source.endswith('.pdf') else None,generator=analysis+'/49_plot_conventional.py' if source.endswith('.pdf') else None,
            note='Matched unchanged controls; one initialization per model. Original conventional forks retained.' if source.endswith('.pdf') else '')
        if order:item['presentation_order']=order
        existing=next((a for a in assets if a['destination']==item['destination']),None)
        if existing:existing.update(item)
        else:assets.append(item)
    catalog['sections'][SECTION]['presentation']='LR schedules, beta2 reductions, gradient clipping, then distributed q50, the sole representation-inspired method. Beta2 covers all three models; clipping covers .5/1 for MLP and Transformer and 1 for ResNet. The quantile/LR ablation remains at the section root as supporting evidence; the TODO records missing experiments.'
    dump(catalog_path,catalog)
    print(json.dumps(dict(status='passed',boundary_rows=len(curves),series=len(checks),catalogue_assets=len(assets))))


if __name__=='__main__':main()
