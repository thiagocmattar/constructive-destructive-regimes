"""Write source-linked seed observations from the verified reduction."""
from pathlib import Path
import hashlib
import json
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.stdout.reconfigure(encoding='utf-8')
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    j=json.loads((HERE/'results.json').read_text())
    lines=['# Observation 001: Scale geometry across three seeds','',
        'Status: completed full passes; raw evidence retrieved and independently verified.','',
        'The user requested two additional seeds at each size to examine a possible single-run effect.',
        'Seeds 1235 and 1236 vary both initialization and training-data permutation, paired across sizes.',
        'Seed1234 comes from the existing paired runs; every size now has three trajectories.','',
        '## Sources and reduction','',
        '[results.json](../results.json) retains exact source/config/environment/acceptance hashes,',
        '[figure_data.csv](../figure_data.csv) retains all 6,408 per-update reductions, and',
        '[01_compare.py](../01_compare.py) generates all statistics and publication PDFs.',
        '[02_record.py](../02_record.py) writes this observation. The pilot reduction functions',
        'are reused with their script hash recorded; no old raw evidence is modified.','']
    for r in j['source_lineage']:
        lines.append(f"- {r['size']}, seed {r['seed']}: [Run{r['run_id']}](../../../runs/{r['folder']}/README.md), {r['attempt']}; {r['parameters']:,} parameters; {r['gpu']}.")
    lines += ['', 'All trajectories match token/cache bytes, 712 updates, effective batch1024/microbatch4,',
        'the AdamW/LR/clipping recipe, FP16 scaling, parameter scope, loss weighting, and validation.',
        'Exact permutation hashes match across sizes within each seed and differ across seed blocks.',
        'Architectural width/head configuration and initialization scale vary with model size.',
        'No pretrained weights are loaded. Each pass consumes all 1,491,711,416 input tokens once.',
        'Initial/final validation includes all 500 documents and the packed short tail, token-weighted.','',
        'For x=g (unscaled raw gradient before clipping) or d (adaptive Adam direction from actual',
        'updated FP32 moments of the clipped gradient, evaluated in FP64), A_t=sum_{i<=t}||x_i||,',
        'B_t=||sum_{i<=t}x_i|| and R=B/A. d excludes learning-rate scaling and weight decay.',
        'Cosines use adjacent accumulated optimizer boundaries. First cosine is undefined;',
        'medians use 711 defined pairs. A/B/R are terminal at update712. Norm medians use all712',
        'updates, with equal update weighting. Steps are not independent seed replicates.','',
        '## Primary replication comparison','',
        'Before the new full passes, the pilot expectation was fixed as decreasing median adjacent',
        'cosine and terminal B/A with size, separately for raw and adaptive streams.','',
        '| Stream | Metric | Strict pilot ordering in new seeds | All three seeds |',
        '| --- | --- | ---: | ---: |']
    primary=[]
    for stream in ('raw','adaptive'):
        for metric in ('median_adjacent_cosine','R'):
            checks=[r for r in j['paired_size_comparisons'] if r['stream']==stream and r['metric']==metric]
            nnew=sum(r['strict_pilot_ordering'] for r in checks if r['seed']!=1234)
            nall=sum(r['strict_pilot_ordering'] for r in checks)
            primary.append({'stream':stream,'metric':metric,'new_seed_blocks_matching':nnew,'all_seed_blocks_matching':nall})
            lines.append(f'| {stream} | {metric} | {nnew}/2 | {nall}/3 |')
    if all(v['new_seed_blocks_matching']==2 for v in primary):
        lines += ['', 'The full-pass median-cosine and terminal-ratio ordering survives both new seed blocks',
            'for both vector streams. This supports reproducibility of the pilot ordering under this',
            'matched recipe. Adaptive cosine varies much less across seeds than raw cosine.']
    lines += ['', 'The counts are descriptive replication outcomes, not a p-value or a validated scaling law.',
        'Any reversal or overlap is retained below rather than selecting successful seeds.','',
        '| Seed | Stream | 14M cosine | 31M cosine | 70M cosine | 14M B/A | 31M B/A | 70M B/A |',
        '| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for seed in (1234,1235,1236):
        for stream in ('raw','adaptive'):
            r=[next(v['all_updates'][stream] for v in j['runs'] if v['seed']==seed and v['size']==size) for size in ('14M','31M','70M')]
            vals=[v['median_adjacent_cosine'] for v in r]+[v['R'] for v in r]
            lines.append(f"| {seed} | {stream} | "+' | '.join(f'{v:.6f}' for v in vals)+' |')
    lines += ['', '## Secondary ordering and phase sensitivity', '',
        'The pilot absolute-geometry directions are checked separately from the primary cosine/ratio',
        'expectations. These dimensional quantities need not share the normalized ordering.', '',
        '| Stream | Metric | Pilot direction | New seed blocks matching | All seed blocks matching |',
        '| --- | --- | --- | ---: | ---: |']
    for stream in ('raw','adaptive'):
        for metric in ('A','B','median_norm'):
            checks=[r for r in j['paired_size_comparisons'] if r['stream']==stream and r['metric']==metric]
            expected=checks[0]['pilot_expectation']
            if expected=='no_monotone_pilot_expectation':
                lines.append(f'| {stream} | {metric} | No monotone pilot expectation | N/A | N/A |')
            else:
                nnew=sum(r['strict_pilot_ordering'] for r in checks if r['seed']!=1234)
                nall=sum(r['strict_pilot_ordering'] for r in checks)
                lines.append(f'| {stream} | {metric} | {expected} | {nnew}/2 | {nall}/3 |')
    for v in j['paired_size_comparisons']:
        if v['stream']=='raw' and v['metric']=='A' and not v['strict_pilot_ordering']:
            lines += ['', f"Raw A is nonmonotone in seed{v['seed']}: 14M/31M/70M values are "
                +'/'.join(f'{x:.6f}' for x in v['values'])+'. This small reversal is retained.']
    lines += ['', 'Full-pass ordering does not imply ordering at every phase. Quarter cosine medians',
        'use pairs whose later update lies in the specified quarter (including the cross-boundary pair).',
        'The following counts examine strict 14M > 31M > 70M ordering within each quarter.', '',
        '| Later updates | Raw cosine ordering | Adaptive cosine ordering |',
        '| --- | ---: | ---: |']
    for q in range(4):
        counts=[]
        for stream in ('raw','adaptive'):
            ordered=0
            for seed in (1234,1235,1236):
                values=[next(r['quarters'][q][stream]['median_adjacent_cosine'] for r in j['runs']
                    if r['seed']==seed and r['size']==size) for size in ('14M','31M','70M')]
                ordered += int(values[0]>values[1]>values[2])
            counts.append(f'{ordered}/3')
        lines.append(f'| {q*178+1}-{(q+1)*178} | '+' | '.join(counts)+' |')
    lines += ['', '## Mean, seed variability, and absolute geometry','',
        'Mean ± sample SD across three per-trajectory summaries; SD is seed spread, not a confidence interval.',
        'The full JSON also provides each seed value, min/max, paired size differences and quarter summaries.','',
        '| Model | Stream | Median cosine | Terminal B/A | Terminal A | Terminal B | Median norm |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for size in ('14M','31M','70M'):
        for stream in ('raw','adaptive'):
            vals=[]
            for key in ('median_adjacent_cosine','R','A','B','median_norm'):
                g=next(v for v in j['across_seeds'] if v['size']==size and v['stream']==stream and v['metric']==key)
                vals.append(f"{g['mean']:.6g} ± {g['sample_sd']:.3g}")
            lines.append(f'| {size} | {stream} | '+' | '.join(vals)+' |')
    lines += ['', '| Run | Seed | Model | Vector | A | B | B/A | Median norm | Negative pairs |',
        '| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in j['runs']:
        for stream in ('raw','adaptive'):
            v=r['all_updates'][stream]
            lines.append(f"| {r['run_id']} | {r['seed']} | {r['size']} | {stream} | {v['A']:.9g} | {v['B']:.9g} | {v['R']:.6f} | {v['median_norm']:.9g} | {v['negative_cosines']}/{v['defined_cosines']} |")
    lines += ['', '| Model | Seed | Median paired ||d||/||g|| | Initial validation loss | Final validation loss | Overflow retries |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in j['runs']:
        lines.append(f"| {r['size']} | {r['seed']} | {r['median_adaptive_to_raw_norm_ratio']:.6f} | {r['initial_validation_loss']:.6f} | {r['final_validation_loss']:.6f} | {r['overflow_retries']} |")
    lines += ['', '## Figures and coverage','',
        '- [Geometry evolution](../figures/geometry_across_seeds.pdf): adjacent cosine, cumulative B/A and individual norm. Bold lines are means across seeds; faint lines are individual trajectories and shading is the seed range. Cosine curves first use a centered31-step median within each seed; adaptive cosine has a zoomed vertical axis. Norm axes are logarithmic.',
        '- [Size summary](../figures/scale_summary_across_seeds.pdf): full-pass per-seed points connected by seed, plus means and sample SD, for cosine, B/A, A, B, norms and negative-pair fractions. Connecting lines are guides, not fitted laws. Absolute norms/A/B use log axes.',
        '- [Cumulative A/B](../figures/cumulative_across_seeds.pdf): raw and adaptive cumulative sums of norms and norms of cumulative sums, with means and seed ranges.',
        '- [Norm ratios](../figures/norm_ratio_across_seeds.pdf): per-step adaptive/raw norm ratios, with seed means/ranges and a logarithmic axis.',
        '', 'All evolution displays use updates1–711; the last partial batch (311 sequences) is omitted',
        'only from display, as previously requested. All raw updates, CSV rows and full terminal',
        'statistics retain update712. results.json also includes update711 statistics for sensitivity.',
        'Envelope widths are descriptive min/max across only three trajectories, not statistical intervals.','',
        '## Interpretation limits','',
        'Three seeds are a small replication check. Pairing makes size differences interpretable',
        'within each seed, but does not isolate initialization from data-order variability.',
        'Seed1234 used RTX4090; the two new seed blocks use RTX5090. Software/precision/optimizer',
        'contracts match, but hardware/kernel floating-point differences can change trajectories.',
        'Within each new seed block all three sizes use the same GPU type.',
        'Width, heads and initialization scale co-vary with parameter count; absolute A/B and norms',
        'depend on dimensionality and parameterization. No universal power law, out-of-sample',
        'prediction, stability, generalization, or causal learning effect is established.',
        'Positive adjacent direction cosine alone does not imply collinearity over the full pass',
        'or A≈B; long-horizon turning can leave a gap. Loss is measured independently.',
        'These are vector-geometry diagnostics. The manuscript\'s constant-step plain-SGD parameter',
        'norm identity is not applied directly to this AdamW trajectory with a varying LR and decay.','']
    (HERE/'observations/001-seed-replication.md').write_text('\n'.join(lines),encoding='utf-8')
    (HERE/'observations/INDEX.md').write_text('# Observations\n\n- [001: Scale seed replication](001-seed-replication.md)\n',encoding='utf-8')
    record={'script_sha256':sha(Path(__file__)),'results_sha256':sha(HERE/'results.json'),'primary_replication_counts':primary,
        'observation_sha256':sha(HERE/'observations/001-seed-replication.md')}
    (HERE/'record_generation.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(primary,indent=2))

if __name__=='__main__': main()
