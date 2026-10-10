"""Secondary analysis of temporal interactions from the already requested scalars."""
import argparse
import hashlib
import json
from pathlib import Path


def reduce_prefixes(rows):
    energy = adjacent = 0.0
    previous = None
    result = []
    for row in rows:
        norm = row['gradient_norm']
        energy += norm * norm
        if previous is not None and previous > 0 and norm > 0:
            if row['adjacent_cosine'] is None:
                raise ValueError('A nonzero adjacent pair has undefined cosine.')
            adjacent += 2 * row['adjacent_cosine'] * previous * norm
        exact = row['cumulative_gradient_norm'] ** 2
        interaction = exact - energy
        omitted = interaction - adjacent
        approximation = energy + adjacent
        result.append({'update':row['update'],'gradient_energy':energy,
                       'exact_sum_norm_squared':exact,'all_cross_step_interactions':interaction,
                       'adjacent_interactions':adjacent,'nonadjacent_interactions':omitted,
                       'adjacent_only_sum_norm_squared':approximation,
                       'relative_squared_norm_error':(approximation-exact)/exact if exact else None})
        previous = norm
    return result


def analyze(root, destination):
    root, destination = Path(root), Path(destination)
    assert json.loads((root/'verification.json').read_text())['verified']
    rows = [json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
    rows = [row for row in rows if row['event'] == 'update']
    reduced = reduce_prefixes(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(2,1,figsize=(10,6),sharex=True,constrained_layout=True)
    x = [r['update'] for r in reduced]
    axes[0].plot(x,[r['exact_sum_norm_squared'] for r in reduced],color='#2462a3',label='Exact cumulative gradient norm squared')
    axes[0].plot(x,[r['adjacent_only_sum_norm_squared'] for r in reduced],color='#d2771c',ls='--',label='Energy + adjacent interactions')
    axes[0].plot(x,[r['gradient_energy'] for r in reduced],color='#737373',ls=':',label='Sum of squared gradient norms')
    axes[0].set_ylabel('Squared gradient norm')
    axes[0].legend(fontsize=8)
    for key,color,label in [('adjacent_interactions','#28794d','Adjacent pairs'),
                            ('nonadjacent_interactions','#8052a0','All nonadjacent pairs')]:
        axes[1].plot(x,[r[key] for r in reduced],color=color,label=label)
    axes[1].axhline(0,color='black',lw=.8,ls='--')
    axes[1].set(ylabel='Signed cross-step interaction',xlabel='Optimizer update')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.18)
    (destination/'figures').mkdir(exist_ok=True)
    (destination/'scratch').mkdir(exist_ok=True)
    fig.savefig(destination/'figures/temporal_interactions.pdf')
    fig.savefig(destination/'scratch/temporal_interactions.png',dpi=180)
    plt.close(fig)
    with (root/'events.jsonl').open('rb') as stream:
        source_hash = hashlib.file_digest(stream,'sha256').hexdigest()
    with Path(__file__).open('rb') as stream:
        script_hash=hashlib.file_digest(stream,'sha256').hexdigest()
    value={'source_events_sha256':source_hash,'script_sha256':script_hash,'prefixes':reduced,
           'negative_adjacent_only_squared_norm_prefixes':sum(r['adjacent_only_sum_norm_squared']<0 for r in reduced)}
    (destination/'figures/temporal_data.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    final = reduced[-1]
    relative_error_label=(f"{final['relative_squared_norm_error']:.6f}"
                          if final['relative_squared_norm_error'] is not None
                          else 'undefined because the exact squared norm is zero')
    text=f"""# Observation 002: adjacent versus full temporal interactions

This secondary reduction uses only the requested full-pass scalar records; it adds no training intervention or gradient-history requirement. Source SHA-256: `{source_hash}`. Script: `../05_temporal_reduction.py`; all prefix values: `../figures/temporal_data.json`. Coverage: all {len(rows)} accepted updates of seed 1234, including the smaller final update. No smoothing, filtering, or uncertainty band.

For each prefix define E = sum ||g_u||², L = 2 sum of adjacent rho_u ||g_(u-1)|| ||g_u||, and J = ||sum g_u||² − E. Then J − L is the signed sum of all nonadjacent pair interactions (twice each pair dot product), up to floating-point reduction error. E + L is an adjacent-only estimate of the squared raw-gradient resultant norm; it is kept signed and never clamped before analysis.

At the final update: exact squared resultant norm {final['exact_sum_norm_squared']:.6f}; energy E {final['gradient_energy']:.6f}; all interactions J {final['all_cross_step_interactions']:.6f}; adjacent interactions L {final['adjacent_interactions']:.6f}; omitted nonadjacent interactions J−L {final['nonadjacent_interactions']:.6f}. The adjacent-only value is {final['adjacent_only_sum_norm_squared']:.6f}, a relative squared-norm error of {relative_error_label}. There are {value['negative_adjacent_only_squared_norm_prefixes']} prefixes with a negative adjacent-only squared-norm estimate.

Caption (`temporal_interactions.pdf`): Exact cumulative raw-gradient squared norm compared with energy plus lag-one interactions, and the signed adjacent/nonadjacent interaction sums along Run001. Values come from the saved whole-model norms and adjacent cosines, without reconstructing every pair separately. One trajectory, no replicate uncertainty.

Interpretation: a ratio ||sum g|| / sum ||g|| below one alone does not establish net negative temporal interaction; orthogonality also reduces the ratio. J supplies the net sign for raw gradients. This reduction examines the cumulative raw-gradient norm term only. Under AdamW and a varying learning rate, it is not a reconstruction of parameter norm, displacement, or the manuscript's full constant-step SGD identity, and it does not establish stability or causality.
"""
    (destination/'observations/002-temporal-interactions.md').write_text(text,encoding='utf-8')
    with (destination/'observations/INDEX.md').open('a',encoding='utf-8') as stream:
        stream.write('- [002: Adjacent versus full temporal interactions](002-temporal-interactions.md)\n')
    print(json.dumps({k:v for k,v in value.items() if k!='prefixes'},indent=2))
    print(json.dumps(final,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--destination',default=str(Path(__file__).resolve().parent))
    args=parser.parse_args()
    analyze(args.root,args.destination)
