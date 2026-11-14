"""Refresh current captions/catalog; preserve the first revision and SVD fits."""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OLD = HERE.parent / 'figure_revision_20261114'
A63 = ROOT / 'analyses/063-2026-11-07-architecture-instability-tracing'
OBS = A63 / 'observations/007-resnet-major-peak.md'
NAME = '00-three-model-loss-representation-motion.pdf'

FIGURE_SECTION = '''## Figure 6: combined loss and representation motion

- **Current source:** [larger ResNet event](../../../../analyses/063-2026-11-07-architecture-instability-tracing/observations/007-resnet-major-peak.md), re-reduced from the unchanged Run273 control. MLP/Transformer retain the exact original arrays. Displayed completed steps are **451--501 / 390--450 / 553--601**. The original ResNet 281--311 view remains in the first figure-revision folder.
- **Selection:** the last recovered >100% loss excursion before step 512 in the native control. Baseline **0.1070888 at 404**, peak **0.4407024 at 415 (+311.53%)**, recovery **439**. Selection uses loss only; the window includes the pre-rise and recovery. The former peak at 293 rose 31.93%.
- **Stage reduction:** unchanged 5/6/6 measured endpoints, without averaging layer cosines. ResNet uses the first convolution, the outputs of residual stages 1/2/3, pooling and all-class log probabilities. Source boundary s already contains the incoming change A_s-A_(s-1); no additional +1 shift is applied to that source.
- **Outputs and scale:** raw MLP outputs; all-class/all-vocabulary log probabilities for ResNet/Transformer. Cosine spans [-1,1]. Each RMS row divides by its own maximum within the displayed window, comparing time within a stage rather than absolute amplitudes between stages.
- **Probe coverage:** all 5,000 MLP images; the first intact original 1,000-image ResNet BatchNorm group out of 50,000; the original 64 Transformer sequences/all 2,240 supervised positions out of 2,088,600 tokens. Full/probe loss curves remain separate for ResNet/Transformer. Model modes, fixed inputs, stochastic context and full training objectives are unchanged.
- **Dashed markers:** blue marks the first recorded negative output-change cosine in the display; black marks the first full loss at least 10% above the pre-peak minimum. Markers have identical completed-step coordinates across rows. Neither is an established alarm or causal onset.

| Model | First displayed negative output cosine | Baseline | First increase from baseline | First loss >=1.10 baseline | Peak | Recovery |
|---|---:|---:|---:|---:|---:|---:|
| MLP | <=451 (negative at entry) | 465 | 466 | 470 | 474 | See original trace |
| ResNet | 391 | 404 | 405 | 406 | 415 | 439 |
| Transformer | 556 | 557 | 558 | 561 | 562 | 596 |

- **ResNet interpretation:** the first displayed negative value at 391 is an earlier reversal, not a unique onset of the later major excursion. First-convolution movement peaks at 415; the other five measured endpoints peak at 416. Alignment-series correlations with output are .764/.828/.905/.995 for first convolution/stages 1/2/3. These dependent observations establish participation and timing, not causal transmission.
- **Trajectory qualification:** 313 loss states exactly match the native Run223 control. The prior Run243 trace agrees on 48 loss states and 372 stage measurements (maximum difference 6.67e-16). This is the same qualified continuation as the old tracing view, not bitwise replay of Run215's original overview. No new training or SVD is performed.
- **Caption:** Loss and functional motion across three unchanged-Adam continuations. Columns show MLP, ResNet and Transformer; rows show full/probe loss, consecutive representation-change cosine and displacement RMS divided by each stage's own window maximum. Each change A_s-A_(s-1) is placed at completed step s, alongside the loss at that state, without smoothing. Displayed steps are 451--501, 390--450 and 553--601. The ResNet window contains the +311.53% excursion peaking at 415 and recovering at 439. Blue lines mark the first recorded negative output-change cosine in the display; MLP is already negative at entry. Black lines mark the first full loss at least 10% above the minimum preceding the largest displayed peak. Fixed probes contain 5,000 images, 1,000 images and 2,240 token positions. RMS colors compare time within a stage. These are selected single-initialization windows; the separate SVD companion retains the original ResNet window 281--311.

'''

TRACING_CAPTION = r'''\newcommand{\consolidationTracingCaption}{%
Loss and functional motion across three unchanged-Adam continuations. Columns
show MLP, ResNet and Transformer; rows show full/probe loss, cosine between
successive representation changes and displacement coordinate RMS divided by
each stage's own window maximum. Heatmap rows are measured stage endpoints.
Each change $A_s-A_{s-1}$ is placed at completed step $s$, alongside the loss at
that state, without smoothing. Blue dashed lines mark the first observed
negative output-change cosine within the display; MLP is already negative at
entry, so its onset is unobserved. Black dashed lines mark the first full loss
at least 10\% above the minimum preceding the largest displayed peak. Both
markers continue through all rows. Fixed probes contain 5,000 images, 1,000
images and 2,240 token positions. MLP probe and full loss coincide; the other
probes are subsets. Displayed steps are 451--501, 390--450 and 553--601.
ResNet's selected excursion rises 311.53\% from step 404 to peak 415 and
recovers at 439. It exactly matches the native control used for the previous
trace, rather than bitwise replay of the original full-run overview.
Normalized RMS colors compare time within a stage, not absolute magnitudes
across stages. These are selected descriptive windows from one initialization
each. The separate temporal-structure companion retains the original ResNet
window 281--311; its SVD results do not describe this larger excursion.}

'''


def relative(path):
    return path.relative_to(ROOT).as_posix()


def main():
    spec = importlib.util.spec_from_file_location('package_helpers', HERE.parent / 'package_evidence.py')
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    notes = helpers.rebase_markdown(OLD / 'OBSERVATIONS.md', HERE / 'OBSERVATIONS.md', {}).split('\n\n', 1)[1]
    old_line = next(line for line in notes.splitlines() if line.startswith('- **Generator:**'))
    notes = notes.replace(old_line, '- **Current Figure 6:** [generator](01_figure.py), [manifest](figure_manifest.json), [verification](verification.json). Other views retain the [first revision generator](../figure_revision_20261114/01_figures.py) and [checks](../figure_revision_20261114/verification.json). [Current editable captions](captions.tex).')
    start, stop = notes.index('## Figure 6:'), notes.index('## Companion:')
    notes = notes[:start] + FIGURE_SECTION + notes[stop:]
    notes = notes.replace('the exact same original window fits as Figure 6: 51/31/49 displacement rows',
                          'the preserved original fits at completed steps **451--501 / 281--311 / 553--601**: 51/31/49 displacement rows (the ResNet fit is distinct from the revised Figure 6 window 390--450)')
    notes = notes.replace('Temporal structure of the representation-change windows in Figure 6.',
                          'Temporal structure of the original representation-change windows: completed steps 451--501, 281--311 and 553--601. ResNet retains its earlier small-excursion fit; it does not characterize Figure 6\'s new 390--450 window.')
    notes = notes[:notes.index('## Verification and limits')] + '''## Verification and limits

- The [first verification](../figure_revision_20261114/verification.json) remains the record for the initial seven views, preserved in their original work folder.
- The [current Figure 6 verification](verification.json) checks raw boundary indexing, all plotted values, marker crossings, source hashes, embedded fonts and page bounds. MLP/Transformer panel pixels and numerical arrays remain identical.
- The revised Poppler render is visually reviewed. All other PDFs, all intervention files and the intervention PLAN section remain byte-identical; manuscript/human input preservation is also checked by the package verifier.
- No new training, independent sample, run, hypothesis, confidence interval, SVD or causal experiment is created by this reduction.
'''
    (HERE / 'OBSERVATIONS.md').write_text(notes, encoding='utf-8', newline='\n')
    captions = (OLD / 'captions.tex').read_text(encoding='utf-8')
    start = captions.index(r'\newcommand{\consolidationTracingCaption}')
    stop = captions.index(r'\newcommand{\consolidationTemporalStructureCaption}')
    captions = captions[:start] + TRACING_CAPTION + captions[stop:]
    captions = captions.replace('Temporal structure of the same representation-change windows.',
                                'Temporal structure at completed steps 451--501 (MLP), 281--311 (ResNet)\nand 553--601 (Transformer). The ResNet fit retains its original small-excursion\nwindow, distinct from the revised loss/motion figure at 390--450.')
    (HERE / 'captions.tex').write_text(captions, encoding='utf-8', newline='\n')

    catalog_path = HERE.parent / 'evidence_catalog.json'
    catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
    for item in catalog['assets']:
        destination = item['destination']
        if destination == '05-adam/b-tracing/figures/' + NAME:
            item.update(source=relative(HERE / 'figures' / NAME), generator=relative(HERE / '01_figure.py'),
                        observation=relative(OBS), note='ResNet now spans completed steps 390--450 around the +311.53% peak at 415; unchanged native control and probe. MLP/Transformer panels remain identical. No new training or SVD.')
        elif destination == '05-adam/b-tracing/figures/01-three-model-temporal-structure.pdf':
            item['note'] = 'Preserved original fixed-window fits: completed steps 451--501 / 281--311 / 553--601. The ResNet SVD describes the earlier small excursion, not the revised Figure 6 window 390--450.'
        elif destination == 'FIGURE_NOTES.md':
            item['source'] = relative(HERE / 'OBSERVATIONS.md')
        elif destination == 'figure_captions.tex':
            item.update(source=relative(HERE / 'captions.tex'), observation=relative(HERE / 'OBSERVATIONS.md'))
        elif destination == 'figure_revision_manifest.json':
            item.update(title='Initial seven-view figure manifest', note='Authenticates the first revision; Figure 6 now has an additional current manifest in V-B/data.')
        elif destination == 'figure_revision_verification.json':
            item.update(title='Initial seven-view verification', note='Passed checks for the first revision; the larger ResNet peak has additional current verification in V-B/data.')
    catalog['sections']['05-adam/b-tracing']['presentation'] = 'Figure 6 uses ResNet steps 390--450 and the +311.53% peak at 415. MLP/Transformer remain unchanged. The separate SVD companion retains its original ResNet window 281--311; historical all-stage PDFs remain in background/figures.'
    additions = [
        (OBS, 'observations/63-007-resnet-major-peak.md', 'Larger ResNet loss-event trace', 'current observation', A63 / '11_resnet_major_peak.py'),
        (A63 / 'resnet_390_450_motion.json', 'data/resnet_390_450_motion.json', 'ResNet 390--450 measurements and source checks', 'current source record', A63 / '11_resnet_major_peak.py'),
        (HERE / 'figure_manifest.json', 'data/resnet_peak_figure_manifest.json', 'Current Figure 6 numerical manifest', 'source record', HERE / '01_figure.py'),
        (HERE / 'verification.json', 'data/resnet_peak_figure_verification.json', 'Current Figure 6 verification', 'verification', HERE / '02_verify.py'),
    ]
    for source, target, title, role, generator in additions:
        item = dict(section='05-adam/b-tracing', source=relative(source), destination='05-adam/b-tracing/' + target,
                    title=title, role=role, observation=relative(OBS), generator=relative(generator),
                    note='Larger native-control event; original SVD fits and intervention assets preserved.')
        existing = next((i for i, a in enumerate(catalog['assets']) if a['destination'] == item['destination']), None)
        if existing is None:
            catalog['assets'].append(item)
        else:
            catalog['assets'][existing] = item
    catalog_path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'catalog_assets': len(catalog['assets']), 'unique_sources': len({a['source'] for a in catalog['assets']})}))


if __name__ == '__main__':
    main()
