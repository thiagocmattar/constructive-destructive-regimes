"""Consolidate the matching temporal fit, preserving historical source records."""
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
PRIOR=HERE.parent/'resnet_peak_revision_20261114'
A63=ROOT/'analyses/063-2026-11-07-architecture-instability-tracing'
OBS=A63/'observations/008-resnet-major-peak-temporal.md'
NAME='01-three-model-temporal-structure.pdf'

CAPTION=r'''\newcommand{\consolidationTemporalStructureCaption}{%
Temporal structure of the same windows as the loss/motion figure: completed
steps 451--501 (MLP), 390--450 (ResNet) and 553--601 (Transformer), with
51/61/49 fitted displacement rows. The ResNet SVD is freshly fitted to the
replay-qualified larger excursion, peaking at 415 and recovering at 439.
Top: cumulative singular-value energy for output changes and raw gradients;
the dotted line marks 95\%. Middle: first-mode energy, first-ten-mode energy
and top-ten temporal subspace overlap with output at the selected stages.
Bottom: the leading output coefficient divided by the square root of its
coordinate count. Each coefficient uses one fixed pattern and sign convention
over its stated window; representation changes $A_s-A_{s-1}$ are placed at
completed step $s$. Raw-gradient rows are the undifferenced $g_{s-1}$ used by
the update into $s$. Predecessor rows for adjacent cosine are excluded from
the fits. Outputs are raw linear values for MLP and log probabilities for
ResNet and Transformer. No temporal centering, example averaging or row
normalization is applied to the primary SVD. Temporal overlap compares
time-domain subspaces, not identical neuron directions. Different probes and
window lengths preclude a controlled intrinsic-dimension comparison. Spectral
concentration alone is time-permutation invariant and establishes neither
causal origin nor spike specificity.}
'''


def rel(path): return path.relative_to(ROOT).as_posix()
def read(path): return json.loads(path.read_text(encoding='utf-8'))


def main():
    spec=importlib.util.spec_from_file_location('package_helpers',HERE.parent/'package_evidence.py')
    helper=importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    r=read(A63/'resnet_390_450_results.json'); output=next(s for s in r['stages'] if s['source_name']=='output')
    entry=r['stages'][0]; final=next(s for s in r['stages'] if s['source_name']=='layer3')
    notes=helper.rebase_markdown(PRIOR/'OBSERVATIONS.md',HERE/'OBSERVATIONS.md',{}).split('\n\n',1)[1]
    notes=notes.replace('[Current editable captions](../resnet_peak_revision_20261114/captions.tex)',
                        '[Current editable captions](captions.tex)')
    notes=notes.replace('Seven publication PDFs; existing measurements, no training or new captures.',
                        'Seven publication PDFs. The current ResNet temporal companion adds Run 292 captures from a verified replay; all other plotted measurements are retained.')
    notes=notes.replace('fit rows are unchanged. Only coefficient display coordinates move by +1. No cropped basis, refit, pooling, or sample averaging is introduced.',
                        'MLP/Transformer fits are unchanged. ResNet is freshly fitted over 61 incoming changes at completed steps 390--450. Coefficients use a fixed basis fitted to each whole stated window; no cropped basis, pooling or sample averaging is introduced.')
    notes=notes.replace('Other views retain the', 'The temporal companion now uses [its own generator](01_figure.py); the first five views retain the')
    notes=notes.replace('No new training or SVD is performed.', 'Figure 6 itself uses retained scalar records; the companion now adds the independently replay-qualified Run292 SVD for the same window.')
    notes=notes.replace('the separate SVD companion retains the original ResNet window 281--311.', 'the temporal companion now uses the same completed-step windows, including ResNet390--450.')
    begin=notes.index('## Companion:'); end=notes.index('## Verification and limits')
    notes=notes[:begin]+f'''## Companion: temporal structure and SVD

- **Current sources:** [new ResNet fit](../../../../analyses/063-2026-11-07-architecture-instability-tracing/observations/008-resnet-major-peak-temporal.md), [generator](01_figure.py), [manifest](figure_manifest.json), [verification](verification.json). MLP/Transformer preserve their previous arrays and panel pixels.
- **Same event windows as Figure 6:** completed steps **451--501 /390--450 /553--601**, with **51/61/49** fitted incoming displacement rows. Raw-gradient rows are the undifferenced gradients used for those updates: states450--500 /389--449 /552--600. Predecessors used for adjacent cosines do not enter the fits.
- **Fresh ResNet reduction:** exact FP64 Grams over all coordinates of the original1,000-image probe; six displayed endpoints, no averaging of layers/examples, no centering or primary row normalization. All187 replay losses match the native control exactly; the initial gradient and state296 parameter/gradient audit match bitwise. All six cosine/RMS histories match Figure 6.
- **Central message:** the larger excursion also has concentrated, shared temporal structure and reversals along a fixed output pattern. Output top-one/top-ten energy is **{100*output['top1']:.2f}%/{100*output['top10']:.2f}%**, with rank95 **{output['rank95']}/61**. First-convolution/final-stage output overlap is **{entry['temporal_top10_overlap']:.3f}/{final['temporal_top10_overlap']:.3f}**. The leading coefficient changes sign **{output['leading_sign_flips']}** times over60 adjacent pairs; amplitude and full-vector cosine determine their significance.
- **Caption:** Temporal structure of the same representation-change windows as Figure 6. Top: cumulative squared-singular-value energy for output changes and raw gradients, with95% reference. Middle: top-one/top-ten stage energy and top-ten temporal subspace overlap with output. Bottom: leading signed output coefficient divided by sqrt(P), using one fixed fitted pattern for the whole window. Steps, probe coverage and output spaces match Figure 6; all ResNet quantities are freshly refitted for390--450. Energy is squared movement, not explained loss; temporal overlap is shared time-domain structure, not identical neuron directions or causal propagation.

''' + '''## Verification and limits

- [Run292 and reduction](../../../../runs/292-2026-11-14-resnet-major-peak-tracing/verification.json): authenticated transfers, exact continuation qualification, independent raw-gradient Gram, all-stage scalar agreement, PSD/trace/reconstruction and spectral controls.
- [Current companion verification](verification.json): independent eigendecomposition, all plotted values, unchanged MLP/Transformer arrays and panel pixels, embedded fonts, page bounds and Poppler visual review.
- Figure 6 and every other existing PDF are byte-identical; all intervention files and the intervention PLAN section remain unchanged. The package checks manuscript/human input preservation.
- Run292 reconstructs one existing native trajectory to recover missing diagnostics; it is not an independent seed, intervention, causal test or held-out evaluation. Spectral fitting and event selection remain retrospective. The original281--311 ResNet fit and report remain historical background.
'''
    for old,new in [('Run292','Run 292'),('ResNet390','ResNet 390'),('original1,000','original 1,000'),
                    ('All187','All 187'),('state296','state 296'),('over60','over 60'),('with95%','with 95%'),
                    ('for390','for 390'),('original281','original 281')]:
        notes=notes.replace(old,new)
    (HERE/'OBSERVATIONS.md').write_text(notes,encoding='utf-8',newline='\n')
    captions=(PRIOR/'captions.tex').read_text(encoding='utf-8')
    old='each. The separate temporal-structure companion retains the original ResNet\nwindow 281--311; its SVD results do not describe this larger excursion.'
    assert old in captions
    captions=captions.replace(old,'each. The temporal-structure companion now uses the same windows; its ResNet\nSVD is freshly fitted to replay-qualified captures for steps 390--450.')
    captions=captions[:captions.index(r'\newcommand{\consolidationTemporalStructureCaption}')]+CAPTION
    (HERE/'captions.tex').write_text(captions,encoding='utf-8',newline='\n')
    catalog_path=HERE.parent/'evidence_catalog.json'; catalog=read(catalog_path)
    for item in catalog['assets']:
        target=item['destination']
        if target=='05-adam/b-tracing/figures/'+NAME:
            item.update(source=rel(HERE/'figures'/NAME),generator=rel(HERE/'01_figure.py'),observation=rel(OBS),note='Fresh ResNet390--450 SVD matches Figure6; all187 replay losses exact. MLP/Transformer unchanged.')
        elif target=='FIGURE_NOTES.md': item['source']=rel(HERE/'OBSERVATIONS.md')
        elif target=='figure_captions.tex': item.update(source=rel(HERE/'captions.tex'),observation=rel(HERE/'OBSERVATIONS.md'))
        elif target=='05-adam/b-tracing/data/spectral_controls.json':
            item.update(source=rel(A63/'spectral_controls_current.json'),generator=rel(A63/'12_resnet_major_svd.py'),observation=rel(OBS),note='Current displayed windows; ResNet controls refitted to390--450, original MLP/Transformer controls preserved.')
    catalog['sections']['05-adam/b-tracing']['presentation']='Both selected figures use completed steps451--501 /390--450 /553--601. The ResNet temporal SVD is freshly fitted to the same replay-qualified +311.53% excursion as Figure6. Original all-stage fits/reports remain in background/figures.'
    additions=[(OBS,'observations/63-008-resnet-major-peak-temporal.md','Matching larger-event temporal structure','current observation',A63/'12_resnet_major_svd.py'),
        (A63/'resnet_390_450_results.json','data/resnet_390_450_results.json','ResNet390--450 full temporal fit and exact Grams','current source record',A63/'12_resnet_major_svd.py'),
        (A63/'resnet_390_450_stages.csv','data/resnet_390_450_stages.csv','ResNet390--450 stage spectral summaries','current source record',A63/'12_resnet_major_svd.py'),
        (HERE/'figure_manifest.json','data/resnet_temporal_figure_manifest.json','Current temporal companion manifest','source record',HERE/'01_figure.py'),
        (HERE/'verification.json','data/resnet_temporal_figure_verification.json','Current temporal companion verification','verification',HERE/'02_verify.py')]
    for source,target,title,role,generator in additions:
        item=dict(section='05-adam/b-tracing',source=rel(source),destination='05-adam/b-tracing/'+target,title=title,role=role,observation=rel(OBS),generator=rel(generator),note='Fresh fit for Figure6\'s existing ResNet window; unchanged MLP/Transformer and intervention assets.')
        index=next((i for i,a in enumerate(catalog['assets']) if a['destination']==item['destination']),None)
        if index is None: catalog['assets'].append(item)
        else: catalog['assets'][index]=item
    for item in catalog['assets']:
        if item['destination'].startswith('05-adam/b-tracing/'):
            for key in ['title','note']:
                if key in item:
                    for old,new in [('ResNet390','ResNet 390'),('Figure6','Figure 6'),('all187','all 187'),('to390','to 390')]:
                        item[key]=item[key].replace(old,new)
    catalog['sections']['05-adam/b-tracing']['presentation']=catalog['sections']['05-adam/b-tracing']['presentation'].replace('steps451','steps 451').replace('/390','/ 390').replace('/553','/ 553').replace('Figure6','Figure 6')
    catalog_path.write_text(json.dumps(catalog,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
    evidence=ROOT/'manuscript/consolidation/evidence'
    readme=(evidence/'README.md').read_text(encoding='utf-8')
    lines=readme.splitlines()
    for index,line in enumerate(lines):
        if line.startswith('- **Contents:**'):
            lines[index]=f"- **Contents:** {len(catalog['assets'])} packaged assets from {len({a['source'] for a in catalog['assets']})} unique source files, plus four regenerated compact tables, four equation sets, two algorithms and eight bibliography additions."
        elif line.startswith('- **14 November figures:**'):
            lines[index]='- **14 November figures:** seven revised views are prepared and checked. Both selected ResNet tracing figures now use steps 390–450 around the peak at 415. The [fresh temporal fit and verification](05-adam/b-tracing/data/resnet_temporal_figure_verification.json) match the [loss/motion view](05-adam/b-tracing/data/resnet_peak_figure_verification.json); original fits remain historical. [Current notes](FIGURE_NOTES.md), [editable captions](figure_captions.tex). MLP/Transformer panels, original manuscript and intervention PDFs are preserved.'
    (evidence/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    artifacts=(evidence/'ARTIFACTS.md').read_text(encoding='utf-8').splitlines()
    for index,line in enumerate(artifacts):
        if line.startswith('- **Tracing companion:**'):
            artifacts[index]='- **Tracing companion:** [SVD, stage overlap and output coefficients](05-adam/b-tracing/figures/01-three-model-temporal-structure.pdf) now matches Figure 6: completed steps 451–501 / 390–450 / 553–601. All ResNet spectral quantities are freshly fitted from replay-qualified captures; MLP/Transformer are unchanged. The original detailed fits and three all-stage PDFs remain under `05-adam/b-tracing/background/figures/` and their canonical sources.'
    (evidence/'ARTIFACTS.md').write_text('\n'.join(artifacts)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(assets=len(catalog['assets']))))


if __name__=='__main__': main()
