"""Register verified H051 outputs in V-C, retaining all historical evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
BASE=ROOT/'manuscript/consolidation';VC=BASE/'evidence/05-adam/c-intervention';DATA=HERE/'matched';FIG=HERE/'figures/matched'
OBS=HERE/'observations/016-matched-intervention-matrix.md'
CAPTIONS={
 '17-matched-intervention-loss.pdf':('Figure 7: matched intervention training losses','prepared main',
    'Matched full-objective Adam continuations for CIFAR-10/MLP, CIFAR-10/ResNet-32 and WikiText-2/Transformer (columns). Rows change only the immediate learning rate, beta2, global raw-gradient clipping, or distributed affine-displacement cap. Gray is the same unchanged Adam control in every row of a column. All requested settings and every training state are shown without smoothing; loss axes are logarithmic and identical within each model. Each treatment starts from the complete common state at f=512/256/544 and stops at e=800/576/928. The dashed line marks the fork. One initialization per model; settings do not imply equal effective strength.'),
 '18-matched-spike-learning-tradeoff.pdf':('Figure 8: spike burden versus learning progress','prepared main',
    'Every intervention setting is compared with the common unchanged Adam baseline at the origin. Horizontal position is the percentage change in mean training loss over f+1 through e; vertical position is 100 times one minus the treatment-to-baseline ratio of positive loss variation, PV=sum max(L_s-L_(s-1),0). Leftward and upward changes are favorable. Family shapes/colors and setting labels retain the complete grid; large negative vertical values denote increased excursions. Axes are model-specific. The observations are paired trajectories at one initialization, not independent estimates with seed-level confidence intervals.'),
 '19-matched-q50-stage-heatmaps.pdf':('Figure 9: matched q50 loss and functional motion','prepared main',
    'Unchanged Adam versus the preselected distributed q50 affine-displacement cap at the common forks. Rows show full training loss, baseline/cap consecutive-change cosine, and baseline/cap displacement RMS at 5/6/6 measured stage endpoints. Changes are placed at their incoming endpoint s and cover f+1 through e. Both RMS maps use each stage\'s same maximum under unchanged Adam in that post-fork window; they are not separately normalized or divided by activation magnitude. The cosine scale is [-1,1]; the shared RMS scale includes all observed values. Dashed markers locate the peak of the baseline\'s largest relative loss excursion in the fixed window, using the same pre-rise reference and first-return rule as the outcome table. They are descriptive reference markers, not intervention triggers. MLP uses all 5000 examples; ResNet uses an intact 1000-image normalization group; Transformer uses 64 sequences/2240 positions. Outputs are raw MLP coordinates or cross-entropy log probabilities. No smoothing, SVD refit, or whole-network displacement bound is implied.'),
 '20-matched-all-setting-stage-heatmaps.pdf':('Supporting stage motion for every setting','prepared support',
    'Eleven pages retain the complete intervention grid, in the same LR, beta2, clipping and cap order as the loss matrix. Each page uses the Figure 9 layout, full fixed windows, the unchanged baseline, and the same baseline stage RMS denominators. RMS color limits are shared across the three models within each page and may differ between pages to include every value. Page headers identify the treatment. These maps test functional motion under all controls; they do not identify a unique causal source of instability.'),
 '21-matched-heldout-loss.pdf':('Full held-out test trajectories','prepared support',
    'Complete held-out test loss at f, f+32, midpoint, e-32 and e, with the same family/setting order as Figure 7. The fork evaluation is shared; every later evaluation is complete, with 10000 CIFAR-10 images or 245550 WikiText-2 test tokens. The Transformer\'s additional 217630-token validation results are retained in the accompanying data. Lines join only the five prespecified observations; no held-out checkpoint, setting or window selection is performed. Training improvements must be interpreted together with these held-out outcomes.'),
 '22-matched-update-and-controller-activity.pdf':('Applied movement and controller activity','prepared support',
    'Page 1 shows the actual stored parameter-increment norm for every treatment and the unchanged baseline, on logarithmic axes shared within each model. Page 2 shows the retained raw-gradient factor for clipping and the fraction of targeted units capped at each update for q25/q50/q75. Values are placed at the completed-update endpoint. Clipping acts before Adam; caps act on the native stored proposal while retaining its updated moments and clock. Actual trajectories diverge after treatment, so parameter norms alone are not matched functional strength. Component timings and memory are reported separately, without a cross-host method-speed ranking.'),
 '23-matched-intervention-outcomes.pdf':('All 36 matched intervention outcomes','prepared table',
    'One model per page, all 12 conditions, identical windows and full test coverage. PV reduction and mean-loss change are relative to unchanged Adam; absolute final and final-32-state mean losses preserve endpoint context. Green highlights the preselected q50 method rather than an outcome-selected winner. The accompanying CSV includes all event, activity, validation, endpoint and relative-effect columns. Undefined effects use N/A; there are no seed error bars from this initialization-0 study.')}


def text(p,value):p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((value.rstrip()+'\n').encode())
def rel(p):return p.relative_to(ROOT).as_posix()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def tex_caption(caption):
    escaped=caption.replace('%',r'\%').replace('_',r'\_').replace('&',r'\&')
    replacements={
        r'PV=sum max(L\_s-L\_(s-1),0)':r'$\mathrm{PV}=\sum_{s=f+1}^{e}\max(L_s-L_{s-1},0)$',
        'beta2':r'$\beta_2$', '[-1,1]':r'$[-1,1]$',
        'f=512/256/544':r'$f=512/256/544$', 'e=800/576/928':r'$e=800/576/928$',
        'f+1 through e':r'$f+1$ through $e$',
        'at f, f+32, midpoint, e-32 and e':r'at $f$, $f+32$, midpoint, $e-32$ and $e$'}
    for plain,latex in replacements.items():escaped=escaped.replace(plain,latex)
    return escaped


def package():
    results=json.loads((DATA/'results.json').read_text());verification=json.loads((DATA/'verification.json').read_text())
    figures=json.loads((FIG/'figure_manifest.json').read_text());visual=json.loads((FIG/'visual_verification.json').read_text())
    assert results['complete_matrix'] and results['conditions']==36 and verification['status']=='passed' and visual['status']=='passed'
    assert figures['complete_matrix'] and visual['complete_matrix']
    assert visual['figure_manifest_sha256']==sha(FIG/'figure_manifest.json')
    assert verification['result_sha256']==sha(DATA/'results.json')
    assert figures['results_sha256']==sha(DATA/'results.json')
    assert figures['generator_sha256']==sha(HERE/'61_plot_matched.py')
    for filename,digest in visual['editable_sources'].items():assert sha(FIG/filename)==digest
    for filename,record in results['outputs'].items():assert sha(DATA/filename)==record['sha256']
    for item in figures['outputs']:assert sha(FIG/item['file'])==item['sha256']
    summary=results['summary'];q50=[v for v in summary if v['arm']=='cap_q50']
    tracing=json.loads((DATA/'tracing_overlap.json').read_text())
    closeout=json.loads((DATA/'compute_closeout.json').read_text())
    assert closeout['status']=='passed', 'Verified retention and compute cleanup are required'
    assert closeout['source_result_sha256']==sha(DATA/'results.json')
    assert tracing['status']=='reproduced', 'Reconcile the historical ResNet tracing window first'
    assert tracing['current_curves_sha256']==sha(DATA/'curves.npz')
    interpretation=json.loads((DATA/'interpretation.json').read_text())
    assert interpretation['status']=='reviewed'
    assert interpretation['source_result_sha256']==sha(DATA/'results.json')
    conclusions=interpretation['conclusions']
    lines=['# H051: matched four-family intervention comparison','',
        '- **Status:** all 36 initialization-0 conditions completed; 11,904 post-fork updates, 11,940 full-gradient boundaries, 180 recorded held-out evaluation states and 108 exact gradient/update audits. Saved terminal states additionally reproduce all 36 training gradients/losses and all 48 full terminal held-out split losses exactly.',
        '- **Question:** at identical starting state and horizon, which controls reduce excursions while preserving learning and how does functional motion change?',
        '- **Comparison:** unchanged Adam plus immediate LR .75/.5/.1; beta2 .95/.9/.5; raw global L2 clipping .5/1; distributed cap q25/q50/q75. The preselected main representation-inspired method remains q50.',
        '- **Matched setup:** MLP 512-800 (5000-image full objective and probe); ResNet 256-576 (50000-image objective; intact 1000-image probe); Transformer 544-928 (2088600-token objective; 64-sequence/2240-position probe). Same full optimizer/RNG/data state and runtime within each model. Native prefixes start at initialization 0.',
        '- **Calibration:** fixed per-module unit-score quantiles over untreated outgoing states f-8 through f-1; no treatment outcome or held-out information enters the budgets.',
        '- **Runtime:** PyTorch 2.11.0; CUDA 13.0/GB10 for MLP and CUDA 12.8/RTX4090 or RTX5090 for ResNet/Transformer. Exact predecessor/fork and eight-update cross-worker identities are verified. These are newly qualified native trajectories, not assumed historical bitwise reproductions.','',
        '| Case: q50 versus Adam | PV reduction % | Mean loss change % | Final loss change % | Last-32 change % | Terminal test change % |',
        '|---|---:|---:|---:|---:|---:|']
    for v in q50:
        lines.append('| '+v['case']+' | '+' | '.join(f'{v[k]:+.2f}' for k in ['pv_reduction_percent','mean_loss_change_percent','final_loss_change_percent','last32_loss_change_percent','terminal_test_loss_change_percent'])+' |')
    lines += ['', '## Main findings', '']
    for key,title in [('LR_FINDING','Immediate LR'),('BETA_FINDING','Late beta2'),
                      ('CLIP_FINDING','Global raw-gradient clipping'),('CAP_FINDING','Cap quantile sensitivity'),
                      ('MOTION_FINDING','Functional motion'),('MSE_FINDING','Exact MLP loss accounting')]:
        lines.append('- **'+title+':** '+conclusions[key])
    if tracing['status']=='reproduced':
        lines += ['',f"- **Connection to V-B:** the new ResNet control exactly reproduces all {tracing['loss_states']} retained losses over states {tracing['loss_window'][0]}-{tracing['loss_window'][1]}. All {tracing['stage_values']} stage cosine/RMS values in the selected 390-450 tracing window agree within 1e-12. This is the same measured +311.53% excursion at 415. [Fixed-window comparison](../matched/tracing_overlap.json)."]
        lines += ['- **Intervention in the traced window:** '+conclusions['TRACING_CONNECTION']]
    lines += ['', '- **Full outcomes:** [36-row CSV](../matched/summary.csv), [readable table](../figures/matched/23-matched-intervention-outcomes.pdf), [editable TeX](../figures/matched/matched-outcomes.tex). All settings are retained, including unfavorable outcomes. [Editable component-cost table](../figures/matched/matched-costs.tex); all 36 timing rows remain in the data.',
        '- **Reduction:** primary mean excludes the shared fork; PV sums all positive post-fork loss increments. Final and last-32 losses are distinct. Excursions use the complete untreated prefix plus each treated continuation, the original fixed pre-rise reference, earliest maximum and first return; carried-in and right-censored events remain explicit. Event counts do not create independent replicates.',
        '- **Motion:** stage statistics reduce all fixed-probe coordinates. Squared-motion energy sums coordinate count times squared RMS; reversal energy uses negative consecutive-change cosine. Summaries exclude the common incoming fork change. Auxiliary correct-label log probabilities reproduce probe cross-entropy; MLP output-space loss decomposition is checked separately.',
        '- **Held-out coverage:** full CIFAR-10 test 10000; WikiText-2 validation 217630/test 245550 supervised tokens. Five fixed states per arm; the fork record is shared. There are 147 distinct evaluation states across the three model trajectories (196 full split evaluations), not 180 separately recomputed fork evaluations.',
        '- **Timing:** synchronized gradient, probe capture, optimizer, controller and evaluation components plus remaining wall time. Probe time covers the forward capture and cap inputs; cosine/RMS reductions, other scalar diagnostics, logging and checkpoint I/O lie in remaining wall time. Controller time alone is not the full deployable method overhead. Memory is the process high-water allocation through each arm, an upper bound rather than an isolated per-arm peak. GPU class is fixed within model, but cloud hosts differ. No efficiency advantage is claimed.',
        '- **Uncertainty:** one initialization/model and one prespecified continuation window. No seed confidence intervals, independent-step tests, universal optimizer claim, or selection of a favorable endpoint. Settings have unequal effective strength/tuning grids. The cap bounds direct affine displacement, not total propagated motion or loss.',
        '- **Adversarial interpretation:** post-fork trajectory changes establish controlled intervention effects in these cases; they do not establish that representation reversal is the unique cause of spikes. Retained-moment late beta2 switching also changes native bias correction and is not a test of training from initialization with lower beta2. The study does not compare against AGC, FOOF or Kosson\'s method.',
        '- **Deferred:** initializations 1/2 and cap x LR x initialization remain separate TODOs with previously recorded ETC/costs; neither is run or needed to report this bounded comparison.',
        '- **Provenance:** [source hashes and reductions](../matched/results.json), [independent checks](../matched/verification.json), [figure coordinates and sources](../figures/matched/figure_manifest.json), [visual review](../figures/matched/visual_verification.json). Canonical raw tensors/histories remain in Runs293-295 on owned Spark, with run-level retention inventories.',
        '- **Reproduction:** `60_reduce_matched.py`, `64_verify_matched_reductions.py`, `61_plot_matched.py`, then `62_package_matched.py`; source training worker is `src/eos_tasks/matched_interventions.py`. Raw replay audit: `59_verify_matched.py`.','## Figure captions','']
    caption_tex=[]
    for filename,(title,role,caption) in CAPTIONS.items():
        lines += ['### '+title,'','- **Asset:** ['+filename+'](../figures/matched/'+filename+').','- **Caption:** '+caption,'']
        caption_tex += ['% '+filename,r'\caption{'+tex_caption(caption)+'}','']
    text(OBS,'\n'.join(lines));text(FIG/'captions.tex','\n'.join(caption_tex))
    catpath=BASE/'work/evidence_catalog.json';cat=json.loads(catpath.read_text(encoding='utf-8'))
    section='05-adam/c-intervention';moved=[]
    for item in cat['assets']:
        if item['section']==section and Path(item['destination']).parent.as_posix()==section+'/figures' and '/figures/matched/' not in item['source']:
            old=item['destination'];item['destination']=section+'/background/figures/'+Path(old).name
            item['title']='Historical: '+item['title']
            item['role']='historical support';item['note']='Historical comparison; superseded by the common-fork H051 matrix. '+item['note'];moved.append((old,item['destination']))
    # Repeated calls update these registrations, without duplicating them.
    cat['assets']=[a for a in cat['assets'] if not a.get('h051')]
    def add(source,destination,title,role,generator):
        cat['assets'].append(dict(section=section,source=rel(source),destination=section+'/'+destination,
            title=title,role=role,observation=rel(OBS),generator=rel(generator),note='H051: same forks, full fixed windows and all 36 initialization-0 conditions; sources and checks retained.',h051=True))
    for filename,(title,role,_) in CAPTIONS.items():add(FIG/filename,'figures/'+filename,title,role,HERE/'61_plot_matched.py')
    add(OBS,'observations/65-016-matched-intervention-matrix.md','Matched outcomes, methods, captions and limits','current observation',HERE/'62_package_matched.py')
    for filename in ['summary.csv','events.csv','stage_metrics.csv','heldout.csv','costs.csv','calibration.csv','calibration.npz','curves.npz','results.json','verification.json','compute_closeout.json','tracing_overlap.json','interpretation.json']:
        destination='tables/matched-outcomes.csv' if filename=='summary.csv' else 'data/matched-'+filename
        generator={'verification.json':'64_verify_matched_reductions.py','compute_closeout.json':'70_close_matched_compute.py','tracing_overlap.json':'71_check_matched_tracing.py','interpretation.json':'62_package_matched.py'}.get(filename,'60_reduce_matched.py')
        add(DATA/filename,destination,'Matched '+filename,'current source record',HERE/generator)
    for filename in ['figure_manifest.json','visual_verification.json','captions.tex','matched-outcomes.tex','matched-costs.tex']:
        destination=('tables/' if filename in ['matched-outcomes.tex','matched-costs.tex'] else 'data/' if filename.endswith('.json') else '')+('matched-'+filename if filename.endswith('.json') else filename)
        add(FIG/filename,destination,'Matched '+filename,'current publication source',HERE/'61_plot_matched.py' if filename!='captions.tex' else HERE/'62_package_matched.py')
    cat['sections'][section].update(claim='A matched comparison of LR, beta2, raw-gradient clipping and local affine-displacement caps separates excursion burden, learning progress and functional motion.',
        boundary='One initialization/model and unequal effective strengths. Retain all endpoints and held-out tradeoffs; local displacement bounds do not imply global stability, generalization or universal superiority.',
        presentation='H051 complete: 36 verified conditions. Main views are the four-family loss matrix, spike/learning tradeoff and matched q50 stage heatmaps. All-setting motion, full test trajectories, activity and outcomes accompany them. Historical figures remain in background/figures; seeds1/2 and cap x LR stay deferred.')
    text(catpath,json.dumps(cat,indent=2,ensure_ascii=False))
    for old,new in moved:
        source=BASE/'evidence'/old;dest=BASE/'evidence'/new
        if source.exists():
            dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():assert sha(source)==sha(dest)
            else:dest.write_bytes(source.read_bytes())
            assert sha(source)==sha(dest);source.unlink()
    text(VC/'figures/README.md','\n'.join(['# Matched intervention figures','',
        '- **All 36 conditions are complete and verified.** Initialization 0; common starts 512/256/544 and ends 800/576/928. All families retain unchanged Adam.',
        '- **Main reading order:** training loss matrix, spike/learning tradeoff, then the q50 stage heatmaps. The loss matrix preserves LR -> beta2 -> clipping -> cap order.',
        '- **Supporting pages:** every setting\'s stage motion, full held-out test trajectories, applied-update/controller activity, and a three-page outcome table.','',
        '| Use | PDF |','|---|---|']+['| '+title+' | ['+filename+']('+filename+') |' for filename,(title,_,_) in CAPTIONS.items()]+['',
        '- **Captions, reductions and limitations:** [current observation](../observations/65-016-matched-intervention-matrix.md).',
        '- **Historical assets:** [background](../background/README.md). The older quantile/LR ablation remains at the section root.',
        '- **Deferred experiments:** [TODO](../TODO_ABLATIONS.md).']))
    subprocess.run([sys.executable,str(BASE/'work/package_evidence.py'),'--build'],cwd=ROOT,check=True)
    print(json.dumps(dict(status='packaged',new_assets=sum(bool(v.get('h051')) for v in cat['assets']),historical_figures_archived=len(moved))))


if __name__=='__main__':package()
