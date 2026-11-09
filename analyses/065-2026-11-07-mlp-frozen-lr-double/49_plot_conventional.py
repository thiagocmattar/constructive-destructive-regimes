"""Complete three-model V-C beta2 and clipping figures from verified frozen runs."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterSciNotation, LogLocator, MaxNLocator, NullFormatter
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SOURCES={};CURVES=[];SUMMARY=[];HISTORICAL={};CONTROL_CONTEXT=[]
GRAY='#777777';BLUE='#0072B2';ORANGE='#D55E00';GREEN='#009E73'
START={'mlp':512,'resnet':256,'transformer':544};END={'mlp':800,'resnet':576,'transformer':928}
BEGIN={'mlp':448,'resnet':192,'transformer':480}
TITLE={'mlp':'CIFAR-10 / MLP','resnet':'CIFAR-10 / ResNet-32','transformer':'WikiText-2 / Transformer'}


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def source(p):
    rel=p.relative_to(ROOT).as_posix();digest=sha(p)
    if rel in HISTORICAL:assert digest==HISTORICAL[rel],('Historical source changed',rel)
    SOURCES[rel]=dict(bytes=p.stat().st_size,sha256=digest);return p


def read(p):return json.loads(source(p).read_text(encoding='utf-8'))
def rows(p):return [json.loads(x) for x in source(p).read_text(encoding='utf-8').splitlines() if x.strip()]
def run(n):return next((ROOT/'runs').glob(f'{n:03d}-*'))
def old(n):return run(n)/'artifacts/spark/payload/worker'
def new(n,name):return run(n)/'artifacts/attempt-01'/name


def check_new():
    control=read(new(290,'trans-control')/'qualification.json');result=[]
    for n,name in [(289,'resnet'),(290,'trans-control'),(290,'trans-b95'),(290,'trans-b90'),(290,'trans-b50'),(291,'trans-clip05'),(291,'trans-clip10')]:
        p=new(n,name);v=read(p/'verification.json');q=read(p/'qualification.json');a=read(p/'complete_prefix_audit.json');assert v['status']==a['status']=='passed'
        manifest=read(p/'manifest.json');assert manifest['status']=='completed'
        assert sha(source(run(n)/'config.json'))==manifest['config_sha256']
        assert sha(source(run(n)/'input_manifest.json'))==manifest['manifest_sha256']
        inventory=read(p/'transfer_inventory.json')
        for item in inventory['files']:
            local=p/item['path']
            if local.exists():assert local.stat().st_size==item['bytes'] and sha(local)==item['sha256'],local
        assert {r['arm'] for r in a['results']}=={r['arm'] for r in v['arms']}
        assert all(r['initial_model_moments_clock_rng_previous_gradient_exact'] for r in a['results'])
        if n!=289:
            for key in ['fork_fingerprint','eight_step_fingerprint','environment','eight_step_trace']:
                assert q[key]==control[key],(name,key,'Unmatched control: do not plot')
        expected_units={'resnet':{'test':(10000,10)},'transformer':{'valid':(217630,6220),'test':(245550,7020)}}['resnet' if n==289 else 'transformer']
        for arm in v['arms']:
            for split,(units,objectives) in expected_units.items():
                assert arm['evaluation'][split]['loss_units']==units
                assert arm['evaluation'][split]['objective_units']==objectives
        result.append(dict(run=n,worker=name,status='passed',qualification=q,verification=v,original_fork_audit=a))
    return result


def load_branch(p):
    b=rows(p/'boundaries.jsonl');updates=rows(p/'metrics.jsonl');u={r['state']:r for r in updates}
    assert len(updates)==len(u) and [r['state'] for r in updates]==[r['state'] for r in b[:-1]]
    for r in b:
        m=u.get(r['state'],{});r.update(clip_factor=m.get('clip_factor',1. if m else None),optimizer_input_norm=m.get('optimizer_input_norm'),beta2=m.get('beta2',r.get('beta2')))
    return b,read(p/'status.json')


def branches(family,case):
    if family=='beta2':
        if case=='mlp':
            paths={'control':old(223)/'control',**{a:old(232)/a for a in ['beta2_0p95','beta2_0p9','beta2_0p5']}}
        elif case=='resnet':paths={a:new(289,'resnet')/a for a in ['control','beta2_0p95','beta2_0p9','beta2_0p5']}
        else:paths={a:new(290,w)/a for a,w in [('control','trans-control'),('beta2_0p95','trans-b95'),('beta2_0p9','trans-b90'),('beta2_0p5','trans-b50')]}
    elif case=='mlp':paths={'control':old(223)/'control','clip_1p0':old(228)/'global_clip1','clip_0p5':old(229)/'global_clip05'}
    elif case=='resnet':paths={'control':old(231)/'control','clip_1p0':old(231)/'clipping'}
    else:paths={'control':new(290,'trans-control')/'control','clip_1p0':new(291,'trans-clip10')/'clip_1p0','clip_0p5':new(291,'trans-clip05')/'clip_0p5'}
    return {a:load_branch(p) for a,p in paths.items()}


def historical(case,full=False):
    host={'mlp':'spark','resnet':'duqoihms8w1gle','transformer':'159vjccj7w3996'}[case]
    p=run(223)/f'artifacts/{host}/payload/worker/prefix/boundaries.jsonl'
    return [r for r in rows(p) if (0 if full else BEGIN[case])<=r['state']<START[case]]


def characterize_control(family,case,series):
    """Descriptive portability check; never a criterion for retaining an outcome."""
    path=source(ROOT/'analyses/036-2026-11-02-controlled-adam-diagnostics/02_extract_loss_spikes.py')
    spec=importlib.util.spec_from_file_location('loss_excursions',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    trace=historical(case,full=True)+series
    assert [r['state'] for r in trace]==list(range(END[case]+1))
    loss=[r['loss'] for r in trace];events=[]
    for event in module.excursions(loss):
        if event['peak_state']<=START[case]:continue
        event.update(peak_loss=loss[event['peak_state']],before_loss=loss[event['before_state']],
                     peak_to_before_ratio=loss[event['peak_state']]/loss[event['before_state']])
        events.append(event)
    target={'mlp':703,'resnet':422,'transformer':794}[case]
    CONTROL_CONTEXT.append(dict(family=family,model=case,events=events,
        excursions_over_10_percent=sum(e['peak_to_before_ratio']>1.1 for e in events),
        excursions_at_least_double=sum(e['peak_to_before_ratio']>=2 for e in events),
        historical_target_state=target,
        double_excursion_within_32_states_of_historical_target=any(
            e['peak_to_before_ratio']>=2 and abs(e['peak_state']-target)<=32 for e in events),
        interpretation='Descriptive check of the new control against the historical episode; no outcome exclusion or new acceptance threshold.'))


def summarize(family,case,data):
    characterize_control(family,case,data['control'][0])
    baseline=np.array([r['loss'] for r in data['control'][0] if START[case]<r['state']<=END[case]])
    for arm,(series,status) in data.items():
        series=[r for r in series if START[case]<=r['state']<=END[case]]
        assert [r['state'] for r in series]==list(range(START[case],END[case]+1))
        losses=np.array([r['loss'] for r in series[1:]])
        assert np.isfinite(losses).all() and (losses>0).all()
        test=status['evaluation']['test']['loss'];base_test=data['control'][1]['evaluation']['test']['loss']
        record=dict(family=family,model=case,arm=arm,seed=0,fork=START[case],end=END[case],
            states=len(losses),mean_loss=float(losses.mean()),late32_loss=float(losses[-32:].mean()),
            final_loss=float(losses[-1]),maximum_loss=float(losses.max()),
            mean_change_percent=float(100*(losses.mean()/baseline.mean()-1)),
            late32_change_percent=float(100*(losses[-32:].mean()/baseline[-32:].mean()-1)),
            maximum_change_percent=float(100*(losses.max()/baseline.max()-1)),
            test_loss=float(test),test_change_percent=float(100*(test/base_test-1)),
            clipping_fraction=float(np.mean([r['clip_factor']<1 for r in series[:-1]])) if family=='clipping' else None,
            evaluation=status.get('evaluation',{}))
        SUMMARY.append(record)
        for r in series:
            CURVES.append(dict(family=family,model=case,arm=arm,**{k:r.get(k) for k in ['state','loss','raw_gradient_norm','rho','eta_kappa_hat','optimizer_input_norm','clip_factor','beta2']}))


def plot(family):
    names=['control','beta2_0p95','beta2_0p9','beta2_0p5'] if family=='beta2' else ['control','clip_1p0','clip_0p5']
    labels=['Unchanged Adam',r'$\beta_2=0.95$',r'$\beta_2=0.9$',r'$\beta_2=0.5$'] if family=='beta2' else ['Unclipped Adam','Global clip 1.0','Global clip 0.5']
    colors=dict(zip(names,[GRAY,BLUE,ORANGE,GREEN]));metric=['loss','eta_kappa_hat','rho','raw_gradient_norm','beta2' if family=='beta2' else 'clip_factor']
    ylabels=['Training loss',r'$\eta\widehat\kappa_s$',r'$\rho_s$',r'$\|g_s\|$',r'$\beta_2$' if family=='beta2' else 'Gradient scale']
    plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix','font.size':8.5,'axes.titlesize':9,'axes.labelsize':8.5,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'pdf.fonttype':42})
    fig,axes=plt.subplots(5,3,figsize=(7.16,6.35),sharex='col')
    fig.subplots_adjust(left=.085,right=.985,top=.88,bottom=.13,hspace=.17,wspace=.30)
    for col,case in enumerate(['mlp','resnet','transformer']):
        data=branches(family,case);summarize(family,case,data);prefix=historical(case)
        for row,(key,label) in enumerate(zip(metric,ylabels)):
            ax=axes[row,col];ax.axvspan(BEGIN[case],START[case],color='#eeeeee',zorder=-5);ax.axvline(START[case],color='#999999',lw=.65,ls=':')
            py=[(.999 if family=='beta2' else 1.) if row==4 else r.get(key) for r in prefix]
            ax.plot([r['state'] for r in prefix],py,color=GRAY,lw=.65,alpha=.6)
            for arm in names:
                if arm not in data:continue
                series=data[arm][0];series=[r for r in series if START[case]<=r['state']<=END[case]]
                if row==4:series=[r for r in series if r['state']<END[case]]
                if row==4 and family=='beta2':
                    value=.999 if arm=='control' else float(arm.removeprefix('beta2_').replace('p','.'));yy=[value]*len(series)
                else:yy=[r.get(key) for r in series]
                ax.plot([r['state'] for r in series],yy,color=colors[arm],lw=.75,alpha=.95)
                if family=='clipping' and key=='raw_gradient_norm' and arm!='control':
                    ax.plot([r['state'] for r in series],[r.get('optimizer_input_norm') for r in series],color=colors[arm],lw=.7,ls='--')
            if key in ['loss','raw_gradient_norm']:ax.set_yscale('log')
            else:ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            if key=='raw_gradient_norm' and case=='mlp':
                ax.yaxis.set_major_locator(LogLocator(base=10,subs=[1,2,5]))
                ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10,labelOnlyBase=False,minor_thresholds=(np.inf,np.inf)))
                ax.yaxis.set_minor_formatter(NullFormatter())
            if key=='rho':ax.set_ylim(-1.08,1.08);ax.set_yticks([-1,0,1]);ax.axhline(0,color='#aaaaaa',lw=.5)
            if key=='eta_kappa_hat':ax.axhline(2,color='#aaaaaa',lw=.5,ls='--')
            if row==4:
                ax.set_ylim((.45,1.035) if family=='beta2' else (-.04,1.08))
                ax.set_yticks([.5,.75,1] if family=='beta2' else [0,.5,1]);ax.set_xlabel('Completed updates s')
            ax.grid(alpha=.16,lw=.45);ax.spines[['top','right']].set_visible(False);ax.set_xlim(BEGIN[case],END[case]);ax.xaxis.set_major_locator(MaxNLocator(nbins=4,integer=True))
            if col==0:ax.set_ylabel(label)
            if row==0:ax.set_title(TITLE[case],pad=7)
        if family=='clipping':axes[3,col].text(.98,.98,'Dashed: optimizer input',ha='right',va='top',transform=axes[3,col].transAxes,fontsize=6)
        if family=='clipping' and case=='resnet':axes[4,col].text(.02,.12,'Clip 0.5: not yet run',transform=axes[4,col].transAxes,fontsize=6.5)
    fig.suptitle('Native Adam second-moment reductions' if family=='beta2' else 'Global gradient clipping before Adam moments',fontsize=11,y=.985)
    fig.legend([Line2D([0],[0],color=colors[a],lw=1.2) for a in names],labels,loc='upper center',bbox_to_anchor=(.53,.961),ncol=len(names),frameon=False,fontsize=8)
    foot='One initialization per model. Shaded: historical prefix. Each treatment uses its matched unchanged control.\nLoss/gradient at state s precede update t=s+1. Raw-gradient secant is not an Adam stability bound. No smoothing.'
    fig.text(.085,.018,foot,fontsize=6.6,va='bottom')
    filename=('15-beta2-reductions-cifar10-mlp-resnet-wikitext2-transformer.pdf' if family=='beta2' else '16-gradient-clipping-cifar10-mlp-resnet-wikitext2-transformer.pdf')
    p=HERE/'figures'/filename;fig.savefig(p);plt.close(fig)
    scratch=ROOT/'tmp/pdfs/h050';scratch.mkdir(parents=True,exist_ok=True)
    with fitz.open(p) as pdf:
        assert len(pdf)==1;pdf[0].get_pixmap(matrix=fitz.Matrix(2,2)).save(scratch/(p.stem+'.png'))
    return dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),pages=1)


def main():
    old58=read(ROOT/'analyses/058-2026-11-06-intervention-consolidation/manifest.json')
    old59=read(ROOT/'analyses/059-2026-11-07-mlp-beta2-reduction/results.json')
    HISTORICAL.update({k:v['sha256'] for k,v in old58['sources'].items()})
    HISTORICAL.update(old59['source_sha256'])
    for n in [228,229,232]:assert read(run(n)/'verification_final.json')['status']=='passed'
    checks=check_new();figures=[plot(f) for f in ['beta2','clipping']]
    out=HERE/'conventional_curves.csv'
    with out.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(CURVES[0]));writer.writeheader();writer.writerows(CURVES)
    (HERE/'conventional_results.json').write_text(json.dumps(dict(status='passed',summaries=SUMMARY,control_context=CONTROL_CONTEXT,checks=checks,sources=SOURCES,figures=figures,
        exported_branch_boundary_rows=len(CURVES),historical_prefix_boundary_rows=2*sum(START[c]-BEGIN[c] for c in START),
        display_clock='Branch CSV includes fork and terminal boundaries. Policy rows omit the terminal boundary because it has no outgoing update; each shaded prefix is displayed once per model/family.'),indent=2)+'\n',encoding='utf-8',newline='\n')
    with (HERE/'conventional_summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=[k for k in SUMMARY[0] if k!='evaluation']);writer.writeheader();writer.writerows({k:v for k,v in r.items() if k!='evaluation'} for r in SUMMARY)
    print(json.dumps(dict(status='passed',figures=figures,rows=len(CURVES))))


if __name__=='__main__':main()
