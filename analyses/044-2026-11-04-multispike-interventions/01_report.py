"""Run223 paired figures and source-audited multi-spike comparisons."""
import argparse,csv,hashlib,importlib.util,json,math,os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];RUN=ROOT/'runs/223-2026-11-04-multispike-interventions'
spec=importlib.util.spec_from_file_location('events',ROOT/'analyses/036-2026-11-02-controlled-adam-diagnostics/02_extract_loss_spikes.py');ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(folder):return rows(folder/'boundaries.jsonl')
def historical(case,sp):
    p=ROOT/'runs/215-2026-11-02-controlled-full-batch-adam/artifacts/attempt-01'/sp['worker']/('lr-'+format(sp['lr'],'g'))
    bs=rows(p/'boundaries.jsonl');ms={r['gradient_state']:r for r in rows(p/'metrics.jsonl')}
    out=[]
    for r in bs:
        if r['state']>sp['end']:break
        m=ms.get(r['state']);raw=m['streams']['raw'] if m else {}
        out.append(dict(state=r['state'],loss=r['loss'],raw_gradient_norm=r['gradient_norm'],eta_kappa_hat=r['projected_gradient_change'],rho=raw.get('cosine'),applied_lr=sp['lr'] if r['state']<sp['end'] else None))
    return out

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    cfg=read(RUN/'config.json');cohorts={};files=[]
    base=Path('\\\\?\\'+str((RUN/'artifacts').resolve())) if os.name=='nt' else RUN/'artifacts'
    for p in base.glob('*/payload/worker/manifest.json'):
        m=read(p)
        if m.get('status')!='completed':continue
        # Receipt is stored beside payload, not inside it.
        receipt=p.parents[2]/'receipt.json'
        if not receipt.exists():raise AssertionError('Missing verified retrieval: '+str(receipt))
        assert read(receipt)['verified'];cohorts.setdefault(m['case'],[]).append(p.parent)
    results=[];observed={};checks=0
    for case,sp in cfg['cases'].items():
        folders=cohorts.get(case,[]);leaders=[p for p in folders if (p/'READY.json').exists()]
        if not leaders:
            if args.partial:continue
            raise AssertionError('Missing complete leader '+case)
        assert len(leaders)==1;leader=leaders[0];prefix=canonical(leader/'prefix')
        assert len(prefix)==sp['start']+1 and read(leader/'event_gate.json')['passed']
        assert read(leader/'qualification/qualification.json')['status']=='passed'
        files.append(leader/'prefix/boundaries.jsonl')
        if read(leader/'historical_qualification.json')['reuse_historical']:
            control=historical(case,sp);control_label='historical trace reuse qualified'
        else:
            control=prefix[:-1]+canonical(leader/'control');control_label='fresh same-worker control'
            files.append(leader/'control/boundaries.jsonl')
        series={'unchanged':control}
        for arm in cfg['arms']:
            candidates=[p/arm['id'] for p in folders if (p/arm['id']/'end.pt').exists()]
            if len(candidates)!=1:
                if args.partial:continue
                raise AssertionError((case,arm['id'],len(candidates)))
            branch=candidates[0];data=canonical(branch)
            if branch.parent!=leader:
                gate=read(branch.parent/'cross_worker_qualification.json');ready=read(leader/'READY.json')
                assert gate['status']=='passed' and gate['fork_raw_gradient_exact'] and gate['two_update_replay_exact']
                assert gate['parent_sha256']==ready['parent_sha256'] and gate['reference_sha256']==ready['reference_sha256']
            assert [r['state'] for r in data]==list(range(sp['start'],sp['end']+1))
            assert data[0]['loss']==prefix[-1]['loss'] and data[0]['raw_gradient_norm']==prefix[-1]['raw_gradient_norm']
            applied=rows(branch/'metrics.jsonl');assert len(applied)==sp['end']-sp['start']
            for k,r in enumerate(applied):
                ratio=.5 if arm['kind']=='fixed' else .5+.25*(1+math.cos(math.pi*min(k,32)/32))
                assert abs(r['applied_lr']/sp['lr']-ratio)<1e-14;checks+=1
            series[arm['id']]=prefix[:-1]+data
            files.extend([branch/'boundaries.jsonl',branch/'metrics.jsonl'])
        observed[case]=(sp,series,control_label)
        ctrl=np.array([r['loss'] for r in control[sp['start']+1:sp['end']+1]])
        for name,data in series.items():
            losses=np.array([r['loss'] for r in data]);tail=losses[sp['start']+1:sp['end']+1]
            events=[]
            for e in ev.excursions(losses):
                if sp['start']<e['peak_state']<=sp['end']:
                    ratio=float(losses[e['peak_state']]/losses[e['before_state']])
                    if ratio>1.1:events.append(dict(e,peak_ratio=ratio))
            for i,r in enumerate(data):
                if i and r.get('rho') is not None and data[i-1]['raw_gradient_norm']:
                    expected=1-r['rho']*r['raw_gradient_norm']/data[i-1]['raw_gradient_norm']
                    assert abs(expected-r['eta_kappa_hat'])<1e-9;checks+=1
            results.append(dict(case=case,arm=name,start=sp['start'],end=sp['end'],control=control_label,mean_loss=float(tail.mean()),final_loss=float(tail[-1]),last32_mean=float(tail[-32:].mean()),mean_change_pct=float(100*(tail.mean()/ctrl.mean()-1)),final_change_pct=float(100*(tail[-1]/ctrl[-1]-1)),maximum_loss=float(losses[sp['start']:].max()),events_gt10=len(events),censored_gt10=sum(e['right_censored'] for e in events),largest_own_reference_ratio=max([e['peak_ratio'] for e in events],default=1),events=events))
    status='completed' if len(results)==9 else 'partial'
    out={'status':status,'source_checks':checks,'results':results,'source_files':[dict(path=Path(str(p).removeprefix('\\\\?\\')).relative_to(ROOT).as_posix(),sha256=sha(p)) for p in files]}
    (HERE/('partial_results.json' if args.partial else 'results.json')).write_text(json.dumps(out,indent=2)+'\n')
    if args.partial:print(json.dumps({'status':status,'cases':list(observed),'checks':checks}));return
    assert status=='completed'
    figdir=HERE/'figures';figdir.mkdir(exist_ok=True);values=[]
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
    metrics=[('loss','Training loss (log)'),('rho',r'Raw $\rho_t$'),('raw_gradient_norm',r'$\|g_t\|$'),('eta_kappa_hat',r'$\eta\widehat\kappa_t$'),('applied_lr','Applied/base LR')]
    labels={'fixed_half':'Immediate and permanent half','cosine32_half':'Cosine32 to half'}
    colors={'fixed_half':'#0072B2','cosine32_half':'#009E73'}
    titles={'mlp':'MLP | Spark GB10 | 5,000 images','resnet':'ResNet-32 | RTX4090 | 50,000 images','transformer':'Transformer | L40S | 2,088,600 targets'}
    plans=[(arm['id'],[(case,arm['id']) for case in observed]) for arm in cfg['arms']]
    plans.append(('all_results_preliminary',[(case,arm['id']) for case in observed for arm in cfg['arms']]))
    pdfs=[]
    for filename,panels in plans:
        fig,axes=plt.subplots(5,len(panels),figsize=(22 if len(panels)==6 else 13.5,11),sharex='col',layout='constrained')
        for col,(case,arm) in enumerate(panels):
            sp,series,label=observed[case]
            for row,(metric,title) in enumerate(metrics):
                ax=axes[row,col]
                for name,color,width in [('unchanged','#777777',1.0),(arm,colors[arm],.85)]:
                    data=[r for r in series[name] if sp['plot_start']<=r['state']<=sp['end'] and r.get(metric) is not None]
                    xx=[r['state'] for r in data];yy=[r[metric] if metric!='applied_lr' else r[metric]/sp['lr'] for r in data]
                    values.extend(dict(figure=filename,case=case,series=name,metric=metric,state=x,value=y) for x,y in zip(xx,yy))
                    if metric=='applied_lr':xx.append(sp['end']);yy.append(yy[-1])
                    ax.plot(xx,yy,color=color,lw=width,linestyle='-',label='Unchanged' if name=='unchanged' else labels[arm],drawstyle='steps-post' if metric=='applied_lr' else 'default')
                ax.axvline(sp['start'],color='#555555',lw=.7);ax.axvspan(sp['plot_start'],sp['start'],color='#eeeeee',zorder=-2);ax.grid(alpha=.14);ax.set_ylabel(title);ax.set_xlim(sp['plot_start'],sp['end']);ax.spines[['top','right']].set_visible(False)
                if metric=='loss':ax.set_yscale('log')
                if row==0:ax.set_title(titles[case]+'\n'+labels[arm]+' | base LR '+str(sp['lr']),fontsize=9);ax.legend(fontsize=7)
                if row==4:
                    ax.set_xlabel('Completed optimizer updates');ax.set_ylim(.45,1.06)
                    if case=='transformer' and arm=='cosine32_half':ax.text(sp['start']+8,.68,'Worker switch\n(exact replay)',fontsize=7)
        fig.suptitle('Run223 complete | '+('Both interventions' if len(panels)==6 else labels[filename])+'\nEach paired with a fresh unchanged control | seed 0 | solid unsmoothed curves | independent scales',fontsize=13)
        fig.supxlabel('Gray region: shared prefix; vertical line: intervention fork. Transformer cosine switches L40S workers at544 after exact replay.\nOther curves stay on one worker. Historical control reuse failed in all three cases; no historical bitwise replay is claimed.',fontsize=9)
        pdf=figdir/(filename+'.pdf');fig.savefig(pdf);plt.close(fig)
        import fitz
        doc=fitz.open(pdf);assert len(doc)==1 and not doc[0].get_images();(HERE/'scratch').mkdir(exist_ok=True);doc[0].get_pixmap(matrix=fitz.Matrix(1.25,1.25)).save(HERE/'scratch'/(filename+'.png'))
        pdfs.append(dict(path=pdf.relative_to(ROOT).as_posix(),sha256=sha(pdf)))
    for r in values:
        sp,series,_=observed[r['case']];source=series[r['series']][r['state']];v=source[r['metric']]
        if r['metric']=='applied_lr':v/=sp['lr']
        assert r['value']==v
    with (HERE/'figure_values.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    out.update(plotted_scalar_checks=len(values),pdfs=pdfs);(HERE/'results.json').write_text(json.dumps(out,indent=2)+'\n')
    combined=[r for r in values if r['figure']=='all_results_preliminary']
    with (HERE/'all_results_preliminary_values.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(combined[0]));writer.writeheader();writer.writerows(combined)
    (HERE/'all_results_preliminary.json').write_text(json.dumps(dict(status='completed',coverage={case:{arm:sp['end'] for arm in series} for case,(sp,series,_) in observed.items()},plotted_values_verified=len(combined),source_report='results.json',pdf_sha256=pdfs[-1]['sha256']),indent=2)+'\n')
    lines=['# Multi-spike intervention results','', 'Single seed, fixed full-objective windows. Historical controls are reused only where recorded qualification passes; otherwise comparisons use the fresh same-worker control. No historical intermediate-tensor identity is claimed.','', '| Case | Policy | Mean loss change | Final loss change | >10% excursions | Censored |','|---|---|---:|---:|---:|---:|']
    for r in results:lines.append(f"| {r['case']} | {r['arm']} | {r['mean_change_pct']:+.2f}% | {r['final_change_pct']:+.2f}% | {r['events_gt10']} | {r['censored_gt10']} |")
    lines+=['','All three figures show complete fixed windows, raw diagnostics, solid unsmoothed curves and independent Y ranges. Smaller oscillations alone do not establish better training progress. A finite window can leave delayed events censored.','',f"Verified {checks} source-derived schedule/diagnostic checks and retained {len(values)} plotted values. Every plotted scalar was checked against its source; vector PDFs were rendered for final visual review."]
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'status':status,'checks':checks,'plot_values':len(values)}))
if __name__=='__main__':main()
