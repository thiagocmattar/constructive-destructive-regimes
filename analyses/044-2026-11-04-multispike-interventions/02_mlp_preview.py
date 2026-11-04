"""Verified MLP-only preview; no training and no change to the active cloud suite."""
import csv,hashlib,importlib.util,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SRC=ROOT/'runs/223-2026-11-04-multispike-interventions/artifacts/spark/payload/worker'
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    receipt=json.loads((SRC.parent.parent/'receipt.json').read_text());assert receipt['verified']
    inventory=json.loads((SRC.parent/'transfer_inventory.json').read_text())
    for item in inventory['files']:
        p=SRC.parent/item['path'];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
    prefix=rows(SRC/'prefix/boundaries.jsonl')[:-1]
    series={name:prefix+rows(SRC/folder/'boundaries.jsonl') for name,folder in [('unchanged','control'),('fixed_half','fixed_half'),('cosine32_half','cosine32_half')]}
    for name,data in series.items():
        assert [r['state'] for r in data]==list(range(801))
        assert data[512]['loss']==series['unchanged'][512]['loss']
    spec=importlib.util.spec_from_file_location('events',ROOT/'analyses/036-2026-11-02-controlled-adam-diagnostics/02_extract_loss_spikes.py');events=importlib.util.module_from_spec(spec);spec.loader.exec_module(events)
    summary={};control=np.array([r['loss'] for r in series['unchanged']])
    for name,data in series.items():
        loss=np.array([r['loss'] for r in data]);exc=[dict(e,ratio=float(loss[e['peak_state']]/loss[e['before_state']])) for e in events.excursions(loss) if e['peak_state']>512 and loss[e['peak_state']]>1.1*loss[e['before_state']]]
        summary[name]=dict(mean_loss=float(loss[513:].mean()),final_loss=float(loss[-1]),last32_mean=float(loss[-32:].mean()),mean_delta_pct=float(100*(loss[513:].mean()/control[513:].mean()-1)),last32_delta_pct=float(100*(loss[-32:].mean()/control[-32:].mean()-1)),events_gt10=exc)
    metrics=[('loss','Training loss'),('rho',r'Raw gradient $\rho_t$'),('raw_gradient_norm',r'Raw gradient norm $\|g_t\|$'),('eta_kappa_hat',r'$\eta\widehat\kappa_t$'),('applied_lr','Applied LR / base LR')]
    plt.rcParams.update({'font.size':10,'axes.titlesize':12,'pdf.fonttype':42})
    fig,axes=plt.subplots(5,2,figsize=(11.5,10.7),sharex='col',layout='constrained');values=[]
    for col,(arm,label,color) in enumerate([('fixed_half','Immediate and permanent 0.5x','#0072B2'),('cosine32_half','Cosine32 to 0.5x, then held','#009E73')]):
        for row,(metric,ylabel) in enumerate(metrics):
            ax=axes[row,col]
            for name,c,lw in [('unchanged','#666666',1.05),(arm,color,.9)]:
                data=[r for r in series[name] if 448<=r['state']<=800 and (metric!='applied_lr' or r['state']<800)]
                xx=[r['state'] for r in data];yy=[r[metric] if metric!='applied_lr' else r[metric]/.001 for r in data]
                values.extend(dict(column=arm,series=name,metric=metric,state=x,value=y) for x,y in zip(xx,yy))
                if metric=='applied_lr':xx=xx+[800];yy=yy+[yy[-1]] # close the last applied interval; not another update
                ax.plot(xx,yy,color=c,lw=lw,ls='-',label='Unchanged' if name=='unchanged' else label,drawstyle='steps-post' if metric=='applied_lr' else 'default')
            ax.axvspan(448,512,color='#eeeeee',zorder=-3);ax.axvline(512,color='#333333',lw=.8)
            ax.grid(alpha=.15);ax.set_ylabel(ylabel);ax.set_xlim(448,800);ax.spines[['top','right']].set_visible(False)
            if row==0:
                ax.set_yscale('log');ax.set_title(label);ax.legend(loc='lower left',fontsize=8,framealpha=.94)
                for state in [522,683,716,799]:
                    v=series['unchanged'][state]['loss'];ax.plot(state,v,'o',color='#666666',ms=3)
                    ax.annotate(str(state),(state,v),xytext=(-8 if state==799 else 0,5),textcoords='offset points',fontsize=8,ha='right' if state==799 else 'center',color='#555555')
            if row==4:ax.set_xlabel('Completed optimizer updates');ax.set_ylim(.45,1.06)
    fig.suptitle('MLP / CIFAR-10 | base LR = 0.001 | seed 0\nIntervention at state 512; matched Spark control; unsmoothed observations',fontsize=13)
    out=HERE/'figures';out.mkdir(exist_ok=True);scratch=HERE/'scratch';scratch.mkdir(exist_ok=True)
    pdf=out/'mlp_preliminary_paired.pdf';fig.savefig(pdf);plt.close(fig)
    import fitz
    doc=fitz.open(pdf);doc[0].get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(scratch/'mlp_preliminary_paired.png')
    with (HERE/'mlp_preliminary_values.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(values[0]));wr.writeheader();wr.writerows(values)
    # Direct source lookup audit for every plotted observation.
    for r in values:
        v=series[r['series']][r['state']][r['metric']]
        if r['metric']=='applied_lr':v/=.001
        assert r['value']==v
    result=dict(status='MLP_completed_other_cases_pending',summary=summary,source_file_checks=len(inventory['files']),plotted_scalar_checks=len(values),pdf_sha256=sha(pdf),source_files=[dict(path=(SRC/k/'boundaries.jsonl').relative_to(ROOT).as_posix(),sha256=sha(SRC/k/'boundaries.jsonl')) for k in ['prefix','control','fixed_half','cosine32_half']])
    (HERE/'mlp_preliminary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['summary'],indent=2));print('Verified plotted scalars',len(values))
if __name__=='__main__':main()
