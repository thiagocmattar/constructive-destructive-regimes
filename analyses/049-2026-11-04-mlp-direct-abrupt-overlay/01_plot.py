"""Direct overlay of two completed MLP schedules; no baseline curve or new training."""
import csv,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PREFIX=ROOT/'runs/223-2026-11-04-multispike-interventions/artifacts/spark/payload/worker'
DIRECT=ROOT/'runs/226-2026-11-04-mlp-cosine75/artifacts/spark/payload/worker'
ABRUPT=ROOT/'runs/224-2026-11-04-mlp-lr-return/artifacts/spark/payload/worker'
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    checked=0
    for root in [PREFIX,DIRECT,ABRUPT]:
        assert json.loads((root.parent.parent/'receipt.json').read_text())['verified']
        for item in json.loads((root.parent/'transfer_inventory.json').read_text())['files']:
            p=root.parent/item['path'];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'];checked+=1
    prefix=rows(PREFIX/'prefix/boundaries.jsonl')[:-1]
    paths={'direct075':DIRECT/'cosine32_three_quarters/boundaries.jsonl','half_then_abrupt075':ABRUPT/'cosine32_half_return075/boundaries.jsonl'}
    series={name:prefix+rows(p) for name,p in paths.items()}
    assert all([r['state'] for r in v]==list(range(801)) for v in series.values())
    metrics=[('loss','Training loss'),('rho',r'Raw gradient $\rho_t$'),('raw_gradient_norm',r'Raw gradient norm $\|g_t\|$'),('eta_kappa_hat',r'$\eta\widehat\kappa_t$'),('applied_lr','Applied LR / base LR')]
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
    fig,axes=plt.subplots(5,1,figsize=(8.8,10.8),sharex=True,layout='constrained');values=[]
    for ax,(metric,label) in zip(axes,metrics):
        for name,color,title in [('direct075','#0072B2','Cosine32 to 0.75x; held from 544'),('half_then_abrupt075','#AA3377','Cosine32 to 0.5x; abrupt rise to 0.75x at 650')]:
            data=[r for r in series[name] if 448<=r['state']<=800 and (metric!='applied_lr' or r['state']<800)]
            xx=[r['state'] for r in data];yy=[r[metric] if metric!='applied_lr' else r[metric]/.001 for r in data]
            values.extend(dict(series=name,metric=metric,state=x,value=y) for x,y in zip(xx,yy))
            if metric=='applied_lr':xx=xx+[800];yy=yy+[yy[-1]]
            ax.plot(xx,yy,color=color,lw=.95,ls='-',label=title,drawstyle='steps-post' if metric=='applied_lr' else 'default')
        ax.axvspan(448,512,color='#eeeeee',zorder=-3);ax.axvline(512,color='#444444',lw=.7);ax.axvline(650,color='#AA3377',lw=.65,alpha=.6)
        ax.set_ylabel(label);ax.set_xlim(448,800);ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    axes[0].set_yscale('log');axes[0].legend(loc='lower left',fontsize=8,framealpha=.95)
    axes[-1].set_xlabel('Completed optimizer updates');axes[-1].set_ylim(.45,1.05);axes[-1].set_xticks([448,512,544,608,650,720,800])
    fig.suptitle('MLP / CIFAR-10 | direct decay versus abrupt restoration\nBase LR 0.001 | seed 0 | same Spark | solid unsmoothed curves',fontsize=12)
    (HERE/'figures').mkdir(exist_ok=True);(HERE/'scratch').mkdir(exist_ok=True)
    pdf=HERE/'figures/mlp_direct075_vs_half_abrupt075.pdf';fig.savefig(pdf);plt.close(fig)
    import fitz
    doc=fitz.open(pdf);assert len(doc)==1 and not doc[0].get_images();doc[0].get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(HERE/'scratch/mlp_direct075_vs_half_abrupt075.png')
    for r in values:
        v=series[r['series']][r['state']][r['metric']]
        if r['metric']=='applied_lr':v/=.001
        assert r['value']==v
    with (HERE/'plotted_values.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    result=dict(status='completed',source_file_checks=checked,plotted_scalar_checks=len(values),displayed_series=list(series),pdf_sha256=sha(pdf),sources=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in [PREFIX/'prefix/boundaries.jsonl',*paths.values()]])
    (HERE/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
