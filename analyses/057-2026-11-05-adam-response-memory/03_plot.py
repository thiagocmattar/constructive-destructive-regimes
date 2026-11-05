"""Publication figures and tables from verified scalar reanalysis."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DEST=ROOT/'manuscript/draft/figures/analysis057'
TASKS=['mlp','resnet','transformer']
NAMES=['MLP','ResNet-32','Transformer']
COLORS=['#3172b2','#df8b2c','#32876c']
CURVES=[]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csvrows(n):return list(csv.DictReader((HERE/n).open()))
def curve(tag,x,y):
    x,y=np.asarray(x,dtype='<f8'),np.asarray(y,dtype='<f8')
    CURVES.append(dict(tag=tag,x=x.tolist(),y=y.tolist(),
        x_sha256=hashlib.sha256(x.tobytes()).hexdigest(),y_sha256=hashlib.sha256(y.tobytes()).hexdigest()))


def save(fig,name):
    fig.savefig(HERE/'figures'/name,bbox_inches='tight',pad_inches=.035,
                metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(HERE/'scratch'/name.replace('.pdf','.png'),dpi=180,bbox_inches='tight',pad_inches=.035)
    plt.close(fig)
    shutil.copy2(HERE/'figures'/name,DEST/name)


def main():
    for p in [HERE/'figures',HERE/'scratch',DEST]:p.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'axes.titlesize':9,
        'axes.labelsize':8,'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,
        'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    rows=csvrows('event_features.csv');fits=json.loads((HERE/'regressions.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(7.16,2.35),layout='constrained')
    for task,name,ax in zip(TASKS,NAMES,axes):
        rs=[r for r in rows if r['task']==task]
        for lr,col in zip(sorted({float(r['lr']) for r in rs}),COLORS[-len({r['lr'] for r in rs}):]):
            es=[r for r in rs if float(r['lr'])==lr]
            x=np.array([float(e['rho'])*float(e['A']) for e in es]);y=np.exp([float(e['log_loss_ratio']) for e in es])
            ax.scatter(x,y,s=13,c=col,alpha=.85,label=f'{lr:g} (n={len(es)})',linewidths=0)
            f=next(f for f in fits if f['task']==task and f['lr']==lr and f['model']=='rho*A' and f['scope']=='in_sample')
            xx=np.linspace(x.min(),x.max(),100);yy=np.exp(f['coefficients'][0]+f['coefficients'][1]*xx)
            ax.plot(xx,yy,'--',color=col,lw=1)
            curve(f'{task}:{lr}:points',x,y);curve(f'{task}:{lr}:fit',xx,yy)
        ax.set_title(name);ax.set_xlabel(r'Pre-rise $\rho$ $\times$ norm ratio $A$')
        ax.set_yscale('log');ax.set_yticks([1,2,4,8],['1','2','4','8']);ax.set_ylim(.95,11)
        ax.grid(alpha=.16);ax.legend(loc='upper right',frameon=False,handletextpad=.2)
    axes[0].set_ylabel('Peak / pre-rise loss')
    save(fig,'adam_severity_product_supported.pdf')
    models=['rho','rho*A','logA','rho+logA','rho+logA+rho*logA']
    labels=[r'$\rho$',r'$\rho A$',r'$\log A$',r'$\rho+\log A$',r'$\rho+\log A+\rho\log A$']
    styles=[('#9d9d9d','s'),('#3172b2','o'),('#df8b2c','^'),('#32876c','D'),('#8a5d9a','v')]
    fig,axes=plt.subplots(1,3,figsize=(7.16,2.5),layout='constrained')
    for task,name,ax in zip(TASKS,NAMES,axes):
        lrs=sorted({float(e['lr']) for e in rows if e['task']==task})
        for i,(model,(color,marker)) in enumerate(zip(models,styles)):
            x=np.arange(len(lrs))+(i-2)*.13
            y=[next(f['r2'] for f in fits if f['task']==task and f['lr']==lr and f['model']==model and f['scope']=='in_sample') for lr in lrs]
            ax.scatter(x,y,s=18,c=color,marker=marker,label=labels[i])
            curve(f'ablation:{task}:{model}',x,y)
        ticks=[f'{lr:g}\n(n={sum(e["task"]==task and float(e["lr"])==lr for e in rows)})' for lr in lrs]
        ax.set_xticks(np.arange(len(lrs)),ticks);ax.set_ylim(0,1.03);ax.set_yticks([0,.25,.5,.75,1])
        ax.grid(axis='y',alpha=.2);ax.set_title(name);ax.set_xlabel('Learning rate')
    axes[0].set_ylabel(r'Within-trajectory $R^2$')
    handles,leglabels=axes[0].get_legend_handles_labels()
    fig.legend(handles,leglabels,loc='outside lower center',ncol=5,frameon=False,columnspacing=.9)
    save(fig,'adam_severity_ablation.pdf')
    incidence=csvrows('event_incidence.csv')
    table=['\\begin{tabular}{llrrrrrrr}','\\toprule',
        r'Model & $\eta$ & $N_{5}$ & $N_{10}$ & $N_{20}$ & $C_{10}$ & Occupancy (\%) & $\rho_{\rm pre}<0$ & Mean $\Delta\mathcal L$ \\',r'\midrule']
    for task,name in zip(TASKS,NAMES):
        for lr in sorted({float(r['lr']) for r in incidence if r['task']==task}):
            es={float(r['threshold']):r for r in incidence if r['task']==task and float(r['lr'])==lr};e=es[.1]
            ratio=f"{e['negative_prerise']}/{e['defined_prerise']}" if int(e['defined_prerise']) else '--'
            absolute=f"{float(e['mean_absolute_peak_increase']):.4f}" if e['mean_absolute_peak_increase'] else '--'
            table.append(f"{name} & {lr:g} & {es[.05]['recovered']} & {e['recovered']} & {es[.2]['recovered']} & {e['censored']} & {100*float(e['occupancy']):.1f} & {ratio} & {absolute} \\\\")
    table+=['\\bottomrule','\\end{tabular}']
    out=ROOT/'manuscript/draft/tables/analysis057_incidence.tex'
    out.write_text('\n'.join(table)+'\n',encoding='utf-8',newline='\n')
    record=dict(generator_sha256=sha(HERE/'03_plot.py'),results_sha256=sha(HERE/'results.json'),
        source_hashes={n:sha(HERE/n) for n in ['event_features.csv','event_incidence.csv','regressions.json']},curves=CURVES,
        figures=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),manuscript_copy=(DEST/p.name).relative_to(ROOT).as_posix()) for p in sorted((HERE/'figures').glob('*.pdf'))],
        table=dict(path=out.relative_to(ROOT).as_posix(),sha256=sha(out)))
    (HERE/'figure_manifest.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('Generated two vector PDFs and nine-row incidence table.')


if __name__=='__main__':main()
