# Three baseline/intervention columns: raw rho, solid lines, independent Y axes.
import csv, hashlib, json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import fitz
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SOURCE=ROOT/'runs/220-2026-11-03-mlp-lr-cooldown/artifacts/attempts/002-audit-continuation/conditions'
INTERVENTIONS=('always_half','hold8','hold8_cosine8')
TITLES=('Half learning rate','8-update cooldown','Cooldown + cosine recovery')
COLORS=('#0072B2','#009E73','#8E44AD')
METRICS=('loss','rho','gradient_norm','eta_kappa_hat','lr_over_base')
LABELS=('Training loss (log)',r'$\rho_t$ (raw adjacent)',r'$\|g_t\|$ (log)',r'$\eta\widehat\kappa_t$','Actual LR / base LR')
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def main():
    inventory={r['path']:r for r in csv.DictReader((HERE/'raw_inventory.csv').open(encoding='utf-8'))}
    raw={};actions={};sources=[]
    for condition in ('unchanged',)+INTERVENTIONS:
        for name,target in [('boundaries.jsonl',raw),('decisions.jsonl',actions)]:
            path=SOURCE/condition/name;rel=path.relative_to(ROOT).as_posix();digest=sha(path)
            assert digest==inventory[rel]['sha256']
            target[condition]=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            sources.append(dict(path=rel,sha256=digest,bytes=path.stat().st_size))
        assert [r['state'] for r in raw[condition]]==list(range(657,1025))
        assert [r['state'] for r in actions[condition]]==list(range(657,1024))
        for r in raw[condition]:
            rho=r['raw_adjacent_cosine'];assert rho is not None and -1.000000000001<=rho<=1.000000000001
            assert r['previous_gradient_state']==r['state']-1
            alternate=1-rho*r['gradient_norm']/r['previous_gradient_norm']
            assert abs(r['eta_kappa_hat']-alternate)<=2e-12*(1+abs(alternate))
    table=[];series={}
    for column,condition in enumerate(INTERVENTIONS):
        for arm in ('unchanged',condition):
            rows=raw[arm];decisions=actions[arm]
            values={k:np.array([r[k] for r in rows]) for k in ['loss','gradient_norm','eta_kappa_hat']}
            values['rho']=np.array([r['raw_adjacent_cosine'] for r in rows])
            values['lr_over_base']=np.array([r['lr_over_base'] for r in decisions]+[decisions[-1]['lr_over_base']])
            series[(column,arm)]=values
            for i,r in enumerate(rows):
                table.append(dict(column_intervention=condition,condition=arm,role='reference' if arm=='unchanged' else 'intervention',
                    state=r['state'],manuscript_gradient_index=r['state']+1,loss=r['loss'],rho=r['raw_adjacent_cosine'],
                    raw_gradient_norm=r['gradient_norm'],eta_kappa_hat=r['eta_kappa_hat'],
                    applied_lr=decisions[i]['applied_lr'] if i<len(decisions) else None,
                    lr_over_base=decisions[i]['lr_over_base'] if i<len(decisions) else None,
                    lr_plot_value=values['lr_over_base'][i],terminal_lr_value_closes_previous_interval=i==len(decisions)))
    assert len(table)==2208
    with (HERE/'paired_comparison_values.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    checked=0;limits={};pdfs={}
    for bounds,name in [((657,1024),'paired_three_column'),((673,745),'paired_three_column_zoom')]:
        fig,axes=plt.subplots(5,3,figsize=(15,11.4),sharex=True,sharey=False,gridspec_kw=dict(height_ratios=[1.08,1.,1.,1.,.75]))
        fig.subplots_adjust(left=.067,right=.99,bottom=.07,top=.89,wspace=.25,hspace=.16)
        x=np.arange(657,1025);visible=(x>=bounds[0])&(x<=bounds[1]);limits[name]={}
        for column,condition in enumerate(INTERVENTIONS):
            axes[0,column].set_title(TITLES[column],pad=30,fontsize=11);limits[name][condition]={}
            for row,metric in enumerate(METRICS):
                ax=axes[row,column];ax.set_xlim(*bounds);ax.grid(alpha=.14,linestyle='-',linewidth=.5)
                for arm,color,label,lw in [('unchanged','#5B5B5B','Unchanged Adam',.85),(condition,COLORS[column],'Intervention',.95)]:
                    y=series[(column,arm)][metric];kwargs=dict(color=color,label=label,linestyle='-',linewidth=lw)
                    line,=ax.step(x[visible],y[visible],where='post',**kwargs) if metric=='lr_over_base' else ax.plot(x[visible],y[visible],**kwargs)
                    assert line.get_linestyle()=='-'
                    assert np.array_equal(line.get_xdata(),x[visible]) and np.array_equal(line.get_ydata(),y[visible]);checked+=int(visible.sum())
                if metric in ('loss','gradient_norm'):ax.set_yscale('log')
                if metric=='rho':ax.axhline(0,color='.75',linewidth=.55,linestyle='-',zorder=0)
                if metric=='eta_kappa_hat':ax.axhline(1.8,color='.67',linewidth=.65,linestyle='-',zorder=0)
                ax.margins(y=.07)
                if column==0:ax.set_ylabel(LABELS[row],fontsize=10)
                ax.tick_params(labelsize=8)
                if row==4:ax.set_xlabel('Parameter state s',fontsize=10)
                limits[name][condition][metric]=list(ax.get_ylim())
            axes[0,column].legend(loc='lower left',bbox_to_anchor=(-.015,1.005),ncol=2,frameon=False,fontsize=8.5,handlelength=2.5)
        for row in range(5):
            for column in range(3):
                for other in range(column+1,3):assert not axes[row,column].get_shared_y_axes().joined(axes[row,column],axes[row,other])
        fig.suptitle('Unchanged Adam versus each intervention',y=.995,fontsize=15)
        fig.text(.5,.963,'Seed 0; states '+str(bounds[0])+'-'+str(bounds[1])+'. Each panel has its own Y range; solid, unsmoothed trajectories.',ha='center',fontsize=10)
        fig.text(.5,.018,r'$\rho_t=\cos(g_t,g_{t-1})$ for raw full-objective gradients.  '+r'$\eta\widehat\kappa_t$ threshold 1.8. LR at s applies to s -> s+1.',ha='center',fontsize=9)
        path=HERE/'figures'/(name+'.pdf');fig.savefig(path);fig.savefig(HERE/'scratch'/(name+'.png'),dpi=155);plt.close(fig);pdfs[path.name]=sha(path)
        with fitz.open(path) as doc:
            assert len(doc)==1 and 'One-step reduction' not in doc[0].get_text()
            assert all(d['dashes'] in (None,'[] 0') for d in doc[0].get_drawings())
            doc[0].get_pixmap(matrix=fitz.Matrix(1.15,1.15)).save(HERE/'scratch'/(name+'_pdf_review.png'))
    assert checked==13230
    assert limits['paired_three_column']['always_half']['eta_kappa_hat']!=limits['paired_three_column']['hold8']['eta_kappa_hat']
    v=dict(status='passed',source_sha256=sha(__file__),sources=sources,conditions=list(INTERVENTIONS),reference='unchanged',columns=3,rows=list(METRICS),
        all_trajectory_lines_solid=True,independent_y_axes=True,y_limits=limits,rho_definition='cosine between current and immediately previous raw full-objective gradient',
        unique_source_boundary_rows=1472,paired_table_rows=2208,plotted_scalar_checks=checked,no_smoothing_or_data_changes=True,no_training=True,
        terminal_lr_is_plot_closure_only=True,values_sha256=sha(HERE/'paired_comparison_values.csv'),pdf_sha256=pdfs)
    (HERE/'paired_comparison_verification.json').write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='passed',plotted_scalar_checks=checked,paired_table_rows=len(table),pdfs=pdfs,y_ranges=limits['paired_three_column']),indent=2))
if __name__=='__main__':main()
