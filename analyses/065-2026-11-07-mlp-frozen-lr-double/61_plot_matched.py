"""Publication views of the complete matched intervention matrix."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import string

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('reduce_matched',HERE/'60_reduce_matched.py');r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
TITLES={'mlp':'CIFAR-10 / MLP','resnet':'CIFAR-10 / ResNet-32','transformer':'WikiText-2 / Transformer'}
FAMILIES=[('lr','Immediate LR reduction',['lr_0p75','lr_0p5','lr_0p1']),
 ('beta2',r'$\beta_2$ reduction',['beta2_0p95','beta2_0p9','beta2_0p5']),
 ('clipping','Global gradient clipping',['clip_0p5','clip_1p0']),
 ('cap','Distributed displacement cap',['cap_q25','cap_q50','cap_q75'])]
COLORS={'control':'#93999e','lr_0p75':'#80b8dd','lr_0p5':'#2879b5','lr_0p1':'#083e70',
 'beta2_0p95':'#f1bd70','beta2_0p9':'#d78024','beta2_0p5':'#8b4407',
 'clip_0p5':'#a174b1','clip_1p0':'#593375','cap_q25':'#85c9ab','cap_q50':'#23885b','cap_q75':'#064c32'}
SHORT={'control':'Adam','lr_0p75':r'$0.75\eta$','lr_0p5':r'$0.5\eta$','lr_0p1':r'$0.1\eta$',
 'beta2_0p95':'.95','beta2_0p9':'.9','beta2_0p5':'.5','clip_0p5':'.5','clip_1p0':'1.0',
 'cap_q25':'q25','cap_q50':'q50','cap_q75':'q75'}
POINT_LABELS={**SHORT,'beta2_0p95':r'$\beta_2=.95$','beta2_0p9':r'$\beta_2=.9$',
 'beta2_0p5':r'$\beta_2=.5$','clip_0p5':'Clip .5','clip_1p0':'Clip 1'}
DISPLAY={**r.LABELS,'beta2_0p95':r'$\beta_2=0.95$','beta2_0p9':r'$\beta_2=0.9$',
 'beta2_0p5':r'$\beta_2=0.5$','lr_0p75':r'LR $\times0.75$',
 'lr_0p5':r'LR $\times0.5$','lr_0p1':r'LR $\times0.1$'}
LABEL_COLORS={**COLORS,'lr_0p75':'#477b9d','beta2_0p95':'#9b651d',
 'cap_q25':'#3c8065','clip_0p5':'#82518f'}
MARKERS={'lr':'o','beta2':'s','clipping':'^','cap':'D'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.4,'axes.labelsize':7.4,'axes.titlesize':8.1,
 'xtick.labelsize':6.7,'ytick.labelsize':6.7,'legend.fontsize':6.5,'axes.linewidth':.55,
 'lines.linewidth':.85,'pdf.fonttype':42,'ps.fonttype':42,'savefig.pad_inches':.025})


def tidy(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(width=.5,length=2.5,pad=2)


def fmt(x,digits=3):return '--' if x is None or not np.isfinite(x) else f'{0. if x==0 else x:.{digits}g}'


class Figures:
    def __init__(self,input_dir,output,partial):
        self.input=input_dir;self.output=output;output.mkdir(parents=True,exist_ok=True)
        self.report=json.loads((input_dir/'results.json').read_text());self.curves=np.load(input_dir/'curves.npz')
        assert partial or self.report['complete_matrix']
        self.cases=[v['case'] for v in self.report['cases']];self.rows=self.report['summary']
        self.held=list(csv.DictReader((input_dir/'heldout.csv').open()))
        self.events=list(csv.DictReader((input_dir/'events.csv').open()))
        self.costs=list(csv.DictReader((input_dir/'costs.csv').open()))
        self.checked=0;self.records=[]
    def v(self,case,arm,metric):return self.curves[case+'__'+arm+'__'+metric]
    def line(self,ax,x,y,**kw):
        line,=ax.plot(x,y,**kw);np.testing.assert_array_equal(line.get_xdata(),x);np.testing.assert_array_equal(line.get_ydata(),y)
        self.checked+=np.size(y);return line
    def save(self,fig,name,**meta):
        fig.savefig(self.output/name,metadata={'Title':meta.get('title',name),'Author':'Constructive-destructive regimes'})
        plt.close(fig);self.records.append(dict(file=name,pages=1,**meta))
    def family_grid(self,kind='loss'):
        n=len(self.cases);fig,axes=plt.subplots(4,n,figsize=(7.15,7.0),sharex='col',squeeze=False)
        for col,case in enumerate(self.cases):
            state=self.v(case,'control','state');all_values=[]
            for row,(_,title,arms) in enumerate(FAMILIES):
                ax=axes[row,col]
                for name in ['control']+arms:
                    if kind=='loss':x,y=state,self.v(case,name,'loss')
                    elif kind=='heldout':
                        data=[v for v in self.held if v['case']==case and v['arm']==name and v['split']=='test']
                        x=np.array([int(v['state']) for v in data]);y=np.array([float(v['loss']) for v in data])
                    else:x,y=state[1:],self.v(case,name,'actual_norm')
                    self.line(ax,x,y,color=COLORS[name],label=SHORT[name],lw=1 if name=='control' else .8,
                              marker='.' if kind=='heldout' else None,ms=3,zorder=2 if name=='control' else 3)
                    all_values.extend(y.tolist())
                ax.axvline(state[0],color='#444444',ls='--',lw=.6)
                ax.legend(frameon=False,loc='upper right',ncol=2,handlelength=1.25,columnspacing=.65)
                ax.set_xlim(state[0],state[-1]);tidy(ax)
                if row==0:ax.set_title(TITLES[case],pad=6)
                if col==0:ax.set_ylabel(title+'\n'+{'loss':'Training loss','heldout':'Test loss','norm':'Applied update norm'}[kind],fontsize=7.2)
                ax.text(.015,.97,'('+string.ascii_lowercase[row*n+col]+')',transform=ax.transAxes,va='top',fontsize=7)
                if row==3:ax.set_xlabel('Training step $s$')
            low=min(all_values);high=max(all_values)
            for ax in axes[:,col]:
                if kind in ['loss','norm']:
                    assert low>0;ax.set_yscale('log')
                    # Reserve a common upper strip for the two-row legends;
                    # otherwise the highest clipping excursion crosses text.
                    bottom=low/1.12
                    ax.set_ylim(bottom,high*np.exp(max((.32 if kind=='norm' else .23)*np.log(high/bottom),.2)))
                else:ax.set_ylim(max(0,low-.08*(high-low)),high+.27*(high-low))
        fig.subplots_adjust(left=.105,right=.98,bottom=.065,top=.955,hspace=.19,wspace=.29)
        return fig
    def loss(self):self.save(self.family_grid(),'17-matched-intervention-loss.pdf',title='Matched intervention training losses',logarithmic_y=True)
    def heldout(self):self.save(self.family_grid('heldout'),'21-matched-heldout-loss.pdf',title='Full held-out test loss at five fixed states')
    def tradeoff(self):
        fig,axes=plt.subplots(1,len(self.cases),figsize=(7.15,3.5),squeeze=False)
        labels=[]
        # Fixed callout positions for this complete comparison. Only text moves;
        # every scientific point retains its exact mean/PV coordinates.
        layout={
            'mlp': {'lr_0p75':(.42,.79),'lr_0p5':(.36,.97),'lr_0p1':(.56,.90),
                'beta2_0p95':(.45,.50),'beta2_0p9':(.45,.65),'beta2_0p5':(.75,.40),
                'clip_0p5':(.03,.64),'clip_1p0':(.40,.06),
                'cap_q25':(.03,.77),'cap_q50':(.03,.90),'cap_q75':(.16,.98)},
            'resnet': {'lr_0p75':(.52,.58),'lr_0p5':(.74,.68),'lr_0p1':(.69,.94),
                'beta2_0p95':(.35,.04),'beta2_0p9':(.09,.43),'beta2_0p5':(.46,.87),
                'clip_0p5':(.62,.79),'clip_1p0':(.52,.35),
                'cap_q25':(.23,.84),'cap_q50':(.04,.64),'cap_q75':(.05,.94)},
            'transformer': {'lr_0p75':(.81,.40),'lr_0p5':(.63,.57),'lr_0p1':(.68,.96),
                'beta2_0p95':(.25,.08),'beta2_0p9':(.14,.30),'beta2_0p5':(.69,.72),
                'clip_0p5':(.46,.84),'clip_1p0':(.06,.56),
                'cap_q25':(.32,.95),'cap_q50':(.025,.86),'cap_q75':(.025,.97)}}
        for col,case in enumerate(self.cases):
            ax=axes[0,col];rows=[v for v in self.rows if v['case']==case and v['arm']!='control']
            x=np.array([v['mean_loss_change_percent'] for v in rows]);y=np.array([v['pv_reduction_percent'] for v in rows])
            for i,row in enumerate(rows):
                arm=row['arm'];ax.scatter(x[i],y[i],s=25,c=COLORS[arm],marker=MARKERS[row['family']],edgecolors='white',linewidths=.35,zorder=4)
                ax.annotate(POINT_LABELS[arm],(x[i],y[i]),xytext=layout[case][arm],textcoords='axes fraction',
                    fontsize=6.5,ha='left',va='center',color=LABEL_COLORS[arm],zorder=6,
                    bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=.1),
                    arrowprops=dict(arrowstyle='-',color=LABEL_COLORS[arm],lw=.4,alpha=.6,shrinkA=2,shrinkB=4))
                labels.append(dict(case=case,arm=arm,x=float(x[i]),y=float(y[i])))
            ax.scatter([0],[0],c=COLORS['control'],marker='x',s=25,zorder=5)
            ax.axhline(0,color='#9fa4a8',lw=.6);ax.axvline(0,color='#9fa4a8',lw=.6)
            ax.set_title(TITLES[case]);ax.set_xlabel('Mean training-loss change (%)')
            if col==0:ax.set_ylabel('Positive-variation reduction (%)')
            ax.margins(x=.16,y=.16);tidy(ax)
        handles=[Line2D([],[],color=COLORS[arms[1]],marker=MARKERS[f],ls='',label=title,markersize=5) for f,title,arms in FAMILIES]
        handles.append(Line2D([],[],color=COLORS['control'],marker='x',ls='',label='Unchanged Adam'))
        fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.52,-.005),columnspacing=1.3)
        fig.subplots_adjust(left=.09,right=.98,bottom=.25,top=.91,wspace=.36)
        self.save(fig,'18-matched-spike-learning-tradeoff.pdf',title='Spike burden versus learning progress',points=labels)
    def heatmap(self,arm):
        n=len(self.cases);fig,axes=plt.subplots(5,n,figsize=(7.15,7.5),sharex='col',squeeze=False,
            gridspec_kw={'height_ratios':[1.05,1.45,1.45,1.45,1.45]})
        data=[];maximum=1.
        for case in self.cases:
            rms=[self.v(case,a,'stage_change_rms')[:,1:] for a in ['control',arm]]
            den=rms[0].max(axis=1);assert (den>0).all()
            relative=[v/den[:,None] for v in rms];maximum=max(maximum,max(float(v.max()) for v in relative))
            data.append((relative,den))
        metadata=[]
        for col,case in enumerate(self.cases):
            state=self.v(case,'control','state');x=state[1:];stages=r.STAGES[case];relative,den=data[col]
            for name in ['control',arm]:self.line(axes[0,col],state,self.v(case,name,'loss'),color=COLORS[name],label=DISPLAY[name],lw=.85)
            axes[0,col].set_title(TITLES[case]);axes[0,col].set_ylim(0,1.3*max(self.v(case,a,'loss').max() for a in ['control',arm]));axes[0,col].legend(frameon=True,facecolor='white',edgecolor='none',framealpha=1,borderpad=.2,fontsize=6,handlelength=1.3)
            events=[v for v in self.events if v['case']==case and v['arm']=='control' and v['window_height_percent']]
            selected=max(events,key=lambda v:float(v['window_height_percent'])) if events else None
            peak=int(selected['window_peak_state']) if selected else None
            mats=[self.v(case,a,'stage_rho')[:,1:] for a in ['control',arm]]+relative
            for row,values in enumerate(mats,1):
                im=axes[row,col].imshow(values,aspect='auto',interpolation='none',
                    extent=[x[0]-.5,x[-1]+.5,len(stages)-.5,-.5],cmap='RdBu_r' if row<3 else 'viridis',
                    vmin=-1 if row<3 else 0,vmax=1 if row<3 else maximum)
                np.testing.assert_array_equal(im.get_array().data,values);self.checked+=values.size
                axes[row,col].set_yticks(range(len(stages)),[v[1] for v in stages],fontsize=5.8)
                if row<3:cosine_image=im
                else:rms_image=im
            for ax in axes[:,col]:
                if peak is not None:ax.axvline(peak,color='#555555',ls='--',lw=.6)
                ax.set_xlim(state[0],state[-1]);tidy(ax)
            axes[-1,col].set_xlabel('Training step $s$')
            metadata.append(dict(case=case,arm=arm,window=[int(state[0]),int(state[-1])],baseline_peak_state=peak,
                baseline_selected_loss_event=selected,baseline_stage_rms_maximum=den.tolist(),stages=stages))
        rowlabels=['Training loss','Adam: change cosine',SHORT[arm]+': change cosine','Adam: relative RMS',SHORT[arm]+': relative RMS']
        for row,label in enumerate(rowlabels):axes[row,0].set_ylabel(label,labelpad=4,fontsize=6.6)
        fig.subplots_adjust(left=.12,right=.915,bottom=.065,top=.945,hspace=.22,wspace=.48)
        cax=fig.add_axes([.938,.49,.011,.25]);fig.colorbar(cosine_image,cax=cax,ticks=[-1,0,1]);cax.tick_params(labelsize=6,width=.5,length=2)
        cax=fig.add_axes([.938,.09,.011,.25]);fig.colorbar(rms_image,cax=cax);cax.tick_params(labelsize=6,width=.5,length=2)
        return fig,dict(columns=metadata,rms_color_maximum=maximum)
    def heatmaps(self):
        fig,meta=self.heatmap('cap_q50');self.save(fig,'19-matched-q50-stage-heatmaps.pdf',title='Matched distributed q50 loss and stage motion',**meta)
        pages=[]
        with PdfPages(self.output/'20-matched-all-setting-stage-heatmaps.pdf') as pdf:
            for name in [v['id'] for v in self.report['cases'][0]['config']['arms'] if v['id']!='control']:
                fig,meta=self.heatmap(name);fig.suptitle(DISPLAY[name]+' versus unchanged Adam',fontsize=9,y=.985)
                pdf.savefig(fig);plt.close(fig);pages.append(meta)
        self.records.append(dict(file='20-matched-all-setting-stage-heatmaps.pdf',pages=len(pages),page_records=pages))
    def activity(self):
        name='22-matched-update-and-controller-activity.pdf'
        with PdfPages(self.output/name) as pdf:
            fig=self.family_grid('norm');pdf.savefig(fig);plt.close(fig)
            fig,axes=plt.subplots(2,len(self.cases),figsize=(7.15,3.7),sharex='col',squeeze=False)
            for col,case in enumerate(self.cases):
                x=self.v(case,'control','state')[1:]
                for name0 in FAMILIES[2][2]:self.line(axes[0,col],x,self.v(case,name0,'clip_factor'),color=COLORS[name0],label=SHORT[name0])
                for name0 in FAMILIES[3][2]:self.line(axes[1,col],x,self.v(case,name0,'cap_active'),color=COLORS[name0],label=SHORT[name0])
                for ax in axes[:,col]:ax.set_ylim(-.025,1.07);ax.set_xlim(x[0]-1,x[-1]);ax.legend(frameon=False,ncol=3,handlelength=1.1);tidy(ax)
                axes[0,col].set_title(TITLES[case]);axes[1,col].set_xlabel('Training step $s$')
            axes[0,0].set_ylabel('Raw-gradient retained factor');axes[1,0].set_ylabel('Fraction of capped units')
            fig.subplots_adjust(left=.095,right=.98,bottom=.12,top=.93,hspace=.2,wspace=.26)
            pdf.savefig(fig);plt.close(fig)
        self.records.append(dict(file=name,pages=2))
    def tables(self):
        name='23-matched-intervention-outcomes.pdf';tex=[]
        tablecols=['Setting','PV\nreduction %','Mean loss\nchange %','Final\nloss','Last-32\nloss','Test loss\nchange %']
        with PdfPages(self.output/name) as pdf:
            for case in self.cases:
                rows=[v for v in self.rows if v['case']==case]
                values=[[DISPLAY[v['arm']],fmt(v['pv_reduction_percent'],4),fmt(v['mean_loss_change_percent'],4),fmt(v['final_loss'],4),fmt(v['last32_loss'],4),fmt(v['terminal_test_loss_change_percent'],4)] for v in rows]
                fig,ax=plt.subplots(figsize=(7.15,4.1));ax.axis('off')
                tab=ax.table(cellText=values,colLabels=tablecols,cellLoc='right',colLoc='right',bbox=[0,.16,1,.79],colWidths=[.24,.15,.16,.14,.14,.17])
                tab.auto_set_font_size(False);tab.set_fontsize(8)
                for (row,col),cell in tab.get_celld().items():
                    cell.set_linewidth(.25);cell.set_edgecolor('#d7dce0')
                    if row==0:cell.set_facecolor('#edf1f3');cell.set_text_props(weight='bold')
                    elif rows[row-1]['arm']=='cap_q50':cell.set_facecolor('#eaf4ee')
                    if col==0:cell.get_text().set_ha('left')
                ax.set_title(TITLES[case]+f" | initialization 0 | steps {rows[0]['fork']}-{rows[0]['end']}",fontsize=10,pad=10)
                ax.text(0,.075,'PV: sum of positive loss increments. Higher PV reduction is favorable; lower loss change is favorable.\n'
                    'Mean excludes the shared fork. Last-32 uses the final 32 states. Full test split; no held-out selection.\n'
                    'Green marks the preselected q50 method, not an outcome-selected winner. One initialization; no error bars.',fontsize=7,va='center',transform=ax.transAxes)
                fig.subplots_adjust(left=.035,right=.985,bottom=.015,top=.9);pdf.savefig(fig);plt.close(fig)
                tex += [r'\begin{table*}[t]',r'\centering',r'\small',r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{lrrrrr}',r'\toprule',
                    r'Setting & \shortstack{PV reduction\\(\%)} & \shortstack{Mean loss\\change (\%)} & \shortstack{Final\\loss} & \shortstack{Last-32\\loss} & \shortstack{Test loss\\change (\%)} \\',r'\midrule']
                tex += [' & '.join(v)+r' \\' for v in values]
                tex += [r'\bottomrule',r'\end{tabular}',r'\caption{'+TITLES[case]+f". Matched initialization-0 interventions at states {rows[0]['fork']}--{rows[0]['end']}. "+
                    r'PV is positive loss variation. Mean excludes the common fork; late loss averages the final 32 states. Test changes use the complete held-out split. All changes compare with unchanged Adam; no held-out selection.}',r'\end{table*}','']
        (self.output/'matched-outcomes.tex').write_bytes(('\n'.join(tex).rstrip()+'\n').encode())
        self.records.append(dict(file=name,pages=len(self.cases)))
        # The main method's synchronized components are supplied independently
        # of trajectory quality. Full-grid timings remain in costs.csv.
        cost_tex=[r'\begin{table*}[t]',r'\centering\small',r'\setlength{\tabcolsep}{4pt}',
            r'\begin{tabular}{llrrrrr}',r'\toprule',
            r'Case & Setting & \shortstack{Gradient\\ms/boundary} & \shortstack{Probe capture\\ms/boundary} & \shortstack{Adam\\ms/update} & \shortstack{Controller\\ms/update} & \shortstack{Peak\\GiB} \\',r'\midrule']
        for case in self.cases:
            for arm in ['control','cap_q50']:
                item=next(v for v in self.costs if v['case']==case and v['arm']==arm)
                boundaries=int(item['full_gradient_passes']);updates=int(item['updates'])
                components=[1000*float(item[k])/den for k,den in [
                    ('gradient_seconds',boundaries),('probe_seconds',boundaries),
                    ('optimizer_seconds',updates),('controller_seconds',updates)]]
                values=[{'mlp':'MLP','resnet':'ResNet-32','transformer':'Transformer'}[case],
                        'Adam' if arm=='control' else 'Cap q50',
                        *[fmt(v,4) for v in components],fmt(int(item['process_peak_cuda_bytes_through_arm'])/2**30,3)]
                cost_tex.append(' & '.join(values)+r' \\')
        cost_tex += [r'\bottomrule',r'\end{tabular}',
            r'\caption{Synchronized components on the recorded GPU class: GB10 for MLP, RTX4090 for ResNet-32 and RTX5090 for Transformer. Gradient/probe means include every boundary, including the terminal state; Adam/controller means cover updates only. Probe time covers forward capture and cap inputs. Cosine/RMS reductions, other scalar diagnostics, logging and checkpoint I/O lie in the separately retained remaining wall time. Controller time alone is not deployable overhead. Peak is the process high-water allocation through the arm, not an isolated per-method peak. Hosts may differ within a GPU class; no cross-host speed ranking is implied. Full-grid timings are retained separately.}',r'\end{table*}','']
        (self.output/'matched-costs.tex').write_bytes(('\n'.join(cost_tex).rstrip()+'\n').encode())
    def run(self):
        self.loss();self.tradeoff();self.heatmaps();self.heldout();self.activity();self.tables()
        for item in self.records:
            p=self.output/item['file'];item.update(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        r.write(self.output/'figure_manifest.json',dict(status='generated',complete_matrix=self.report['complete_matrix'],
            results_sha256=r.sha(self.input/'results.json'),curves_sha256=r.sha(self.input/'curves.npz'),
            plotted_values_checked=self.checked,outputs=self.records,generator_sha256=r.sha(Path(__file__)),
            visual_review='pending',clock='Training step s; incoming functional changes at their endpoint; outgoing control activity at s+1.'))
        print(json.dumps(dict(figures=len(self.records),pages=sum(v['pages'] for v in self.records),plotted_values_checked=self.checked)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=HERE/'matched');p.add_argument('--output',type=Path,default=HERE/'figures/matched');p.add_argument('--partial',action='store_true')
    a=p.parse_args();Figures(a.input,a.output,a.partial).run()
