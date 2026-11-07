"""One standardized two-page vector report per architecture."""
import hashlib
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

HERE=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'mathtext.fontset':'stix','font.size':10,'pdf.fonttype':42})

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def short(name):
    return name.replace('transformer_encoder.layers.','block').replace('.call0.output','').replace('.call1.output','.relu2').replace('.call0.input','.in').replace('.call1.input','.sum').replace('output.log_probabilities','log probabilities').replace('output.correct_label_log_probability','correct-label log p')
def axis(ax):
    ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)

def main(case_filter=None):
    (HERE/'figures').mkdir(exist_ok=True);audit={}
    for case in ['resnet','transformer']:
        if case_filter and case!=case_filter:continue
        r=json.loads((HERE/(case+'_results.json')).read_text());t=np.array(r['states']);stages=[s for s in r['stages'] if s['name']!='raw_gradient'];names=[s['name'] for s in stages];curves=r['curves']
        title='ResNet-32 / CIFAR-10' if case=='resnet' else 'Transformer / WikiText-2'
        probe='1,000 fixed images; complete original BatchNorm group' if case=='resnet' else '64 fixed sequences; all supervised positions and output coordinates'
        path=HERE/'figures'/(('01' if case=='resnet' else '02')+'-'+case+'-instability-trace.pdf')
        with PdfPages(path) as pdf:
            height=max(12,5+len(stages)*.17)
            fig=plt.figure(figsize=(14,height));grid=fig.add_gridspec(3,3,height_ratios=[1.4,1.2,max(4,len(stages)*.15)],width_ratios=[2.7,2.7,1.4],hspace=.22,wspace=.12)
            ax=fig.add_subplot(grid[0,:]);lookup={q['state']:q for q in r['boundaries']};loss=np.array([lookup[int(i)]['loss'] for i in t]);ploss=np.array([lookup[int(i)]['probe_loss'] for i in t])
            ax.plot(t,loss,color='black',label='Full training objective');ax.plot(t,ploss,color='#b15928',label='Fixed probe');ax.set_ylabel('Loss');ax.legend(frameon=False,ncol=2);axis(ax)
            ax=fig.add_subplot(grid[1,:]);ax.plot(t,curves['raw_gradient']['rho'],color='#2166ac',label='Raw gradient');ax.plot(t,curves['output.log_probabilities']['rho'],color='#b2182b',label='All log-probability changes');ax.plot(t,curves['output.correct_label_log_probability']['rho'],color='#e08214',ls='--',label='Correct-label log-probability changes');ax.axhline(0,color='.5',lw=.6);ax.set_ylim(-1.05,1.05);ax.set_ylabel('Consecutive cosine');ax.legend(frameon=False,ncol=3,fontsize=9);axis(ax)
            a=fig.add_subplot(grid[2,0]);rho=np.array([[np.nan if x is None else x for x in curves[n]['rho']] for n in names]);im=a.imshow(rho,aspect='auto',vmin=-1,vmax=1,cmap='RdBu',extent=[t[0]-.5,t[-1]+.5,len(names)-.5,-.5],interpolation='nearest');a.set_yticks(range(len(names)),[short(n) for n in names],fontsize=7);a.set_title('Consecutive representation-change cosine');a.set_xlabel('Completed optimizer updates');fig.colorbar(im,ax=a,orientation='horizontal',pad=.12 if case=='transformer' else .025,fraction=.025)
            b=fig.add_subplot(grid[2,1]);rms=np.array([curves[n]['change_rms'] for n in names]);relative=np.divide(rms,rms.max(1)[:,None],out=np.zeros_like(rms),where=rms.max(1)[:,None]>0);im=b.imshow(relative,aspect='auto',vmin=0,vmax=1,cmap='magma',extent=[t[0]-.5,t[-1]+.5,len(names)-.5,-.5],interpolation='nearest');b.set_yticks([]);b.set_title('Displacement RMS / own window maximum');b.set_xlabel('Completed optimizer updates');fig.colorbar(im,ax=b,orientation='horizontal',pad=.12 if case=='transformer' else .025,fraction=.025)
            c=fig.add_subplot(grid[2,2]);y=np.arange(len(stages));c.plot([s['top1'] for s in stages],y,'o-',ms=2,lw=.6,label='Top1 energy');c.plot([s['top10'] for s in stages],y,'o-',ms=2,lw=.6,label='Top10 energy');c.plot([s['temporal_top10_overlap'] for s in stages],y,'o-',ms=2,lw=.6,label='Output overlap');c.set_ylim(len(stages)-.5,-.5);c.set_xlim(0,1.02);c.set_yticks([]);c.set_title('SVD / temporal overlap');c.legend(frameon=False,fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.015));c.grid(alpha=.2)
            fig.suptitle(title+' - complete measured architecture trace',y=1-.16/height,fontsize=17,va='top')
            fig.text(.5,1-.52/height,probe+'\nFixed inputs: zero change; motion cosine undefined. Unchanged Adam continuation; no causal intervention.',ha='center',va='top',fontsize=10)
            fig.subplots_adjust(left=.25,right=.97,bottom=.08 if case=='transformer' else .055,top=1-1.1/height)
            # Colorbars shorten the heatmap axes. Keep spectral rows aligned
            # with the identical architecture rows in both heatmaps.
            fig.canvas.draw()
            heat_box=a.get_position();spectral_box=c.get_position()
            c.set_position([spectral_box.x0,heat_box.y0,spectral_box.width,heat_box.height])
            c.legend(frameon=False,fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.35/(height*heat_box.height)))
            pdf.savefig(fig);plt.close(fig)
            landmarks=(['conv1.call0.output','bn1.call0.output','relu.call0.output','layer1.4.relu.call1.output','layer2.4.relu.call1.output','layer3.4.relu.call1.output','avgpool.call0.output','fc.call0.output','output.log_probabilities'] if case=='resnet' else ['encoder.call0.output','pos_encoder.call0.output','transformer_encoder.layers.0.self_attn.call0.output','transformer_encoder.layers.0.norm1.call0.output','transformer_encoder.layers.0.norm2.call0.output','transformer_encoder.layers.1.self_attn.call0.output','transformer_encoder.layers.1.norm2.call0.output','decoder.call0.output','output.log_probabilities'])
            landmarks=[n for n in landmarks if n in names]+['output.correct_label_log_probability'];fig,axes=plt.subplots(len(landmarks),3,figsize=(14,2+len(landmarks)*1.35),sharex='col')
            for i,name in enumerate(landmarks):
                curve=curves[name];record=next(s for s in stages if s['name']==name)
                axes[i,0].plot(t,curve['rho'],color='#2166ac');axes[i,0].axhline(0,color='.5',lw=.6);axes[i,0].set_ylim(-1.05,1.05);axes[i,0].set_ylabel(short(name),fontsize=9,rotation=0,ha='right',va='center',labelpad=12)
                axes[i,1].plot(t,curve['change_rms'],color='#542788')
                # Dividing by sqrt(coordinates) keeps coefficient units aligned
                # with coordinate RMS; it does not normalize each time row.
                alpha=np.array(curve['alpha'])/np.sqrt(record['coordinates']);axes[i,2].plot(t,alpha,color='#b2182b');axes[i,2].axhline(0,color='.5',lw=.6)
                for a in axes[i]:axis(a)
            for a,title2 in zip(axes[0],['Consecutive change cosine','Displacement coordinate RMS','Leading fixed SVD coefficient / sqrt(P)']):a.set_title(title2)
            for a in axes[-1]:a.set_xlabel('Completed optimizer updates')
            fig.suptitle(title+' - selected architecture landmarks',fontsize=16,y=.99)
            fig.text(.5,.967,'Basis fitted once over this window; signs fixed once. Each layer has its own spatial basis.',ha='center',fontsize=10)
            fig.tight_layout(rect=[.04,.02,1,.945]);pdf.savefig(fig);plt.close(fig)
        # The same arrays used by artists are explicitly checked against the table.
        assert rho.shape==(len(stages),len(t)) and rms.shape==rho.shape
        for i,s in enumerate(stages):assert int(np.sum(rho[i]<0))==s['negative']
        audit[case]=dict(pdf_sha256=sha(path),pages=2,stages=len(stages),time_rows=len(t),heatmap_values=int(rho.size+rms.size),source_sha256=sha(HERE/(case+'_results.json')))
    (HERE/'figure_verification.json').write_text(json.dumps(dict(status='numeric_checks_passed',figures=audit,generator_sha256=sha(Path(__file__))),indent=2)+'\n',newline='\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['resnet','transformer']);a=p.parse_args();main(a.case)
