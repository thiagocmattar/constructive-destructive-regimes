"""Executive activation evidence and a separate all-site matrix atlas."""
from pathlib import Path
import csv,hashlib,importlib.util,json,textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize,TwoSlopeNorm
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('reduction',HERE/'02_reduce.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
CASES=r.CASES;ARMS=r.ARMS;NAMES={'mlp':'CIFAR-10 / MLP','resnet':'CIFAR-10 / ResNet-32','transformer':'WikiText-2 / Transformer'}
ARM={'control':'Adam','cap_q25':'Cap q25','cap_q50':'Cap q50'};COLORS=['#66717d','#259b65','#075b39']
LAST={'mlp':'4','resnet':'layer3.4.relu#2','transformer':'transformer_encoder.layers.1.linear2'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
RECORD={'arrays':[],'figures':{},'matrix_color_limits':[]}
def note(fig,title,caption,page):
    fig.suptitle(title,fontsize=15,x=.06,ha='left',y=.97)
    fig.text(.06,.025,'\n'.join(textwrap.wrap(caption,130)),fontsize=8,va='bottom')
    fig.text(.96,.025,str(page),fontsize=8,ha='right')
def save(pdf,fig):pdf.savefig(fig);plt.close(fig)
def record(tag,a):RECORD['arrays'].append(dict(tag=tag,shape=list(np.shape(a)),sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()))
def site(c,arm,state,scope,name):
    d,a=r.load(c,arm,state,scope);s=next(s for s in d['sites'] if s['site']==name);p=s['array_prefix']+'_';return s,{k[len(p):]:a[k] for k in a.files if k.startswith(p)}
def primary(c):
    d,_=r.load(c,'control',r.END[c],'full_training');return [s['site'] for s in d['sites'] if r.primary(s)]
def scale(c,n):return site(c,'control',r.FORK[c],'full_training',n)[0]['rms']
def main():
    final=[]
    for c in CASES:
        for arm in ARMS:
            s,a=site(c,arm,r.END[c],'full_training',LAST[c]);b=scale(c,LAST[c])
            final.append(dict(case=c,arm=arm,site=LAST[c],units=s['units'],zero_entries=100*s['zero_entry_fraction'],zero_units=100*s['dead_zero_fraction'],near_units=100*np.mean(a['maximum']<=.01*b),weak_units=100*np.mean(a['rms']<=.01*b),top10_energy=100*s['top10pct_energy'],effective_units=100*s['effective_unit_fraction']))
    with (HERE/'final-hidden-summary.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(final[0]),lineterminator='\n');w.writeheader();w.writerows(final)
    with PdfPages(HERE/'activation-summary.pdf',metadata={'Title':'Activation distributions and inactive units','CreationDate':None,'ModDate':None}) as pdf:
        fig=plt.figure(figsize=(11.7,8.3));ax=fig.add_axes([.06,.40,.88,.35]);ax.axis('off')
        cells=[[NAMES[z['case']].split(' / ')[-1],ARM[z['arm']],str(z['units'])]+[f'{z[k]:.1f}' for k in ['zero_entries','zero_units','near_units','weak_units','top10_energy','effective_units']] for z in final]
        table=ax.table(cellText=cells,colLabels=['Model','Arm','Units','Zero\nentries %','Zero\nunits %','Near-zero\nunits %','Weak\nunits %','Top 10%\nenergy %','Effective\nunits %'],loc='center',cellLoc='center',colWidths=[.12,.10,.06,.105,.105,.105,.10,.10,.105]);table.auto_set_font_size(False);table.set_fontsize(8);table.scale(1,1.7)
        for (i,j),cell in table.get_celld().items():
            cell.set_edgecolor('#ddd');cell.set_linewidth(.5)
            if i==0:cell.set_facecolor('#e9eef2');cell.set_text_props(weight='bold')
        fig.text(.06,.83,'Final saved state, complete training set, last hidden nonlinearity',fontsize=12,weight='bold')
        fig.text(.06,.79,'MLP: Tanh 2 (t=800)  |  ResNet: final residual ReLU (t=576)  |  Transformer: Block 2 FF ReLU (t=928)',fontsize=9)
        fig.text(.06,.385,'ResNet: the zero-channel pattern predates treatment and is shared by Adam and both caps; see the structural audit.',fontsize=9,color='#953c27')
        fig.text(.06,.36,'Transformer: both caps spread final-layer activation energy across more units, without changing the zero-unit count.',fontsize=9,color='#075b39')
        definitions=[('Inactive unit','Every measured activation of the unit is zero (or below the stated magnitude threshold).'),('Near-zero versus weak','Near-zero uses the maximum; weak uses RMS. Both use 1% of the common-fork layer RMS.'),('Concentrated energy','Top 10% is the share of squared activation in the strongest tenth of units. Effective units is inverse concentration divided by width.'),('Scope','One initialization; actual retained checkpoints; 21 complete training-set passes and 48 reference-set state measurements. No new optimizer updates.'),('Read the plots','Pages 2-4 compare thresholds, depth and distributions. Pages 5-7 show every hidden nonlinearity over the six saved states. The separate atlas shows all measured sites.')]
        y=.325
        for title,text in definitions:
            fig.text(.06,y,title,weight='bold',fontsize=9);fig.text(.23,y,'\n'.join(textwrap.wrap(text,112)),fontsize=9,va='top');y-=.058
        note(fig,'Activation audit: Adam versus distributed caps','All zero/inactivity claims refer to the declared training examples and positions. They do not establish permanent neuron death or behavior on unseen data. Signed negative values count as active by magnitude.',1);save(pdf,fig)

        fig,axes=plt.subplots(2,3,figsize=(11.7,8.3));grid=np.array(r.CFG['relative_thresholds'][1:]+[3,10,30,100])
        for col,c in enumerate(CASES):
            for arm,color in zip(ARMS,COLORS):
                s,a=site(c,arm,r.END[c],'full_training',LAST[c]);b=scale(c,LAST[c])
                for row,key in enumerate(['maximum','rms']):
                    y=np.array([100*np.mean(a[key]<=t*b) for t in grid]);record(f'threshold/{c}/{arm}/{key}',y);axes[row,col].plot(grid,y,'o-',color=color,ms=3,label=ARM[arm])
            axes[0,col].set_title(NAMES[c]+'\n'+r.label(c,LAST[c]))
            for row in [0,1]:
                ax=axes[row,col];ax.set_xscale('log');ax.set_ylim(-2,102);ax.set_xlabel('Threshold / common-fork layer RMS');ax.grid(alpha=.15)
            axes[0,col].legend(fontsize=8)
        axes[0,0].set_ylabel('Inactive on every row (%)');axes[1,0].set_ylabel('Weak RMS units (%)')
        fig.subplots_adjust(left=.075,right=.97,top=.82,bottom=.15,hspace=.42,wspace=.30)
        note(fig,'A threshold sweep separates inactivity from weak average responses','Last hidden nonlinearities, final states, complete training set. Common-fork full-training RMS is fixed across arms. Maximum-based inactivity requires no above-threshold response anywhere; RMS permits rare responses. Thresholds above 1 are included only to complete the distributions, not to call large activations near zero.',2);save(pdf,fig)

        fig,axes=plt.subplots(2,3,figsize=(11.7,9.8))
        for col,c in enumerate(CASES):
            names=primary(c)
            for row,key in enumerate(['dead_zero_fraction','effective_unit_fraction']):
                z=np.array([[100*site(c,a,r.END[c],'full_training',n)[0][key] for a in ARMS] for n in names]);record(f'depth/{c}/{key}',z)
                im=axes[row,col].imshow(z,aspect='auto',vmin=0,vmax=100,cmap='viridis' if row else 'magma')
                axes[row,col].set_xticks(range(3),[ARM[a] for a in ARMS]);axes[row,col].set_yticks(range(len(names)),[r.label(c,n) for n in names],fontsize=5.5 if c=='resnet' else 8)
                if len(names)<4:
                    for (i,j),v in np.ndenumerate(z):axes[row,col].text(j,i,f'{v:.1f}',ha='center',va='center',color='white' if v<50 else 'black')
                fig.colorbar(im,ax=axes[row,col],fraction=.045,pad=.025,label='Percent')
            axes[0,col].set_title(NAMES[c]);axes[0,col].set_xlabel('Exactly inactive units');axes[1,col].set_xlabel('Effective unit fraction')
        fig.subplots_adjust(left=.12,right=.93,top=.85,bottom=.15,hspace=.3,wspace=.75)
        note(fig,'Layer depth: completely inactive units and energy concentration','All hidden nonlinearities, final state, complete training set. Each ResNet ReLU call is separate: first inside a residual branch, then after addition. Effective fraction is 100/(width * sum(unit energy shares squared)); it is not a spectral rank.',3);save(pdf,fig)

        fig,axes=plt.subplots(3,3,figsize=(11.7,8.3));edges=np.linspace(-10,3,105);centers=(edges[1:]+edges[:-1])/2
        for col,c in enumerate(CASES):
            for arm,color in zip(ARMS,COLORS):
                s,a=site(c,arm,r.END[c],'full_training',LAST[c]);hist=a['hist'][1:-1]/(s['n_rows']*s['units']);record(f'hist/{c}/{arm}',hist)
                axes[0,col].plot(centers,hist,color=color,label=ARM[arm]);axes[0,col].set_xlim(-6,2.5)
                x=np.sort(a['rms']/scale(c,LAST[c]));axes[1,col].step(x,100*np.arange(1,len(x)+1)/len(x),where='post',color=color);axes[1,col].set_xscale('symlog',linthresh=1e-4)
                share=np.sort(a['share'])[::-1];xx=100*np.arange(1,len(share)+1)/len(share);yy=100*share.cumsum();record(f'energy/{c}/{arm}',yy)
                axes[2,col].plot(xx,yy,color=color)
            axes[0,col].set_title(NAMES[c]);axes[0,col].legend(fontsize=7);axes[0,col].set_xlabel('log10 absolute activation')
            axes[1,col].set_xlabel('Unit RMS / common-fork RMS');axes[2,col].set_xlabel('Units, ranked by energy (%)');axes[2,col].plot([0,100],[0,100],':',color='#bbb')
            for row in range(3):axes[row,col].grid(alpha=.12)
        axes[0,0].set_ylabel('Fraction in log bin');axes[1,0].set_ylabel('Cumulative units (%)');axes[2,0].set_ylabel('Cumulative energy (%)')
        fig.subplots_adjust(left=.08,right=.97,top=.87,bottom=.13,hspace=.60,wspace=.35)
        note(fig,'Distributions: entry magnitude, unit strength, and concentration','Last hidden nonlinearities, final states, full training data. Fixed log-magnitude histograms; exact zeros and tails outside the displayed range remain in the data (zero fraction on page 1). RMS CDF includes zero units. Ranked energy curves are uncentered coordinate statistics.',4);save(pdf,fig)

        for pi,c in enumerate(CASES):
            names=primary(c);states=[r.FORK[c],r.FORK[c]+1,r.FORK[c]+32,r.FORK[c]+33,r.END[c]-1,r.END[c]]
            fig,axes=plt.subplots(3,3,figsize=(11.7,12 if c=='resnet' else 8.3))
            for col,arm in enumerate(ARMS):
                for row,metric in enumerate(['dead','weak','energy']):
                    z=[]
                    for n in names:
                        vals=[]
                        for t in states:
                            s,a=site(c,arm,t,'reference',n);b=scale(c,n)
                            value=np.mean(a['maximum']<=.01*b) if metric=='dead' else np.mean(a['rms']<=.01*b) if metric=='weak' else s['top10pct_energy']
                            vals.append(100*value)
                        z.append(vals)
                    z=np.array(z);record(f'time/{c}/{arm}/{metric}',z)
                    im=axes[row,col].imshow(z,aspect='auto',vmin=0,vmax=100,cmap='viridis')
                    axes[row,col].set_xticks(range(6),states,rotation=45,fontsize=7)
                    axes[row,col].set_yticks(range(len(names)),[r.label(c,n) for n in names] if col==0 else [],fontsize=5.5 if c=='resnet' else 8)
                    axes[row,col].set_xlabel('Saved training step t (uneven intervals)')
                    if col==0:axes[row,col].set_ylabel(['Inactive units\n(max <= 1% scale)','Weak units\n(RMS <= 1% scale)','Top 10% unit\nenergy share'][row])
                    if row==0:axes[row,col].set_title(ARM[arm])
            fig.subplots_adjust(left=.20,right=.91,top=.90 if c=='resnet' else .87,bottom=.10 if c=='resnet' else .15,hspace=.28 if c=='resnet' else .50,wspace=.15)
            cb=fig.add_axes([.93,.22,.012,.55]);fig.colorbar(im,cax=cb,label='Percent')
            note(fig,NAMES[c]+': layer activity at the saved states','Original reference examples; the six columns are selected saved states, NOT equally spaced updates. Threshold scale is frozen at the full-training common fork. No inference about intermediate steps is made. Same fork is shared by all arms.',5+pi);save(pdf,fig)

    # Separate atlas: all sites, fixed unit order, no sample or spatial averaging.
    page=0
    with PdfPages(HERE/'activation-matrices.pdf',metadata={'Title':'Activation matrices: every measured site','CreationDate':None,'ModDate':None}) as pdf:
        for c in CASES:
            d,_=r.load(c,'control',r.END[c],'full_training');names=[s['site'] for s in d['sites']]
            for start in range(0,len(names),4):
                selected=names[start:start+4];fig,axes=plt.subplots(len(selected),3,figsize=(11.7,8.3),squeeze=False);images=[]
                for row,n in enumerate(selected):
                    values=[]
                    for arm in ARMS:
                        s,a=site(c,arm,r.END[c],'full_training',n);m=a['matrix_values'];idx=np.linspace(0,len(m)-1,min(512,len(m))).astype(int);values.append(m[idx]);record(f'matrix/{c}/{arm}/{n}',m[idx])
                    q=float(np.quantile(np.concatenate([v.ravel() for v in values]),.995)) if s['kind'] in ['relu','pooled'] else float(np.quantile(np.abs(np.concatenate([v.ravel() for v in values])),.995))
                    q=max(q,1e-12);positive=all(np.min(v)>=0 for v in values);norm=Normalize(0,q) if positive else TwoSlopeNorm(0,-q,q)
                    RECORD['matrix_color_limits'].append(dict(case=c,site=n,vmin=0 if positive else -q,vmax=q,pooled_clipped_fraction=float(np.mean(np.abs(np.concatenate([v.ravel() for v in values]))>q))))
                    for col,(arm,z) in enumerate(zip(ARMS,values)):
                        ax=axes[row,col];im=ax.imshow(z.T,aspect='auto',interpolation='none',cmap='viridis' if positive else 'RdBu_r',norm=norm)
                        if row==0:ax.set_title(ARM[arm])
                        ax.set_xlabel('Selected input-position row (original order)',fontsize=7)
                        ax.set_xticks([0,len(z)-1],[0,len(z)-1]);ax.tick_params(labelsize=6)
                        if col==0:ax.set_ylabel(r.label(c,n)+'\nUnit index',fontsize=7)
                    images.append(im)
                # Reserve labels and captions without shrinking the site matrices.
                fig.subplots_adjust(left=.18,right=.88,top=.86,bottom=.14,hspace=.50,wspace=.18)
                for row,im in enumerate(images):
                    pos=axes[row,-1].get_position();cb=fig.add_axes([.90,pos.y0,.012,pos.height]);fig.colorbar(im,cax=cb,label='Activation');cb.tick_params(labelsize=6)
                page+=1
                note(fig,NAMES[c]+f': activation matrices at t={r.END[c]} (sites {start+1}-{start+len(selected)})','Rows in each image are units; columns are deterministic input-position selections from the first original physical batch. No averaging. All unit columns retained; up to 512 input-position rows shown from the saved 4,096-row subset. Color limits shared across arms per site, clipped at the pooled 99.5th magnitude percentile; full-data statistics never clip.',page);save(pdf,fig)
    for n in ['activation-summary.pdf','activation-matrices.pdf']:RECORD['figures'][n]=hashlib.sha256((HERE/n).read_bytes()).hexdigest()
    (HERE/'figure-manifest.json').write_text(json.dumps(RECORD,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps({'pdfs':2,'summary_pages':7,'atlas_pages':page,'plotted_arrays':len(RECORD['arrays'])}))
if __name__=='__main__':main()
