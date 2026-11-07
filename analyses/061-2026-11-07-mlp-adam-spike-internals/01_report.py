"""Verified layer-level report and publication figures from exact Adam replay."""
import csv
import hashlib
import json
from pathlib import Path
import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
RUN=ROOT/'runs/236-2026-11-07-mlp-adam-spike-internals'
RAW=RUN/'artifacts/spark/payload/worker'
SOURCES={};PLOTTED=[]
EVENTS=['nearby_large','last_complete_excursion','last_local_peak','post512_control']
COLORS=['#CC79A7','#009E73','#0072B2','#D55E00']
LABELS=['Peak 474','Peak 493','Peak 503','Peak 522']
LAYERS=['1.weight','3.weight','5.weight']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    SOURCES[p.relative_to(ROOT).as_posix()]=sha(p)
    return json.loads(p.read_text())
def rows(p):
    SOURCES[p.relative_to(ROOT).as_posix()]=sha(p)
    return [json.loads(s) for s in p.read_text().splitlines()]
def line(ax,x,y,figure,panel,series,**kwargs):
    artist,=ax.plot(x,y,**kwargs)
    np.testing.assert_array_equal(artist.get_xdata(),x);np.testing.assert_array_equal(artist.get_ydata(),y)
    PLOTTED.extend(dict(figure=figure,panel=panel,series=series,x=float(a),y=float(b)) for a,b in zip(x,y))
    return artist
def bars(ax,x,y,figure,panel,series,**kwargs):
    artists=ax.bar(x,y,**kwargs)
    # Matplotlib subtracts the stacked baseline when storing bar heights.
    np.testing.assert_allclose([r.get_height() for r in artists],y,rtol=1e-12,atol=1e-12)
    PLOTTED.extend(dict(figure=figure,panel=panel,series=series,x=float(a),y=float(b)) for a,b in zip(x,y))
def save(fig,name):
    fig.canvas.draw();box=fig.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans)
    assert box.x0>=0 and box.y0>=0 and box.x1<=fig.bbox.width and box.y1<=fig.bbox.height
    path=HERE/'figures'/name;fig.savefig(path,metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
    scratch=ROOT/'tmp/pdfs/analysis061';scratch.mkdir(parents=True,exist_ok=True)
    with fitz.open(path) as doc:
        assert len(doc)==1 and not doc[0].get_images()
        for f in doc[0].get_fonts(full=True):assert f[2]!='Type3' and doc.extract_font(f[0])[3]
        for word in doc[0].get_text('words'):assert doc[0].rect.contains(fitz.Rect(word[:4]))
        doc[0].get_pixmap(matrix=fitz.Matrix(2,2)).save(scratch/(Path(name).stem+'.png'))
    return sha(path)


def main():
    assert read(RUN/'verification_final.json')['status']=='passed'
    assert read(HERE/'probe_verification.json')['status']=='passed'
    data=rows(RAW/'internals.jsonl');lookup={r['state']:r for r in data}
    counter=rows(RAW/'counterfactuals.jsonl');rises={r['event']:r for r in counter if r['phase']=='rise'}
    curves={r['event']:r for r in rows(RAW/'step_curves.jsonl')}
    feature={r['event']:r for r in read(HERE/'feature_probe.json')['events']}
    norm={r['event']:r for r in read(HERE/'norm_direction_probe.json')['events']}
    config=read(RUN/'config.json');summary={}
    for e in config['events']:
        name=e['name'];r=rises[name];delta=r['hybrid']['total_loss_change'];a,b,c=[lookup[e[k]] for k in ['before','peak','recovery']]
        shares={key:100*value/delta for key,value in r['hybrid']['shapley'].items()}
        summary[name]=dict(states=[e['before'],e['peak'],e['recovery']],loss=[q['loss'] for q in [a,b,c]],
            peak_increase_pct=100*(b['loss']/a['loss']-1),weight_norm_change_pct={k:100*r['changes']['parameters'][k]['relative_norm_change'] for k in LAYERS},
            bias_norm_change_pct={k:100*r['changes']['parameters'][k]['relative_norm_change'] for k in r['changes']['parameters'] if k.endswith('bias')},
            activation_rms={k:[q['activations'][k]['rms'] for q in [a,b,c]] for k in ['z1','h1','z2','h2','output']},
            saturation95_pct={k:[100*q['activations'][k]['saturated95'] for q in [a,b,c]] for k in ['h1','h2']},
            mean_tanh_slope={k:[q['activations'][k]['mean_tanh_slope'] for q in [a,b,c]] for k in ['h1','h2']},
            shapley_loss_change_pct=shares,bias_combined_shapley_pct=sum(v for k,v in shares.items() if k.endswith('bias')),
            hidden_feature_loss_allocation_pct=100*r['output']['terms']['hidden_features']['loss_allocation']/r['output']['loss_change_fp64_head'],
            native_output_change_rms=r['output']['native_output_change_rms'],mean_residual_loss=[q['output']['mean_residual_loss'] for q in [a,b,c]],
            accuracy=[q['output']['accuracy'] for q in [a,b,c]],
            norm_probe_losses=norm[name]['losses'],norm_restored_excess_pct=100*(norm[name]['losses']['peak_restore_all_norms']-a['loss'])/delta,
            old_head_excess_pct=100*(norm[name]['losses']['peak_old_head']-a['loss'])/delta,
            feature_probe=feature[name],incoming_step=curves[name],best_sampled_step_scale=min(curves[name]['curve'],key=lambda q:q['loss']))
    plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'stix','font.size':8.5,
        'axes.labelsize':8.5,'axes.titlesize':9,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
        'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'path.simplify':False})
    (HERE/'figures').mkdir(exist_ok=True)
    fig,axes=plt.subplots(3,2,figsize=(7.16,7.1),sharex=True)
    fig.subplots_adjust(left=.105,right=.98,bottom=.11,top=.91,hspace=.46,wspace=.30)
    x=[r['state'] for r in data]
    for ax in axes.flat:
        ax.axvspan(502,505,color='#0072B2',alpha=.09,lw=0)
        ax.axvline(512,color='.4',ls=':',lw=.7)
        ax.set_xlim(448,535);ax.set_xticks([448,474,493,512,535]);ax.grid(alpha=.15,lw=.4)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    line(axes[0,0],x,[r['loss'] for r in data],'01','a','training_loss',color='.25',lw=1)
    for name,color,label in zip(EVENTS,COLORS,LABELS):
        peak=summary[name]['states'][1]
        line(axes[0,0],[peak],[lookup[peak]['loss']],'01','a',label,color=color,marker='o',ms=3,ls='none',label=label)
    axes[0,0].set(title='(a) Loss peaks and recovery',ylabel='Full training loss')
    axes[0,0].legend(frameon=False,fontsize=6.6,ncol=1,loc='upper left')
    for key,color,label in zip(LAYERS,['#0072B2','#D55E00','#009E73'],['W1','W2','W3 (output)']):
        values=[100*(r['parameters'][key]['norm']/data[0]['parameters'][key]['norm']-1) for r in data]
        line(axes[0,1],x,values,'01','b',label,color=color,lw=.9,label=label)
        bias=key.replace('weight','bias');values=[100*(r['parameters'][bias]['norm']/data[0]['parameters'][bias]['norm']-1) for r in data]
        line(axes[1,0],x,values,'01','c',label.replace('W','b'),color=color,lw=.9,label=label.replace('W','b'))
    axes[0,1].set(title='(b) Weight norms grow smoothly',ylabel='Change from state 448 (%)')
    axes[1,0].set(title='(c) Bias norms grow smoothly',ylabel='Change from state 448 (%)')
    for ax in [axes[0,1],axes[1,0]]:ax.legend(frameon=False,fontsize=7,loc='upper left')
    for key,color,label in [('h1','#0072B2','Hidden 1'),('h2','#D55E00','Hidden 2'),('output','#009E73','Output')]:
        values=[100*(r['activations'][key]['rms']/data[0]['activations'][key]['rms']-1) for r in data]
        line(axes[1,1],x,values,'01','d',key,color=color,lw=.9,label=label)
        if key!='output':
            line(axes[2,0],x,[100*r['activations'][key]['saturated95'] for r in data],'01','e',key,color=color,lw=.9,label=label)
            line(axes[2,1],x,[r['activations'][key]['mean_tanh_slope'] for r in data],'01','f',key,color=color,lw=.9,label=label)
    axes[1,1].set(title='(d) Hidden RMS barely changes',ylabel='RMS change from state 448 (%)')
    axes[1,1].legend(frameon=False,fontsize=7,loc='center left')
    axes[2,0].set(title=r'(e) Saturation: $|h|>0.95$',ylabel='Fraction of activations (%)',yscale='log')
    axes[2,0].set_yticks([.1,1,10,100]);axes[2,0].set_yticklabels(['0.1','1','10','100'])
    axes[2,1].set(title='(f) Hidden 2 remains responsive',ylabel=r'Mean $1-h^2$',ylim=(0,1))
    for ax in axes[2]:ax.set_xlabel('Parameter state');ax.legend(frameon=False,fontsize=7,loc='center left')
    fig.suptitle('Adam control: internal magnitudes around the loss peaks',y=.975,fontsize=11)
    fig.text(.5,.025,'Blue band: last local peak and return (502–505). Dotted line: state 512. All 5,000 training examples.',ha='center',fontsize=7)
    pdfs={'01-internal-time-course.pdf':save(fig,'01-internal-time-course.pdf')}
    fig,axes=plt.subplots(2,2,figsize=(7.16,6.4))
    fig.subplots_adjust(left=.10,right=.98,bottom=.12,top=.89,hspace=.42,wspace=.32)
    positions=np.arange(4)
    bottom=np.zeros(4)
    for keys,color,label in [( ['1.weight'],'#0072B2','W1'),(['3.weight'],'#D55E00','W2'),(['5.weight'],'#009E73','W3'),(['1.bias','3.bias','5.bias'],'#777777','Biases')]:
        values=np.array([sum(summary[e]['shapley_loss_change_pct'][k] for k in keys) for e in EVENTS])
        bars(axes[0,0],positions,values,'02','a',label,bottom=bottom,width=.65,color=color,label=label)
        bottom+=values
    axes[0,0].set(title='(a) Loss-change allocation by parameter group',ylabel='Shapley share of loss rise (%)',ylim=(-2,115))
    axes[0,0].legend(frameon=False,ncol=4,fontsize=6.5,loc='upper center',columnspacing=.7)
    for shift,key,color,label in [(-.25,'peak','#0072B2','Actual peak'),(0,'peak_restore_all_norms','#D55E00','Restore norms'),(.25,'before_grow_all_norms','#009E73','Only norm growth')]:
        values=[summary[e]['norm_probe_losses'][key]/summary[e]['loss'][0] for e in EVENTS]
        bars(axes[0,1],positions+shift,values,'02','b',key,width=.23,color=color,label=label)
    axes[0,1].axhline(1,color='.3',ls=':',lw=.7)
    axes[0,1].set(title='(b) Removing norm growth leaves the spike',ylabel='Loss / pre-rise loss',ylim=(0,2.65))
    axes[0,1].legend(frameon=False,fontsize=6.7,loc='upper left')
    for shift,key,color,label in [(-.18,'baseline_feature_rowspace_energy','#777777','Baseline features'),(.18,'feature_change_rowspace_energy','#0072B2','Feature change')]:
        values=[100*feature[e]['readout'][key] for e in EVENTS]
        bars(axes[1,0],positions+shift,values,'02','c',key,width=.34,color=color,label=label)
    axes[1,0].axhline(5,color='#D55E00',ls=':',lw=.9,label='Isotropic reference: 5%')
    axes[1,0].set(title='(c) Changes align with output-sensitive directions',ylabel='Energy in output-head row space (%)',ylim=(0,100))
    axes[1,0].legend(frameon=False,fontsize=6.7,loc='upper left')
    for event,color,label in zip(EVENTS,COLORS,LABELS):
        curve=curves[event]['curve'];base=curve[0]['loss']
        line(axes[1,1],[r['scale'] for r in curve],[r['loss']/base for r in curve],'02','d',event,color=color,lw=1,label=label)
    axes[1,1].axvline(1,color='.3',ls=':',lw=.8);axes[1,1].axhline(1,color='.6',lw=.5)
    axes[1,1].set(title='(d) The full incoming step overshoots',xlabel='Scale of actual incoming Adam step',ylabel='Loss / loss before the step',xlim=(0,1.5))
    axes[1,1].legend(frameon=False,fontsize=6.8,ncol=2,loc='upper left')
    for ax in [axes[0,0],axes[0,1],axes[1,0]]:
        ax.set_xticks(positions);ax.set_xticklabels(['474','493','503','522']);ax.set_xlabel('Peak state')
    for ax in axes.flat:ax.grid(axis='y',alpha=.15,lw=.4);ax.set_axisbelow(True)
    fig.suptitle('Static probes: feature displacement and finite-step overshoot',y=.98,fontsize=11)
    fig.text(.5,.035,'Same retained checkpoints and full objective. Parameter recombinations are not retrained interventions.',ha='center',fontsize=7)
    pdfs['02-mechanism-probes.pdf']=save(fig,'02-mechanism-probes.pdf')
    with (HERE/'plotted_values.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(PLOTTED[0]),lineterminator='\n');writer.writeheader();writer.writerows(PLOTTED)
    result=dict(status='completed',events=summary,source_sha256=SOURCES,generator_sha256=sha(Path(__file__)),
        pdf_sha256=pdfs,plotted_points=len(PLOTTED),plotted_values_sha256=sha(HERE/'plotted_values.csv'),
        smoothing=None,scope='One selected seed/window; static recombinations and exact replay, not universal causal identification.')
    (HERE/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(status='completed',figures=list(pdfs),plotted_points=len(PLOTTED),
        summaries={k:{x:v for x,v in r.items() if x in ['states','loss','shapley_loss_change_pct','hidden_feature_loss_allocation_pct','norm_restored_excess_pct','best_sampled_step_scale']} for k,r in summary.items()}),indent=2))


if __name__=='__main__':main()
