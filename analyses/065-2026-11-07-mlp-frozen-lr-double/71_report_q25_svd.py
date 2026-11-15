"""Verified tables and scientific plots from retained temporal/cross Grams."""
import csv,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
HERE=Path(__file__).resolve().parent;OUT=HERE/'svd_q25';RAW=OUT/'raw'
LABELS={
 'mlp':{'1':'Affine 1','2':'Tanh 1','3':'Affine 2','4':'Tanh 2','output':'Output'},
 'resnet':{'conv1':'Conv 1','layer1':'Group 1','layer2':'Group 2','layer3':'Group 3','avgpool':'Pool','output':'Log-probs','correct_label_log_probability':'Target log-prob'},
 'transformer':{'encoder':'Embedding','transformer_encoder.layers.0.self_attn':'Attention 1','transformer_encoder.layers.0':'Block 1','transformer_encoder.layers.1.self_attn':'Attention 2','transformer_encoder.layers.1':'Block 2','output':'Log-probs','correct_label_log_probability':'Target log-prob'}}
COLORS={'control':'#555d66','cap_q25':'#c55b20'}
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(fig,pdf,case,page):
    fig.text(.01,.012,'Fixed probes; uncentered per-update changes; one initialization. Dimension describes movement, not model capacity.',fontsize=8)
    pdf.savefig(fig);fig.savefig(OUT/'preview'/f'{case}-{page}.png',dpi=125);plt.close(fig)
def main():
    (OUT/'preview').mkdir(exist_ok=True);table=[];comparisons=[];checks=[];sources={};figures=[];dense_checks={};provenance={};all_full={}
    for case in ['mlp','resnet','transformer']:
        root=RAW/case
        if not (root/'reduction/results.json').exists():continue
        data=json.loads((root/'reduction/results.json').read_text());gram=np.load(root/'reduction/grams.npz');sites=list(data['layout']['sites'])
        sources[case]=dict(results_sha256=sha(root/'reduction/results.json'),gram_sha256=sha(root/'reduction/grams.npz'),inventory_sha256=sha(root/'reduction/inventory.json'))
        direct=root/'reduction/dense_verification.json'
        if direct.exists():
            dense_checks[case]=json.loads(direct.read_text());assert dense_checks[case]['status']=='passed'
        histories={arm:[json.loads(t) for t in (root/arm/'boundaries.jsonl').read_text().splitlines()] for arm in COLORS}
        provenance[case]=dict(layout=data['layout'],capture=json.loads((root/'capture-complete.json').read_text()),
            environment=json.loads((root/'environment.json').read_text())['environment'],
            loss_summary={arm:dict(initial=histories[arm][0]['loss'],post_update_mean=float(np.mean([x['loss'] for x in histories[arm][1:]])),final=histories[arm][-1]['loss']) for arm in COLORS},
            qualifications={arm:json.loads((root/arm/'qualification.json').read_text()) for arm in COLORS},
            source_hashes=json.loads((root/'sources.json').read_text()))
        for r in data['results']:
            j=sites.index(r['site']);lo,hi=map(int,r['window'].split(':'));P=r['coordinates']
            comparison=dict(case=case,site=r['site'],window=r['window'])
            comparison.update({k:v for k,v in r['comparison'].items() if not isinstance(v,list)})
            comparisons.append(comparison)
            for arm in COLORS:
                K=gram[f'{j}_{arm}'][lo:hi,lo:hi];ev=np.maximum(np.linalg.eigvalsh((K+K.T)/2)[::-1],0);s=r[arm]
                assert np.allclose(ev,s['eigenvalues'],rtol=1e-8,atol=max(ev[0],1e-30)*1e-12)
                norms=np.array([x['representations'][r['site']]['change_rms'] for x in histories[arm][lo+1:hi+1]])
                assert np.allclose(np.diag(K),norms**2*P,rtol=1e-8,atol=1e-24)
                assert int(np.searchsorted(ev.cumsum()/ev.sum(),.95)+1)==s['rank95']
                record=dict(case=case,site=r['site'],label=LABELS[case][r['site']],window=r['window'],first_change_state=r['first_change_state'],last_change_state=r['last_change_state'],updates=r['updates'],coordinates=P,arm=arm)
                record.update({k:s[k] for k in ['rank95','rank99','energy','coordinate_rms','participation_rank','entropy_rank','top1','top5','top10','retention','reversal_fraction']});table.append(record)
                checks.append(dict(case=case,site=r['site'],window=r['window'],arm=arm,spectrum_trace_diagonal_verified=True))
        full=[r for r in data['results'] if r['window']=='0:64'];labels=[LABELS[case][r['site']] for r in full];x=np.arange(len(full));n=len(full)
        all_full[case]=full
        pdfpath=OUT/f'{case}-q25-svd.pdf'
        with PdfPages(pdfpath) as pdf:
            fig,axs=plt.subplots(2,3,figsize=(13,8.4));fig.subplots_adjust(left=.07,right=.98,top=.88,bottom=.18,hspace=.62,wspace=.36)
            fig.suptitle(f'{case.upper()}: unchanged Adam versus distributed Q25\n64 incoming changes, states {full[0]["first_change_state"]}–{full[0]["last_change_state"]}',fontsize=14)
            ax=axs[0,0]
            for arm,offset in [('control',-.18),('cap_q25',.18)]:ax.bar(x+offset,[r[arm]['rank95'] for r in full],.36,label='Adam' if arm=='control' else 'Q25',color=COLORS[arm])
            ax.set(title='Modes covering 95% of squared movement',ylabel='Rank95',xticks=x,xticklabels=labels);ax.tick_params(axis='x',rotation=45);ax.legend(frameon=False)
            ax=axs[0,1];ratio=np.array([r['comparison']['energy_ratio'] for r in full]);ax.scatter(x,ratio,color=COLORS['cap_q25'],s=38);ax.axhline(1,color='#555',ls='--');ax.set(yscale='log',title='Absolute motion: Q25 / Adam',ylabel='Squared-motion ratio',xticks=x,xticklabels=labels);ax.tick_params(axis='x',rotation=45)
            output=next(r for r in full if r['site']=='output');ax=axs[0,2]
            for arm in COLORS:ax.plot(np.arange(1,65),output[arm]['cumulative'],label=arm,color=COLORS[arm])
            ax.axhline(.95,color='#999',ls=':');ax.set(title='Output spectral concentration',xlabel='Number of modes',ylabel='Cumulative energy fraction',ylim=(0,1.03))
            ax=axs[1,0]
            for arm in COLORS:ax.semilogy(np.arange(1,65),np.maximum(output[arm]['eigenvalues'],1e-30),color=COLORS[arm],label=arm)
            ax.set(title='Output absolute spectrum',xlabel='Mode index',ylabel='Squared singular value')
            ax=axs[1,1]
            for arm,off in [('control',-.18),('cap_q25',.18)]:
                e=[r['comparison']['baseline_energy_outside_own95'] if arm=='control' else r['comparison']['cap_energy_outside_baseline95'] for r in full]
                ax.scatter(x+off,np.array(e)/np.array([r['control']['energy'] for r in full]),color=COLORS[arm],s=32)
            ax.set(yscale='log',title='Motion outside Adam’s fixed 95% subspace',ylabel='Energy / total Adam energy',xticks=x,xticklabels=labels);ax.tick_params(axis='x',rotation=45)
            ax=axs[1,2]
            for arm in COLORS:ax.plot([r['state'] for r in histories[arm]],[r['loss'] for r in histories[arm]],color=COLORS[arm],label=arm)
            ax.set(title='Full-training loss in this window',xlabel='Completed updates',ylabel='Loss')
            save(fig,pdf,case,1)
            fig,axs=plt.subplots(2,4,figsize=(13,8.4));fig.subplots_adjust(left=.06,right=.98,top=.89,bottom=.11,hspace=.38,wspace=.32)
            fig.suptitle(f'{case.upper()}: normalized spectra at every measured site',fontsize=14)
            for ax,r in zip(axs.flat,full):
                for arm in COLORS:ax.plot(np.arange(1,65),r[arm]['cumulative'],color=COLORS[arm],label='Adam' if arm=='control' else 'Q25')
                ax.axhline(.95,color='#999',ls=':');ax.set(title=f'{LABELS[case][r["site"]]} | rank95 {r["control"]["rank95"]} → {r["cap_q25"]["rank95"]}',xlabel='Number of modes',ylabel='Cumulative energy',ylim=(0,1.03));ax.legend(frameon=False,fontsize=8)
            for ax in list(axs.flat)[len(full):]:ax.axis('off')
            save(fig,pdf,case,2)
            fig,axs=plt.subplots(1,3,figsize=(13,5.5));fig.subplots_adjust(left=.07,right=.98,top=.82,bottom=.27,wspace=.3)
            fig.suptitle(f'{case.upper()}: window sensitivity of the number of modes',fontsize=14)
            for ax,window in zip(axs,['0:32','32:64','0:64']):
                rr=[r for r in data['results'] if r['window']==window]
                for arm,offset in [('control',-.18),('cap_q25',.18)]:ax.bar(x+offset,[r[arm]['rank95'] for r in rr],.36,color=COLORS[arm],label='Adam' if arm=='control' else 'Q25')
                ax.set(title=f'{rr[0]["updates"]} changes: {rr[0]["first_change_state"]}–{rr[0]["last_change_state"]}',ylabel='Rank95',xticks=x,xticklabels=labels);ax.tick_params(axis='x',rotation=45);ax.legend(frameon=False)
            save(fig,pdf,case,3)
        figures.append(dict(case=case,path=pdfpath.name,pages=3,sha256=sha(pdfpath)))
    if len(all_full)==3:
        pdfpath=OUT/'comparison-q25-svd.pdf'
        with PdfPages(pdfpath) as pdf:
            fig,axs=plt.subplots(3,2,figsize=(12,12));fig.subplots_adjust(left=.08,right=.97,top=.92,bottom=.15,hspace=.67,wspace=.3)
            fig.suptitle('Q25 versus unchanged Adam: representation-motion spectra',fontsize=15)
            for row,(case,full) in enumerate(all_full.items()):
                labels=[LABELS[case][r['site']] for r in full];x=np.arange(len(full));ax=axs[row,0]
                for arm,off in [('control',-.18),('cap_q25',.18)]:ax.bar(x+off,[r[arm]['rank95'] for r in full],.36,color=COLORS[arm],label='Adam' if arm=='control' else 'Q25')
                ax.set(title=f'{case.upper()} | changes {full[0]["first_change_state"]}-{full[0]["last_change_state"]}',ylabel='Rank95 (squared movement)',xticks=x,xticklabels=labels);ax.tick_params(axis='x',rotation=35);ax.legend(frameon=False,fontsize=8)
                output=next(r for r in full if r['site']=='output');ax=axs[row,1]
                for arm in COLORS:ax.plot(np.arange(1,65),output[arm]['cumulative'],color=COLORS[arm])
                ax.axhline(.95,color='#999',ls=':');ax.set(title=f'Output: rank95 {output["control"]["rank95"]} to {output["cap_q25"]["rank95"]}; energy ratio {output["comparison"]["energy_ratio"]:.4f}',xlabel='Number of modes',ylabel='Cumulative squared movement',ylim=(0,1.03))
            fig.text(.02,.045,'Energy ratio = Q25 / Adam. Compare arms within each row; datasets and coordinate sets differ between rows.',fontsize=9)
            save(fig,pdf,'comparison',1)
        figures.append(dict(case='comparison',path=pdfpath.name,pages=1,sha256=sha(pdfpath)))
    if table:
        with (OUT/'metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
        with (OUT/'comparisons.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(comparisons[0]));w.writeheader();w.writerows(comparisons)
    (OUT/'verification.json').write_text(json.dumps(dict(status='passed',checks=checks,direct_dense_checks=dense_checks,sources=sources),indent=2)+'\n')
    (OUT/'figure-manifest.json').write_text(json.dumps(dict(generator=Path(__file__).name,figures=figures,sources=sources),indent=2)+'\n')
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(dict(cases=list(sources),rows=len(table),pages=sum(x['pages'] for x in figures),checks=len(checks))))
if __name__=='__main__':main()
