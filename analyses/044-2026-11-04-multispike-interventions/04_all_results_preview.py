"""One preliminary Run223 figure; complete cases plus hash-verified live snapshot."""
import base64,csv,datetime,hashlib,importlib.util,json,shlex
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fitz
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];RUN=ROOT/'runs/223-2026-11-04-multispike-interventions'
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    spec=importlib.util.spec_from_file_location('cloud223',RUN/'02_cloud.py');cloud=importlib.util.module_from_spec(spec);spec.loader.exec_module(cloud)
    snapshot=HERE/'scratch/all_results_snapshot';snapshot.mkdir(exist_ok=True,parents=True)
    cfg=read(RUN/'config.json');sources=[];verified=0;series={};provenance={}
    for case,folder in [('mlp','spark'),('resnet','duqoihms8w1gle')]:
        base=RUN/'artifacts'/folder;worker=base/'payload/worker'
        assert read(base/'receipt.json')['verified']
        for item in read(base/'payload/transfer_inventory.json')['files']:
            f=base/'payload'/item['path'];assert f.stat().st_size==item['bytes'] and sha(f)==item['sha256'];verified+=1
        assert read(worker/'qualification/qualification.json')['status']=='passed'
        assert read(worker/'manifest.json')['control']=='fresh_same_worker'
        prefix=rows(worker/'prefix/boundaries.jsonl')[:-1];series[case]={}
        for name,sub in [('unchanged','control'),('fixed_half','fixed_half'),('cosine32_half','cosine32_half')]:
            f=worker/sub/'boundaries.jsonl';series[case][name]=prefix+rows(f)
            sources.append(dict(path=f.relative_to(ROOT).as_posix(),sha256=sha(f)))
        sources.append(dict(path=(worker/'prefix/boundaries.jsonl').relative_to(ROOT).as_posix(),sha256=sha(worker/'prefix/boundaries.jsonl')))
        provenance[case]=read(worker/'manifest.json')['gpu']
    pods=[p for p in cloud.state()['pods'] if p['case']=='transformer' and not p.get('deleted_epoch')]
    series['transformer']={};live=[]
    for pod in pods:
        code="from pathlib import Path; import json,base64,hashlib,datetime; p=Path("+repr(cloud.OUT)+"); files=[]\nfor f in sorted(p.rglob('*')):\n if f.is_file() and (f.name in ['boundaries.jsonl','manifest.json','historical_qualification.json','READY.json','event_gate.json','cross_worker_qualification.json','qualification.json']):\n  b=f.read_bytes()\n  if f.suffix=='.jsonl': b=b[:b.rfind(b'\\n')+1]\n  files.append(dict(path=f.relative_to(p).as_posix(),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),data=base64.b64encode(b).decode()))\nprint(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files)))"
        packet=json.loads(cloud.x.ssh(pod,'python3 -c '+shlex.quote(code)+'\n',timeout=30));dest=snapshot/pod['id'];dest.mkdir(exist_ok=True)
        for item in packet['files']:
            b=base64.b64decode(item.pop('data'));assert len(b)==item['bytes'] and hashlib.sha256(b).hexdigest()==item['sha256']
            f=dest/item['path'];f.parent.mkdir(exist_ok=True,parents=True);f.write_bytes(b);verified+=1;sources.append(dict(path=f.relative_to(ROOT).as_posix(),sha256=item['sha256']))
        (dest/'snapshot_inventory.json').write_text(json.dumps(packet,indent=2)+'\n');live.append(dict(pod=pod['id'],role=pod['role'],utc=packet['utc'],files=packet['files']))
        provenance['transformer']=pod['gpu']
        if pod['role']=='leader':
            prefix=rows(dest/'prefix/boundaries.jsonl');series['transformer']['unchanged']=prefix
            if (dest/'control/boundaries.jsonl').exists():series['transformer']['unchanged']=prefix[:-1]+rows(dest/'control/boundaries.jsonl')
            leader_prefix=prefix
        for arm in ['fixed_half','cosine32_half']:
            if (dest/arm/'boundaries.jsonl').exists():
                if pod['role']=='follower':assert read(dest/'cross_worker_qualification.json')['two_update_replay_exact']
                series['transformer'][arm]=leader_prefix[:-1]+rows(dest/arm/'boundaries.jsonl')
    metrics=[('loss','Training loss (log)'),('rho',r'Raw $\rho_t$'),('raw_gradient_norm',r'$\|g_t\|$'),('eta_kappa_hat',r'$\eta\widehat\kappa_t$'),('applied_lr','Applied / base LR')]
    arms=[('fixed_half','Immediate half','#0072B2'),('cosine32_half','Cosine32 to half','#009E73')]
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
    fig,axes=plt.subplots(5,6,figsize=(22,11),sharex='col',layout='constrained');values=[];coverage={}
    for ci,(case,sp) in enumerate(cfg['cases'].items()):
        coverage[case]={n:max(r['state'] for r in data) for n,data in series[case].items()}
        for ai,(arm,title,color) in enumerate(arms):
            col=ci*2+ai
            for ri,(metric,label) in enumerate(metrics):
                ax=axes[ri,col]
                for name,c,legend in [('unchanged','#777777','Unchanged'),(arm,color,title)]:
                    if name not in series[case]:continue
                    data=[r for r in series[case][name] if sp['plot_start']<=r['state']<=sp['end'] and r.get(metric) is not None]
                    xx=[r['state'] for r in data];yy=[r[metric]/sp['lr'] if metric=='applied_lr' else r[metric] for r in data]
                    for x,y in zip(xx,yy):values.append(dict(case=case,panel=arm,series=name,metric=metric,state=x,value=y))
                    if metric=='applied_lr' and xx and xx[-1]==sp['end']-1:xx.append(sp['end']);yy.append(yy[-1])
                    ax.plot(xx,yy,color=c,lw=.85,ls='-',label=legend,drawstyle='steps-post' if metric=='applied_lr' else 'default')
                ax.axvspan(sp['plot_start'],sp['start'],color='#eeeeee',zorder=-3);ax.axvline(sp['start'],color='#444444',lw=.65)
                ax.set_xlim(sp['plot_start'],sp['end']);ax.grid(alpha=.13);ax.spines[['top','right']].set_visible(False)
                if ai==0:ax.set_ylabel(label)
                if ri==0:
                    ax.set_yscale('log');ax.set_title(case.upper()+' | '+title+'\n'+provenance.get(case,'pending')+' | base LR '+str(sp['lr']),fontsize=10);ax.legend(fontsize=8,loc='best')
                    if arm not in series[case]:ax.text(.5,.20,'Intervention pending',ha='center',transform=ax.transAxes,fontsize=10,bbox=dict(facecolor='white',edgecolor='#bbbbbb',alpha=.95))
                if ri==4:ax.set_ylim(.45,1.06);ax.set_xlabel('Completed optimizer updates')
    stamp=datetime.datetime.now(datetime.timezone.utc).astimezone(datetime.timezone(datetime.timedelta(hours=-3))).strftime('%d %b %Y, %H:%M BRT')
    fig.suptitle('Run223 preliminary | '+stamp+'\nEach intervention versus unchanged baseline | seed 0 | solid, unsmoothed curves | independent panel scales',fontsize=14)
    fig.supxlabel('Gray region: shared prefix. Vertical line: intervention fork. Each shown trajectory stays on its named GPU; Transformer snapshot is incomplete.\nMLP: 5,000 images; ResNet: 50,000 images; Transformer: 2,088,600 targets. No historical control reuse.',fontsize=10)
    for row in values:
        source=next(r for r in series[row['case']][row['series']] if r['state']==row['state']);value=source[row['metric']]
        if row['metric']=='applied_lr':value/=cfg['cases'][row['case']]['lr']
        assert row['value']==value
    pdf=HERE/'figures/all_results_preliminary.pdf';fig.savefig(pdf);plt.close(fig)
    doc=fitz.open(pdf);assert len(doc)==1 and not doc[0].get_images();doc[0].get_pixmap(matrix=fitz.Matrix(1.2,1.2)).save(HERE/'scratch/all_results_preliminary.png')
    with (HERE/'all_results_preliminary_values.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    result=dict(status='preliminary',snapshot_brt=stamp,coverage=coverage,artifact_files_verified=verified,plotted_values_verified=len(values),sources=sources,live_snapshots=live,pdf_sha256=sha(pdf))
    (HERE/'all_results_preliminary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['sources','live_snapshots']},indent=2))
if __name__=='__main__':main()
