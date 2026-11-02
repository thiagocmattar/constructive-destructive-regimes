"""Audit every retrospective pair and render separate GD/Adam endpoint figures."""
import csv
import hashlib
import json
from pathlib import Path

import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).parent
RUN=ROOT/'runs/212-2026-11-02-eos-backward-anchors'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(path,value):
    path.parent.mkdir(exist_ok=True,parents=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def main():
    closeout=json.loads((HERE/'closeout.json').read_text())
    assert closeout['status']=='complete' and closeout['cloud_storage_remaining_gb']==0
    cfg=json.loads((RUN/'config.json').read_text())
    profiles=[];audits=[];flat=[];receipts=[]
    for receiptpath in sorted(HERE.glob('retrieval-212-worker*-attempt*.json')):
        receipt=json.loads(receiptpath.read_text())
        assert receipt['verified']
        receipts.append(receipt)
        root=ROOT/receipt['artifact_root']
        assert sha(root/'inventory.json')==receipt['inventory_sha256']
        if receipt['scientific_status']!='complete':continue
        manifest=json.loads((root/'manifest.json').read_text())
        assert manifest['status']=='complete' and manifest['config_sha256']==sha(RUN/'config.json')
        for job in [j for j in cfg['jobs'] if j['worker']==receipt['worker']]:
            out=root/job['id'];r=json.loads((out/'result.json').read_text())
            assert r['job']==job and r['status']=='complete'
            assert r['gradients']==257 and r['applied_updates']==256 and r['endpoint']==job['start']+256
            checks=r['boundary_checks']
            assert checks['initial_tensor_sha256']==job['tensor_sha256']
            assert checks['source_gradient_relative_error']<1e-5 and checks['compute_chunk_relative_error']<1e-5
            assert abs(checks['source_loss_error'])<1e-5 and abs(checks['compute_chunk_loss_error'])<1e-5
            rows=r['rows'];scalars=r['scalar_rows']
            assert [x['lag'] for x in rows]==list(range(1,257))
            assert [x['state'] for x in scalars]==list(range(job['start'],job['start']+257))
            assert len(scalars)==257 and len({x['gradient_sha256'] for x in scalars})>1
            for row in rows:
                assert row['reference_state']==r['endpoint'] and row['compared_state']==r['endpoint']-row['lag']
                assert row['defined'] and np.isfinite(row['cosine']) and abs(row['cosine'])<=1+1e-12
                for product in [row['cpu'],row['gpu']]:
                    assert product['norm']>0 and product['anchor_norm']>0
                    assert abs(product['cosine']-product['dot']/(product['norm']*product['anchor_norm']))<1e-12
                assert abs(row['cosine']-row['cpu']['cosine'])<1e-10
                assert abs(row['gpu']['norm']/scalars[256-row['lag']]['norm']-1)<1e-10
                flat.append(dict(optimizer=job['optimizer'],task=job['task'],parent_run_id=job['parent_run_id'],
                    parent_gd_lr=job['parent_gd_lr'],lr=job['lr'],start=job['start'],endpoint=r['endpoint'],**row))
            assert abs(rows[0]['cosine']-scalars[-1]['adjacent_cosine'])<1e-10
            assert abs(rows[-1]['cosine']-scalars[-1]['forward_cosine'])<1e-10
            vectors=np.load(out/'audit_gradients.npy',mmap_mode='r')
            assert vectors.shape==(6,r['parameter_count']) and vectors.dtype==np.float32
            for offset,vector in zip(r['retained_offsets'],vectors):
                assert np.isfinite(vector).all()
                assert hashlib.sha256(vector.tobytes()).hexdigest()==scalars[offset]['gradient_sha256']
            endpoint=np.asarray(vectors[-1],dtype=np.float64)
            for offset,vector in zip(r['retained_offsets'][:-1],vectors[:-1]):
                earlier=np.asarray(vector,dtype=np.float64)
                actual=float(np.dot(endpoint,earlier)/(np.linalg.norm(endpoint)*np.linalg.norm(earlier)))
                error=abs(actual-rows[256-offset-1]['cosine'])
                assert error<1e-10
                audits.append(dict(job=job['id'],lag=256-offset,error=error))
            cp=torch.load(out/'end.pt',map_location='cpu',weights_only=False)
            assert cp['step']==r['endpoint'] and cp['offset']==256 and cp['last_gradient_unapplied']
            if job['optimizer']=='adam':
                states=cp['optimizer']['state'].values()
                assert states and all(int(s['step'])==256 for s in states)
            group=cp['optimizer']['param_groups'][0]
            assert group['lr']==job['lr'] and group['weight_decay']==0
            profiles.append(dict(job=job,endpoint=r['endpoint'],rows=rows,scalar_rows=scalars,
                                 artifact_root=out.relative_to(ROOT).as_posix(),boundary_checks=checks))
            del vectors,endpoint,earlier,cp
    assert len(profiles)==18 and len(flat)==4608 and len(audits)==90
    assert len({p['job']['id'] for p in profiles})==18
    save(HERE/'results.json',dict(profiles=profiles))
    historical={opt:{(x['parent_run_id'],x['anchor']):x for x in json.loads((ROOT/path/'results.json').read_text())['profiles']}
                for opt,path in [('gd','analyses/033-2026-11-01-eos-long-lags'),
                                 ('adam','analyses/034-2026-11-01-eos-adam-continuations')]}
    replay=[]
    for profile in profiles:
        job=profile['job'];old=historical[job['optimizer']][(job['parent_run_id'],job['start'])]
        field='cosine' if job['optimizer']=='gd' else 'anchor_cosine'
        pairs=list(zip(profile['scalar_rows'],old['rows'][:257]))
        assert len(pairs)==257 and all(new['offset']==before['lag'] for new,before in pairs)
        replay.append(dict(job=job['id'],boundaries=257,
            maximum_loss_difference=max(abs(a['loss']-b['loss']) for a,b in pairs),
            maximum_norm_relative_difference=max(abs(a['norm']/b['norm']-1) for a,b in pairs),
            maximum_forward_cosine_difference=max(abs(a['forward_cosine']-b[field]) for a,b in pairs)))
    save(HERE/'historical_comparison.json',dict(profiles=replay,
        interpretation='Same source tensors and full objective; later FP32 replay differences are reported, not silently treated as identical historical trajectories. Not a stopping or selection criterion.'))
    with (HERE/'lag_profiles.csv').open('w',newline='',encoding='utf-8') as f:
        fields=['optimizer','task','parent_run_id','parent_gd_lr','lr','start','endpoint','lag','reference_state','compared_state','cosine','defined']
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(flat)
    with (HERE/'lag_profiles.csv').open(newline='',encoding='utf-8') as f:
        independent={(x['optimizer'],x['task'],int(x['start']),int(x['lag'])):float(x['cosine']) for x in csv.DictReader(f)}
    save(HERE/'verification.json',dict(status='passed',profiles=18,gradients=4626,updates=4608,backward_pairs=4608,
        local_full_vector_pair_audits=audits,endpoint_first_and_last_pair_identities=True,
        maximum_cpu_gpu_cosine_error=max(abs(x['cosine']-x['cpu']['cosine']) for x in flat),
        all_start_and_chunk_qualifications_pass=True,all_recovery_clocks_pass=True,
        transferred_files=sum(x['files'] for x in receipts),transferred_bytes=sum(x['bytes'] for x in receipts),
        source_config_sha256=sha(RUN/'config.json'),source_manifest_sha256=sha(RUN/'input_manifest.json')))
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'path.simplify':False,
                         'axes.spines.top':False,'axes.spines.right':False})
    figure_records=[]
    for optimizer in ['gd','adam']:
        fig,axes=plt.subplots(3,3,figsize=(12,8.8),layout='constrained')
        lines=[]
        for i,start in enumerate([500,2000,3000]):
            for j,(task,title) in enumerate(zip(['mlp','resnet','transformer'],
                ['CIFAR-10 / MLP','CIFAR-10 / ResNet-32','WikiText-2 / Transformer'])):
                p=next(x for x in profiles if x['job']['optimizer']==optimizer and x['job']['task']==task and x['job']['start']==start)
                job=p['job'];x=np.array([r['lag'] for r in p['rows']]);y=np.array([r['cosine'] for r in p['rows']])
                assert np.array_equal(x,np.arange(1,257))
                assert all(independent[(optimizer,task,start,int(k))]==value for k,value in zip(x,y))
                ax=axes[i,j];artist,=ax.plot(x,y,color='#246b9a',linewidth=.7,drawstyle='default')
                assert np.array_equal(artist.get_xdata(),x) and np.array_equal(artist.get_ydata(),y)
                assert len(ax.lines)==1
                ax.axhline(0,color='.55',linewidth=.65,zorder=0)
                ax.set(xlim=(1,256),ylim=(-1.05,1.05),xticks=[1,64,128,192,256],yticks=[-1,-.5,0,.5,1])
                if i==0:
                    rate=f'GD LR {job["lr"]:g}' if optimizer=='gd' else f'Adam LR {job["lr"]:g}; source GD LR {job["parent_gd_lr"]:g}'
                    ax.set_title(f'{title}\n{rate}',fontsize=11)
                if j==0:ax.set_ylabel(f'Anchor t = {p["endpoint"]:,}\nstart {start:,}\ncos(gₜ, gₜ₋ₖ)')
                if i==2:ax.set_xlabel('Backward lag k (optimizer updates)')
                lines.append(dict(job=job['id'],points=256,endpoint=p['endpoint'],start=start,
                    exact_json_csv_artist_match=True,plotted_values_sha256=hashlib.sha256(
                        json.dumps(list(zip(x.tolist(),y.tolist())),separators=(',',':')).encode()).hexdigest()))
        fig.suptitle(f'Backward anchored raw-gradient cosine along {"GD" if optimizer=="gd" else "Adam"}',fontsize=14)
        fig.supxlabel('Fixed endpoint t = retained start + 256; compare gₜ with gₜ₋ₖ for every k = 1–256.',fontsize=9)
        pdf=HERE/'figures'/f'{optimizer}_backward_lag256.pdf';pdf.parent.mkdir(exist_ok=True)
        fig.savefig(pdf,bbox_inches='tight',metadata={'Creator':Path(__file__).name});plt.close(fig)
        png=HERE/'scratch'/f'{optimizer}_backward_lag256.png'
        with fitz.open(pdf) as doc:
            assert len(doc)==1 and not doc[0].get_images()
            for block in doc[0].get_text('dict')['blocks']:
                for line in block.get('lines',[]):
                    for span in line['spans']:assert doc[0].rect.contains(fitz.Rect(span['bbox']))
            doc[0].get_pixmap(matrix=fitz.Matrix(1.6,1.6)).save(png)
        figure_records.append(dict(figure=pdf.relative_to(HERE).as_posix(),sha256=sha(pdf),
            render=png.relative_to(HERE).as_posix(),render_sha256=sha(png),profiles=lines,plotted_points=2304,
            vector_pdf=True,text_within_page=True,visual_review='pending actual PDF render inspection'))
    save(HERE/'figure_review.json',dict(status='numerical/structural passed; visual pending',figures=figure_records,
        generator=Path(__file__).name,generator_sha256=sha(Path(__file__)),
        source_results_sha256=sha(HERE/'results.json'),source_csv_sha256=sha(HERE/'lag_profiles.csv')))
    print(json.dumps(dict(profiles=len(profiles),pairs=len(flat),audits=len(audits),figures=2)))


if __name__=='__main__':main()
