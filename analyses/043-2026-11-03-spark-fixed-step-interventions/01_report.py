"""Compile verified partial/final intervention snapshots into before/after responses."""
import csv,hashlib,json,os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import fitz
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
RUN=ROOT/'runs/222-2026-11-03-spark-matched-interventions';PARENT=ROOT/'runs/221-2026-11-03-spark-fixed-step-interventions'
CASES=('mlp','resnet','transformer');TITLES=('MLP / CIFAR-10','ResNet-32 / CIFAR-10','Transformer / WikiText-2')
METRICS=('loss','rho','raw_gradient_norm','eta_kappa_hat','lr_over_base')
LABELS=('Training loss (log)',r'$\rho_t$ (raw)',r'$\|g_t\|$ (log)',r'$\eta\widehat\kappa_t$','Actual LR / base LR')
def read(path):return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def csv_out(path,rows):
 if not rows:return
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 receipt=json.loads((RUN/'artifacts/latest_snapshot.json').read_text());assert receipt['verified']
 snapshot=ROOT/receipt['payload_root']
 if os.name=='nt':snapshot=Path('\\\\?\\'+str(snapshot.resolve()))
 raw=snapshot/'runs/222-2026-11-03-spark-matched-interventions/artifacts/attempts/001-spark'
 manifest=json.loads((raw/'manifest.json').read_text());cfg=json.loads((RUN/'config.json').read_text());parents=json.loads((PARENT/'parent_manifest.json').read_text())
 data={};actions={};statuses={};aligned=[];summary=[];plotted=0;figure_hashes={}
 for case in CASES:
  history=json.loads((ROOT/parents[case]['history']).read_text());assert sha(ROOT/parents[case]['history'])==parents[case]['history_sha256']
  for r in history[960:1024]:aligned.append(dict(case=case,condition='shared_history',origin='historical_preperiod',state=r['state'],relative_update=r['state']-1024,loss=r['loss'],rho=r['rho'],raw_gradient_norm=r['gradient_norm'],eta_kappa_hat=r['eta_kappa_hat'],lr_over_base=1.,parameter_norm=None))
  for arm in cfg['arms']:
   folder=raw/'conditions'/case/arm['id'];status=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else dict(status='queued')
   statuses[case+'/'+arm['id']]=status['status'];rows=read(folder/'boundaries.jsonl');metrics=read(folder/'metrics.jsonl')
   if status['status']!='completed':continue
   assert [r['state'] for r in rows]==list(range(1024,1089)) and len(metrics)==64
   for r,m in zip(rows,metrics):assert r['state']==m['state'] and m['update']==r['state']+1 and r['applied_lr']==m['applied_lr']
   for i,r in enumerate(rows):
    previous_norm=history[1023]['gradient_norm'] if i==0 else rows[i-1]['raw_gradient_norm']
    if r['eta_kappa_hat'] is not None and r['rho'] is not None and previous_norm>0:
     alternate=1-r['rho']*r['raw_gradient_norm']/previous_norm
     assert abs(r['eta_kappa_hat']-alternate)<=2e-12*(1+abs(alternate))
    if i<64:
     expected=cfg['cases'][case]['lr']
     if arm['kind']=='fixed':expected*=arm['ratio']
     if arm['kind']=='cosine':expected*=arm['ratio']+(1-arm['ratio'])*(1+np.cos(np.pi*min(i,arm['duration'])/arm['duration']))/2
     assert abs(r['applied_lr']-expected)<1e-15
     expected_clip=min(1.,parents[case]['clipping_threshold_reference_median64']*arm['factor']/r['raw_gradient_norm']) if arm['kind']=='clip' and r['raw_gradient_norm'] else 1.
     assert abs(metrics[i]['clip_factor']-expected_clip)<1e-14

   data[(case,arm['id'])]=rows;actions[(case,arm['id'])]=metrics
   for i,r in enumerate(rows):
    aligned.append(dict(case=case,condition=arm['id'],origin='Spark',state=r['state'],relative_update=r['state']-1024,loss=r['loss'],rho=r['rho'],raw_gradient_norm=r['raw_gradient_norm'],eta_kappa_hat=r['eta_kappa_hat'],lr_over_base=metrics[i]['lr_over_base'] if i<64 else None,parameter_norm=r['parameter_norm']))
 for (case,condition),rows in data.items():
  if (case,'unchanged') not in data:continue
  baseline=data[(case,'unchanged')];metrics=actions[(case,condition)];loss=np.array([r['loss'] for r in rows]);base=np.array([r['loss'] for r in baseline])
  for key in ['loss','rho','raw_gradient_norm','eta_kappa_hat','parameter_norm']:assert rows[0][key]==baseline[0][key],(case,condition,key)
  fork=base[0];excess=np.maximum(loss[1:]/fork-1,0);values=np.array([r['eta_kappa_hat'] for r in rows[1:]],dtype=float)
  status=json.loads((raw/'conditions'/case/condition/'status.json').read_text())
  summary.append(dict(case=case,condition=condition,status='completed',updates=64,fork_loss=fork,mean_post_loss=float(loss[1:].mean()),final_loss=float(loss[-1]),
   mean_loss_change_vs_unchanged_percent=100*(float(loss[1:].mean()/base[1:].mean())-1),final_loss_change_vs_unchanged_percent=100*(float(loss[-1]/base[-1])-1),
   peak_post_loss=float(loss[1:].max()),positive_excess_area=float(excess.sum()),mean_raw_rho=float(np.nanmean([r['rho'] for r in rows[1:]])),
   eta_kappa_hat_std=float(np.nanstd(values)),mean_raw_gradient_norm=float(np.mean([r['raw_gradient_norm'] for r in rows[1:]])),
   final_parameter_norm=rows[-1]['parameter_norm'],mean_actual_increment_norm=float(np.mean([m['actual_increment_norm'] for m in metrics])),
   clipped_updates=sum(m['clip_factor']<1 for m in metrics),mean_lr_over_base=float(np.mean([m['lr_over_base'] for m in metrics])),
   mean_decay_increment_norm=float(np.mean([m['decay_increment_norm'] for m in metrics])),
   uphill_actual_updates=sum(m['gradient_dot_actual_step']>0 for m in metrics),runner_seconds=status['elapsed_seconds'],endpoint_evaluation=json.dumps(status.get('evaluation',{}),sort_keys=True)))
 (HERE/'figures').mkdir(exist_ok=True);(HERE/'scratch').mkdir(exist_ok=True)
 for arm in cfg['arms']:
  condition=arm['id']
  if condition=='unchanged' or not all((c,condition) in data and (c,'unchanged') in data for c in CASES):continue
  fig,axes=plt.subplots(5,3,figsize=(15,11.4),sharex=True,sharey=False,gridspec_kw=dict(height_ratios=[1.08,1.,1.,1.,.75]))
  fig.subplots_adjust(left=.067,right=.99,bottom=.075,top=.89,wspace=.26,hspace=.17)
  for col,case in enumerate(CASES):
   axes[0,col].set_title(TITLES[col],pad=29,fontsize=11)
   historic=json.loads((ROOT/parents[case]['history']).read_text())[960:1024]
   pre={k:np.array([r['gradient_norm' if k=='raw_gradient_norm' else k] for r in historic],dtype=float) for k in METRICS if k!='lr_over_base'};pre['lr_over_base']=np.ones(64)
   for row,key in enumerate(METRICS):
    ax=axes[row,col];ax.grid(alpha=.13,linestyle='-');ax.axvline(0,color='.65',lw=.7,linestyle='-');ax.set_xlim(-64,64)
    for cond,color,label in [('unchanged','#5B5B5B','Unchanged Adam'),(condition,'#0072B2','Intervention')]:
     rows=data[(case,cond)];values=np.array([r[key] for r in rows],dtype=float) if key!='lr_over_base' else np.array([m['lr_over_base'] for m in actions[(case,cond)]]+[actions[(case,cond)][-1]['lr_over_base']])
     x=np.arange(65);y=values
     if cond=='unchanged':x=np.concatenate([np.arange(-64,0),x]);y=np.concatenate([pre[key],y])
     opts=dict(color=color,linewidth=.9,linestyle='-',label=label)
     line,=ax.step(x,y,where='post',**opts) if key=='lr_over_base' else ax.plot(x,y,**opts)
     assert np.array_equal(line.get_xdata(),x) and np.array_equal(line.get_ydata(),y,equal_nan=True);assert line.get_linestyle()=='-';plotted+=len(y)
    if key in ('loss','raw_gradient_norm'):ax.set_yscale('log')
    if key=='rho':ax.axhline(0,color='.8',lw=.5)
    if col==0:ax.set_ylabel(LABELS[row],fontsize=10)
    ax.tick_params(labelsize=8)
    if row==4:ax.set_xlabel('Updates relative to intervention',fontsize=10)
   axes[0,col].legend(loc='lower left',bbox_to_anchor=(-.015,1.005),ncol=2,frameon=False,fontsize=8.5)
  for row in range(5):
   assert not axes[row,0].get_shared_y_axes().joined(axes[row,0],axes[row,1])
  fig.suptitle('Fixed-step response: '+condition,y=.995,fontsize=14)
  fig.text(.5,.962,'Intervention at saved state1024;64 further full-objective updates; seed0; independent Y ranges.',ha='center',fontsize=10)
  fig.text(.5,.02,'Gray before0: shared historical trajectory. At0: hardware handover to Spark for ALL arms. Solid, unsmoothed observations.',ha='center',fontsize=9)
  path=HERE/'figures'/(condition+'.pdf');fig.savefig(path,metadata={'CreationDate':None,'ModDate':None});fig.savefig(HERE/'scratch'/(condition+'.png'),dpi=150);plt.close(fig);figure_hashes[path.name]=sha(path)
  with fitz.open(path) as pdf:pdf[0].get_pixmap(matrix=fitz.Matrix(1.,1.)).save(HERE/'scratch'/(condition+'_pdf_review.png'))
  if arm['kind']=='decay':
   comp,axs=plt.subplots(3,3,figsize=(15,8),sharex=True,sharey=False)
   comp.subplots_adjust(left=.07,right=.99,bottom=.09,top=.87,wspace=.25,hspace=.2)
   for col,case in enumerate(CASES):
    axs[0,col].set_title(TITLES[col],fontsize=11)
    for row,key,label in [(0,'parameter_norm','Parameter norm'),(1,'actual_increment_norm','Actual increment norm (log)'),(2,'decay_increment_norm','Decay increment norm')]:
     ax=axs[row,col];ax.grid(alpha=.14)
     for cond,color,title in [('unchanged','#5B5B5B','Unchanged Adam'),(condition,'#0072B2','Intervention')]:
      values=np.array([r[key] for r in data[(case,cond)]]) if row==0 else np.array([m[key] for m in actions[(case,cond)]])
      x=np.arange(65) if row==0 else np.arange(1,65)
      line,=ax.plot(x,values,color=color,ls='-',lw=1.,label=title)
      assert np.array_equal(line.get_ydata(),values);plotted+=len(values)
     if row==1:ax.set_yscale('log')
     if col==0:ax.set_ylabel(label)
     if row==2:ax.set_xlabel('Updates after intervention')
     ax.tick_params(labelsize=8)
    axs[0,col].legend(frameon=False,fontsize=8)
   comp.suptitle('Decoupled weight-decay response: '+condition,y=.98)
   comp.text(.5,.025,'Post-intervention observations only; weight norms were not stored throughout the historical before period.',ha='center',fontsize=9)
   path=HERE/'figures'/(condition+'_increments.pdf');comp.savefig(path,metadata={'CreationDate':None,'ModDate':None});comp.savefig(HERE/'scratch'/(condition+'_increments.png'),dpi=150);plt.close(comp);figure_hashes[path.name]=sha(path)
   with fitz.open(path) as pdf:pdf[0].get_pixmap(matrix=fitz.Matrix(1.,1.)).save(HERE/'scratch'/(condition+'_increments_pdf_review.png'))
 csv_out(HERE/'comparison.csv',summary);csv_out(HERE/'aligned_values.csv',aligned)
 primary=[a['id'] for a in cfg['arms'] if a['primary']];expected=[c+'/'+a for a in primary for c in CASES]
 main_updates=sum(len(read(p)) for p in raw.glob('conditions/*/*/metrics.jsonl'));qa_updates=sum(len(read(p)) for p in raw.glob('qualification/*/*/metrics.jsonl'))
 gradients=sum(len(read(p)) for p in raw.rglob('boundaries.jsonl'))
 result=dict(status=manifest['status'],snapshot=receipt,primary_completed=sum(statuses[k]=='completed' for k in expected),primary_total=30,
  statuses=statuses,skipped_optional=manifest.get('skipped_optional',[]),qualification=manifest['qualification'],main_updates=main_updates,
  qualification_updates=qa_updates,Run222_full_gradient_calls=gradients,comparison=summary,plotted_scalar_checks=plotted,figure_sha256=figure_hashes,
  source_sha256=sha(__file__),no_smoothing=True,independent_y_ranges=True,solid_lines=True,historical_hardware_boundary_explicit=True)
 (HERE/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 lines=['# Spark fixed-step intervention responses','',f"Status: {manifest['status']}. Completed primary branches: {result['primary_completed']}/30.",'',
  'All interventions start from the same saved state1024, with warm Adam moments. Contemporary unchanged Spark continuations are the controls. The historical before period uses prior hardware and is explicitly separated at the intervention boundary.','',
  'Ten primary conditions test fixed LR reductions, cosine floors, global clipping and decoupled weight decay; optional cosine durations depend only on remaining time. One selected seed/state;64 new full-objective updates measure short-term responses.','',
  '| Case | Intervention | Mean loss change vs control | Final loss change vs control | Clipped updates |','| --- | --- | ---: | ---: | ---: |']
 for r in summary:lines.append(f"| {r['case']} | {r['condition']} | {r['mean_loss_change_vs_unchanged_percent']:+.3f}% | {r['final_loss_change_vs_unchanged_percent']:+.3f}% | {r['clipped_updates']}/64 |")
 lines+=['','Negative percentages mean lower unregularized training loss. Complete held-out endpoints are descriptive; no tuning or generalization claim. Queued, failed and interrupted branches are retained in results.json.','',
  '[Numeric comparison](comparison.csv), [aligned values](aligned_values.csv), and [status/verification](results.json). Main figures have three model/dataset columns, solid curves, independent Y ranges and five rows: loss, raw rho, raw gradient norm, projected raw change and LR. No smoothing or unreported clipping of curves. Weight-decay companions show post-intervention parameter, actual-increment and decay-increment norms; no unrecorded historical weight norms are invented.','',
  f"Run222 actual updates: {main_updates} main + {qa_updates} qualification; recorded full gradient calls: {gradients}. Separate Run221 mechanics:9 updates and3 zero-update warm-parent gradient calls. Cloud spendUSD0; owned Spark.",'',
  ('[Eight verified findings and full interpretation](FINDINGS.md); [checkpoint and provenance checks](verification.json). The failed Run221 cross-GPU gradient gate remains part of the evidence.' if manifest['status']=='completed' and (HERE/'FINDINGS.md').exists() else 'Final interpretation and5-10 factual findings are added after terminal retrieval and numerical/visual verification. The failed Run221 cross-GPU gradient gate remains part of the evidence.')]
 lines+=['','Available response figures:','']+[f"- [{name}](figures/{name})" for name in sorted(figure_hashes)]
 (HERE/'REPORT.md').write_text('\n'.join(lines).rstrip()+'\n',encoding='utf-8')
 print(json.dumps(dict(status=manifest['status'],primary_completed=result['primary_completed'],main_updates=main_updates,qualification_updates=qa_updates,figures=len(figure_hashes),plotted_scalar_checks=plotted),indent=2))
if __name__=='__main__':main()
