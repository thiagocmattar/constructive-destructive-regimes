"""H024 matched LR forks; native AdamW updates with non-mutating diagnostics."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

from baseline.continuation import load_parent, protected_evaluate, rng_state, restore_rng
from baseline.data_protocol import block_order, read_cache, sha256
from baseline.descent_diagnostics import fixed_probe
from baseline.gradient_geometry import GradientGeometry
from baseline.momentum_diagnostics import momentum_vectors
from baseline.recipe import recipe_learning_rate
from baseline.training import append_event, now, optimizer_boundary, synchronize, write_json


@torch.no_grad()
def dot(xs, ys):
    return sum(float((x.double()*y.double()).sum()) for x,y in zip(xs,ys))

def norm(xs):
    return math.sqrt(dot(xs,xs))

def cosine(xs,ys):
    den=norm(xs)*norm(ys)
    return dot(xs,ys)/den if den else None

def reset_prefix(geometry):
    for value in geometry.cumulative:
        value.zero_()
    geometry.norm_sum=0.


class RawDiagnostics:
    """Read incoming g and the actual old m at the existing pre-clip boundary."""
    def __init__(self,geometry,optimizer,beta1):
        self.geometry,self.optimizer,self.beta1=geometry,optimizer,beta1
    def __getattr__(self,name):
        return getattr(self.geometry,name)
    @torch.no_grad()
    def consume(self):
        ps=[p for _,p in self.geometry.parameters]
        gradients=[p.grad for p in ps]
        old_m=[self.optimizer.state[p]['exp_avg'] for p in ps]
        old_norm=norm(old_m)
        alignment=cosine(gradients,old_m)
        previous_norm=self.geometry.previous_norm
        out=self.geometry.consume()
        out['incoming_previous_moment_cosine']=alignment
        out['previous_moment_norm']=old_norm
        rho=out['adjacent_cosine']
        out['projected_gradient_change']=1-rho*out['gradient_norm']/previous_norm if rho is not None else None
        return out


def initialize_streams(model,optimizer,raw,adaptive,junction):
    reset_prefix(raw)
    reset_prefix(adaptive)
    moment=GradientGeometry(model.named_parameters())
    moment.consume(momentum_vectors(optimizer,model.named_parameters(),junction))
    reset_prefix(moment)
    moment.updates=junction
    clipped=GradientGeometry(model.named_parameters())
    actual=GradientGeometry(model.named_parameters(),previous_dtype=torch.float64)
    return dict(raw=raw,adaptive=adaptive,moment=moment,clipped=clipped,actual=actual)


@torch.no_grad()
def set_rate(optimizer,config,update,replay=False):
    t,d=config['training'],config['decision']
    if replay:
        lr=recipe_learning_rate(update,peak=t['peak_learning_rate'],max_steps=config['resume']['full_updates'],warmup_fraction=t['warmup_fraction'],minimum=t['minimum_learning_rate'])
        decay=t['weight_decay']
    else:
        lr=d['base_lr']*d['multiplier']
        decay=t['weight_decay']/d['multiplier']
    for group in optimizer.param_groups:
        group['lr']=lr
        # Native recipe has one decay group and one zero-decay group.
        if group['weight_decay'] !=0: group['weight_decay']=decay
    return lr


def step(model,optimizer,scaler,streams,tokens,indices,config,device,callback=None):
    parameters=[p for p in model.parameters() if p.requires_grad]
    before=[p.detach().clone() for p in parameters]
    beta=config['training']['adamw_betas'][0]
    wrapped=RawDiagnostics(streams['raw'],optimizer,beta)
    result=optimizer_boundary(model,optimizer,scaler,wrapped,streams['adaptive'],tokens,indices,config,device,callback,momentum_geometry=streams['moment'])
    start=time.perf_counter()
    with torch.no_grad():
        clipped=streams['clipped'].consume([p.grad for p in parameters])
        delta=[p.double()-old.double() for p,old in zip(parameters,before)]
        actual=streams['actual'].consume(delta)
        result.update(clipped=clipped,actual_parameter_update=actual,
            incoming_to_retained_moment_ratio=((1-beta)*clipped['gradient_norm']/(beta*result['previous_moment_norm']) if result['previous_moment_norm'] else None),
            raw_adaptive_cosine=cosine(streams['raw'].previous,streams['adaptive'].previous),
            raw_actual_update_cosine=cosine(streams['raw'].previous,delta),
            clip_coefficient=min(1.,config['training']['gradient_clip_norm']/(result['gradient_clip_returned_norm']+1e-6)))
    synchronize(device)
    result['additional_diagnostic_seconds']=time.perf_counter()-start
    result['diagnostic_seconds']+=result['additional_diagnostic_seconds']
    return result


def save_state(path,model,optimizer,scaler,streams,position,update,order_hash,config_hash):
    path.parent.mkdir(parents=True,exist_ok=True)
    state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
        streams={k:v.state_dict() for k,v in streams.items()},position=position,update=update,
        order_sha256=order_hash,config_sha256=config_hash,**rng_state())
    tmp=path.with_suffix('.pt.tmp')
    torch.save(state,tmp)
    tmp.replace(path)


def restore_state(path,model,optimizer,scaler,streams,order_hash,config_hash):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['order_sha256']!=order_hash or state['config_sha256']!=config_hash:
        raise ValueError('Recovery identity mismatch')
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scaler.load_state_dict(state['scaler'])
    for k,v in streams.items():v.load_state_dict(state['streams'][k])
    restore_rng(state)
    return state['position'],state['update']


def execute(args):
    config_path=Path(args.config).resolve()
    config=json.loads(config_path.read_text())
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='preparing',run_id=config['run_id'],started_at=now(),scope='mechanics' if args.mechanics else 'scientific',config_sha256=sha256(config_path))
    write_json(output/'manifest.json',manifest)
    write_json(output/'resolved_config.json',config)
    events=output/'events.jsonl'
    try:
        device=torch.device(args.device)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        train,train_meta=read_cache(Path(args.cache)/'train',config['data']['train_sha256'],config['data'])
        val,val_meta=read_cache(Path(args.cache)/'validation',config['data']['validation_sha256'],config['data'])
        order=block_order(len(train),config['data']['sequence_length'],config['seeds']['data_order'])
        order_hash=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest()
        if order_hash!=config['resume']['order_sha256']:raise ValueError('Order identity mismatch')
        parent=Path(args.parent or config['resume']['remote_checkpoint'])
        model,optimizer,scaler,raw,adaptive,position,identity=load_parent(config,parent,device)
        junction=config['resume']['junction_update']
        streams=initialize_streams(model,optimizer,raw,adaptive,junction)
        start_update=junction
        if args.resume:
            position,start_update=restore_state(args.resume,model,optimizer,scaler,streams,order_hash,sha256(config_path))
        probe_path=Path(config['decision']['probe_path'])
        if sha256(probe_path)!=config['decision']['probe_sha256']:raise ValueError('Probe identity mismatch')
        probe=json.loads(probe_path.read_text())
        indices=np.array(probe['indices'],dtype=np.int64)
        budget=args.mechanics or config['decision']['updates']
        finish=junction+budget
        effective=config['training']['effective_batch_sequences']
        assert not np.intersect1d(indices,order[:(junction+config['decision']['updates'])*effective]).size
        assert position==start_update*effective
        write_json(output/'resume_identity.json',identity)
        write_json(output/'input_identity.json',dict(train=train_meta,validation=val_meta,order_sha256=order_hash,probe=probe,parent=identity))
        write_json(output/'environment.json',dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,transformers=__import__('transformers').__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,tf32=False,source_sha256={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}))
        validations=[];probes=[];times=[]
        def evaluate_at(update):
            local=update-junction
            if local in config['decision']['validation_local_updates'] or args.mechanics:
                record=dict(event='validation',update=update,local_update=local,**protected_evaluate(model,val,config,device))
                validations.append(record);append_event(events,record)
            if local in config['decision']['probe_local_updates'] or args.mechanics:
                t=time.perf_counter()
                loss,_=fixed_probe(model,train,indices,config['data']['sequence_length'],config['training']['micro_batch_size'],device)
                record=dict(event='fixed_probe',update=update,local_update=local,loss=loss,supervised_tokens=probe['supervised_tokens'],seconds=time.perf_counter()-t)
                probes.append(record);append_event(events,record)
        evaluate_at(start_update)
        manifest.update(status='running',completed_updates=start_update,planned_updates=finish)
        write_json(output/'manifest.json',manifest)
        for update in range(start_update+1,finish+1):
            if args.deadline and time.time()>args.deadline:
                save_state(output/'checkpoints/deadline.pt',model,optimizer,scaler,streams,position,update-1,order_hash,sha256(config_path))
                raise TimeoutError('Training deadline reached; recovery retained')
            synchronize(device);t=time.perf_counter()
            batch=order[position:position+effective]
            lr=set_rate(optimizer,config,update,args.replay)
            metrics=step(model,optimizer,scaler,streams,train,batch,config,device,lambda e:append_event(events,dict(update=update,**e)))
            synchronize(device);elapsed=time.perf_counter()-t;times.append(elapsed)
            position+=len(batch)
            local=update-junction
            record=dict(event='update',update=update,local_update=local,learning_rate=lr,
                weight_decay_groups=[g['weight_decay'] for g in optimizer.param_groups],
                batch_indices_sha256=hashlib.sha256(batch.astype('<i8').tobytes()).hexdigest(),
                update_seconds=elapsed,tokens_per_second=metrics['input_tokens']/elapsed,
                continuation_input_tokens=local*effective*config['data']['sequence_length'],
                remaining_training_seconds=(finish-update)*float(np.median(times[-10:])),**metrics)
            if device.type=='cuda':record.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device))
            append_event(events,record)
            manifest.update(completed_updates=update,latest_loss=metrics['loss'],latest_event_at=now(),remaining_training_seconds=record['remaining_training_seconds'],latest_update_seconds=elapsed)
            write_json(output/'manifest.json',manifest)
            print(json.dumps(dict(update=update,loss=metrics['loss'],seconds=elapsed)),flush=True)
            if local in set(config['decision']['validation_local_updates']+config['decision']['probe_local_updates']):evaluate_at(update)
            if not args.mechanics and local in config['decision']['checkpoint_local_updates']:
                save_state(output/f'checkpoints/update-{update}.pt',model,optimizer,scaler,streams,position,update,order_hash,sha256(config_path))
        if args.mechanics:evaluate_at(finish)
        # Independent displacement check against saved fork weights, tensor by tensor.
        from safetensors import safe_open
        residual2=distance2=0.
        with safe_open(parent/'model/model.safetensors',framework='pt',device='cpu') as source:
            for (name,p),total in zip(streams['actual'].parameters,streams['actual'].cumulative):
                displacement=p.detach().cpu().double()-source.get_tensor(name).double()
                distance2+=float(displacement.square().sum())
                residual2+=float((total.cpu()-displacement).square().sum())
        relative=math.sqrt(residual2/max(distance2,1e-300))
        if relative>1e-10:raise ValueError('Actual displacement prefix fails independent endpoint check')
        summary=dict(completed_updates=finish,local_updates=budget,validations=validations,probes=probes,
            update_seconds=times,last_update=record,actual_displacement_relative_residual=relative,
            actual_distance_from_fork=math.sqrt(distance2),order_sha256=order_hash)
        write_json(output/'summary.json',summary)
        manifest.update(status='completed',finished_at=now())
        write_json(output/'manifest.json',manifest)
    except Exception as exc:
        manifest.update(status='failed',finished_at=now(),error=repr(exc))
        write_json(output/'manifest.json',manifest)
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--cache',required=True);p.add_argument('--parent');p.add_argument('--resume')
    p.add_argument('--device',default='cuda');p.add_argument('--mechanics',type=int)
    p.add_argument('--replay',action='store_true');p.add_argument('--deadline',type=float)
    args=p.parse_args()
    if args.replay and not args.mechanics:p.error('Replay is a mechanics check only')
    execute(args)

if __name__=='__main__':main()
