"""H023: frozen gradient sampling and paired moving SGD on the complete CIFAR MLP objective."""
from pathlib import Path
import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.cifar_eos.core import (Geometry, append_json, cosine, evaluate, file_hash,
    flat, full_gradient, make_model, stamp, tensor_hash, write_json)
from src.eos_tasks.task import Task, configure
from src.eos_tasks.noise import mlp_unit_energy


def save_state(path, model, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(dict(model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, **extra), tmp)
    tmp.replace(path)


def load_model(path, device):
    model = make_model().to(device)
    state = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(state.get('model', state))
    return model


def gradient(task, ids=None):
    x, y = task.data['x_train'], task.data['y_train']
    if ids is not None:
        x, y = x[ids], y[ids]
    # Division and physical accumulation match the historical Run062 implementation.
    loss, accuracy, g = full_gradient(task.model, x, y, task.physical, 'mse')
    return loss, g.detach().clone()


@torch.no_grad()
def step(model, g, eta):
    offset = 0
    for p in model.parameters():
        p.add_(g[offset:offset+p.numel()].reshape_as(p), alpha=-eta)
        offset += p.numel()


def covariance(task, mu):
    energy = mlp_unit_energy(task) / task.n
    signal = float(mu.double().square().sum())
    variance = energy - signal
    tolerance = 1e-5 * max(energy, signal, 1)
    if variance < -tolerance:
        raise ValueError('Negative population covariance beyond rounding tolerance')
    return dict(signal_energy=signal, unit_mean_square_energy=energy,
                trace_population_covariance=max(0., variance),
                covariance_roundoff_clamp=max(0., -variance), measured_units=task.n)


def noise_pair(previous, g, old_noise, noise, old_mu, mu):
    if previous is None:
        return dict(q=None, epsilon=None, noise_projection=None, corrected=None,
                    correction_identity_residual=None, rho_full=None, norm_product=None,
                    undefined_reason='first_gradient')
    dot = float(previous.dot(g))
    denom = float(previous.square().sum())
    projection = float(previous.dot(noise-old_noise))
    q = 1-dot/denom if denom else None
    epsilon = projection/denom if denom else None
    change = float(previous.dot(old_mu-mu))/denom if denom else None
    residual = q+epsilon-change if denom else None
    if residual is not None and abs(residual) > 2e-8 * max(1., abs(q), abs(epsilon), abs(change)):
        raise ValueError('Noise correction identity failed')
    return dict(q=q, epsilon=epsilon, noise_projection=projection,
                corrected=None if q is None else q+epsilon,
                correction_identity_residual=residual, rho_full=cosine(old_mu, mu),
                norm_product=float(previous.norm()*g.norm()),
                undefined_reason=None if denom else 'zero_previous_gradient')


def mean(rows, field):
    values = [r[field] for r in rows if r.get(field) is not None]
    return float(np.mean(values)) if values else None


class Experiment:
    def __init__(self, config, output, device='cuda:0', deadline=None):
        configure()
        self.cfg, self.out, self.device, self.deadline = config, Path(output), device, deadline
        self.out.mkdir(parents=True, exist_ok=True)
        self.start = time.time()
        self.data = torch.load(ROOT/config['data_path'], map_location='cpu', weights_only=True)
        self.data = {k: v.to(device) for k, v in self.data.items() if k in ('x_train', 'y_train')}
        if len(self.data['x_train']) != 5000:
            raise ValueError('Complete 5000-example objective required')
        self.completed = dict(reference_updates=0, frozen_draws=0, continuation_updates=0)

    def task(self, model):
        return Task('mlp', model, self.data, self.cfg['physical_batch_size'])

    def progress(self, phase, **extra):
        self.limit()
        record = dict(utc=stamp(), phase=phase, elapsed_seconds=time.time()-self.start,
                      **self.completed, **extra)
        if torch.device(self.device).type == 'cuda':
            record['peak_allocated_gb'] = torch.cuda.max_memory_allocated(self.device)/1e9
        write_json(self.out/'progress.json', record)
        append_json(self.out/'events.jsonl', record)
        print(json.dumps(record), flush=True)

    def limit(self):
        if self.deadline and time.time() > self.deadline:
            raise TimeoutError('Training deadline reached; recovery evidence retained')

    def reference(self, seed):
        folder = self.out/f'reference-seed{seed}'
        folder.mkdir(exist_ok=True)
        summary_path = folder/'summary.json'
        if summary_path.exists():
            self.completed['reference_updates'] += self.cfg['reference_updates']
            return
        model = load_model(ROOT/self.cfg['initial_paths'][str(seed)], self.device)
        task = self.task(model)
        geometry = Geometry(sum(p.numel() for p in model.parameters()), self.device)
        begin = 0
        recovery = folder/'recovery.pt'
        if recovery.exists():
            state = torch.load(recovery, map_location='cpu', weights_only=False)
            model.load_state_dict(state['model']); begin = state['update']
            geometry.load_state_dict(state['geometry'])
            truncate_rows(folder/'metrics.jsonl', 'update', begin)
        self.completed['reference_updates'] += begin
        initial_hash = tensor_hash(model.parameters()) if begin == 0 else self.cfg['initial_tensor_hashes'][str(seed)]
        if initial_hash != self.cfg['initial_tensor_hashes'][str(seed)]:
            raise ValueError('Initialization tensor identity differs')
        if begin == 0:
            save_state(folder/'state-0000.pt', model, state_step=0, initialization_seed=seed)
        for index in range(begin, self.cfg['reference_updates']):
            loss, g = gradient(task)
            stats = geometry.consume(g)
            step(model, g, self.cfg['eta'])
            append_json(folder/'metrics.jsonl', dict(update=index+1, loss=loss, **stats))
            self.completed['reference_updates'] += 1
            if index+1 in self.cfg['checkpoints']:
                save_state(folder/f'state-{index+1:04d}.pt', model, state_step=index+1, initialization_seed=seed)
            if (index+1) % self.cfg['recovery_every'] == 0:
                save_state(recovery, model, update=index+1, geometry=geometry.state_dict())
                self.progress('reference', seed=seed, step=index+1, full_loss=loss)
        final = evaluate(model, self.data['x_train'], self.data['y_train'], task.physical, 'mse')
        write_json(summary_path, dict(seed=seed, updates=self.cfg['reference_updates'], final=final,
                    final_tensor_hash=tensor_hash(model.parameters())))

    def frozen(self, seed, checkpoint):
        model = load_model(self.out/f'reference-seed{seed}/state-{checkpoint:04d}.pt', self.device)
        task = self.task(model)
        before_hash = tensor_hash(model.parameters())
        full_loss, mu32 = gradient(task); mu = mu32.double()
        moments = covariance(task, mu)
        folder = self.out/f'frozen-seed{seed}-state{checkpoint}'
        folder.mkdir(exist_ok=True)
        write_json(folder/'population.json', dict(seed=seed, checkpoint=checkpoint, full_loss=full_loss,
                    parameter_hash=before_hash, **moments))
        for b in self.cfg['batches']:
            condition = folder/f'b{b}'; condition.mkdir(exist_ok=True)
            V = (task.n-b)/(b*(task.n-1))*moments['trace_population_covariance']
            S = moments['signal_energy']
            summaries = []
            for rep in range(self.cfg['frozen_repetitions']):
                dest = condition/f'rep{rep:02d}'
                done = dest/'summary.json'
                if done.exists():
                    summaries.append(json.loads(done.read_text()))
                    self.completed['frozen_draws'] += self.cfg['frozen_draws']
                    continue
                dest.mkdir(exist_ok=True)
                # Each repetition uses a distinct stream; b is deliberately absent.
                rng = np.random.default_rng(np.random.SeedSequence([self.cfg['frozen_sampling_seed'], seed, checkpoint, rep]))
                stream = Geometry(mu.numel(), self.device)
                old_noise = None
                rows = []
                # Interrupted unfinished sequences restart from their frozen state.
                with (dest/'indices.u32').open('wb') as index_file, (dest/'metrics.jsonl').open('w') as log:
                    for k in range(1, self.cfg['frozen_draws']+1):
                        if b == task.n:
                            g = mu
                        else:
                            ids = rng.permutation(task.n)[:b].astype('<u4')
                            index_file.write(ids.tobytes())
                            _, raw = gradient(task, torch.from_numpy(ids.astype(np.int64)).to(self.device))
                            g = raw.double()
                        noise = g-mu
                        pair = noise_pair(stream.previous, g, old_noise, noise, mu, mu)
                        stats = stream.consume(g)
                        row = dict(draw=k, **stats, **pair, noise_energy=float(noise.square().sum()))
                        log.write(json.dumps(row, allow_nan=False)+'\n'); rows.append(row)
                        old_noise = noise
                        self.completed['frozen_draws'] += 1
                norms = np.array([r['norm'] for r in rows])
                products = np.array([r['norm_product'] for r in rows if r['norm_product'] is not None])
                result = dict(seed=seed, checkpoint=checkpoint, batch=b, replicate=rep,
                    mean_rho=mean(rows,'cosine'), R=rows[-1]['B_over_A'], mean_q=mean(rows,'q'),
                    mean_noise_projection=mean(rows,'noise_projection'), mean_noise_energy=mean(rows,'noise_energy'),
                    norm_cv=float(norms.std()/norms.mean()) if norms.mean() else None,
                    norm_product_cv=float(products.std()/products.mean()) if products.mean() else None,
                    valid_cosines=sum(r['cosine'] is not None for r in rows), predicted_variance=V,
                    predicted_noise_projection=-V, predicted_rho=S/(S+V) if S+V else None,
                    signal_energy=S, frozen=True,
                    max_identity_residual=max(abs(r['correction_identity_residual'] or 0) for r in rows))
                write_json(done, result); summaries.append(result)
                self.progress('frozen', seed=seed, checkpoint=checkpoint, batch=b, repetition=rep+1)
            write_json(condition/'summary.json', dict(replicates=summaries, predicted_variance=V,
                predicted_rho=S/(S+V) if S+V else None, signal_energy=S))
        if tensor_hash(model.parameters()) != before_hash:
            raise ValueError('Frozen parameters changed')

    def moving(self, seed, checkpoint, batch, rep):
        dest = self.out/f'moving-seed{seed}-state{checkpoint}-b{batch}-rep{rep:02d}'
        dest.mkdir(exist_ok=True)
        done = dest/'summary.json'
        if done.exists():
            self.completed['continuation_updates'] += json.loads(done.read_text())['updates']
            return
        initial_path = self.out/f'reference-seed{seed}/state-{checkpoint:04d}.pt'
        model = load_model(initial_path, self.device); task = self.task(model)
        origin = flat(list(model.parameters())).detach().double()
        count = origin.numel()
        raw_stream, update_stream = Geometry(count,self.device), Geometry(count,self.device)
        rng = np.random.default_rng(np.random.SeedSequence([self.cfg['moving_sampling_seed'],seed,checkpoint,rep]))
        old_mu = old_noise = old_hvp = None
        begin = 0
        recovery = dest/'recovery.pt'
        if recovery.exists():
            state = torch.load(recovery,map_location='cpu',weights_only=False)
            model.load_state_dict(state['model']); begin=state['update']; rng.bit_generator.state=state['rng']
            raw_stream.load_state_dict(state['raw_stream']); update_stream.load_state_dict(state['update_stream'])
            old_mu=state['old_mu'].to(self.device);old_noise=state['old_noise'].to(self.device);old_hvp=state['old_hvp']
            truncate_rows(dest/'metrics.jsonl','update',begin)
            if batch != task.n:
                with (dest/'indices.u32').open('r+b') as f:f.truncate(begin*batch*4)
        self.completed['continuation_updates'] += begin
        previous_rows = list(read_rows(dest/'metrics.jsonl')) if begin else []
        rows = previous_rows
        mode = 'ab' if begin else 'wb'
        with (dest/'indices.u32').open(mode) as ids_file, (dest/'metrics.jsonl').open('a' if begin else 'w') as log:
            for index in range(begin,self.cfg['continuation_updates']):
                full_loss, mu32=gradient(task);mu=mu32.double()
                if batch == task.n:
                    batch_loss,g32=full_loss,mu32
                else:
                    ids=rng.permutation(task.n)[:batch].astype('<u4');ids_file.write(ids.tobytes())
                    batch_loss,g32=gradient(task,torch.from_numpy(ids.astype(np.int64)).to(self.device))
                g=g32.double();noise=g-mu
                pair=noise_pair(raw_stream.previous,g,old_noise,noise,old_mu,mu)
                hvp=None;moments=None
                if index+1 in self.cfg['hvp_predecessor_gradients']:
                    save_state(dest/f'probe-before-{index+1:03d}.pt',model,update=index,
                               batch_indices=None if batch==task.n else ids.astype(np.int64))
                    product=task.hvp(g32)
                    denominator=float(g.square().sum())
                    hvp=self.cfg['eta']*float(g.dot(product.double()))/denominator if denominator else None
                    moments=covariance(task,mu)
                stats=raw_stream.consume(g)
                before=flat(list(model.parameters())).detach().double()
                step(model,g32,self.cfg['eta'])
                after=flat(list(model.parameters())).detach().double()
                actual=after-before;actual_stats=update_stream.consume(actual)
                nominal=-self.cfg['eta']*raw_stream.total
                distance=after-origin
                discrepancy=float((distance-nominal).norm())
                row=dict(update=index+1,full_loss=full_loss,batch_loss=batch_loss,**stats,**pair,
                         full_gradient_norm=float(mu.norm()),noise_energy=float(noise.square().sum()),
                         hvp_eta_kappa=hvp,previous_hvp_eta_kappa=old_hvp,
                         taylor_residual=None if old_hvp is None or pair['corrected'] is None else pair['corrected']-old_hvp,
                         population=moments,actual=actual_stats,actual_displacement=float(distance.norm()),
                         displacement_identity_absolute_error=discrepancy,
                         displacement_identity_relative_error=discrepancy/max(float(distance.norm()),1e-30),
                         actual_prefix_identity_error=float((distance-update_stream.total).norm()))
                log.write(json.dumps(row,allow_nan=False)+'\n');rows.append(row)
                old_mu=mu;old_noise=noise;old_hvp=hvp
                self.completed['continuation_updates']+=1
                if (index+1)%self.cfg['recovery_every']==0 or index+1==self.cfg['continuation_updates']:
                    log.flush();ids_file.flush()
                    save_state(recovery,model,update=index+1,rng=rng.bit_generator.state,
                               raw_stream=raw_stream.state_dict(),update_stream=update_stream.state_dict(),
                               old_mu=old_mu.cpu(),old_noise=old_noise.cpu(),old_hvp=old_hvp)
                    self.progress('moving',seed=seed,checkpoint=checkpoint,batch=batch,repetition=rep,
                                  step=index+1,full_loss=full_loss,rho=row['cosine'],R=row['B_over_A'])
        final=evaluate(model,self.data['x_train'],self.data['y_train'],task.physical,'mse')
        save_state(dest/'final.pt',model,update=len(rows))
        # Recovery contains diagnostic/RNG state; retain it for reproducibility/resumption.
        result=dict(seed=seed,checkpoint=checkpoint,batch=batch,replicate=rep,updates=len(rows),
            initial_loss=rows[0]['full_loss'],final_loss=final['loss'],loss_change=final['loss']-rows[0]['full_loss'],
            mean_rho=mean(rows,'cosine'),mean_full_rho=mean(rows,'rho_full'),mean_q=mean(rows,'q'),
            mean_noise_projection=mean(rows,'noise_projection'),
            mean_previous_noise_energy=mean(rows[:-1],'noise_energy'),
            R=rows[-1]['B_over_A'],actual_R=rows[-1]['actual']['B_over_A'],
            absolute_displacement=rows[-1]['actual_displacement'],sampled_examples=batch*len(rows),
            all_pair_twice=rows[-1]['all_pair'],lag_one_twice=rows[-1]['lag_one'],
            max_correction_identity_error=max(abs(r['correction_identity_residual'] or 0) for r in rows),
            max_displacement_identity_relative_error=max(r['displacement_identity_relative_error'] for r in rows),
            hvp_comparison_count=sum(r['taylor_residual'] is not None for r in rows),
            max_actual_prefix_identity_error=max(r['actual_prefix_identity_error'] for r in rows),
            valid_cosines=sum(r['cosine'] is not None for r in rows))
        write_json(done,result)


def read_rows(path):
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            if line.strip():yield json.loads(line)


def truncate_rows(path,key,last):
    rows=[r for r in read_rows(path) if r[key]<=last]
    Path(path).write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--deadline-epoch',type=float)
    p.add_argument('--device',default='cuda:0');args=p.parse_args()
    cfg=json.loads(args.config.read_text())
    for relative,expected in cfg['input_files_sha256'].items():
        if file_hash(ROOT/relative)!=expected:raise ValueError('Input hash differs: '+relative)
    args.output.mkdir(parents=True,exist_ok=True)
    existing=args.output/'manifest.json'
    if existing.exists():
        prior=json.loads(existing.read_text())
        if prior['config_sha256']!=file_hash(args.config) or prior['experiment_sha256']!=file_hash(__file__):
            raise ValueError('Resume requires unchanged scientific config and source')
        if prior['status']=='complete':
            print(json.dumps({'status':'already_complete','output':str(args.output)}));return
    lock=args.output/'process.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    manifest=dict(status='running',started_utc=stamp(),config_sha256=file_hash(args.config),
                  experiment_sha256=file_hash(__file__),torch=torch.__version__,cuda=torch.version.cuda,
                  python=platform.python_version(),gpu=torch.cuda.get_device_name(args.device),
                  input_files_sha256=cfg['input_files_sha256'],deadline_epoch=args.deadline_epoch)
    write_json(args.output/'manifest.json',manifest)
    try:
        experiment=Experiment(cfg,args.output,args.device,args.deadline_epoch)
        for seed in cfg['seeds']:
            experiment.reference(seed)
            for checkpoint in cfg['checkpoints']:
                experiment.frozen(seed,checkpoint)
                for b in cfg['batches']:
                    for rep in range(1 if b==5000 else cfg['moving_repetitions']):
                        experiment.moving(seed,checkpoint,b,rep)
        experiment.progress('complete')
        manifest.update(status='complete',completed_utc=stamp(),completed=experiment.completed)
    except BaseException as error:
        manifest.update(status='failed',failed_utc=stamp(),error=repr(error))
        raise
    finally:
        write_json(args.output/'manifest.json',manifest)
        lock.unlink(missing_ok=True)


if __name__=='__main__':main()
