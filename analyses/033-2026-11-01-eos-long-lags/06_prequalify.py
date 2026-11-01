"""Read-only early qualification of later starts; never applies an update."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.cifar_eos.core import file_hash,tensor_hash,write_json
from src.eos_tasks.production import ProductionTask
from src.eos_tasks.task import configure,model_for


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();configure();device=torch.device('cuda:0')
    cfgpath=ROOT/'runs/206-2026-11-01-eos-long-lags/config.json';cfg=json.loads(cfgpath.read_text())
    rows=[]
    for job in cfg['jobs']:
        if job['worker']!=args.worker or job['task']!='resnet' or job['anchor']==500:continue
        data={k:v.to(device) for k,v in torch.load(ROOT/job['data'],weights_only=True).items() if k.startswith(('x_train','y_train'))}
        model=model_for('resnet',ROOT/cfg['task_cache']).to(device)
        state=torch.load(ROOT/job['checkpoint'],map_location='cpu',weights_only=False)
        model.load_state_dict(state['model']);before=tensor_hash(model.parameters())
        assert before==job['resume_tensor_sha256']
        task=ProductionTask('resnet',model,data,1000)
        begin=time.perf_counter();loss,g=task.gradient()
        reference=torch.from_numpy(np.load(ROOT/job['boundary_gradient'])).to(device)
        error=float((g-reference).double().norm())/float(reference.double().norm())
        assert tensor_hash(model.parameters())==before
        rows.append(dict(parent_run_id=job['parent_run_id'],anchor=job['anchor'],relative_gradient_error=error,
            passed=error<cfg['gradient_relative_tolerance'],loss_error=loss-job['prefix_rows'][-1]['loss'],seconds=time.perf_counter()-begin,
            parameters_unchanged=True,applied_updates=0))
        del task,model,data,state,g,reference;torch.cuda.empty_cache()
    write_json(args.output,dict(config_sha256=file_hash(cfgpath),worker=args.worker,gpu=torch.cuda.get_device_name(),checks=rows))
    print(json.dumps(rows),flush=True)


if __name__=='__main__':main()
