"""Production-shaped mechanics/timing only; no intervention outcomes."""
import json,sys,time,hashlib,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch,numpy,scipy
from src.eos_tasks.task import configure,model_for
from src.eos_tasks.production import ProductionTask
from src.cifar_eos.core import tensor_hash
HERE=Path(__file__).resolve().parent
configure();torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
cache=ROOT/'cache/eos-multitask';shared=json.loads((cache/'inputs.json').read_text())
results={}
for name,physical,lr in [('mlp',1000,.001),('resnet',1000,.003),('transformer',64,.003)]:
 torch.manual_seed(0);torch.cuda.reset_peak_memory_stats()
 path=ROOT/'cache/cifar10-eos/data.pt' if name=='mlp' else cache/(name+'_data.pt')
 data={k:v.to('cuda:0') for k,v in torch.load(path,weights_only=True).items() if k.startswith(('x_','y_'))}
 model=model_for(name,cache,shared['wikitext']['vocabulary_size']).to('cuda:0')
 initial=shared['initial'][name]['0'];model.load_state_dict(torch.load(cache/initial['path'],weights_only=True))
 assert tensor_hash(model.parameters())==initial['initial_tensor_sha256']
 task=ProductionTask(name,model,data,physical);opt=torch.optim.Adam(model.parameters(),lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False,fused=False)
 timings=[]
 for index in range(3):
  torch.cuda.synchronize();start=time.perf_counter();loss,g=task.gradient();opt.step();torch.cuda.synchronize()
  timings.append(time.perf_counter()-start)
 result=dict(name=name,lr=lr,physical=physical,update_seconds=timings,peak_allocated_bytes=torch.cuda.max_memory_allocated(),objective_units=task.units,parameters=sum(p.numel() for p in model.parameters()))
 results[name]=result;print(json.dumps(result),flush=True)
 del task,opt,model,data,g;torch.cuda.empty_cache()
receipt=dict(status='completed',mechanics_only=True,updates=9,results=results,torch=torch.__version__,numpy=numpy.__version__,scipy=scipy.__version__,cuda=torch.version.cuda)
(HERE/'artifacts').mkdir(exist_ok=True);(HERE/'artifacts/profile.json').write_text(json.dumps(receipt,indent=2)+'\n')
