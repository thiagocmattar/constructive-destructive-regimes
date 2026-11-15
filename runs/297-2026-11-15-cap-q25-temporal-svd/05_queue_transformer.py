"""Run the remaining architecture sequentially after successful ResNet completion."""
import importlib.util,json,shlex,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('transport',HERE/'03_spark.py');t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
rel=t.REL;remote=t.REMOTE;py=t.L['python'];scratch=HERE/'scratch';scratch.mkdir(exist_ok=True)
script='#!/bin/bash\nset -e\ncd '+shlex.quote(remote)+'\n'
script+='export PYTHONPATH='+shlex.quote(remote+':'+t.L['pythonpath'])+' CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4\n'
script+='deadline=$((SECONDS+10800))\nwhile ! test -f '+rel+'/scratch/production-resnet.exit; do\n  if ((SECONDS>deadline)); then echo "ResNet prerequisite timeout"; exit 1; fi\n  sleep 30\ndone\n'
for smoke in [True,False]:
    cmd=[py,rel+'/01_capture.py','--case','transformer','--source-root',remote,'--data-root',t.L['data_root'],'--out',rel+'/artifacts/'+('smoke' if smoke else 'attempt-01')]
    if smoke:cmd+=['--smoke']
    script+=' '.join(map(shlex.quote,cmd))+'\n'
script+=' '.join(map(shlex.quote,[py,rel+'/02_reduce.py','--case-root',rel+'/artifacts/attempt-01/transformer']))+'\n'
script+='echo 0 > '+rel+'/scratch/production-transformer.exit\n'
f=scratch/'queued-transformer.sh';f.write_text(script,encoding='utf8',newline='\n')
subprocess.run(['scp','-q',str(f),t.HOST+':'+remote+'/'+rel+'/scratch/'+f.name],check=True)
pid=t.ssh('cd '+shlex.quote(remote)+'\nsetsid bash '+rel+'/scratch/'+f.name+' </dev/null >'+rel+'/scratch/production-transformer.log 2>&1 &\necho $!\n').strip()
(HERE/'production-transformer-launch.json').write_text(json.dumps(dict(pid=pid,queued_after='production-resnet.exit',root=remote,output=rel+'/artifacts/attempt-01',case='transformer',smoke_first=True),indent=2)+'\n')
print('Queued Transformer PID',pid)
