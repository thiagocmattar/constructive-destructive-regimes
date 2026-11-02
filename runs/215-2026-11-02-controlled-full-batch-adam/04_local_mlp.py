import json,subprocess,sys
from pathlib import Path
here=Path(__file__).resolve().parent
cfg=json.loads((here/'config.json').read_text())
for lr in cfg['learning_rates_by_task']['mlp']:
 output=here/'artifacts/attempt-01/local-mlp'/('lr-'+str(lr))
 with (here/'scratch'/('mlp-'+str(lr)+'.log')).open('w') as log:
  subprocess.run([sys.executable,str(here/'01_train.py'),'--task','mlp','--lr',str(lr),'--output',str(output)],stdout=log,stderr=subprocess.STDOUT,check=True)
print('All three local MLP arms finished',flush=True)
