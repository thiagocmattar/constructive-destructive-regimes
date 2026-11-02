"""Keep relevant provider evidence public; preserve full reads in ignored scratch."""
import json
from pathlib import Path

HERE=Path(__file__).parent
FIELDS=['id','name','costPerHr','createdAt','gpuCount','gpuId','imageName','containerDiskInGb',
    'volumeInGb','volumeMountPath','desiredStatus','runtimeStatus','dataCenterId','computeType',
    'minCudaVersion','machineId']


def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')


def main():
    assert json.loads((HERE/'closeout.json').read_text())['status']=='complete'
    private=HERE/'scratch/control-reads';private.mkdir(parents=True,exist_ok=True)
    for pattern in ['create-worker*.json','final-pod-worker*.json']:
        for path in HERE.glob(pattern):
            original=private/path.name
            if not original.exists():original.write_bytes(path.read_bytes())
            raw=json.loads(path.read_text())
            save(path,{k:raw[k] for k in FIELDS if k in raw})
    path=HERE/'live_resources.json';raw=json.loads(path.read_text())
    original=private/path.name
    if not original.exists():original.write_bytes(path.read_bytes())
    if 'template' in raw:
        raw['template']={k:v for k,v in raw['template'].items() if k in
            ['id','name','imageName','containerDiskInGb','volumeInGb','volumeMountPath','ports']}
    save(path,raw)
    print('Relevant resource IDs/prices/configuration retained; full provider reads remain in ignored scratch')


if __name__=='__main__':main()
