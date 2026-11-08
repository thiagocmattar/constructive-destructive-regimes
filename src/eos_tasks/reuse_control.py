"""Reuse an immutable same-worker control only after exact native-state qualification."""
import json
from pathlib import Path
import shutil
import torch
from src.eos_tasks import representation_continuation as c


def install():
    original=c.run_arm
    def run_arm(context,fork,cfg,arm,folder,thresholds,ema,groups,entry,trunk,probe_history,parameter_history,mechanics=False):
        if arm['id']!='control' or mechanics:
            return original(context,fork,cfg,arm,folder,thresholds,ema,groups,entry,trunk,probe_history,parameter_history,mechanics)
        reference=cfg['baseline_reference'];worker=c.ROOT/reference['worker'];source=worker/'control'
        assert worker.resolve().is_relative_to(c.ROOT/'runs')
        source_run=c.ROOT/reference['worker'].split('/artifacts/')[0]
        assert c.w.sha(source_run/'input_manifest.json')==reference['input_manifest_sha256']
        old_cfg=json.loads((source_run/'config.json').read_text())
        for key in ['case','source_state','fork','end','base_lr','physical','probe','calibration_updates','seed']:assert cfg[key]==old_cfg[key],key
        new_run=next((c.ROOT/'runs').glob(str(cfg['run_id'])+'-*'))
        old_inputs={r['path']:r['sha256'] for r in json.loads((source_run/'input_manifest.json').read_text())['files']}
        new_inputs={r['path']:r['sha256'] for r in json.loads((new_run/'input_manifest.json').read_text())['files']}
        common={p:h for p,h in old_inputs.items() if p.startswith(('cache/','src/','runs/223-'))}
        assert common and all(new_inputs.get(p)==h for p,h in common.items())
        saved=torch.load(worker/'shared/fork.pt',map_location='cpu',weights_only=False)
        c.w.equal(fork,saved);assert c.fingerprint(fork)==reference['fork_fingerprint']
        new_qualification=json.loads((folder.parent/'shared/qualification.json').read_text())
        old_qualification=json.loads((worker/'shared/qualification.json').read_text())
        for q in [new_qualification,old_qualification]:assert q['native_two_step_fingerprint']==reference['native_two_step_fingerprint']
        status=json.loads((source/'manifest.json').read_text());assert status['status']=='completed' and not status['mechanics_only']
        assert status['completed_state']==cfg['end']+1 and status['training_units']==context['task'].units and status['probe_units']==cfg['probe']
        assert status['environment']==dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda)
        shutil.copytree(source,folder,ignore=shutil.ignore_patterns('transfer_inventory.json'))
        files=[]
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                relative=path.relative_to(folder);digest=c.w.sha(path);assert digest==c.w.sha(source/relative)
                files.append(dict(path=relative.as_posix(),bytes=path.stat().st_size,sha256=digest))
        c.w.write(folder.parent/'control_reuse.json',dict(status='passed',reference=reference,matched_config_fields=['case','source_state','fork','end','base_lr','physical','probe','calibration_updates','seed'],matched_common_inputs=common,
            exact_full_fork=True,exact_native_two_step=True,source_fork_sha256=c.w.sha(worker/'shared/fork.pt'),files=files,native_updates_executed_this_run=0))
        return dict(status,reused_from=reference['worker']+'/control',native_updates_executed_this_run=0)
    c.run_arm=run_arm
