"""Read live catalog and recorded timings; estimate costs without provisioning."""
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.cifar_eos.core import file_hash, stamp, write_json

HERE = Path(__file__).resolve().parent
RUN = ROOT/'runs/213-2026-11-02-eos-adam-from-init-calibration'


def main():
    version = subprocess.check_output(['runpodctl', 'version'], text=True).strip()
    catalog = json.loads(subprocess.check_output(
        ['runpodctl', 'gpu', 'list', '--include-unavailable'], text=True))
    selected = [row for row in catalog if row['displayName'] in ('RTX 4090', 'RTX 5090')]
    price = next(row['securePricePerHr'] for row in selected if row['displayName'] == 'RTX 4090')
    source = ROOT/'analyses/034-2026-11-01-eos-adam-continuations/results.json'
    profiles = json.loads(source.read_text())['profiles']
    references = []
    for profile in profiles:
        folder = ROOT/profile['artifact_root']
        env = json.loads((folder.parent/'environment.json').read_text())
        if env['gpu'] != 'NVIDIA GeForce RTX 4090':
            continue
        arm = next(row for row in json.loads((folder/'result.json').read_text())['arms']
            if row['lr'] == profile['adam_lr'])
        references.append(dict(task=profile['task'], horizon=profile['horizon'],
            seconds_per_update=arm['elapsed_seconds']/profile['horizon'],
            source=(folder/'result.json').relative_to(ROOT).as_posix(),
            sha256=file_hash(folder/'result.json')))
    local_file = RUN/'scratch/transformer-four-stream-timing.json'
    local = json.loads(local_file.read_text())
    samples = local['four_stream_updates']['measured_seconds']
    assert len(samples) == 2 and local['mechanics_only']
    assumptions = dict(mlp=[.05, .10], resnet=[2.8, 3.5], transformer=[8., 16.])
    cases = []
    for side in (0, 1):
        step = {name: bounds[side] for name, bounds in assumptions.items()}
        overhead_factor = [1.10, 1.20][side]
        setup_hours = [.25, .50][side]
        retrieval_hours = [.10, .25][side]
        transition_hours = [1/12, 1/6][side]
        cal_arm_work = {name: 256*seconds/3600*overhead_factor for name, seconds in step.items()}
        main_arm_work = {name: 1024*seconds/3600*overhead_factor for name, seconds in step.items()}
        # Seven workers, one LR per worker; each screens all three tasks.
        cal_wall = sum(cal_arm_work.values())+setup_hours+retrieval_hours
        cal_gpu_hours = 7*(sum(cal_arm_work.values())+setup_hours+retrieval_hours)
        # Three Transformer and three ResNet arms run concurrently. MLP arms
        # execute on the ResNet workers; the seventh worker can be released.
        main_wall = max(main_arm_work['transformer'], main_arm_work['resnet']+main_arm_work['mlp'])
        main_wall += transition_hours+retrieval_hours
        main_gpu_hours = 3*sum(main_arm_work.values())+6*(transition_hours+retrieval_hours)
        storage_per_gpu_hour = 100*.10/(30*24)  # 100GB running disk/worker.
        cases.append(dict(case=['lower_planning_scenario', 'upper_planning_scenario'][side],
            step_seconds=step, evaluation_checkpoint_allowance_factor=overhead_factor,
            setup_hours_per_worker=setup_hours, retrieval_hours_per_worker=retrieval_hours,
            phase_transition_hours_per_main_worker=transition_hours,
            calibration=dict(wall_hours=cal_wall, gpu_hours=cal_gpu_hours,
                cost_usd=cal_gpu_hours*(price+storage_per_gpu_hour)),
            production=dict(wall_hours=main_wall, gpu_hours=main_gpu_hours,
                cost_usd=main_gpu_hours*(price+storage_per_gpu_hour)),
            combined_wall_hours=cal_wall+main_wall,
            combined_cost_usd=(cal_gpu_hours+main_gpu_hours)*(price+storage_per_gpu_hour)))
    result = dict(checked_utc=stamp(), runpodctl_version=version, live_catalog=selected,
        price_source='runpodctl gpu list --include-unavailable; Secure on-demand',
        pricing_url='https://www.runpod.io/pricing',
        storage_url='https://docs.runpod.io/pods/pricing',
        historical_reference_seconds={name: dict(count=len(values), median=statistics.median(values),
            minimum=min(values), maximum=max(values)) for name in assumptions
            if (values := [row['seconds_per_update'] for row in references if row['task'] == name])},
        historical_sources=references, historical_results_sha256=file_hash(source),
        local_mechanics=dict(source=local_file.relative_to(ROOT).as_posix(),
            sha256=file_hash(local_file), environment=local['environment'],
            measured_full_update_seconds=samples,
            peak_allocated_bytes=local['four_stream_updates']['peak_allocated_bytes']),
        planning_assumptions=dict(maximum_concurrent_gpus=7, screening_arms=21,
            screening_updates_each=256, production_arms=9, production_updates_each=1024,
            new_remote_step_seconds=assumptions, disk_gb_per_worker=100,
            running_storage_usd_per_gb_month=.10),
        scenarios=cases, suggested_calibration_ceiling_usd=20,
        suggested_combined_ceiling_usd=50, authorization='estimate_only_no_launch',
        limits=['Catalog availability is point-in-time without exact GPU counts; seven workers are not reserved.',
            'Transformer chunk64 is unmeasured on Runpod with the new capture. The 8--16s interval is an explicit planning assumption, not a confidence interval.',
            'Old cloud Transformer uses chunk512; local full-update measurement is on a 5070Ti Laptop and is not a cloud throughput measurement.',
            'A local chunk512 attempt was stopped under VRAM pressure and is excluded from throughput inference.',
            'Estimates assume completed workers are verified and deleted promptly; no paid idle fleet is held for the slowest worker.',
            'Includes evaluation/checkpoint allowances, setup, retrieval and ephemeral disk; no HVP audit, extra seeds, grid expansion or beyond-1024 production.',
            'Prices can change. Real timing, retrieval and aggregate spend must be rechecked at launch. No resource was provisioned.'])
    write_json(HERE/'runpod_estimate.json', result)
    print(json.dumps(dict(price_per_gpu_hour=price, scenarios=cases,
        suggested_calibration_ceiling_usd=20, suggested_combined_ceiling_usd=50)))


if __name__ == '__main__':
    main()
