"""Write descriptive closeout records from verified, full-horizon evidence."""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def readable(text):
    # Presentation spacing only; never applied to scientific configs/evidence.
    text = re.sub(r'\b(All|all|each|every|through|fixed|took|was|across|within|complete|at|about|first|using|plus|cover|the|and|with)(?=\d)', r'\1 ', text)
    text = re.sub(r'\bLR(?=\d)', 'LR ', text)
    text = re.sub(r'\b(completed|updates|last|Last|accuracy|in|spans|reference|only|near|is|verifies|records|use|a|approximately|to)(?=-?\d)', r'\1 ', text)
    text = re.sub(r',(?=[A-Za-z])', ', ', text)
    text = re.sub(r'(\d)(?=(?:updates|minutes|minute|ordinary|additional|complete|eigenpairs|spectra|pairs|images|parameters|states|training|retained|files|MiB|GB)\b)', r'\1 ', text)
    for old, new in [('preselectedGD', 'preselected GD'), ('preselectedADAM', 'preselected ADAM'), ('withinGD', 'within GD'), ('withinADAM', 'within ADAM'), ('atLR', 'at LR'), ('seed0', 'seed 0'),
                     ('<=0.005', '<= 0.005'), ('<=0.35', '<= 0.35'),
                     ('every100', 'every 100'), ('with512', 'with 512'), ('using512', 'using 512'),
                     ('and4,816', 'and 4,816'), ('all160,000', 'all 160,000'), ('ETC0', 'ETC 0'),
                     ('GD2/eta', 'GD 2/eta'), ('Adam38/eta', 'Adam 38/eta'), ('top2', 'top 2'), (',201', ', 201'), ('45C', '45 C')]:
        text = text.replace(old, new)
    return text


def main():
    results = json.loads((HERE / 'results.json').read_text())
    verification = json.loads((HERE / 'verification.json').read_text())
    retrieval = json.loads((HERE / 'retrieval.json').read_text())
    figures = json.loads((HERE / 'figure_manifest.json').read_text())
    assert results['status'] == verification['status'] == 'passed'
    assert retrieval['status'] == 'complete'
    runs = results['runs']
    table = ['| Run | Method | LR | Final train loss | Train accuracy | Test loss | Test accuracy |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    geometry = ['| Run | Raw negative pairs | Late raw cosine | Late mhat cosine | Raw final B/A | mhat final B/A | Final eta*lambda1 | Final eta*lambda1(P) | Late eta-kappa |',
                '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in runs:
        table.append(f"| {r['run_id']} | {r['optimizer'].upper()} | {r['lr']:.6g} | {r['final_train']['loss']:.7g} | {r['final_train']['accuracy']:.2%} | {r['final_test']['loss']:.7g} | {r['final_test']['accuracy']:.2%} |")
        raw = r['streams']['raw']
        momentum = r['streams'].get('unbiased_momentum')
        mcos = f"{momentum['last_1000_cosine_mean']:.5f}" if momentum else 'n/a'
        mba = f"{momentum['B_over_A']:.6g}" if momentum else 'n/a'
        preconditioned = f"{r['final_eta_preconditioned_lambda_max']:.6g}" if momentum else 'n/a'
        geometry.append(f"| {r['run_id']} | {raw['negative_adjacent_fraction']:.2%} | {raw['last_1000_cosine_mean']:.5f} | {mcos} | {raw['B_over_A']:.6g} | {mba} | {r['final_eta_lambda_max']:.6g} | {preconditioned} | {r['eta_kappa_last_1000_mean']:.6g} |")
    minutes = sum(r['elapsed_seconds'] for r in runs) / 60
    retained_bytes = sum(r['verified_bytes'] for r in retrieval['runs'].values())
    archive_bytes = sum(r['archive_bytes'] for r in retrieval['runs'].values())
    peak_alloc = max(r['memory']['peak_allocated_bytes'] for r in runs) / 2**20
    peak_reserved = max(r['memory']['peak_reserved_bytes'] for r in runs) / 2**20
    best_train = {method: min([r for r in runs if r['optimizer'] == method], key=lambda r: r['final_train']['loss']) for method in ['gd', 'adam']}
    text = '# Analysis019 results: full-batch CIFAR-10 GD/Adam LR sweep\n\n'
    text += f'All eight runs completed20,000 updates on the owned DGX Spark. Training and diagnostics took{minutes:.2f} minutes in total; the observed queue timestamps are in [queue_state.json](queue_state.json). Training ETC is zero. Peak CUDA allocation/reservation across the cohort was{peak_alloc:.1f}/{peak_reserved:.1f}MiB.\n\n'
    text += '\n'.join(table) + '\n\n'
    text += 'These are final-state evaluations on all5,000 training and10,000 test images. Loss is half the summed squared output error to one-hot labels, averaged over images. The loss figure includes all pre-update training states plus the separately evaluated final state. Test performance is reported descriptively; no separate validation/tuning split or seed ensemble was used.\n\n'
    text += '\n'.join(geometry) + '\n\n'
    text += 'Late cosine means cover updates19,001-20,000. Negative-pair fractions cover every19,999 defined pair. A=sum of vector norms and B=norm of the vector sum use the complete prefix; no temporal truncation is applied. Momentum in this table is bias-corrected mhat. Stored m is also retained and separately plotted. Their adjacent cosines agree under positive rescaling, but their early norms and cumulative geometry differ.\n\n'
    text += 'The final recorded raw gradient belongs to theta[19,999], while final train/test evaluations and the last Hessian belong to theta[20,000]. Late eta-kappa uses the last1,000 defined gradient pairs. These operational state indices are preserved; the tables do not pretend that the last pre-update gradient was recomputed at the final checkpoint.\n\n'
    text += '## Descriptive findings\n\n'
    for method in ['gd', 'adam']:
        r = best_train[method]
        text += f"The lowest final training loss within the preselected{method.upper()} grid occurs at LR{r['lr']:.6g}: {r['final_train']['loss']:.7g}, with test accuracy{r['final_test']['accuracy']:.2%}. This is a finite-horizon single-initialization comparison, not an optimized or general method ranking.\n\n"
    gd_edge, adam_runs = runs[1:4], runs[4:]
    text += f"The three larger GD rates finish with eta*lambda1 in{min(r['final_eta_lambda_max'] for r in gd_edge):.3f}-{max(r['final_eta_lambda_max'] for r in gd_edge):.3f}; their late raw cosine means are all approximately-0.999. LR0.004 instead retains a late cosine near+1 and finishes below the spectral reference. The onset moves earlier with larger LR in this cohort.\n\n"
    text += f"Adam's final eta*lambda1(P) spans{min(r['final_eta_preconditioned_lambda_max'] for r in adam_runs):.2f}-{max(r['final_eta_preconditioned_lambda_max'] for r in adam_runs):.2f}, near the frozen-preconditioner reference38. Its late bias-corrected momentum cosine is negative at every rate ({min(r['streams']['unbiased_momentum']['last_1000_cosine_mean'] for r in adam_runs):.3f} to{max(r['streams']['unbiased_momentum']['last_1000_cosine_mean'] for r in adam_runs):.3f}). Ordinary eta*lambda1 is only{min(r['final_eta_lambda_max'] for r in adam_runs):.4f}-{max(r['final_eta_lambda_max'] for r in adam_runs):.4f}, while late eta-kappa remains near1.8-1.9. The gradient proxy therefore does not supply an ordinary-Hessian curvature estimate on these Adam trajectories.\n\n"
    text += '| GD LR | First negative raw pair (update) | First sampled eta*lambda1 >2 (state) | Late sampled eta*lambda1 mean | Last1,000 eta-kappa mean |\n| ---: | ---: | ---: | ---: | ---: |\n'
    for r in runs[:4]:
        text += f"| {r['lr']:.6g} | {r['streams']['raw']['first_negative_adjacent_update']} | {r['first_sampled_state_eta_lambda_above_2']} | {r['last_1000_eta_lambda_max_mean']:.6g} | {r['eta_kappa_last_1000_mean']:.6g} |\n"
    text += '\nThe Hessian is sampled every100 updates, while gradient pairs are measured every update. These event indices have different resolution and do not establish a causal ordering. GD2/eta is the standard local quadratic reference. For Adam, the plotted frozen-preconditioner reference is38/eta at beta1=0.9, using P=(1-beta1^t)(sqrt(vhat)+epsilon) as in the released adaptive-EOS implementation. The moving preconditioner and nonlinear objective prevent treating that line as a universal stability guarantee.\n\n'
    text += 'The gradient-derived quantity eta*kappa=1-rho_t*||g_t||/||g_(t-1)|| is retained from the previous analyses. There is no injected gradient noise here, so its explicit noise term is zero. On Adam it remains a trajectory proxy: the actual step is adaptive and differs from -eta*g. Adjacent sign, parameter norm, B/A or a low training loss alone does not certify stability, convergence or generalization.\n\n'
    text += '## Exact raw-gradient interaction\n\n'
    text += '| Run | Energy E | All-pair B squared minus E | Lag-one interaction | Beyond-lag-one interaction | Absolute lag-one omission / B squared |\n| --- | ---: | ---: | ---: | ---: | ---: |\n'
    for r in runs:
        raw = r['streams']['raw']
        text += f"| {r['run_id']} | {raw['energy']:.7g} | {raw['all_pair']:.7g} | {raw['lag_one']:.7g} | {raw['beyond_lag_one']:.7g} | {abs(raw['beyond_lag_one']) / raw['B']**2:.5g} |\n"
    text += '\nThese are temporal sums of batch-mean gradients, not per-example gradient energy or intra-batch alignment. The last column measures the full-prefix squared-norm error from omitting all interactions beyond lag one, relative to the true B squared. It can exceed1. Strongly negative adjacent cosines do not determine the sign of exact all-pair interaction.\n\n'
    text += '## Evidence and checks\n\n'
    text += f"[Verification](verification.json) checks all{verification['updates_checked']:,} update rows, all160,000 complete test evaluations,201ordinary spectra per run and200additional preconditioned spectra per Adam run: {verification['eigenpairs_checked']:,} eigenpairs in total. Source/config/input identities, initial raw norms/loss/spectra, scalar prefix identities, cosine and ratio bounds, mhat scaling, kappa arithmetic, final optimizer vectors and exact displacement telescoping pass. An independent direct affine/tanh forward computation with512-image chunks reproduces every final train/test result.\n\n"
    text += f"[Figure manifest](figure_manifest.json) checks every plotted line's actual arrays and{figures['mean_values_checked']:,} complete trailing means. Loss remains unsmoothed; the norm grid retains faint original data beneath a10-update raw mean; cosine and kappa panels use100complete-pair means. All20,000 updates are included, without temporal thinning.\n\n"
    text += f"[Retrieval](retrieval.json) verifies{sum(r['files_verified'] for r in retrieval['runs'].values())} retained files ({retained_bytes/1e9:.3f}GB uncompressed) and{archive_bytes/1e9:.3f}GB transfer archives by size/SHA256. Full scalar histories, prepared inputs, recovery/final states and archives remain locally and on Spark outside Git; compact manifests, spectra, results and figures are tracked. No rented cloud resources or billing. Dense intermediate vector histories were intentionally disabled: scalar identities can be audited, but arbitrary long-lag vector profiles and per-image gradient energy cannot be reconstructed.\n\n"
    closeout = json.loads((HERE / 'compute_closeout.json').read_text())
    assert closeout['task_remote_retained_bytes'] < closeout['storage_allowance_bytes']
    text += f"The [final Spark check](compute_closeout.json) records0% GPU utilization and{closeout['gpu_temperature_celsius']}C, with{closeout['task_remote_retained_bytes']/1e9:.3f}GB in the task directory including transfer archives. Prepared inputs from Run058 remain in their separate frozen directory. No unrelated process or storage was changed.\n\n"
    text += '## Figures\n\n'
    for filename, item in figures['figures'].items():
        text += f"- [{filename}](figures/{filename}): {item['caption']}\n"
    text += '\nThe [observation](observations/001-learning-rate-sweep.md) records sources, reductions, captions, coverage and interpretation limits.\n'
    (HERE / 'REPORT.md').write_text(readable(text), encoding='utf-8')
    observation = '# Observation001: CIFAR-10 full-batch learning-rate sweep\n\n'
    observation += 'Sources: frozen Runs059-066 `artifacts/attempt-01-spark/production/{metrics.jsonl,curvature.jsonl,manifest.json,summary.json,final.pt}`, the [cohort/config identity](../cohort.json), [retrieval inventories](../retrieval.json), [independent reductions](../verification.json) and [results](../results.json). The [report](../REPORT.md) contains the complete numerical tables.\n\n'
    observation += 'Primary external references: [original Edge of Stability](https://arxiv.org/abs/2103.00065), [Adaptive Gradient Methods at the Edge of Stability](https://arxiv.org/abs/2207.14484), and their [released implementation](https://github.com/locuslab/edge-of-stability), pinned by Run058. We reproduce the small5k tanh squared-loss setup; the frozen default seed0 initialization, LA selection and fixed20k horizon are explicit choices, not a claim of pixel-identical paper figures.\n\n'
    observation += 'Reduction and generator: [04_reduce_and_verify.py](../04_reduce_and_verify.py) audits all rows, eigenpairs, identities and final states; [05_figures.py](../05_figures.py) produces publication PDFs with source and plotted array hashes. Complete100-pair trailing means are right-aligned at the last update; the first is at update101 because the first cosine/kappa has no previous vector. Norm means use10complete samples and first appear at update10. Undefined first values are not replaced by zero.\n\n'
    observation += 'Coverage: one matched seed0, first5,000 official training examples, all10,000 test examples each update, all20,000 updates in each of eight runs; ordinary spectra at states0..20,000 every100, and Adam preconditioned spectra at100..20,000 every100. All656,810 trainable coordinates contribute to the geometry; no clipping/decay/noise/schedule. FP32 optimizer state and FP64 vector prefixes are distinct. Initial weights/data hashes and runtime match throughout.\n\n'
    observation += 'Uncertainty and caveats: no independent seed replication, no confidence bands, no optimized LR grid, finite horizon only. Spectra pass relative residual<=0.005 but remain iterative estimates on100-step sampled states. Hessian largest algebraic eigenvalues are not the largest absolute eigenvalues. Curvature at the labeled state and the pairwise gradient proxy refer to different operational quantities. The Adam38/eta line is a local frozen-preconditioner reference, not a global nonlinear threshold. No per-image or missing dense long-lag gradient decomposition is inferred.\n\n'
    for filename, item in figures['figures'].items():
        observation += f"## {filename}\n\nCaption: {item['caption']}\n\nSource and reduction: full histories above; actual plot arrays and PDF SHA256 are recorded in [figure_manifest.json](../figure_manifest.json). Training loss is unsmoothed, norms/cumulative lengths are explicitly logarithmic, cosines/B/A use linear Y, and all lines share the complete fixed horizon. Color encodes learning rate within each method; styles identify vector or eigenvalue.\n\n"
    observation += 'Numerical findings are descriptive and preserved in the report. Visual QA and PDF text/page checks are recorded separately in `visual_verification.json`; all figures must pass before closeout.\n'
    (HERE / 'observations').mkdir(exist_ok=True)
    (HERE / 'observations/001-learning-rate-sweep.md').write_text(readable(observation), encoding='utf-8')
    readme = (HERE / 'README.md').read_text()
    if '\n## Figures\n' in readme:
        readme = readme.split('\n## Figures\n')[0] + '\n'
    start = readme.index('Status:')
    stop = readme.index('## Comparison')
    readme = readme[:start] + 'Status: complete. All eight Spark runs reached20,000 fixed updates; results and\nfigures are retrieved, hash-verified and numerically audited. See the\n[results report](REPORT.md), [observation](observations/001-learning-rate-sweep.md),\n[verification](verification.json) and [four figures](#figures). Run058 is unchanged.\n\n' + readme[stop:]
    readme += '\n## Figures\n\n'
    for filename in figures['figures']:
        readme += f'- [{filename}](figures/{filename})\n'
    readme += f'\nCloseout: {minutes:.2f}minutes training/diagnostics; ETC0; all{verification["updates_checked"]:,}updates and{verification["eigenpairs_checked"]:,}eigenpairs audited. Full scalar/checkpoint evidence remains locally and on the owned Spark; no rental/billing. See the report for actual storage and the final device check.\n'
    (HERE / 'README.md').write_text(readable(readme), encoding='utf-8')
    for r in runs:
        run = ROOT / 'runs' / r['run']
        line = f"# Run{r['run_id']}: CIFAR-10 full-batch {r['optimizer'].upper()}, LR{r['lr']:.9g}\n\n"
        line += 'Complete:20,000 fixed updates on DGX Spark; full artifact retrieval, independent\nscalar/eigenpair/state checks and final forward evaluation passed.\n'
        line += f"Final train loss/accuracy: {r['final_train']['loss']:.8g}/{r['final_train']['accuracy']:.2%};\nfinal10k test loss/accuracy: {r['final_test']['loss']:.8g}/{r['final_test']['accuracy']:.2%}.\n\n"
        line += 'Owning [Analysis019](../../analyses/019-2026-10-13-cifar10-eos-lr/README.md)\ndefines the matched comparison and contains results, figures and interpretation.\n'
        line += f"[Reduced evidence](../../analyses/019-2026-10-13-cifar10-eos-lr/results/Run{r['run_id']}.json) and [verification](../../analyses/019-2026-10-13-cifar10-eos-lr/verification/Run{r['run_id']}.json)\ncover all updates and retained final vectors. Immutable config and attempt source\nbytes remain unchanged; scalar histories/checkpoints are retained outside Git.\n"
        (run / 'README.md').write_text(readable(line), encoding='utf-8')
    print(json.dumps({'status': 'records_written', 'training_minutes': minutes, 'retained_bytes': retained_bytes}))


if __name__ == '__main__':
    main()
