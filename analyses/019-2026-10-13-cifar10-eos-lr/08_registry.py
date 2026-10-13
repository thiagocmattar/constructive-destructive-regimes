"""Update own records and stage only own shared-registry hunks against HEAD."""
import difflib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GIT = ['git', '-c', 'safe.directory=' + ROOT.as_posix()]


def git(*args):
    return subprocess.check_output(GIT + list(args), cwd=ROOT).decode('utf-8')


def preserve_endings(original, wanted):
    before = original.replace('\r\n', '\n').splitlines(keepends=True)
    after = wanted.splitlines(keepends=True)
    raw = original.splitlines(keepends=True)
    output = []
    for tag, a, b, c, d in difflib.SequenceMatcher(None, before, after).get_opcodes():
        output.extend(raw[a:b] if tag == 'equal' else after[c:d])
    return ''.join(output)


def main():
    results = json.loads((HERE / 'results.json').read_text())
    qa = json.loads((HERE / 'visual_verification.json').read_text())
    assert results['status'] == qa['status'] == 'passed'
    assert not git('diff', '--cached', '--name-only').strip(), 'Preserve existing staged work; do not include it in this commit.'
    runs = results['runs']
    minutes = sum(r['elapsed_seconds'] for r in runs) / 60
    gd = ','.join(f"{r['final_train']['loss']:.4g}" for r in runs[:4])
    adam = ','.join(f"{r['final_train']['loss']:.4g}" for r in runs[4:])
    analysis_row = '| [019](019-2026-10-13-cifar10-eos-lr/README.md) | H018: full-batch CIFAR-10 GD/Adam LR ablation and geometry versus Hessian spectra | Runs059-066; frozen Run058 inputs | Complete; eight20k-update Spark trajectories,160k rows and4,816 eigenpairs verified; four PDFs reviewed | [Results](019-2026-10-13-cifar10-eos-lr/REPORT.md), [observation](019-2026-10-13-cifar10-eos-lr/observations/001-learning-rate-sweep.md); one seed; no cloud rental |\n\n'
    run_rows = ''.join(f"| [{r['run_id']}]({r['run']}/README.md) | H018: CIFAR-10 5k tanh MLP,{r['optimizer'].upper()} LR{r['lr']:.6g}; full geometry/Hessian spectra | Complete;20,000 fixed updates; retrieved/hashed/state-verified | Attempt01-spark; detached queue | [Analysis019](../analyses/019-2026-10-13-cifar10-eos-lr/REPORT.md); final train loss{r['final_train']['loss']:.5g}; four PDFs reviewed |\n" for r in runs) + '\n'
    hypothesis_row = f'| H018 | How do full-batch CIFAR-10 loss and raw/m/mhat/adaptive/update geometry relate to ordinary/preconditioned Hessian curvature across GD/Adam learning rates? | Completed exploratory single-seed fixed-horizon comparison | Runs059-066; [Analysis019](../analyses/019-2026-10-13-cifar10-eos-lr/README.md) | All160k updates and4,816 eigenpairs verified; four PDFs reviewed. Final train losses GD{gd},Adam{adam} in increasing LR order. Fixed20k horizon,one frozen initialization,full5k/10k coverage; no universal stability/alignment proxy or optimized method ranking |\n\n'
    summary = f'[Analysis019: CIFAR-10 GD/Adam learning-rate sweep](../analyses/019-2026-10-13-cifar10-eos-lr/REPORT.md)\ncompleted Runs059-066 on Spark:20,000 fixed updates each,common5k tanh MLP\ninitialization and full10k test evaluation. All160k rows and4,816 residual-checked\nordinary/preconditioned eigenpairs pass source/scalar/state/forward checks; four\nPDFs pass array/render review. Final train losses GD{gd},\nAdam{adam} in increasing LR order. Actual training/diagnostics{minutes:.1f}min;\nETC0,peak allocation<0.35GiB; all outputs retrieved/hashed,owned Spark idle.\nOne frozen seed,no LR tuning; no global stability/convergence or alignment proxy\nclaim. Full scalar/recovery evidence retained locally/on Spark; no cloud rental.\nNext Run067 / Analysis020 / H019.\n\n'
    transforms = {
        'analyses/INDEX.md': lambda text: text.replace('Next analysis: `019-', analysis_row + 'Next analysis: `020-', 1),
        'runs/INDEX.md': lambda text: text.replace('Next run: `059-', run_rows + 'Next run: `067-', 1),
        'research/HYPOTHESES.md': lambda text: text.replace('Next hypothesis: `H018`.', hypothesis_row + 'Next hypothesis: `H019`.', 1),
        'research/INDEX.md': lambda text: text.replace('## Current state\n\n', '## Current state\n\n' + summary, 1),
    }
    patch = ''
    for relative, transform in transforms.items():
        original = git('show', 'HEAD:' + relative)
        wanted = preserve_endings(original, transform(original.replace('\r\n', '\n')))
        assert original != wanted
        path = ROOT / relative
        working = path.read_text(encoding='utf-8')
        marker = {'analyses/INDEX.md': '| [019]', 'runs/INDEX.md': '| [059]',
                  'research/HYPOTHESES.md': '| H018 |', 'research/INDEX.md': '[Analysis019:'}[relative]
        if marker not in working:
            modified = transform(working)
            assert modified != working
            path.write_text(modified, encoding='utf-8')
        patch += ''.join(difflib.unified_diff(original.splitlines(keepends=True), wanted.splitlines(keepends=True),
                                             fromfile='a/' + relative, tofile='b/' + relative))
    patch_path = ROOT / 'tmp/cifar019-own-registry.patch'
    patch_path.parent.mkdir(exist_ok=True)
    patch_path.write_text(patch, encoding='utf-8', newline='\n')
    subprocess.run(GIT + ['apply', '--cached', str(patch_path)], cwd=ROOT, check=True)
    paths = [HERE.relative_to(ROOT).as_posix(), 'src/cifar_eos', 'src/README.md'] + ['runs/' + r['run'] for r in runs]
    subprocess.run(GIT + ['add', '--'] + paths, cwd=ROOT, check=True)
    print(git('diff', '--cached', '--stat'))


if __name__ == '__main__':
    main()
