"""Two-column overview of the completed seed0 MLP/ResNet 3k GD evidence."""
import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.cifar_eos.core import file_hash, write_json
MANIFEST = HERE / 'completed_3k_figure_manifest.json'


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def sources():
    cohort = json.loads((HERE / 'cohort.json').read_text())
    inputs = json.loads((ROOT / 'cache/eos-multitask/inputs.json').read_text())
    checks = module(ROOT / 'analyses/021-2026-10-15-eos-multitask-lr/03_collect.py', 'completed_3k_checks')
    loader = module(ROOT / 'analyses/020-2026-10-14-eos-multitask/07_plot_training.py', 'completed_3k_loader')
    data, results = [], []
    for rid in ['059', '120']:
        record = next(r for r in cohort['runs'] if r['run_id'] == rid)
        assert record['updates'] == 3000 and record['seed'] == 0 and record['method'] == 'gd'
        if record['kind'] == 'reused_gd_prefix': receipt = checks.verify_legacy(record)
        else:
            assert checks.verified_local(record['run'])
            receipt = json.loads((ROOT / 'runs' / record['run'] / 'retrieval_verification.json').read_text())
        run = ROOT / 'runs' / record['run']
        folder = run / 'artifacts/attempt-01-spark/production'
        config = json.loads((run / 'config.json').read_text())
        manifest = json.loads((folder / 'manifest.json').read_text())
        assert manifest['config_sha256'] == file_hash(run / 'config.json')
        assert manifest['initial_tensor_sha256'] == inputs['initial'][record['task']][0]['initial_tensor_sha256']
        assert record['lr'] == (config['optimizers']['gd']['lr'] if rid == '059' else config['lr'])
        if rid == '120': assert manifest['full_batch'] and manifest['completed_updates'] == 3000
        d = loader.load(record)
        np.testing.assert_array_equal(d['spectral_x'], np.arange(0, 3001, 100))
        assert len(d['loss']) == 3000 and np.isfinite(d['loss']).all() and np.isfinite(d['final_loss'])
        assert np.isnan(d['q'][0]) and np.isfinite(d['q'][1:]).all()
        np.testing.assert_allclose(d['q'][1:], 1-d['cosine'][1:]*d['norm'][1:]/d['norm'][:-1], rtol=2e-10, atol=2e-10)
        np.testing.assert_array_equal(d['cosine'][1:] < 0, d['q'][1:] > 1)
        active = [json.loads(s) for s in (folder / 'metrics.jsonl').read_text().splitlines()][:3000]
        raw = [r['streams']['raw'] for r in active]
        get = lambda key: np.asarray([r[key] for r in raw], dtype=float)
        np.testing.assert_allclose(get('A'), np.cumsum(d['norm']), rtol=2e-10, atol=2e-10)
        np.testing.assert_allclose(get('energy'), np.cumsum(d['norm']**2), rtol=2e-10, atol=2e-10)
        np.testing.assert_allclose(get('B')**2, get('energy')+get('all_pair'), rtol=2e-10, atol=2e-10)
        np.testing.assert_allclose(get('B_over_A'), get('B')/get('A'), rtol=2e-10, atol=2e-10)
        np.testing.assert_allclose(get('lag_one'), np.r_[0, np.cumsum(2*d['cosine'][1:]*d['norm'][1:]*d['norm'][:-1])], rtol=2e-9, atol=2e-9)
        first = np.flatnonzero(d['cosine'][1:] < 0)
        results.append(dict(run_id=rid, run=record['run'], task=record['task'], lr=record['lr'], seed=0,
            updates=3000, initial_loss=float(d['loss'][0]), loss_at_state_3000=float(d['final_loss']),
            first_negative_gradient_update=int(first[0]+2) if len(first) else None,
            late_1000_cosine_mean=float(d['cosine'][-1000:].mean()),
            late_1000_q_mean=float(d['q'][-1000:].mean()), final_B_over_A=float(d['ratio'][-1]),
            maximum_sampled_eta_lambda=float(d['eta_lambda'].max()),
            last_sampled_eta_lambda=float(d['eta_lambda'][-1]), source_sha256=d['source_sha256'],
            config_sha256=file_hash(run / 'config.json'), verified_files=receipt['files'],
            transfer_inventory_sha256=receipt['inventory_sha256']))
        data.append(d)
    return data, results, loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--accept-reviewed', action='store_true')
    args = parser.parse_args()
    if args.accept_reviewed:
        m = json.loads(MANIFEST.read_text())
        assert m['script_sha256'] == file_hash(__file__)
        for key in ['pdf', 'render', 'arrays', 'results']:
            assert file_hash(ROOT / m[key]) == m[key+'_sha256']
        m.update(status='passed', visual_review='Current Poppler render inspected: two columns, six panels, readable LR/seed labels, unsmoothed curves, full/subset spectral labels, reference lines and margins.')
        write_json(MANIFEST, m)
        return
    data, results, loader = sources()
    plt.rcParams.update({'font.family':'serif', 'mathtext.fontset':'stix', 'font.size':10,
        'axes.labelsize':10.5, 'axes.titlesize':11, 'xtick.labelsize':9, 'ytick.labelsize':9,
        'axes.spines.top':False, 'axes.spines.right':False, 'axes.linewidth':.7, 'pdf.fonttype':42})
    fig, axes = plt.subplots(3, 2, figsize=(7.4, 7.0), sharex='col')
    arrays, artists = {}, []
    titles = ['CIFAR-10 / tanh MLP\n5,000 training images', 'CIFAR-10 / ResNet-32\n50,000 training images']
    colours = ['#0072B2', '#D55E00']
    for col, d in enumerate(data):
        axes[0,col].set_title(titles[col], pad=31)
        values = [(np.arange(3001), np.r_[d['loss'], d['final_loss']]),
                  (d['spectral_x'], d['eta_lambda']), (np.arange(1,3000), d['q'][1:])]
        for row,(x,y) in enumerate(values):
            ax = axes[row,col]
            line, = ax.plot(x, y, color=colours[col], lw=.65 if row==2 else 1.15,
                marker='o' if row==1 else None, ms=2.8,
                label=rf'GD, $\eta={d["record"]["lr"]:g}$, seed 0')
            np.testing.assert_array_equal(line.get_xdata(),x)
            np.testing.assert_array_equal(line.get_ydata(),y)
            tag = f'{d["record"]["task"]}_row{row}'
            arrays[tag+'_x'],arrays[tag+'_y'] = x,y
            artists.append(dict(tag=tag, points=len(x),
                x_sha256=hashlib.sha256(np.asarray(x,dtype='<f8').tobytes()).hexdigest(),
                y_sha256=hashlib.sha256(np.asarray(y,dtype='<f8').tobytes()).hexdigest()))
            ax.set_xlim(0,3000); ax.set_xticks([0,1000,2000,3000]); ax.grid(alpha=.16,lw=.55)
            ax.text(.025,.94,f'({chr(97+row*2+col)})',transform=ax.transAxes,va='top',fontsize=9,
                bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=.8))
            if row>0:
                ax.axhline(2 if row==1 else 1,color='.45',ls=(0,(4,3)),lw=.8)
                ax.margins(y=.08)
            if row==1:
                label='Hessian: full 5k' if col==0 else 'Hessian: fixed 5k subset'
                ax.text(.03,.28 if col==0 else .06,label,transform=ax.transAxes,fontsize=8,color='.3',
                    bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1))
            if row==2: ax.set_xlabel('GD iteration')
        axes[0,col].legend(loc='lower center',bbox_to_anchor=(.5,1.01),frameon=False,
            fontsize=9,handlelength=1.6,borderaxespad=0)
    for row,label in enumerate(['Training loss',r'$\eta\lambda_{\max}$',r'$\eta\widehat\kappa_t$']):
        axes[row,0].set_ylabel(label)
    fig.subplots_adjust(left=.105,right=.965,bottom=.08,top=.85,wspace=.28,hspace=.24)
    fig.canvas.draw(); renderer=fig.canvas.get_renderer()
    for artist in fig.findobj(matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text(): continue
        b=artist.get_window_extent(renderer)
        assert b.x0>=-1 and b.y0>=-1 and b.x1<=fig.bbox.width+1 and b.y1<=fig.bbox.height+1, artist.get_text()
    pdf=HERE/'figures/eos_gd_overview_completed_3k.pdf'; pdf.parent.mkdir(exist_ok=True)
    fig.savefig(pdf,metadata={'CreationDate':None,'ModDate':None}); plt.close(fig)
    png=HERE/'scratch/eos_gd_overview_completed_3k.png'; png.parent.mkdir(exist_ok=True)
    subprocess.run([shutil.which('pdftoppm'),'-r','170','-singlefile','-png',str(pdf),str(png.with_suffix(''))],
        check=True,capture_output=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    arrayfile=HERE/'derived/completed_3k_curves.npz'; arrayfile.parent.mkdir(exist_ok=True)
    np.savez_compressed(arrayfile,**arrays)
    resultfile=HERE/'completed_3k_results.json'
    write_json(resultfile,dict(status='passed',scope='Completed MLP/ResNet only; one LR and seed per case; 3000 GD updates.',
        runs=results,gradient_pairs=5998,spectral_points=62,temporal_smoothing='none',seed_aggregation='none',
        checks=['Full used-row and spectral coverage','Verified byte sizes and SHA256 inventories',
                'Frozen initialization/config identities','Exact q and prefix geometry identities','Exact plotted arrays']))
    paths=dict(pdf=pdf,render=png,arrays=arrayfile,results=resultfile)
    record=dict(status='awaiting_visual_review',script_sha256=file_hash(__file__),loader_sha256=file_hash(loader.__file__),
        cohort_sha256=file_hash(HERE/'cohort.json'),curves=artists,visual_review=None)
    for key,path in paths.items():
        record[key]=path.relative_to(ROOT).as_posix(); record[key+'_sha256']=file_hash(path)
    write_json(MANIFEST,record)
    print(json.dumps(dict(status='awaiting_visual_review',results=results),indent=2))


if __name__=='__main__':
    main()
