"""Prepare the two figure changes requested in feedback C12 and C24.

The GD panel preserves the previous generator and all numerical curves; only
its legend changes. The spike diagram is explicitly schematic.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / "analyses/024-2026-10-17-eos-gd-lr-calibration"
DRAFT = ROOT / "manuscript/draft"
TASKS = ["mlp", "resnet", "transformer"]
TITLES = ["CIFAR-10 / MLP", "CIFAR-10 / ResNet-32", "WikiText-2 / Transformer"]
COLORS = ["#0072B2", "#D55E00", "#009E73"]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arrsha(values):
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def plot_gd():
    manifest = dict(status="awaiting_visual_review", generator=Path(__file__).relative_to(ROOT).as_posix(),
                    generator_sha256=sha(__file__), sources={}, curves=[], runs=[])

    def source(path):
        manifest["sources"][path.relative_to(ROOT).as_posix()] = sha(path)
        return path

    singles = read(source(SOURCE / "single_rows_figure_manifest.json"))
    spectra = read(source(SOURCE / "mlp_learning_rates_figure_manifest.json"))
    for entry in [singles, spectra]:
        assert entry["status"] == "passed"
        assert sha(source(ROOT / entry["arrays"])) == entry["arrays_sha256"]
    a = np.load(ROOT / singles["arrays"])
    b = np.load(ROOT / spectra["arrays"])
    results = read(source(SOURCE / "results.json"))
    cohort = read(source(SOURCE / "cohort.json"))
    records = {r["run_id"]: r for r in cohort["runs"]}
    runs = sorted(results["runs"], key=lambda r: (TASKS.index(r["task"]), r["lr"]))
    loader_path = source(ROOT / "analyses/020-2026-10-14-eos-multitask/07_plot_training.py")
    spec = importlib.util.spec_from_file_location("retained_training_loader", loader_path)
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix",
                         "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .6, "pdf.fonttype": 42, "path.simplify": False})
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.5))

    def curve(ax, x, y, tag, **kwargs):
        assert len(x) == len(y) and np.isfinite(y).all()
        line, = ax.plot(x, y, **kwargs)
        np.testing.assert_array_equal(line.get_xdata(), x)
        np.testing.assert_array_equal(line.get_ydata(), y)
        manifest["curves"].append(dict(tag=tag, points=len(x),
                                       x_sha256=arrsha(x), y_sha256=arrsha(y)))

    for col, task in enumerate(TASKS):
        axes[0, col].set_title(f"({chr(97+col)})  {TITLES[col]}", pad=15)
        for color, run in zip(COLORS, [r for r in runs if r["task"] == task]):
            run_id = run["run_id"]
            raw = loader.load(records[run_id])
            assert raw["source_sha256"] == run["source_sha256"]
            loss = a[f"loss_log_run{run_id}_y"]
            np.testing.assert_array_equal(loss, np.r_[raw["loss"], raw["final_loss"]])
            x = a[f"loss_log_run{run_id}_x"]
            np.testing.assert_array_equal(x, np.arange(3001))
            curve(axes[0, col], x, loss, f"loss_run{run_id}", color=color, lw=.85,
                  label=rf"$\eta={run['lr']:g}$")
            x = a[f"eta_kappa_hat_smoothed_run{run_id}_x"]
            y = a[f"eta_kappa_hat_smoothed_run{run_id}_y"]
            # Independent reconstruction: pair t=2..3000, 50 complete pairs.
            np.testing.assert_array_equal(x, np.arange(51, 3001))
            np.testing.assert_allclose(y, np.convolve(raw["q"][1:], np.ones(50)/50, "valid"),
                                       rtol=2e-11, atol=2e-11)
            curve(axes[1, col], x, y, f"eta_kappa_hat_run{run_id}", color=color, lw=.95)
            sx = b[f"run{run_id}_row1_x"]
            sy = b[f"run{run_id}_row1_y"]
            np.testing.assert_array_equal(sx, raw["spectral_x"])
            np.testing.assert_array_equal(sy, raw["eta_lambda"])
            np.testing.assert_array_equal(sx, np.arange(0, 3001, 100))
            # kappa_t is based at theta_(t-2). Keep every historical spectrum.
            curve(axes[1, col], sx+2, sy, f"eta_lambda_max_run{run_id}",
                  color=color, lw=.85, ls="--", marker="o", ms=2, mfc="white", mew=.65)
            manifest["runs"].append(dict(run_id=run_id, task=task, lr=run["lr"],
                raw_source_sha256=raw["source_sha256"], spectral_states=len(sx),
                spectral_min=float(sy.min()), spectral_max=float(sy.max()),
                spectral_final=float(sy[-1]), late_eta_kappa_hat=run["late_1000_q_mean"]))
        for row in range(2):
            ax = axes[row, col]
            ax.set(xlim=(0, 3002), xticks=[0, 1000, 2000, 3000])
            ax.tick_params(length=3, width=.6)
            ax.grid(axis="y", alpha=.15, lw=.5)
        ax = axes[0, col]
        ax.set(yscale="log", xlabel=r"Parameter state $s$")
        ax.yaxis.set_major_locator(FixedLocator([[.2,.3,.4,.6], [.4,.6,1,2], [6,7,8,9,10]][col]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.legend(loc="lower center", bbox_to_anchor=(.5,1.015), borderaxespad=0,
                  ncol=3, fontsize=7, frameon=False, columnspacing=.7,
                  handlelength=1.1, handletextpad=.25)
        ax = axes[1, col]
        ax.text(0,1.025,f"({chr(100+col)})",transform=ax.transAxes,va="bottom")
        ax.axhline(2, color=".4", lw=.65, ls=":", zorder=0)
        ax.set(xlabel=r"Gradient index $t$", ylim=(-.08,[3.4,6.1,3.6][col]),
               yticks=[[0,1,2,3],[0,2,4,6],[0,1,2,3]][col])
        if col == 0:
            axes[0,col].set_ylabel("Training loss")
            axes[1,col].set_ylabel("Scaled curvature")
    fig.legend(handles=[
        Line2D([],[],color=".2",lw=1,label=r"$\eta\widehat\kappa_t$ (50-pair mean)"),
        Line2D([],[],color=".2",lw=.85,ls="--",marker="o",ms=3,mfc="white",
               label=r"$\eta\lambda_{\max}$")],
        loc="lower center",bbox_to_anchor=(.53,.005),ncol=2,frameon=False,
        fontsize=8,handlelength=2.4,columnspacing=2)
    fig.subplots_adjust(left=.085,right=.985,top=.86,bottom=.19,hspace=.60,wspace=.29)
    pdf = HERE / "figures/gd_loss_projected_change_spectral_clean.pdf"
    fig.savefig(pdf, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    target = DRAFT / "figures/analysis024" / pdf.name
    shutil.copyfile(pdf, target)
    manifest.update(pdf=pdf.relative_to(ROOT).as_posix(), pdf_sha256=sha(pdf),
                    manuscript_copy=target.relative_to(ROOT).as_posix(),
                    spectral_points=sum(r["spectral_states"] for r in manifest["runs"]),
                    plotted_points=sum(c["points"] for c in manifest["curves"]),
                    coverage={"mlp": "all 5000 images", "resnet": "first 5 fixed 1000-image groups of 50",
                              "transformer": "first 2500 of 59680 sequences"},
                    clocks="Loss at theta_s; gradient pair t uses states t-2 and t-1; spectrum at state s plotted at t=s+2. Final spectrum t=3002 has no recorded gradient pair.",
                    reduction="Loss and spectra unsmoothed; eta*kappa_hat is unchanged complete trailing 50-pair arithmetic mean.",
                    uncertainty="One shared initialization per model; spectra are retained point estimates, not uncertainty intervals.")
    assert manifest["spectral_points"] == 279 and len(manifest["curves"]) == 27
    baseline = read(source(SOURCE / "curvature_comparison_manifest.json"))
    assert manifest["curves"] == baseline["curves"], "Numerical curves changed"
    manifest["change"] = "C12: delete (retained measurements) from the legend; every curve is identical."
    (HERE / "curvature_comparison_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps({k:manifest[k] for k in ["status","spectral_points","plotted_points","pdf"]}))


def plot_spike_definition():
    # Illustrative values only: this is not a new empirical event or sample.
    x = np.arange(13)
    loss = np.array([1.09, 1.04, 1., 1.11, 1.34, 1.62, 1.82,
                     1.61, 1.34, 1.14, 1.04, .97, .94])
    a, p, c = 2, 6, 11
    assert loss[a+1] > loss[a] and np.all(loss[a+1:c] > loss[a])
    assert loss[c] <= loss[a] and p == a+1+np.argmax(loss[a+1:c])
    fig, ax = plt.subplots(figsize=(3.45, 1.7))
    ax.axvspan(a, c, color='.95', zorder=0)
    ax.axhline(loss[a], color='.45', lw=.7, ls='--', zorder=1)
    ax.plot(x, loss, color=COLORS[0], lw=1, marker='o', ms=2.4)
    for state in (a, p, c):
        ax.plot([state, state], [.82, loss[state]], color='.65', ls=':', lw=.6)
        ax.plot(state, loss[state], 'o', color=COLORS[0], ms=3.6)
    ax.annotate(r'$a_i+1$', (a+1, loss[a+1]), xytext=(9, -10),
                textcoords='offset points', fontsize=9, ha='left', va='top',
                arrowprops=dict(arrowstyle='-', color='.4', lw=.5))
    ax.text(p, loss[p]+.035, 'Peak', ha='center', fontsize=8)
    ax.set(xlim=(-.2, 12.3), ylim=(.82, 1.99),
           xticks=[a, p, c], xticklabels=[r'$a_i$', r'$p_i$', r'$c_i$'],
           yticks=[loss[a]], yticklabels=[r'$\mathcal{L}_{a_i}$'],
           xlabel=r'Parameter state $s$', ylabel='Training loss')
    ax.tick_params(length=2.5, width=.6)
    fig.subplots_adjust(left=.16, right=.98, bottom=.29, top=.93)
    pdf = HERE / 'figures/adam_spike_definition.pdf'
    fig.savefig(pdf, metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    target = DRAFT / 'figures/analysis036' / pdf.name
    shutil.copyfile(pdf, target)
    manifest = dict(status='awaiting_visual_review',
                    generator=Path(__file__).relative_to(ROOT).as_posix(),
                    generator_sha256=sha(__file__),
                    pdf=pdf.relative_to(ROOT).as_posix(), pdf_sha256=sha(pdf),
                    manuscript_copy=target.relative_to(ROOT).as_posix(),
                    source='Explicit synthetic sequence in plot_spike_definition; no training data.',
                    loss=loss.tolist(), a=a, p=p, c=c,
                    reduction='None; connect the illustrative observations with straight lines.',
                    coverage='One schematic recovered event; does not enter any statistic.',
                    uncertainty='Not an estimate; no uncertainty interval.',
                    caption='The fixed pre-rise baseline defines recovery. The first elevated state, peak and first recovered state are marked.')
    (HERE / 'spike_definition_manifest.json').write_text(
        json.dumps(manifest, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(figure=manifest['pdf'], illustrative_boundaries=[a,p,c])))


if __name__ == '__main__':
    (HERE / 'figures').mkdir(exist_ok=True)
    plot_gd()
    plot_spike_definition()
