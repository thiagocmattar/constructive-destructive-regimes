"""A separate single-panel overlay of eta times the requested kappa proxy."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kappa import HERE, ROOT, digest, load_sources, rolling, sha256, write_json

COLORS = ("#0072B2", "#8C4AA8", "#D55E00")
STYLES = ("-", (0, (2, 2)), (0, (7, 3)))
LABELS = (r"$\eta=10^{-2}$", r"$\eta=3.1623\times10^{-3}$", r"$\eta=10^{-3}$")


def nullable(values):
    return [float(v) if np.isfinite(v) else None for v in values]


def main():
    original = HERE / "figures/rosenbrock_kappa_comparison.pdf"
    original_hash = sha256(original)
    assert json.loads((HERE / "verification.json").read_text())["passed"]
    histories, sources, turns = load_sources()
    report = json.loads((HERE / "results.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.labelcolor": "#26313D",
                         "text.color": "#26313D"})
    fig, ax = plt.subplots(figsize=(14, 7.5))
    fig.subplots_adjust(left=.09, right=.975, top=.76, bottom=.20)
    fig.suptitle("Rosenbrock: learning-rate-scaled gradient-identity kappa", fontsize=20, y=.973)
    fig.text(.5, .904, "Three matched AdamW learning rates | Start (0, 1) | 2,000 updates | Weight decay 0.1", ha="center", fontsize=11)
    fig.text(.5, .844, r"$\eta\,\widehat{\kappa}_t=1-\rho_t\,\|g_t\|/\|g_{t-1}\|\quad(\varepsilon_t=0)$", ha="center", fontsize=16)
    records, references, means = [], [], []
    for history, source, row, color, style, label, event in zip(histories, sources, report["sources"], COLORS, STYLES, LABELS, turns["points"]):
        path = ROOT / row["derived_history"]
        assert sha256(path) == row["derived_history_sha256"]
        with np.load(path, allow_pickle=False) as values:
            updates = values["updates"]
            normalized = source["learning_rate"] * values["kappa_proxy"]
            average = rolling(normalized, 100)
            np.testing.assert_allclose(average, source["learning_rate"] * values["kappa_proxy_mean100"], rtol=3e-13, atol=1e-14, equal_nan=True)
        g = history["raw_vectors"]
        direct = np.r_[np.nan, [-float(a @ (b-a)) / float(a @ a) for a,b in zip(g[:-1],g[1:])]]
        np.testing.assert_allclose(normalized, direct, rtol=3e-10, atol=1e-12, equal_nan=True)
        for i in range(100, 2000):
            np.testing.assert_allclose(average[i], sum(float(v) for v in direct[i-99:i+1]) / 100, rtol=2e-10, atol=2e-13)
        assert np.flatnonzero(np.isfinite(average))[0] == 100 and np.isfinite(average).sum() == 1900
        line, = ax.plot(updates, average, color=color, ls=style, lw=2.2, label=label)
        np.testing.assert_array_equal(line.get_ydata(), average)
        ref = ax.axvline(event["update"], color=color, ls="--", lw=.9, alpha=.6)
        references.append(dict(run_id=source["run_id"], update=int(ref.get_xdata()[0])))
        records.append(dict(run_id=source["run_id"], learning_rate=source["learning_rate"],
            normalized_raw=nullable(normalized), normalized_mean100=nullable(average),
            plotted_x_sha256=digest(line.get_xdata()), plotted_y_sha256=digest(line.get_ydata()),
            last100_pair_mean=float(average[-1]), source_history_sha256=source["history_sha256"],
            derived_history_sha256=row["derived_history_sha256"], color=color))
        means.append(average[np.isfinite(average)])
    finite = np.concatenate(means); padding=.06*(finite.max()-finite.min())
    ax.set(xlim=(0,2000), ylim=(min(0,finite.min()-padding),finite.max()+padding),
           xlabel="Optimizer update t", ylabel=r"$\eta\,\widehat{\kappa}_t$ (100-pair trailing mean)")
    ax.grid(axis="y",color="#E2E7EC",lw=.8); ax.set_axisbelow(True)
    ax.legend(loc="upper left",frameon=False,ncol=3,handlelength=3.2)
    fig.text(.09,.113,"Multiply each unsmoothed kappa proxy by its constant learning rate, then average the last 100 finite adjacent pairs.",fontsize=10.5)
    fig.text(.09,.080,"First complete mean at t = 101. All 1,900 means per LR are shown; dashed vertical lines mark the original first y-turns.",fontsize=10.5)
    fig.text(.09,.047,"Dimensionless proxy; epsilon = 0 (exact gradients). The SGD-derived identity is evaluated along AdamW; no stability threshold is implied.",fontsize=10.5)
    destination=HERE/"figures/rosenbrock_kappa_eta_overlay.pdf"
    fig.savefig(destination); fig.savefig(HERE/"scratch/rosenbrock_kappa_eta_overlay.png",dpi=130)
    plt.close(fig)
    assert sha256(original) == original_hash
    data=HERE/"eta_overlay_data.json"
    write_json(data,dict(updates=updates.tolist(),smoothing_window=100,epsilon=0,
        normalization="multiply by eta; no rescaling by an observed maximum",sources=records))
    write_json(HERE/"figure_manifest_rosenbrock_kappa_eta_overlay.json",dict(
        figure=destination.relative_to(ROOT).as_posix(),sha256=sha256(destination),
        panels=1,dimensions_inches=[14,7.5],script_sha256=sha256(__file__),
        metric_implementation_sha256=sha256(HERE/"kappa.py"),data_sha256=sha256(data),
        smoothing_window=100,first_finite_mean_update=101,means_per_lr=1900,
        vertical_reference_lines=references,y_scale="linear",thinning=None,
        original_comparison_sha256_unchanged=original_hash,
        rendered_series=[{k:r[k] for k in ("run_id","learning_rate","plotted_x_sha256","plotted_y_sha256","last100_pair_mean")} for r in records]))
    print(json.dumps({r['run_id']:r['last100_pair_mean'] for r in records}))
    print(destination)


if __name__ == "__main__":
    main()
