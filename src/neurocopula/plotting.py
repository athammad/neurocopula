"""
Plots for inspecting and comparing copula models.

Every function returns a matplotlib ``Figure`` (and optionally saves it), so
they compose into notebooks and reports. All of them share
`neurocopula.theme`, so a mixed figure set still reads as one system, and all
of them follow the same conventions: colour is assigned to the ENTITY
(observed data is always the same blue), a legend appears whenever more than
one series is on screen, and signed quantities use a diverging scale with a
neutral midpoint so "no dependence" reads as absence of colour.

The functions divide by what they answer:

- **Did it train properly?** `plot_training_curves`, `plot_diagnostics`
- **Does the dependence look right?** `plot_copula_scatter`,
  `plot_pairwise_grid`, `plot_dependence_heatmap`, `plot_tau_error_heatmap`
- **Does it get the tails right?** `plot_tail_dependence_curve`,
  `plot_tail_concentration`, `plot_exceedance_comparison`
- **Are the marginals intact?** `plot_marginal_comparison`
- **What does a conditional query look like?** `plot_conditional_distribution`
- **How uncertain is it?** `plot_uncertainty`
- **How do the models stack up?** `plot_benchmark_bars`,
  `plot_model_comparison`
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import (
    kendall_tau_matrix,
    tail_dependence_curve,
)
from .theme import (
    CATEGORICAL,
    DIVERGING_CMAP,
    INK,
    INK_MUTED,
    SERIES,
    apply_theme,
    series_color,
)

__all__ = [
    "plot_training_curves",
    "plot_diagnostics",
    "plot_copula_scatter",
    "plot_pairwise_grid",
    "plot_dependence_heatmap",
    "plot_tau_error_heatmap",
    "plot_tail_dependence_curve",
    "plot_tail_concentration",
    "plot_exceedance_comparison",
    "plot_marginal_comparison",
    "plot_conditional_distribution",
    "plot_uncertainty",
    "plot_benchmark_bars",
    "plot_model_comparison",
]


def _finish(fig, path=None):
    """Tighten layout, optionally save, and return the figure."""
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=140, bbox_inches="tight")
    return fig


def _weighted_quantile(values, weights, q):
    """Quantile of a weighted sample (values need not be sorted)."""
    order = np.argsort(values)
    v, cw = values[order], np.cumsum(weights[order])
    return float(np.interp(q, cw / cw[-1], v))


def _tick_labels(ax, cols) -> None:
    """
    Label a matrix axis, rotating only when the names are long enough to
    collide. Rotating two-character labels just makes them harder to read.
    """
    rotate = max(len(str(c)) for c in cols) > 4
    ax.set_xticks(range(len(cols)), cols, fontsize=7,
                  rotation=45 if rotate else 0,
                  ha="right" if rotate else "center")
    ax.set_yticks(range(len(cols)), cols, fontsize=7)


def _as_frame(x, columns=None) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x
    arr = np.atleast_2d(np.asarray(x))
    return pd.DataFrame(arr, columns=columns or [f"x{i+1}" for i in range(arr.shape[1])])


# =====================================================================
# training
# =====================================================================
def plot_training_curves(model, path=None, ax=None):
    """
    Train and validation NLL per epoch, with the checkpoint epoch marked.

    This is the overfitting check at a glance: the two curves should fall
    together, and the gap that opens after the checkpoint line is exactly
    the capacity the model was prevented from wasting. Note that these are
    always the TRAIN/VALIDATION SPLIT curves, even when `refit_on_full_data`
    later produced the deployed model -- that split is the honest evidence
    about generalisation, so it is what gets shown.
    """
    apply_theme()
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    h = model.history
    ax.plot(h["train"], color=CATEGORICAL[0], label="train NLL")
    ax.plot(h["val"], color=CATEGORICAL[1], label="validation NLL")
    if model.best_epoch is not None:
        ax.axvline(model.best_epoch, color=INK_MUTED, ls="--", lw=1.2,
                   label=f"checkpoint (epoch {model.best_epoch})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("negative log-likelihood")
    ax.set_title("Training curves")
    ax.legend(loc="upper right")
    return _finish(fig, path) if fig is not None else ax


def plot_diagnostics(model, data: pd.DataFrame | None = None, n_samples: int = 3000,
                     path=None):
    """
    A four-panel post-fit summary: training curves, the model's Kendall tau
    matrix, a copula-scale scatter of the two most strongly dependent
    columns, and marginal histograms.

    The first figure to look at after `fit`. If `data` is passed, the tau
    matrix panel shows the model's error against the data's tau instead of
    the raw values, which is the more useful view.
    """
    apply_theme()
    samples = model.sample(n_samples)
    cols = model.columns
    d = len(cols)

    fig = plt.figure(figsize=(13, 7.6), layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.75])

    plot_training_curves(model, ax=fig.add_subplot(gs[0, 0]))

    # tau matrix, or tau error against the data if we have it
    ax = fig.add_subplot(gs[0, 1])
    tau_model = kendall_tau_matrix(samples)
    if data is not None:
        mat = (tau_model - kendall_tau_matrix(data[cols]))
        title, vmax, cmap = "Kendall tau error (model - data)", 0.25, DIVERGING_CMAP
    else:
        mat = tau_model
        title, vmax, cmap = "Kendall tau (model samples)", 1.0, DIVERGING_CMAP
    im = ax.imshow(mat.values, vmin=-vmax, vmax=vmax, cmap=cmap)
    _tick_labels(ax, cols)
    ax.set_title(title)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046)

    # copula-scale scatter of the most dependent pair
    ax = fig.add_subplot(gs[0, 2])
    off = tau_model.where(~np.eye(d, dtype=bool))
    a, b = off.stack().abs().idxmax() if d > 1 else (cols[0], cols[0])
    if model.copula_mode and d > 1:
        u = model.sample_uniform(min(n_samples, 4000))
        ax.scatter(u[a], u[b], s=4, alpha=0.35, color=SERIES["neurocopula"],
                   edgecolors="none")
        ax.set_xlabel(f"u({a})")
        ax.set_ylabel(f"u({b})")
        ax.set_title(f"Copula scale: {a} vs {b}")
    else:
        ax.scatter(samples[a], samples[b], s=4, alpha=0.35,
                   color=SERIES["neurocopula"], edgecolors="none")
        ax.set_xlabel(a)
        ax.set_ylabel(b)
        ax.set_title(f"Strongest pair: {a} vs {b}")

    # marginals
    # Always a 3-slot row, so a 2-column model does not get two enormous
    # histograms; unused slots simply stay empty.
    n_show = min(d, 3)
    for i in range(n_show):
        ax = fig.add_subplot(gs[1, i])
        col = cols[i]
        if data is not None:
            ax.hist(data[col], bins=50, density=True, color=SERIES["data"],
                    alpha=0.55, label="data")
        ax.hist(samples[col], bins=50, density=True, histtype="step",
                color=SERIES["neurocopula"], lw=1.8, label="model")
        ax.set_title(col, fontsize=9)
        ax.set_yticks([])
        if i == 0:
            ax.legend(loc="upper right")

    fig.suptitle("NeuroCopula diagnostics", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    if path:
        fig.savefig(path, dpi=140, bbox_inches="tight")
    return fig


# =====================================================================
# dependence structure
# =====================================================================
def plot_copula_scatter(samples: Mapping[str, pd.DataFrame], col1: str, col2: str,
                        n: int = 3000, path=None, uniform: bool = True):
    """
    Side-by-side scatter of one column pair, one panel per named sample.

    Pass ``uniform=True`` (the default) with copula-scale data and the
    marginals are flat by construction, so every visible feature of the
    cloud is dependence -- the corner clustering that means tail dependence
    is obvious here and easily hidden in original units, where a fat
    marginal spreads points out and disguises it.

    samples: ordered mapping of label -> DataFrame, e.g.
        ``{"observed": u_data, "NeuroCopula": u_flow, "vine": u_vine}``.
        At most three panels; the palette is only validated for three
        simultaneous series in an all-pairs form like a scatter.
    """
    apply_theme()
    labels = list(samples)
    if len(labels) > 3:
        raise ValueError(
            "plot_copula_scatter shows at most 3 samples: the categorical palette "
            "is validated for 3 simultaneous series in scatter form. Split into "
            "multiple figures instead."
        )
    fig, axes = plt.subplots(1, len(labels), figsize=(4.2 * len(labels), 4.3),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, label in zip(axes, labels, strict=True):
        s = samples[label]
        sub = s.iloc[:n] if len(s) > n else s
        color = series_color(label.lower(), fallback_slot=labels.index(label))
        ax.scatter(sub[col1], sub[col2], s=4, alpha=0.3, color=color,
                   edgecolors="none")
        ax.set_title(label)
        ax.set_xlabel(f"u({col1})" if uniform else col1)
        if uniform:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
    axes[0].set_ylabel(f"u({col2})" if uniform else col2)
    fig.suptitle(f"Dependence structure: {col1} vs {col2}"
                 + ("  (copula scale)" if uniform else ""),
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    return _finish(fig, path)


def plot_pairwise_grid(samples: Mapping[str, pd.DataFrame], columns=None,
                       n: int = 2000, path=None):
    """
    A lower-triangular grid of pairwise scatters with every named sample
    overlaid, plus marginal histograms on the diagonal.

    The whole-picture view: use it to spot a pair whose dependence the model
    has missed, then zoom into that pair with `plot_copula_scatter`.
    """
    apply_theme()
    labels = list(samples)
    first = samples[labels[0]]
    cols = list(columns or first.columns)
    d = len(cols)
    fig, axes = plt.subplots(d, d, figsize=(2.3 * d, 2.3 * d))
    axes = np.atleast_2d(axes)

    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            for k, label in enumerate(labels):
                s = samples[label]
                sub = s.iloc[:n] if len(s) > n else s
                color = series_color(label.lower(), fallback_slot=k)
                if i == j:
                    ax.hist(sub[cols[i]], bins=45, density=True, histtype="step",
                            color=color, lw=1.6)
                else:
                    ax.scatter(sub[cols[j]], sub[cols[i]], s=3, alpha=0.25,
                               color=color, edgecolors="none")
            ax.tick_params(labelsize=6)
            if i == j:
                ax.set_yticks([])
            if i == d - 1:
                ax.set_xlabel(cols[j], fontsize=8)
            if j == 0:
                ax.set_ylabel(cols[i], fontsize=8)

    handles = [plt.Line2D([], [], color=series_color(name.lower(), k), lw=2.5, label=name)
               for k, name in enumerate(labels)]
    fig.legend(handles=handles, loc="upper right", ncols=len(labels),
               bbox_to_anchor=(0.99, 0.99))
    fig.suptitle("Pairwise structure", x=0.01, ha="left", fontsize=12,
                 fontweight="bold", color=INK)
    return _finish(fig, path)


def plot_dependence_heatmap(frames: Mapping[str, pd.DataFrame], path=None):
    """
    Kendall tau matrix for each named sample, on a shared diverging scale
    with a neutral midpoint, so zero dependence reads as no colour.

    Rank-based, so these are directly comparable even when the samples have
    different marginals.
    """
    apply_theme()
    labels = list(frames)
    mats = {k: kendall_tau_matrix(v) for k, v in frames.items()}
    cols = list(mats[labels[0]].columns)
    d = len(cols)
    fig, axes = plt.subplots(1, len(labels), figsize=(4.0 * len(labels), 4.0))
    axes = np.atleast_1d(axes)
    for ax, label in zip(axes, labels, strict=True):
        im = ax.imshow(mats[label].values, vmin=-1, vmax=1, cmap=DIVERGING_CMAP)
        _tick_labels(ax, cols)
        ax.set_title(label)
        ax.grid(False)
        for i in range(d):
            for j in range(d):
                v = mats[label].values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color=INK if abs(v) < 0.55 else "#ffffff")
    fig.colorbar(im, ax=axes, fraction=0.025, label="Kendall tau")
    fig.suptitle("Rank dependence", x=0.01, ha="left", fontsize=12,
                 fontweight="bold", color=INK)
    if path:
        fig.savefig(path, dpi=140, bbox_inches="tight")
    return fig


def plot_tau_error_heatmap(reference: pd.DataFrame,
                           models: Mapping[str, pd.DataFrame], path=None):
    """
    Each model's Kendall tau MINUS the reference data's, on a diverging
    scale centred at zero.

    More useful than absolute tau matrices when comparing models: a panel
    that is uniformly pale is a model that has the dependence right, and any
    coloured cell names the pair it got wrong and in which direction.
    """
    apply_theme()
    ref = kendall_tau_matrix(reference)
    cols = list(ref.columns)
    d = len(cols)
    labels = list(models)
    errs = {k: kendall_tau_matrix(v[cols]) - ref for k, v in models.items()}
    vmax = max(0.05, max(np.abs(e.values).max() for e in errs.values()))

    fig, axes = plt.subplots(1, len(labels), figsize=(4.0 * len(labels), 4.0))
    axes = np.atleast_1d(axes)
    for ax, label in zip(axes, labels, strict=True):
        e = errs[label]
        im = ax.imshow(e.values, vmin=-vmax, vmax=vmax, cmap=DIVERGING_CMAP)
        _tick_labels(ax, cols)
        mae = np.abs(e.values[~np.eye(d, dtype=bool)]).mean()
        ax.set_title(f"{label}\nmean abs error {mae:.3f}", fontsize=9.5)
        ax.grid(False)
        for i in range(d):
            for j in range(d):
                ax.text(j, i, f"{e.values[i, j]:+.2f}", ha="center", va="center",
                        fontsize=6.5, color=INK)
    fig.colorbar(im, ax=axes, fraction=0.025, label="tau error (model - data)")
    fig.suptitle("Where each model gets the dependence wrong", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    if path:
        fig.savefig(path, dpi=140, bbox_inches="tight")
    return fig


# =====================================================================
# tails
# =====================================================================
def plot_tail_dependence_curve(samples: Mapping[str, pd.DataFrame], col1: str,
                               col2: str, tail: str = "lower",
                               theoretical: float | None = None,
                               quantiles=None, path=None, ax=None):
    """
    ``lambda(q)`` against `q` for each named sample -- the single most
    informative copula diagnostic.

    Read it by the SHAPE, not the level. A curve sliding toward zero as `q`
    shrinks is asymptotic independence: the pair decouples in the extreme,
    which is what a Gaussian copula always predicts. A curve flattening onto
    a positive plateau is genuine tail dependence: the pair keeps moving
    together however far out you look. A model that reproduces the level at
    q = 0.25 but not the shape at q = 0.01 will misprice exactly the events
    you built it for.

    theoretical: if given, drawn as a dashed reference line -- the true
        asymptotic lambda, which the curves should approach as q -> 0.
    """
    apply_theme()
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
    if theoretical is not None:
        ax.axhline(theoretical, color=SERIES["theory"], ls="--", lw=1.4,
                   label=f"theoretical $\\lambda$ = {theoretical:.3f}")
    for k, (label, s) in enumerate(samples.items()):
        curve = tail_dependence_curve(s[col1].values, s[col2].values,
                                      quantiles=quantiles, tail=tail)
        ax.plot(curve["q"], curve["lambda"], marker="o", markersize=4,
                color=series_color(label.lower(), k), label=label)
    ax.set_xscale("log")
    ax.set_xlabel("tail probability q  (log scale, further into the tail to the left)")
    ax.set_ylabel(f"$\\lambda_{{{tail[0].upper()}}}(q)$")
    ax.set_ylim(0, 1)
    ax.set_title(f"{tail.capitalize()} tail dependence: {col1} vs {col2}")
    ax.legend(loc="best")
    return _finish(fig, path) if fig is not None else ax


def plot_tail_concentration(samples: Mapping[str, pd.DataFrame], col1: str,
                            col2: str, path=None):
    """
    Lower and upper tail dependence curves side by side.

    The pair of panels exposes ASYMMETRY, which a single panel cannot. An
    elliptical copula (Gaussian, Student-t) is forced to produce two
    identical panels; real data very often crashes together harder than it
    rallies together, and a visible left/right difference is direct evidence
    that an elliptical model is the wrong shape.
    """
    apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, tail in zip(axes, ["lower", "upper"], strict=True):
        plot_tail_dependence_curve(samples, col1, col2, tail=tail, ax=ax)
    axes[1].set_ylabel("")
    fig.suptitle(f"Tail concentration: {col1} vs {col2}", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    return _finish(fig, path)


def plot_exceedance_comparison(results: pd.DataFrame, path=None):
    """
    Grouped bars of joint exceedance probabilities by scenario and model.

    results: DataFrame indexed by scenario label, one column per model.

    Bars start at zero and are labelled directly, because these numbers are
    usually small and close together, and a reader comparing risk figures
    needs the values rather than bar lengths.
    """
    apply_theme()
    scenarios = list(results.index)
    models = list(results.columns)
    x = np.arange(len(scenarios))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(max(7, 1.9 * len(scenarios)), 4.6))
    for k, m in enumerate(models):
        pos = x + (k - (len(models) - 1) / 2) * width
        bars = ax.bar(pos, results[m].values, width * 0.92,
                      color=series_color(m.lower(), k), label=m)
        ax.bar_label(bars, fmt="%.4f", fontsize=7, padding=2, color=INK_MUTED)
    ax.set_xticks(x, scenarios, fontsize=8)
    ax.set_ylabel("probability")
    ax.set_title("Joint exceedance probability by scenario")
    ax.legend(loc="upper right")
    ax.set_axisbelow(True)
    return _finish(fig, path)


# =====================================================================
# marginals & conditionals
# =====================================================================
def plot_marginal_comparison(data: pd.DataFrame, models: Mapping[str, pd.DataFrame],
                             columns=None, path=None):
    """
    Per-column histogram of the data with each model's density overlaid,
    plus a Q-Q panel against the data's quantiles.

    In copula mode this is a CONTROL, not a test: the marginals are
    reproduced by construction, so these panels should overlap almost
    exactly. If they do not, something is wrong with the transform rather
    than with the dependence model -- which is precisely what makes it worth
    plotting.
    """
    apply_theme()
    cols = list(columns or data.columns)
    d = len(cols)
    fig, axes = plt.subplots(2, d, figsize=(3.3 * d, 6.4), squeeze=False)

    probs = np.linspace(0.01, 0.99, 99)
    for j, col in enumerate(cols):
        ax = axes[0, j]
        ax.hist(data[col], bins=60, density=True, color=SERIES["data"],
                alpha=0.5, label="data")
        for k, (label, s) in enumerate(models.items()):
            ax.hist(s[col], bins=60, density=True, histtype="step", lw=1.7,
                    color=series_color(label.lower(), k + 1), label=label)
        ax.set_title(col, fontsize=9.5)
        ax.set_yticks([])
        if j == 0:
            ax.legend(loc="upper right")
            ax.set_ylabel("density")

        ax = axes[1, j]
        dq = np.quantile(data[col], probs)
        lim = (dq.min(), dq.max())
        ax.plot(lim, lim, color=INK_MUTED, ls="--", lw=1.2)
        for k, (label, s) in enumerate(models.items()):
            ax.plot(dq, np.quantile(s[col], probs), lw=1.8,
                    color=series_color(label.lower(), k + 1), label=label)
        ax.set_xlabel("data quantile", fontsize=8)
        if j == 0:
            ax.set_ylabel("model quantile", fontsize=8)
    fig.suptitle("Marginal fit  (top: density, bottom: Q-Q)", x=0.01, ha="left",
                 fontsize=12, fontweight="bold", color=INK)
    return _finish(fig, path)


def plot_conditional_distribution(model, target: str, given_list: Sequence[Mapping],
                                  labels: Sequence[str] | None = None,
                                  tol: float = 0.35, n_mc: int = 60_000,
                                  bins: int = 60, path=None):
    """
    Weighted histograms of ``p(target | given)`` for several conditioning
    points on shared axes.

    This is the figure that makes tail dependence tangible. Condition on a
    calm value and on a crash value for the same driver, and the difference
    between the two curves -- how far the target's whole distribution shifts
    and fattens -- is the dependence, expressed in the units of the thing
    you actually care about rather than as a coefficient.

    given_list: conditioning dicts, e.g.
        ``[{"grain_A": 0.0}, {"grain_A": -3.0}]``. At most three, per the
        palette's simultaneous-series limit.
    labels: names for each; defaults to a readable rendering of the dict.

    The x-axis is clipped to the 0.1st-99.9th weighted percentile across all
    series, so a handful of extreme draws from a fat-tailed marginal cannot
    squeeze the informative part of the figure into a corner. A small amount
    of tail mass is therefore off-screen by design.
    """
    apply_theme()
    if len(given_list) > 3:
        raise ValueError("at most 3 conditioning points per figure")
    if labels is None:
        labels = [", ".join(f"{k} = {v:g}" for k, v in g.items()) for g in given_list]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    draws = []
    for k, (given, label) in enumerate(zip(given_list, labels, strict=True)):
        cd = model.conditional_distribution(target, given, tol=tol, n_mc=n_mc)
        v, w = cd["value"].values, cd["weight"].values
        draws.append((v, w))
        ax.hist(v, bins=bins, weights=w, density=True,
                histtype="step", lw=2.0, color=CATEGORICAL[k],
                label=f"{label}   (eff. n = {cd.attrs['effective_sample_size']:.0f})")
        ax.axvline(np.average(v, weights=w), color=CATEGORICAL[k], ls=":", lw=1.4)

    # Clip to a robust range across all series. Fat-tailed marginals put a few
    # draws far out, and letting them set the limits squeezes the part of the
    # figure that carries the information into a narrow strip.
    lo = min(_weighted_quantile(v, w, 0.001) for v, w in draws)
    hi = max(_weighted_quantile(v, w, 0.999) for v, w in draws)
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    ax.set_xlim(lo - pad, hi + pad)

    ax.set_xlabel(target)
    ax.set_ylabel("conditional density")
    ax.set_title(f"Conditional distribution of {target}")
    ax.legend(loc="upper left")
    return _finish(fig, path)


def plot_uncertainty(reports: Mapping[str, dict], path=None):
    """
    Posterior mean and 90% credible interval for each named Bayesian
    `uncertainty_report`, as a dot-and-whisker chart.

    Reach for this before quoting any single tail-dependence number from a
    few thousand rows. If the interval spans half the unit interval, the
    point estimate was never worth the decimal places it was printed with.
    """
    apply_theme()
    labels = list(reports)
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.5, 0.7 * len(labels) + 2.4))
    for i, label in enumerate(labels):
        r = reports[label]
        ax.plot([r["q05"], r["q95"]], [i, i], color=CATEGORICAL[0], lw=2.5,
                solid_capstyle="round", alpha=0.55)
        ax.plot([r["mean"]], [i], "o", color=CATEGORICAL[0], markersize=8,
                markeredgecolor="#ffffff", markeredgewidth=1.5)
        ax.annotate(f"{r['mean']:.3f}  [{r['q05']:.3f}, {r['q95']:.3f}]",
                    xy=(r["q95"], i), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK_MUTED)
    ax.set_yticks(y, labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("estimate  (dot: posterior mean, bar: 90% credible interval)")
    ax.set_title("Epistemic uncertainty")
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.22)
    return _finish(fig, path)


# =====================================================================
# benchmark summaries
# =====================================================================
def plot_benchmark_bars(results: pd.DataFrame, metrics: Sequence[str] | None = None,
                        lower_is_better: Mapping[str, bool] | None = None,
                        path=None):
    """
    Small multiples of benchmark metrics, one panel per metric, bars grouped
    by model.

    Small multiples rather than one combined chart, because these metrics
    live on completely different scales (a log-likelihood, an energy
    distance, a runtime in seconds). Putting them on shared axes would need
    a second y-axis, which is never the right answer -- one panel each keeps
    every bar honestly proportional to its own zero.

    results: tidy DataFrame with a ``model`` column and one column per metric.
    lower_is_better: per-metric flag, used only to annotate the panel titles
        so the direction of "good" is never ambiguous.
    """
    apply_theme()
    metrics = list(metrics or [c for c in results.columns
                               if c != "model" and pd.api.types.is_numeric_dtype(results[c])])
    lower_is_better = lower_is_better or {}
    n = len(metrics)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.5 * nrows),
                             squeeze=False)
    models = list(results["model"])

    for idx, metric in enumerate(metrics):
        ax = axes[idx // ncols][idx % ncols]
        vals = results[metric].values.astype(float)
        colors = [series_color(m.lower(), k) for k, m in enumerate(models)]
        bars = ax.barh(np.arange(len(models)), vals, color=colors, height=0.62)
        ax.bar_label(bars, fmt="%.4g", fontsize=7.5, padding=3, color=INK_MUTED)
        ax.set_yticks(np.arange(len(models)), models, fontsize=8)
        ax.invert_yaxis()
        direction = lower_is_better.get(metric)
        suffix = ("  (lower is better)" if direction is True
                  else "  (higher is better)" if direction is False else "")
        ax.set_title(f"{metric}{suffix}", fontsize=9.5)
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)
        ax.margins(x=0.20)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.suptitle("Benchmark results", x=0.01, ha="left", fontsize=13,
                 fontweight="bold", color=INK)
    return _finish(fig, path)


def plot_model_comparison(data: pd.DataFrame, models: Mapping[str, object],
                          col1: str, col2: str, n: int = 4000,
                          theoretical: float | None = None, path=None):
    """
    The headline three-panel comparison: copula-scale scatters of the data
    and each fitted model, then their lower-tail dependence curves together.

    models: mapping label -> a fitted model exposing `sample_uniform`
        (`NeuroCopula` or `VineCopula`). At most two, so the scatter row
        stays within the validated three-series limit alongside the data.
    """
    apply_theme()
    if len(models) > 2:
        raise ValueError("at most 2 models, so the figure stays within 3 series")

    from .marginals import pseudo_observations
    u_data = pd.DataFrame(pseudo_observations(data.values), columns=data.columns)
    frames = {"data": u_data}
    for label, m in models.items():
        frames[label] = m.sample_uniform(n)

    n_panels = len(frames)
    fig = plt.figure(figsize=(4.2 * n_panels, 8.4), layout="constrained")
    gs = fig.add_gridspec(2, n_panels, height_ratios=[1, 0.95])

    for k, (label, frame) in enumerate(frames.items()):
        ax = fig.add_subplot(gs[0, k])
        sub = frame.iloc[:n] if len(frame) > n else frame
        ax.scatter(sub[col1], sub[col2], s=4, alpha=0.3,
                   color=series_color(label.lower(), k), edgecolors="none")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(label)
        ax.set_xlabel(f"u({col1})")
        if k == 0:
            ax.set_ylabel(f"u({col2})")

    ax = fig.add_subplot(gs[1, :])
    plot_tail_dependence_curve(frames, col1, col2, tail="lower",
                               theoretical=theoretical, ax=ax)
    fig.suptitle(f"Model comparison: {col1} vs {col2}", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    if path:
        fig.savefig(path, dpi=140, bbox_inches="tight")
    return fig
