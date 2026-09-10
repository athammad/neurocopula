"""
A reproducible benchmark comparing `NeuroCopula` against vine copulas.

The design decisions that make the numbers mean something:

**Held-out scoring.** Every model is fitted on a train split and scored on a
test split it never saw. `NeuroCopula.fit` fits its marginal transform on
everything it is given, so the split happens HERE, before either model is
constructed -- otherwise the test rows would have leaked into the empirical
CDF and the comparison would flatter both models equally but meaninglessly.

**Identical marginals.** Both models use the same
`EmpiricalCopulaTransform`, so the marginal half of the factorisation is held
fixed and any log-likelihood difference is attributable to the copula.

**Copula-scale log-likelihood as the headline.** `copula_log_prob` is exact
for both models and needs no density estimate, unlike a log-likelihood in
original units, which would depend on a shared KDE and measure the KDE as
much as the model.

**Tail metrics estimated the same way for both.** Both models are asked for
Monte Carlo samples and the coefficient is computed from those samples. Using
the vine's analytic pair-copula coefficient instead would compare two
different quantities.

**Ground truth where it exists.** On synthetic data from
`neurocopula.datasets` the true tail dependence and Kendall's tau are known
in closed form, so the benchmark reports ERROR AGAINST TRUTH, not just
agreement with the sample.

Run the whole thing with::

    from neurocopula.benchmark import run_benchmark, STANDARD_SUITE
    results = run_benchmark(STANDARD_SUITE)
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import datasets as ds
from .core import NeuroCopula
from .metrics import (
    energy_distance,
    kendall_tau_matrix,
    mmd_rbf,
    tail_dependence,
)

__all__ = ["BenchmarkCase", "STANDARD_SUITE", "run_case", "run_benchmark",
           "summarize", "tail_dependence_table", "paired_comparison"]


@dataclass
class BenchmarkCase:
    """
    One benchmark problem: a data generator plus whatever ground truth is
    known about it.

    name: label used in the results table.
    generator: callable ``(n, seed) -> DataFrame``.
    truth_tail: ``{"lower": ..., "upper": ...}`` if known, else None.
    truth_tau: true Kendall's tau of the first column pair, if known.
    description: one line on what makes this case interesting -- carried
        into the results so a table is readable without the source.
    n: rows generated.
    """

    name: str
    generator: Callable[..., pd.DataFrame]
    truth_tail: dict[str, float] | None = None
    truth_tau: float | None = None
    description: str = ""
    n: int = 4000
    flow_kwargs: dict = field(default_factory=dict)


def _case(name, gen, family=None, params=None, description="", n=4000, **flow_kwargs):
    truth_tail = ds.theoretical_tail_dependence(family, **params) if family else None
    truth_tau = ds.theoretical_kendall_tau(family, **params) if family else None
    return BenchmarkCase(
        name=name, generator=gen, truth_tail=truth_tail, truth_tau=truth_tau,
        description=description, n=n, flow_kwargs=flow_kwargs,
    )


#: The default suite. It is deliberately not stacked in the flow's favour:
#: the first three cases are exactly the shapes a vine's parametric families
#: were designed for, and a flow should be expected to lose or draw on them.
#: The last two are where a flow has something to offer.
STANDARD_SUITE: list[BenchmarkCase] = [
    _case("gaussian_rho0.7",
          lambda n, seed: ds.make_gaussian_copula(n=n, rho=0.7, d=2, seed=seed),
          family="gaussian", params={"rho": 0.7},
          description="No tail dependence. The vine has the exact family available."),
    _case("clayton_theta2",
          lambda n, seed: ds.make_clayton(n=n, theta=2.0, seed=seed),
          family="clayton", params={"theta": 2.0},
          description="Strong lower-tail dependence, zero upper. Exact family in the vine's set."),
    _case("gumbel_theta2",
          lambda n, seed: ds.make_gumbel(n=n, theta=2.0, seed=seed),
          family="gumbel", params={"theta": 2.0},
          description="Upper-tail dependence, zero lower. Exact family in the vine's set."),
    _case("t_copula_df4",
          lambda n, seed: ds.make_t_copula(n=n, rho=0.7, dof=4, d=2, seed=seed),
          family="t", params={"rho": 0.7, "df": 4},
          description="Symmetric tail dependence in both tails. Exact family in the vine's set."),
    _case("mixed_regime",
          lambda n, seed: ds.make_mixed_regime(n=n, seed=seed),
          description="Two-regime mixture: in NO standard parametric family. "
                      "The case a flow should win."),
    _case("asset_panel_5d",
          lambda n, seed: ds.make_asset_panel(n=n, seed=seed),
          description="5-dimensional block-structured t copula. An elliptical structure "
                      "in 5 dimensions, where the vine has both the right family and a "
                      "large statistical-efficiency advantage.",
          n=3000, sampling_mode="coupling"),
    _case("mixed_regime_5d",
          lambda n, seed: ds.make_mixed_regime(n=n, d=5, seed=seed),
          description="5-dimensional two-regime mixture: high dimension AND no matching "
                      "parametric family. Paired with asset_panel_5d, it separates the "
                      "effect of dimension from the effect of family mismatch.",
          n=3000, sampling_mode="coupling"),
]


def _split(df: pd.DataFrame, test_frac: float, seed: int):
    """Random train/test split, done BEFORE any model sees the data."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_test = int(len(df) * test_frac)
    return (df.iloc[idx[n_test:]].reset_index(drop=True),
            df.iloc[idx[:n_test]].reset_index(drop=True))


def run_case(
    case: BenchmarkCase,
    seed: int = 0,
    test_frac: float = 0.25,
    n_mc: int = 100_000,
    q_tail: float = 0.02,
    epochs: int = 600,
    vine_family_sets: tuple[str, ...] = ("elliptical", "parametric"),
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Run one `BenchmarkCase` and return a tidy DataFrame, one row per model.

    Reported per model:

    - ``test_copula_loglik`` -- mean held-out copula log-likelihood. The
      headline number; higher is better.
    - ``lambda_L`` / ``lambda_U`` and, where truth is known, their absolute
      errors at `q_tail`.
    - ``tau_mae`` -- mean absolute Kendall tau error against the TEST data,
      averaged over all column pairs.
    - ``energy_distance`` / ``mmd`` -- two-sample distances between model
      samples and the test data on the copula scale; lower is better.
    - ``fit_seconds`` / ``sample_seconds`` and ``n_parameters``.

    q_tail: the tail probability at which lambda is evaluated. Theory is an
        asymptotic (q -> 0) statement while any estimate needs a positive q,
        so even a perfect model shows some error here. It is held fixed
        across models, so the numbers are comparable to each other.
    """
    from .marginals import pseudo_observations

    df = case.generator(case.n, seed)
    train, test = _split(df, test_frac, seed)
    cols = list(df.columns)
    c1, c2 = cols[0], cols[1]
    u_test = pd.DataFrame(pseudo_observations(test.values), columns=cols)
    tau_test = kendall_tau_matrix(test)
    d = len(cols)
    offdiag = ~np.eye(d, dtype=bool)

    rows = []

    def evaluate(label, model, fit_seconds):
        t0 = time.perf_counter()
        u_model = model.sample_uniform(n_mc)
        sample_seconds = time.perf_counter() - t0

        lam_l = tail_dependence(u_model[c1], u_model[c2], q=q_tail, tail="lower")
        lam_u = tail_dependence(u_model[c1], u_model[c2], q=q_tail, tail="upper")
        tau_model = kendall_tau_matrix(u_model)
        row = {
            "case": case.name,
            "model": label,
            "test_copula_loglik": model.score(test, copula=True),
            "lambda_L": lam_l,
            "lambda_U": lam_u,
            "tau_mae": float(np.abs((tau_model - tau_test).values[offdiag]).mean()),
            "energy_distance": energy_distance(u_model, u_test, seed=seed),
            "mmd": mmd_rbf(u_model, u_test, seed=seed),
            "fit_seconds": fit_seconds,
            "sample_seconds": sample_seconds,
            "n_parameters": _n_params(model),
        }
        if case.truth_tail is not None:
            row["lambda_L_err"] = abs(lam_l - case.truth_tail["lower"])
            row["lambda_U_err"] = abs(lam_u - case.truth_tail["upper"])
        if case.truth_tau is not None:
            row["tau_err_vs_truth"] = abs(tau_model.loc[c1, c2] - case.truth_tau)
        rows.append(row)

    # --- the flow -----------------------------------------------------
    flow_kwargs = dict(transforms=4, hidden_features=(64, 64), copula_mode=True,
                       marginal_density="none", seed=seed)
    flow_kwargs.update(case.flow_kwargs)
    t0 = time.perf_counter()
    flow = NeuroCopula(**flow_kwargs).fit(
        train, epochs=epochs, patience=50, split_method="random", verbose=verbose
    )
    evaluate("NeuroCopula", flow, time.perf_counter() - t0)

    # --- the vines ----------------------------------------------------
    try:
        from .vine import VineCopula
    except ImportError:
        warnings.warn(
            "pyvinecopulib is not installed, so the benchmark ran the flow only. "
            "Install it with: pip install neurocopula[vine]",
            RuntimeWarning, stacklevel=2,
        )
        return pd.DataFrame(rows)

    for fs in vine_family_sets:
        t0 = time.perf_counter()
        vine = VineCopula(family_set=fs, marginal_density="none", seed=seed).fit(train)
        evaluate(f"Vine ({fs})", vine, time.perf_counter() - t0)

    return pd.DataFrame(rows)


def _n_params(model) -> float:
    if isinstance(model, NeuroCopula):
        return float(sum(p.numel() for p in model.flow.parameters()))
    return float(model.vine.npars)


def run_benchmark(
    cases: list[BenchmarkCase] | None = None,
    seeds: tuple[int, ...] = (0,),
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Run every case at every seed and concatenate the results.

    seeds: repeating over several seeds is strongly recommended before
        drawing any conclusion. A single fit of a flow varies noticeably
        with initialisation, and a difference smaller than that spread is
        not a finding. Use `summarize` to collapse seeds into a mean and a
        standard deviation.

    Returns: tidy DataFrame with ``case``, ``model``, ``seed``, and one
        column per metric.
    """
    cases = cases if cases is not None else STANDARD_SUITE
    frames = []
    for case in cases:
        for seed in seeds:
            if verbose:
                print(f"[benchmark] {case.name}  seed={seed} ...", flush=True)
            out = run_case(case, seed=seed, **kwargs)
            out["seed"] = seed
            frames.append(out)
    return pd.concat(frames, ignore_index=True)


def summarize(results: pd.DataFrame, metrics: list[str] | None = None) -> pd.DataFrame:
    """
    Collapse a multi-seed benchmark into mean and standard deviation per
    case and model.

    Read the standard deviation first. Where it overlaps between two models,
    they are tied on that metric however different the means look -- that is
    the whole reason to run several seeds.
    """
    metrics = metrics or [
        "test_copula_loglik", "lambda_L", "lambda_U", "tau_mae",
        "energy_distance", "mmd", "fit_seconds",
    ]
    metrics = [m for m in metrics if m in results.columns]
    agg = results.groupby(["case", "model"])[metrics].agg(["mean", "std"])
    return agg.round(4)


def paired_comparison(
    results: pd.DataFrame,
    baseline: str | None = None,
    challenger: str = "NeuroCopula",
    metric: str = "test_copula_loglik",
) -> pd.DataFrame:
    """
    Compare two models seed by seed, on the same generated dataset each time.

    This is the right way to read the benchmark, and it is worth being precise
    about why. The standard deviation across seeds mixes together two
    completely different things: how much the MODELS vary, and how much the
    randomly generated DATASETS vary. The second term is usually the larger,
    which makes the unpaired standard deviation look alarmingly wide and can
    hide a difference that is in fact perfectly consistent.

    Pairing removes the dataset term. Both models saw the same rows at each
    seed, so their difference isolates the models. A difference that is small
    but lands the same way on every seed is real; one that is large but flips
    sign between seeds is noise.

    baseline: model to compare against. None (default) uses the best-scoring
        other model per case, which is the honest comparison -- it never lets
        the challenger win by being measured against a weak baseline.
    metric: column to compare. Higher is assumed to be better; for a
        lower-is-better metric read the sign in reverse.

    Returns: DataFrame indexed by case with the mean difference, the per-seed
        differences, and how many seeds the challenger won.
    """
    pivot = results.pivot_table(index=["case", "seed"], columns="model", values=metric)
    if challenger not in pivot.columns:
        raise KeyError(f"{challenger!r} is not among the benchmarked models: "
                       f"{list(pivot.columns)}")

    rows = []
    for case in pivot.index.get_level_values("case").unique():
        sub = pivot.loc[case]
        others = [c for c in sub.columns if c != challenger]
        if not others:
            continue
        ref = sub[baseline] if baseline else sub[others].max(axis=1)
        diff = sub[challenger] - ref
        rows.append({
            "case": case,
            "baseline": baseline or "best other model",
            "mean_diff": float(diff.mean()),
            "min_diff": float(diff.min()),
            "max_diff": float(diff.max()),
            "seeds_won": int((diff > 0).sum()),
            "n_seeds": int(len(diff)),
            "consistent": bool((diff > 0).all() or (diff < 0).all()),
        })
    return pd.DataFrame(rows).set_index("case").round(4)


def tail_dependence_table(results: pd.DataFrame,
                          cases: list[BenchmarkCase] | None = None) -> pd.DataFrame:
    """
    A focused view: estimated lambda_L and lambda_U per case and model,
    beside the theoretical values where they are known.

    The table to read when the question is specifically "does this model get
    the tails right", which is usually the question that decided you needed
    a copula in the first place.
    """
    cases = cases if cases is not None else STANDARD_SUITE
    truth = {c.name: c.truth_tail for c in cases}
    out = (results.groupby(["case", "model"])[["lambda_L", "lambda_U"]]
           .mean().round(3).reset_index())
    out["lambda_L_true"] = out["case"].map(
        lambda c: truth.get(c, {}).get("lower") if truth.get(c) else np.nan)
    out["lambda_U_true"] = out["case"].map(
        lambda c: truth.get(c, {}).get("upper") if truth.get(c) else np.nan)
    return out
