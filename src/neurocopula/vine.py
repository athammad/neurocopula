"""
A vine-copula baseline wearing the same API as `NeuroCopula`.

`pyvinecopulib` is the reference implementation of regular vine copulas and
is the model `neurocopula` has to justify itself against. Its native API
works on pseudo-observations and returns numpy arrays; `VineCopula` wraps it
so that `fit`, `sample`, `log_prob`, `copula_log_prob`, `tail_dependence`,
`joint_exceedance_probability` and friends behave identically to
`NeuroCopula`. The benchmark suite can then treat the two interchangeably,
and any difference in the reported numbers is a difference between the
MODELS rather than between two ways of calling them.

`pyvinecopulib` is an optional dependency::

    pip install neurocopula[vine]

Importing this module without it raises with that instruction.

How the two models differ, in one paragraph: a vine factorises a
d-dimensional copula into ``d(d-1)/2`` bivariate pair copulas arranged in a
tree, selecting a parametric family (Gaussian, Student-t, Clayton, Gumbel,
...) for each. That is interpretable, fast, and statistically efficient when
some family in the set is close to right. It also fixes the shape of the
dependence up front and needs a tree structure to be selected greedily. A
flow makes no family assumption and models all `d` dimensions jointly, at
the cost of interpretability and a much larger parameter count.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .marginals import EmpiricalCopulaTransform

try:
    import pyvinecopulib as pv
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "VineCopula needs pyvinecopulib, which is an optional dependency.\n"
        "Install it with:  pip install neurocopula[vine]   (or: pip install pyvinecopulib)"
    ) from exc

__all__ = ["VineCopula", "FAMILY_SETS"]


#: Named family sets, so benchmarks can vary a vine's flexibility from a
#: single Gaussian family up to the full library-default set including
#: nonparametric TLL pair copulas.
FAMILY_SETS: dict[str, list] = {
    "gaussian": [pv.BicopFamily.gaussian],
    "elliptical": [pv.BicopFamily.gaussian, pv.BicopFamily.student],
    "parametric": [
        pv.BicopFamily.indep, pv.BicopFamily.gaussian, pv.BicopFamily.student,
        pv.BicopFamily.clayton, pv.BicopFamily.gumbel, pv.BicopFamily.frank,
        pv.BicopFamily.joe,
    ],
    "all": [],  # empty means "every family pyvinecopulib knows", including tll
}


class VineCopula:
    """
    A regular vine copula baseline with the `NeuroCopula` interface.

    Marginals are handled exactly as in `NeuroCopula`'s copula mode -- the
    same `EmpiricalCopulaTransform` -- so the two models are compared on
    their dependence structure alone, with the marginal half held identical.
    That is deliberate: it is the only way a log-likelihood difference
    between them can be attributed to the copula.

    Parameters
    ----------
    family_set:
        A key of `FAMILY_SETS` (``"gaussian"``, ``"elliptical"``,
        ``"parametric"``, ``"all"``) or an explicit list of
        `pyvinecopulib.BicopFamily` values. Default ``"parametric"`` covers
        the standard families without the nonparametric TLL fits, which are
        slow on larger problems.
    selection_criterion:
        ``"bic"`` (default), ``"aic"``, or ``"mbicv"`` -- how each pair
        copula's family is chosen. BIC penalises parameters more heavily and
        gives simpler, better-generalising vines.
    trunc_lvl:
        Truncate the vine after this many trees, modelling higher-order
        conditional dependence as independent. None fits every tree.
    num_threads:
        Threads for fitting.
    seed:
        Seed used for simulation.
    marginal_density:
        As in `NeuroCopula`: ``"kde"`` enables `log_prob` in original units,
        ``"none"`` restricts you to `copula_log_prob`.
    """

    def __init__(
        self,
        family_set: str | Sequence = "parametric",
        selection_criterion: str = "bic",
        trunc_lvl: int | None = None,
        num_threads: int = 1,
        seed: int = 0,
        marginal_density: str = "kde",
    ) -> None:
        if isinstance(family_set, str):
            if family_set not in FAMILY_SETS:
                raise ValueError(
                    f"unknown family_set {family_set!r}; expected one of "
                    f"{sorted(FAMILY_SETS)} or an explicit list of BicopFamily values"
                )
            self.family_set_name = family_set
            self._families = FAMILY_SETS[family_set]
        else:
            self.family_set_name = "custom"
            self._families = list(family_set)

        self.selection_criterion = selection_criterion
        self.trunc_lvl = trunc_lvl
        self.num_threads = num_threads
        self.seed = seed
        self.marginal_density = marginal_density

        self.vine: pv.Vinecop | None = None
        self.columns: list[str] | None = None
        self.marginals_: EmpiricalCopulaTransform | None = None
        self.fit_report_: dict | None = None

    # -----------------------------------------------------------------
    def fit(self, df: pd.DataFrame, verbose: bool = False) -> VineCopula:
        """
        Fit the vine: map columns to pseudo-observations with the empirical
        CDF, then let `pyvinecopulib` select the tree structure and each
        pair copula's family and parameters.

        Returns: self.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("fit() expects a pandas DataFrame")
        data = df.values.astype(np.float64)
        if not np.isfinite(data).all():
            raise ValueError("data contains NaN or infinite values; clean it first")

        self.columns = list(df.columns)
        self.marginals_ = EmpiricalCopulaTransform(
            marginal_density=self.marginal_density
        ).fit(data)
        u = self.marginals_.to_uniform(data)

        kwargs = dict(
            family_set=self._families,
            selection_criterion=self.selection_criterion,
            num_threads=self.num_threads,
            show_trace=verbose,
        )
        if self.trunc_lvl is not None:
            kwargs["trunc_lvl"] = self.trunc_lvl
        controls = pv.FitControlsVinecop(**kwargs)
        self.vine = pv.Vinecop.from_data(u, controls=controls)

        self.fit_report_ = {
            "n_train": len(data),
            "n_features": data.shape[1],
            "family_set": self.family_set_name,
            "n_parameters": float(self.vine.npars),
            "loglik": float(self.vine.loglik(u)),
            "bic": float(self.vine.bic(u)),
            "families": self.pair_families(),
        }
        if verbose:
            print(self.vine)
        return self

    # -----------------------------------------------------------------
    def sample(self, n: int = 1000, **_) -> pd.DataFrame:
        """
        Draw `n` synthetic rows in ORIGINAL units: simulate copula-scale
        points from the vine, then push them through the empirical quantile
        functions.

        Accepts and ignores extra keyword arguments (such as `batch_size`)
        so it is drop-in interchangeable with `NeuroCopula.sample`.
        """
        u = self.sample_uniform(n).values
        return pd.DataFrame(self.marginals_.from_uniform(u), columns=self.columns)

    def sample_uniform(self, n: int = 1000, **_) -> pd.DataFrame:
        """Draw `n` rows on the copula scale, in (0, 1)^d."""
        self._check_fitted()
        rng = np.random.default_rng(self.seed)
        seeds = [int(s) for s in rng.integers(1, 2 ** 31 - 1, size=5)]
        return pd.DataFrame(self.vine.simulate(n, seeds=seeds), columns=self.columns)

    def pseudo_observations(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Map `df` to copula-scale pseudo-observations with the fitted ECDFs."""
        self._check_fitted()
        data = self.marginals_._sorted if df is None else df[self.columns].values
        return pd.DataFrame(self.marginals_.to_uniform(data), columns=self.columns)

    # -----------------------------------------------------------------
    def copula_log_prob(self, df: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Log copula density of each row. Directly comparable with
        `NeuroCopula.copula_log_prob`.
        """
        self._check_fitted()
        data = self._coerce(df)
        u = self.marginals_.to_uniform(data)
        return np.log(np.clip(self.vine.pdf(u), 1e-300, None))

    def log_prob(self, df: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Log-density in original units: the copula log-density plus the
        KDE-estimated marginal log-densities, using the same estimator
        `NeuroCopula` uses, so the two remain comparable.
        """
        self._check_fitted()
        data = self._coerce(df)
        return self.copula_log_prob(data) + self.marginals_.log_det_jacobian(data)

    def score(self, df, copula: bool = False) -> float:
        """Mean log-likelihood; higher is better."""
        lp = self.copula_log_prob(df) if copula else self.log_prob(df)
        return float(np.mean(lp))

    def aic(self, df, copula: bool = True) -> float:
        """AIC, ``2k - 2 * loglik``, with `k` the vine's parameter count."""
        self._check_fitted()
        lp = self.copula_log_prob(df) if copula else self.log_prob(df)
        return float(2 * self.vine.npars - 2 * lp.sum())

    def bic(self, df, copula: bool = True) -> float:
        """BIC, ``k * log(n) - 2 * loglik``."""
        self._check_fitted()
        lp = self.copula_log_prob(df) if copula else self.log_prob(df)
        return float(self.vine.npars * np.log(len(lp)) - 2 * lp.sum())

    # -----------------------------------------------------------------
    def tail_dependence(self, col1: str, col2: str, q: float = 0.05,
                        tail: str = "lower", n_mc: int = 50_000) -> float:
        """
        Tail dependence between two columns, estimated the same way
        `NeuroCopula.tail_dependence` estimates it -- from the model's own
        Monte Carlo sample rather than from the fitted pair copula's
        closed-form coefficient.

        That choice is deliberate. Using the analytic coefficient would
        measure a different thing (an asymptotic parameter of one pair
        copula, ignoring the rest of the vine) and would give the vine an
        advantage that has nothing to do with fit quality. Estimating both
        models the same way keeps the comparison honest.
        """
        from .metrics import tail_dependence as _td
        self._check_fitted()
        s = self.sample_uniform(n_mc)
        return _td(s[col1].values, s[col2].values, q=q, tail=tail)

    def tail_dependence_curve(self, col1: str, col2: str, quantiles=None,
                              tail: str = "lower", n_mc: int = 100_000) -> pd.DataFrame:
        """`tail_dependence` across a range of `q`, from one shared sample."""
        from .metrics import tail_dependence_curve as _tdc
        self._check_fitted()
        s = self.sample_uniform(n_mc)
        return _tdc(s[col1].values, s[col2].values, quantiles=quantiles, tail=tail)

    def joint_exceedance_probability(
        self, conditions: Mapping[str, tuple[str, float]], n_mc: int = 50_000
    ) -> float:
        """Monte Carlo P(all conditions hold), matching `NeuroCopula`'s method."""
        from .core import NeuroCopula
        self._check_fitted()
        samp = self.sample(n_mc)
        mask = np.ones(len(samp), dtype=bool)
        for col, (op, val) in conditions.items():
            mask &= NeuroCopula._apply_op(samp[col].values, op, val)
        return float(mask.mean())

    # -----------------------------------------------------------------
    def pair_families(self) -> list[list[str]]:
        """
        The family chosen for each pair copula, tree by tree -- the vine's
        interpretable output, and the thing a flow cannot give you. Tree 0
        holds the unconditional pairs; later trees hold conditional ones.
        """
        self._check_fitted()
        return [[str(pc.family).split(".")[-1] for pc in tree]
                for tree in self.vine.pair_copulas]

    def taus(self) -> list[list[float]]:
        """Kendall's tau of each fitted pair copula, tree by tree."""
        self._check_fitted()
        return self.vine.taus

    def summary(self) -> str:
        """`pyvinecopulib`'s own printable description of the fitted vine."""
        self._check_fitted()
        return str(self.vine)

    # -----------------------------------------------------------------
    def _coerce(self, df) -> np.ndarray:
        if isinstance(df, pd.DataFrame):
            return df[self.columns].values.astype(np.float64)
        return np.atleast_2d(np.asarray(df, dtype=np.float64))

    def _check_fitted(self) -> None:
        if self.vine is None:
            raise RuntimeError("this VineCopula is not fitted yet; call fit(df) first.")

    def __repr__(self) -> str:
        if self.vine is None:
            return f"VineCopula(family_set={self.family_set_name!r}, unfitted)"
        return (f"VineCopula(family_set={self.family_set_name!r}, "
                f"d={self.fit_report_['n_features']}, "
                f"npars={self.fit_report_['n_parameters']:.0f})")
