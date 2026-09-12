"""
Semi-parametric marginals: empirical body, optional atom, optional tail model.

The base `EmpiricalCopulaTransform` assumes every variable is continuous and
that the observed range is all you need. Plenty of real variables violate both
-- daily rainfall, insurance claims, river discharge, trade volumes -- in two
ways that quietly corrupt a copula fit if ignored.

**Some variables have a point mass at zero.** Daily rainfall is exactly 0 on a
dry day, and in a monsoon climate that is most days; an insurance policy
usually claims nothing at all. A plain empirical CDF maps
every one of those days to the same pseudo-observation, so after the probit
they all land on one spike. The flow then spends its capacity modelling a
delta function, and the dependence structure it reports for dry days is an
artifact. The fix is the randomized probability integral transform: a dry day
is mapped to a *uniform draw* below the dry probability, which spreads the
mass over the interval it actually occupies. This makes the transform
stochastic by design -- the same dry day gets a different pseudo-observation
each call -- and that is the correct behaviour for a discrete-continuous
mixture, not a bug.

**The empirical CDF cannot extrapolate.** It is bounded by the largest value
ever observed, so it can never assign probability to a wetter day than any
in the record. For compound-extreme work that is precisely the wrong
limitation. A generalised Pareto tail fitted above a high quantile replaces
the upper end of the ECDF, so the model can extrapolate past the sample
maximum on a principled basis (extreme value theory) rather than by
flattening out.

Both behaviours are optional and off by default, so a variable with neither
problem (temperature, say) still gets the plain empirical treatment.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import genpareto, norm

__all__ = ["SemiParametricMarginal", "GroupedMarginalTransform"]

_EPS = 1e-9


class SemiParametricMarginal:
    """
    A one-dimensional marginal: empirical body, optional zero inflation,
    optional generalised Pareto upper tail.

    Parameters
    ----------
    zero_inflated:
        Treat values at or below `zero_threshold` as an atom at zero and use
        the randomized PIT for them. Turn this on for a variable with an atom at zero.
    zero_threshold:
        Values at or below this count as dry. A small positive number (0.1
        mm is the usual choice for daily rainfall) is better than exactly 0,
        because reanalysis and gauge records both contain meaningless trace
        amounts that are not really wet days.
    tail:
        ``None`` for a pure empirical CDF, or ``"gpd"`` to splice a
        generalised Pareto distribution above `tail_quantile`.
    tail_quantile:
        Quantile of the (positive) data above which the GPD takes over.
        0.95 is a common compromise: high enough for the asymptotic
        justification to be reasonable, low enough to leave enough
        exceedances to fit.
    seed:
        Seed for the randomized PIT draws, so a fit is reproducible.
    """

    def __init__(
        self,
        zero_inflated: bool = False,
        zero_threshold: float = 0.0,
        tail: str | None = None,
        tail_quantile: float = 0.95,
        seed: int = 0,
    ) -> None:
        if tail not in (None, "gpd"):
            raise ValueError("tail must be None or 'gpd'")
        if not 0.5 < tail_quantile < 1.0:
            raise ValueError("tail_quantile must be in (0.5, 1)")
        self.zero_inflated = zero_inflated
        self.zero_threshold = zero_threshold
        self.tail = tail
        self.tail_quantile = tail_quantile
        self.seed = seed

        self.p_zero_: float = 0.0
        self.sorted_: np.ndarray | None = None   # sorted wet (or all) values
        self.grid_: np.ndarray | None = None     # matching ECDF levels
        self.tail_threshold_: float | None = None
        self.tail_params_: tuple[float, float] | None = None  # (shape, scale)
        self.tail_prob_: float | None = None     # P(X <= threshold)
        self._rng = np.random.default_rng(seed)

    # -----------------------------------------------------------------
    def fit(self, x: np.ndarray) -> SemiParametricMarginal:
        x = np.asarray(x, dtype=np.float64).ravel()
        x = x[np.isfinite(x)]
        if x.size == 0:
            raise ValueError("cannot fit a marginal to an empty sample")

        if self.zero_inflated:
            wet_mask = x > self.zero_threshold
            self.p_zero_ = float(1.0 - wet_mask.mean())
            wet = x[wet_mask]
            if wet.size < 10:
                warnings.warn(
                    f"only {wet.size} wet values above the threshold; the positive "
                    "part of this marginal is essentially unconstrained.",
                    RuntimeWarning, stacklevel=2,
                )
        else:
            self.p_zero_ = 0.0
            wet = x

        if wet.size == 0:
            # Degenerate but possible (a permanently dry cell). Keep a
            # single placeholder so the transform stays total.
            wet = np.array([self.zero_threshold])

        n = wet.size
        self.sorted_ = np.sort(wet)
        self.grid_ = np.arange(1, n + 1) / (n + 1)

        if self.tail == "gpd":
            self._fit_tail(self.sorted_)
        return self

    def _fit_tail(self, wet_sorted: np.ndarray) -> None:
        """Fit a GPD to exceedances above `tail_quantile` of the wet values."""
        thr = float(np.quantile(wet_sorted, self.tail_quantile))
        exceed = wet_sorted[wet_sorted > thr] - thr
        if exceed.size < 20:
            warnings.warn(
                f"only {exceed.size} exceedances above the {self.tail_quantile:.0%} "
                "quantile; falling back to the empirical tail. Lower tail_quantile "
                "or supply more data if extrapolation matters.",
                RuntimeWarning, stacklevel=3,
            )
            self.tail = None
            return
        # floc=0 because the GPD is fitted to exceedances, which start at 0.
        shape, _, scale = genpareto.fit(exceed, floc=0.0)
        self.tail_threshold_ = thr
        self.tail_params_ = (float(shape), float(scale))
        # Probability of being at or below the threshold, on the FULL scale
        # (including the dry atom), so the splice is continuous.
        self.tail_prob_ = self.p_zero_ + (1 - self.p_zero_) * self.tail_quantile

    # -----------------------------------------------------------------
    def to_uniform(self, x: np.ndarray, randomize: bool = True) -> np.ndarray:
        """
        Map values to pseudo-observations in (0, 1).

        randomize:
            With zero inflation, dry days are spread uniformly over
            ``(0, p_zero)``. Set False to place them all at ``p_zero / 2``
            instead -- deterministic, which is convenient for reproducible
            diagnostics, but it reintroduces the spike, so never use it for
            fitting.
        """
        self._check_fitted()
        x = np.asarray(x, dtype=np.float64).ravel()
        u = np.empty_like(x)

        wet = x > self.zero_threshold if self.zero_inflated else np.ones_like(x, bool)
        dry = ~wet

        if dry.any():
            if randomize:
                u[dry] = self._rng.uniform(_EPS, self.p_zero_, size=int(dry.sum()))
            else:
                u[dry] = self.p_zero_ / 2.0

        if wet.any():
            xw = x[wet]
            # Empirical body, rescaled to sit above the dry atom.
            uw = self.p_zero_ + (1 - self.p_zero_) * np.interp(
                xw, self.sorted_, self.grid_
            )
            if self.tail == "gpd":
                over = xw > self.tail_threshold_
                if over.any():
                    shape, scale = self.tail_params_
                    cdf = genpareto.cdf(xw[over] - self.tail_threshold_, shape, scale=scale)
                    uw[over] = self.tail_prob_ + (1 - self.tail_prob_) * cdf
            u[wet] = uw

        return np.clip(u, _EPS, 1 - _EPS)

    def from_uniform(self, u: np.ndarray) -> np.ndarray:
        """Map pseudo-observations back to physical units."""
        self._check_fitted()
        u = np.asarray(u, dtype=np.float64).ravel()
        x = np.empty_like(u)

        dry = u <= self.p_zero_
        x[dry] = 0.0 if self.zero_inflated else self.sorted_[0]

        wet = ~dry
        if wet.any():
            uw = u[wet]
            # Rescale out of the dry atom, back onto the wet ECDF.
            uu = (uw - self.p_zero_) / max(1 - self.p_zero_, _EPS)
            xw = np.interp(uu, self.grid_, self.sorted_)
            if self.tail == "gpd":
                over = uw > self.tail_prob_
                if over.any():
                    shape, scale = self.tail_params_
                    q = (uw[over] - self.tail_prob_) / max(1 - self.tail_prob_, _EPS)
                    xw[over] = self.tail_threshold_ + genpareto.ppf(
                        np.clip(q, 0, 1 - _EPS), shape, scale=scale
                    )
            x[wet] = xw
        return x

    # -----------------------------------------------------------------
    @property
    def supports_extrapolation(self) -> bool:
        """True when a GPD tail was actually fitted (not silently dropped)."""
        return self.tail == "gpd" and self.tail_params_ is not None

    def _check_fitted(self) -> None:
        if self.sorted_ is None:
            raise RuntimeError("SemiParametricMarginal is not fitted yet; call fit() first.")

    def __repr__(self) -> str:
        bits = [f"p_zero={self.p_zero_:.3f}"] if self.zero_inflated else []
        if self.supports_extrapolation:
            bits.append(f"gpd(shape={self.tail_params_[0]:.3f}, "
                        f"scale={self.tail_params_[1]:.3f})")
        return f"SemiParametricMarginal({', '.join(bits) or 'empirical'})"


class GroupedMarginalTransform:
    """
    A multivariate marginal transform built from per-variable
    `SemiParametricMarginal` objects, optionally fitted separately per group.

    Grouping is the important option. With ``scope="group"`` a separate set
    of marginals is fitted for each group label -- normally a grid cell --
    so the flow never has to represent the fact that Borneo is wetter than
    Myanmar. Every cell's data arrives on the flow's desk already uniform,
    and the flow is left to model *only* how the variables depend on each
    other and how that dependence shifts with context. That is what makes
    this a conditional copula rather than a conditional density estimator.

    Cells with fewer than `min_group_size` rows fall back to marginals
    pooled across all data, so a short record degrades gracefully instead
    of producing a wild per-cell ECDF.

    Parameters
    ----------
    columns:
        Variable names, in the order the flow will see them.
    specs:
        Optional ``{column: kwargs}`` passed to `SemiParametricMarginal`, e.g.
        ``{"precip": dict(zero_inflated=True, zero_threshold=0.1, tail="gpd")}``.
        Columns not mentioned get plain empirical marginals.
    scope:
        ``"group"`` to fit per group label, ``"global"`` to pool everything.
    min_group_size:
        Groups smaller than this use the pooled marginals instead.
    seed:
        Seed for the randomized PIT.
    """

    def __init__(
        self,
        columns: list[str],
        specs: dict[str, dict] | None = None,
        scope: str = "group",
        min_group_size: int = 200,
        seed: int = 0,
    ) -> None:
        if scope not in ("group", "global"):
            raise ValueError("scope must be 'group' or 'global'")
        self.columns = list(columns)
        self.specs = dict(specs or {})
        unknown = set(self.specs) - set(self.columns)
        if unknown:
            raise KeyError(f"specs given for unknown columns: {sorted(unknown)}")
        self.scope = scope
        self.min_group_size = min_group_size
        self.seed = seed

        self.pooled_: dict[str, SemiParametricMarginal] | None = None
        self.groups_: dict[object, dict[str, SemiParametricMarginal]] = {}
        self.group_sizes_: dict[object, int] = {}

    # -----------------------------------------------------------------
    def _make_set(self, offset: int) -> dict[str, SemiParametricMarginal]:
        return {
            c: SemiParametricMarginal(seed=self.seed + offset + i, **self.specs.get(c, {}))
            for i, c in enumerate(self.columns)
        }

    def fit(self, data: np.ndarray, groups: np.ndarray | None = None
            ) -> GroupedMarginalTransform:
        """
        data: array (n, d) in the order of `columns`.
        groups: array (n,) of group labels; required when ``scope="group"``.
        """
        data = np.asarray(data, dtype=np.float64)
        if data.shape[1] != len(self.columns):
            raise ValueError(f"expected {len(self.columns)} columns, got {data.shape[1]}")

        self.pooled_ = self._make_set(0)
        for i, c in enumerate(self.columns):
            self.pooled_[c].fit(data[:, i])

        if self.scope == "group":
            if groups is None:
                raise ValueError("scope='group' requires group labels")
            groups = np.asarray(groups)
            for off, g in enumerate(np.unique(groups)):
                mask = groups == g
                self.group_sizes_[g] = int(mask.sum())
                if mask.sum() < self.min_group_size:
                    continue  # falls back to pooled at transform time
                mset = self._make_set(1000 * (off + 1))
                for i, c in enumerate(self.columns):
                    mset[c].fit(data[mask, i])
                self.groups_[g] = mset
        return self

    def _set_for(self, g) -> dict[str, SemiParametricMarginal]:
        return self.groups_.get(g, self.pooled_) if self.scope == "group" else self.pooled_

    # -----------------------------------------------------------------
    def to_uniform(self, data: np.ndarray, groups: np.ndarray | None = None,
                   randomize: bool = True) -> np.ndarray:
        """Physical units -> pseudo-observations in (0, 1)^d."""
        self._check_fitted()
        data = np.atleast_2d(np.asarray(data, dtype=np.float64))
        out = np.empty_like(data)
        for g, idx in self._group_slices(len(data), groups):
            mset = self._set_for(g)
            for i, c in enumerate(self.columns):
                out[idx, i] = mset[c].to_uniform(data[idx, i], randomize=randomize)
        return out

    def from_uniform(self, u: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
        """Pseudo-observations -> physical units."""
        self._check_fitted()
        u = np.atleast_2d(np.asarray(u, dtype=np.float64))
        out = np.empty_like(u)
        for g, idx in self._group_slices(len(u), groups):
            mset = self._set_for(g)
            for i, c in enumerate(self.columns):
                out[idx, i] = mset[c].from_uniform(u[idx, i])
        return out

    def forward(self, data: np.ndarray, groups: np.ndarray | None = None,
                randomize: bool = True) -> np.ndarray:
        """Physical units -> the flow's Gaussianized training space."""
        return norm.ppf(self.to_uniform(data, groups, randomize=randomize))

    def inverse(self, z: np.ndarray, groups: np.ndarray | None = None) -> np.ndarray:
        """The flow's training space -> physical units."""
        return self.from_uniform(norm.cdf(np.atleast_2d(np.asarray(z))), groups)

    # -----------------------------------------------------------------
    def _group_slices(self, n: int, groups):
        """Yield (label, index array) pairs covering all n rows."""
        if self.scope == "global" or groups is None:
            yield None, np.arange(n)
            return
        groups = np.asarray(groups)
        if len(groups) != n:
            raise ValueError(f"got {len(groups)} group labels for {n} rows")
        for g in np.unique(groups):
            yield g, np.flatnonzero(groups == g)

    @property
    def n_fitted_groups(self) -> int:
        return len(self.groups_)

    def coverage_report(self) -> dict:
        """
        How many groups got their own marginals and how many fell back to
        the pooled ones. Worth checking after a fit: heavy fallback means
        `min_group_size` is too high for the record length.
        """
        total = len(self.group_sizes_)
        own = len(self.groups_)
        return {
            "n_groups": total,
            "fitted_own_marginals": own,
            "fell_back_to_pooled": total - own,
            "smallest_group": min(self.group_sizes_.values()) if self.group_sizes_ else None,
        }

    def _check_fitted(self) -> None:
        if self.pooled_ is None:
            raise RuntimeError("GroupedMarginalTransform is not fitted yet; call fit().")

    def __repr__(self) -> str:
        return (f"GroupedMarginalTransform(d={len(self.columns)}, scope={self.scope!r}, "
                f"groups={self.n_fitted_groups})")
