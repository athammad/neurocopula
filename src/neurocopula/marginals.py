"""
Marginal transforms.

A copula factorises a joint density into marginals and a dependence
structure::

    p(x_1, ..., x_d) = c(F_1(x_1), ..., F_d(x_d)) * prod_j f_j(x_j)

`neurocopula` learns ``c`` with a normalizing flow. This module owns the
other half: mapping each column into (and back out of) the space the flow
trains on, and supplying the marginal log-densities ``log f_j(x_j)`` needed
to turn a copula density into a density in original units.

Two transforms are provided:

``StandardizeTransform``
    Plain per-column ``(x - mean) / std``. The flow then models the FULL
    joint -- marginals included. Exact and cheap, but the flow has to spend
    capacity on marginal shape (skew, fat tails) instead of dependence.

``EmpiricalCopulaTransform``
    The standard copula move: map each column through its own empirical CDF
    to a uniform pseudo-observation, then through the probit to a standard
    normal. Marginally N(0, 1) by construction, so anything the flow learns
    afterwards is purely dependence. Marginals are reproduced exactly on the
    way back out via the empirical quantile function.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde, norm, rankdata

__all__ = [
    "MarginalTransform",
    "StandardizeTransform",
    "EmpiricalCopulaTransform",
    "pseudo_observations",
]

# Pseudo-observations are clipped away from exactly 0 and 1 before the probit,
# since norm.ppf(0) = -inf would poison the flow's training data.
_EPS = 1e-6


def pseudo_observations(data: np.ndarray) -> np.ndarray:
    """
    Map each column of `data` to uniform pseudo-observations via its
    empirical CDF, using the standard ``rank / (n + 1)`` scaling.

    The ``n + 1`` denominator (rather than ``n``) keeps every value strictly
    inside the open interval (0, 1), which is what downstream copula code --
    both this library's probit step and `pyvinecopulib` -- requires.

    data: array of shape (n, d).
    Returns: array of shape (n, d) with entries in (0, 1).
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {data.shape}")
    n = len(data)
    return np.column_stack(
        [rankdata(data[:, j]) / (n + 1) for j in range(data.shape[1])]
    )


class MarginalTransform:
    """
    Interface shared by the marginal transforms.

    A transform is fitted once on the training data, then used to move
    points back and forth between original units and the flow's own
    training space, and to score marginal log-densities.
    """

    def fit(self, data: np.ndarray) -> MarginalTransform:
        raise NotImplementedError

    def forward(self, data: np.ndarray) -> np.ndarray:
        """Original units -> the flow's training space."""
        raise NotImplementedError

    def inverse(self, z: np.ndarray) -> np.ndarray:
        """The flow's training space -> original units."""
        raise NotImplementedError

    def log_det_jacobian(self, data: np.ndarray) -> np.ndarray:
        """
        Per-row ``sum_j log |dz_j / dx_j|`` of `forward`, evaluated at
        `data` (original units). Adding this to the flow's log-density in
        its own training space gives the log-density in original units.

        Returns: array of shape (n,).
        """
        raise NotImplementedError

    @property
    def n_features(self) -> int:
        raise NotImplementedError


class StandardizeTransform(MarginalTransform):
    """
    Per-column standardization: ``z = (x - mean) / std``.

    The flow sees the full joint, marginals included. The Jacobian is
    constant, so `log_det_jacobian` is exact and the resulting `log_prob`
    is an exact density in original units.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> StandardizeTransform:
        data = np.asarray(data, dtype=np.float64)
        self.mean_ = data.mean(axis=0)
        self.std_ = data.std(axis=0)
        # A constant column has zero spread; leave it untouched rather than
        # dividing by zero. The flow will see a constant, which is degenerate
        # but at least not NaN -- NeuroCopula.fit warns about this case.
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        return self

    def forward(self, data: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return (np.asarray(data, dtype=np.float64) - self.mean_) / self.std_

    def inverse(self, z: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return np.asarray(z, dtype=np.float64) * self.std_ + self.mean_

    def log_det_jacobian(self, data: np.ndarray) -> np.ndarray:
        self._check_fitted()
        n = len(np.atleast_2d(data))
        return np.full(n, float(-np.sum(np.log(self.std_))))

    @property
    def n_features(self) -> int:
        self._check_fitted()
        return len(self.mean_)

    def _check_fitted(self) -> None:
        if self.mean_ is None:
            raise RuntimeError("StandardizeTransform is not fitted yet; call fit() first.")


class EmpiricalCopulaTransform(MarginalTransform):
    """
    Gaussianize each column via its empirical CDF: ``z = Phi^-1(F_j(x_j))``.

    This is what makes `neurocopula` a copula model rather than a generic
    density estimator: after this transform every column is marginally
    N(0, 1), so the flow's entire capacity goes into the dependence
    structure. Going back out, the empirical quantile function reproduces
    the observed marginal shape exactly -- heavy tails, skew and all --
    without the flow having to learn any of it.

    Two limitations worth knowing:

    - **The empirical CDF cannot extrapolate.** Beyond the smallest and
      largest observed value of a column, `inverse` returns that observed
      extreme (constant extrapolation, from ``np.interp``). If you need
      probabilities for events more extreme than anything in your sample,
      splice on an explicit tail model (e.g. a generalised Pareto fit)
      rather than trusting this transform out there.
    - **The empirical CDF has no density**, being a step function. Turning
      the learned copula density into a density in original units needs
      ``f_j(x_j)``, so `log_det_jacobian` uses a Gaussian KDE per column,
      fitted at the same time as the ECDF. That makes `log_prob` in
      original units an estimate, not an exact quantity. `copula_log_prob`
      -- the copula density on pseudo-observations -- is exact and is the
      right number to compare against a vine copula.

    marginal_density: ``"kde"`` (default) fits a Gaussian KDE per column so
        `log_det_jacobian` (and hence `NeuroCopula.log_prob`) is available.
        ``"none"`` skips it, which is cheaper to fit and fine when you only
        ever need sampling, conditional queries, and `copula_log_prob`;
        `log_det_jacobian` then raises.
    kde_bw: bandwidth passed to ``scipy.stats.gaussian_kde`` (its
        ``bw_method``). None uses Scott's rule.
    """

    def __init__(self, marginal_density: str = "kde", kde_bw=None) -> None:
        if marginal_density not in ("kde", "none"):
            raise ValueError("marginal_density must be 'kde' or 'none'")
        self.marginal_density = marginal_density
        self.kde_bw = kde_bw
        self._sorted: np.ndarray | None = None   # sorted column values
        self._grid: np.ndarray | None = None     # matching rank grid in (0, 1)
        self._kdes: list | None = None

    def fit(self, data: np.ndarray) -> EmpiricalCopulaTransform:
        data = np.asarray(data, dtype=np.float64)
        n, d = data.shape
        self._sorted = np.sort(data, axis=0)
        self._grid = np.arange(1, n + 1) / (n + 1)
        if self.marginal_density == "kde":
            self._kdes = []
            for j in range(d):
                col = data[:, j]
                # A constant column has no KDE; record None and fall back to a
                # zero log-density contribution for it.
                if np.ptp(col) == 0:
                    self._kdes.append(None)
                else:
                    self._kdes.append(gaussian_kde(col, bw_method=self.kde_bw))
        return self

    def to_uniform(self, data: np.ndarray) -> np.ndarray:
        """
        Original units -> uniform pseudo-observations in (0, 1), by linear
        interpolation of the fitted empirical CDF.

        Unlike `pseudo_observations`, this uses the CDF fitted at `fit`
        time, so it can be applied to new/held-out rows consistently.
        """
        self._check_fitted()
        data = np.atleast_2d(np.asarray(data, dtype=np.float64))
        u = np.empty_like(data)
        for j in range(data.shape[1]):
            u[:, j] = np.interp(data[:, j], self._sorted[:, j], self._grid)
        return np.clip(u, _EPS, 1 - _EPS)

    def from_uniform(self, u: np.ndarray) -> np.ndarray:
        """Uniform pseudo-observations -> original units (empirical quantiles)."""
        self._check_fitted()
        u = np.atleast_2d(np.asarray(u, dtype=np.float64))
        out = np.empty_like(u)
        for j in range(u.shape[1]):
            out[:, j] = np.interp(u[:, j], self._grid, self._sorted[:, j])
        return out

    def forward(self, data: np.ndarray) -> np.ndarray:
        return norm.ppf(self.to_uniform(data))

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return self.from_uniform(norm.cdf(np.atleast_2d(np.asarray(z, dtype=np.float64))))

    def log_det_jacobian(self, data: np.ndarray) -> np.ndarray:
        """
        ``sum_j [ log f_j(x_j) - log phi(z_j) ]``, the log-Jacobian of
        ``z_j = Phi^-1(F_j(x_j))``, with ``f_j`` estimated by the per-column
        KDE fitted in `fit`.
        """
        self._check_fitted()
        if self.marginal_density != "kde":
            raise RuntimeError(
                "This transform was built with marginal_density='none', so marginal "
                "densities are unavailable and log_prob in original units cannot be "
                "computed. Rebuild with marginal_density='kde', or use "
                "copula_log_prob(), which needs no marginal density."
            )
        data = np.atleast_2d(np.asarray(data, dtype=np.float64))
        z = self.forward(data)
        total = np.zeros(len(data))
        for j, kde in enumerate(self._kdes):
            if kde is None:
                continue
            log_f = np.log(np.clip(kde(data[:, j]), 1e-300, None))
            total += log_f - norm.logpdf(z[:, j])
        return total

    def marginal_log_pdf(self, data: np.ndarray) -> np.ndarray:
        """
        Per-column marginal log-densities ``log f_j(x_j)`` from the fitted
        KDEs. Shape (n, d). Useful for diagnosing which half of a joint
        density -- marginals or dependence -- a model is getting wrong.
        """
        self._check_fitted()
        if self.marginal_density != "kde":
            raise RuntimeError("marginal_density='none': no marginal densities available.")
        data = np.atleast_2d(np.asarray(data, dtype=np.float64))
        out = np.zeros_like(data)
        for j, kde in enumerate(self._kdes):
            if kde is not None:
                out[:, j] = np.log(np.clip(kde(data[:, j]), 1e-300, None))
        return out

    @property
    def n_features(self) -> int:
        self._check_fitted()
        return self._sorted.shape[1]

    def _check_fitted(self) -> None:
        if self._sorted is None:
            raise RuntimeError("EmpiricalCopulaTransform is not fitted yet; call fit() first.")
