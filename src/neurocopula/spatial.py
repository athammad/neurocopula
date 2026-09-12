"""
Spatial dependence: linking locations so that a joint distribution over any
subset of them is available on demand.

A per-location copula tells you what happens *at* a point. It says nothing
about whether two points go wrong together, so anything aggregated over an
area -- a regional mean, a count of sites over threshold, the probability
that a basin is hit anywhere -- comes out wrong. Under independence the
aggregate barely moves: averaging n locations divides its standard deviation
by sqrt(n), so region-wide extremes become impossible by construction. This
module supplies the missing piece.

The design constraint is that the region is chosen *after* fitting. Nothing
can be precomputed per region, so the model has to be a genuine spatial
process: correlation is a function of distance, and the joint distribution
over any finite set of locations follows from it directly. Cost then scales
with the region asked for, not with the size of the domain.

**Why Student-t rather than Gaussian.** A Gaussian random field has exactly
zero tail dependence: however strongly two locations correlate in the body of
the distribution, they become independent in the extremes. For work on joint
extremes that assumption answers the question before the data is consulted. A
t field keeps the same correlation structure, adds one parameter (the degrees
of freedom), and retains tail dependence. It is the smallest change that
avoids assuming the conclusion.

The module is domain-neutral: "locations" are any points with coordinates.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import multivariate_normal, multivariate_t, norm
from scipy.stats import t as student_t

__all__ = [
    "haversine_matrix",
    "euclidean_matrix",
    "CORRELATION_MODELS",
    "SpatialDependence",
]

_EPS = 1e-10
EARTH_RADIUS_KM = 6371.0088


# ---------------------------------------------------------------------
# distances
# ---------------------------------------------------------------------
def haversine_matrix(coords_a: np.ndarray, coords_b: np.ndarray | None = None) -> np.ndarray:
    """
    Great-circle distance in kilometres between two sets of (lat, lon) points.

    Great-circle rather than Euclidean-in-degrees because a degree of
    longitude shrinks with latitude: treating degrees as a plane would make
    the same physical separation look larger near the equator than near the
    poles, and the fitted correlation range would absorb that distortion.

    coords_a: array (n, 2) of latitude, longitude in degrees.
    coords_b: array (m, 2), or None to use `coords_a`.
    Returns: array (n, m) of distances in km.
    """
    a = np.atleast_2d(np.asarray(coords_a, dtype=np.float64))
    b = a if coords_b is None else np.atleast_2d(np.asarray(coords_b, dtype=np.float64))
    if a.shape[1] != 2 or b.shape[1] != 2:
        raise ValueError("coordinates must have two columns: latitude, longitude")

    lat_a, lon_a = np.deg2rad(a[:, 0])[:, None], np.deg2rad(a[:, 1])[:, None]
    lat_b, lon_b = np.deg2rad(b[:, 0])[None, :], np.deg2rad(b[:, 1])[None, :]
    dlat, dlon = lat_b - lat_a, lon_b - lon_a
    h = np.sin(dlat / 2) ** 2 + np.cos(lat_a) * np.cos(lat_b) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def euclidean_matrix(coords_a: np.ndarray, coords_b: np.ndarray | None = None) -> np.ndarray:
    """Plain Euclidean distance, for projected coordinates or synthetic tests."""
    a = np.atleast_2d(np.asarray(coords_a, dtype=np.float64))
    b = a if coords_b is None else np.atleast_2d(np.asarray(coords_b, dtype=np.float64))
    return np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))


# ---------------------------------------------------------------------
# correlation functions
# ---------------------------------------------------------------------
def _exponential(d, rng_):
    return np.exp(-d / rng_)


def _gaussian(d, rng_):
    return np.exp(-((d / rng_) ** 2))


def _matern32(d, rng_):
    s = np.sqrt(3) * d / rng_
    return (1 + s) * np.exp(-s)


def _matern52(d, rng_):
    s = np.sqrt(5) * d / rng_
    return (1 + s + s ** 2 / 3) * np.exp(-s)


#: Correlation shapes, all reaching 1 at zero distance and decaying to 0.
#: They differ in how smooth the field is at short range: ``exponential`` is
#: rough (non-differentiable at the origin), ``gaussian`` is very smooth and
#: often unrealistically so, and the Matern pair sits between. When in doubt
#: fit several and compare -- `SpatialDependence` does that automatically
#: when ``model="auto"``.
CORRELATION_MODELS = {
    "exponential": _exponential,
    "matern32": _matern32,
    "matern52": _matern52,
    "gaussian": _gaussian,
}


# ---------------------------------------------------------------------
def _t_tail_dependence(rho: np.ndarray, df: float) -> np.ndarray:
    """
    ASYMPTOTIC tail dependence of a bivariate t copula:

        lambda = 2 * T_{df+1}( -sqrt((df+1)(1-rho)/(1+rho)) )

    This is the q -> 0 limit. It is reported by `SpatialDependence.tail_dependence`
    as the headline summary, but it must NOT be compared against an empirical
    co-exceedance rate measured at a finite threshold -- see `_coexceedance`.
    """
    rho = np.clip(np.asarray(rho, dtype=np.float64), -0.999999, 0.999999)
    arg = -np.sqrt((df + 1) * (1 - rho) / (1 + rho))
    return 2 * student_t.cdf(arg, df + 1)


def _coexceedance(rho: float, df: float, q: float) -> float:
    """
    Co-exceedance rate at a FINITE threshold: ``P(U2 < q | U1 < q)``.

    This is the quantity an empirical estimate actually measures, and it is
    not the asymptotic lambda. The gap is large and systematic: a Gaussian
    field with rho = 0.76 has asymptotic lambda exactly 0, yet still shows a
    co-exceedance rate near 0.45 at q = 0.05, because the two only become
    independent in the limit. Fitting degrees of freedom by matching an
    empirical rate against the asymptotic formula therefore drives df down
    hard and invents tail dependence in data that has none -- including in
    data generated from a Gaussian field.

    ``df=inf`` gives the Gaussian case.
    """
    rho = float(np.clip(rho, -0.999, 0.999))
    cov = [[1.0, rho], [rho, 1.0]]
    if np.isfinite(df):
        x = float(student_t.ppf(q, df))
        c = float(multivariate_t.cdf([x, x], loc=[0.0, 0.0], shape=cov, df=df))
    else:
        x = float(norm.ppf(q))
        c = float(multivariate_normal.cdf([x, x], mean=[0.0, 0.0], cov=cov))
    return c / q


def _tetrachoric(p_both: float, a1: float, a2: float) -> float:
    """
    Latent correlation implied by how often two censored variables are censored
    together.

    Under a latent Gaussian field, "dry" means the latent fell below a threshold
    ``a = Phi^-1(P(dry))``. The probability that BOTH locations are dry is then
    the bivariate normal CDF at those thresholds, which depends on the latent
    correlation. Inverting that relationship recovers the correlation from the
    dry/wet co-occurrence table alone -- the classical tetrachoric correlation.

    This matters because the randomized PIT deliberately replaces every dry day
    with an independent uniform draw. That is correct for the marginal but it
    destroys exactly the co-occurrence information needed here, so the spatial
    correlation has to be recovered from the censoring pattern instead of from
    the randomized values.

    p_both: observed fraction of times both locations are censored.
    a1, a2: the two latent thresholds.
    Returns: correlation in (-1, 1).
    """
    # Degenerate cases: one location always or never censored carries no signal.
    if not np.isfinite(a1) or not np.isfinite(a2):
        return 0.0

    def gap(rho):
        cov = [[1.0, rho], [rho, 1.0]]
        return float(multivariate_normal.cdf([a1, a2], mean=[0.0, 0.0], cov=cov)) - p_both

    g_lo, g_hi = gap(-0.999), gap(0.999)
    if g_lo > 0 or g_hi < 0:
        # Observed co-occurrence outside what any correlation can produce.
        return -0.999 if g_lo > 0 else 0.999
    return float(brentq(gap, -0.999, 0.999, xtol=1e-4))


class SpatialDependence:
    """
    A Student-t random field fitted to gridded observations.

    Fitting is a three-step procedure, each step cheap and robust:

    1. **Empirical correlogram.** Correlate every pair of locations across
       time and bin those correlations by separation distance. Binning keeps
       the cost manageable and averages away the noise in individual pairs.
    2. **Correlation function.** Fit ``nugget + (1 - nugget) * rho(d / range)``
       to the binned correlogram by weighted least squares, weighting by the
       number of pairs in each bin. Least squares on the correlogram rather
       than full maximum likelihood: ML on an n x n covariance costs O(n^3)
       per evaluation and buys little here, where three parameters are being
       estimated from thousands of fields.
    3. **Degrees of freedom.** Measure how often nearby locations are jointly
       extreme and choose the `df` whose theoretical tail dependence matches.
       This is what separates a t field from a Gaussian one, so it is
       estimated from the tails rather than from the body of the data.

    Parameters
    ----------
    model:
        A key of `CORRELATION_MODELS`, or ``"auto"`` (default) to fit all of
        them and keep the best by weighted residual error.
    distance:
        ``"haversine"`` for (lat, lon) degrees, ``"euclidean"`` for projected
        or synthetic coordinates.
    n_bins:
        Number of distance bins for the correlogram.
    max_distance:
        Ignore pairs beyond this separation when fitting. Useful because the
        correlogram at very long range is dominated by a handful of pairs and
        can drag the fit. None uses the median pairwise distance.
    fit_df:
        Estimate the t degrees of freedom. False pins it to `df`, and
        ``df=inf`` gives a Gaussian field -- worth fitting deliberately as a
        baseline, since the difference between them *is* the tail dependence.
    df:
        Initial or fixed degrees of freedom.
    max_pairs:
        Cap on the number of location pairs used for the correlogram; pairs
        are subsampled beyond this. The full set is O(n^2) and unnecessary.
    seed:
        Seed for subsampling and for sampling.
    """

    def __init__(
        self,
        model: str = "auto",
        distance: str = "haversine",
        n_bins: int = 25,
        max_distance: float | None = None,
        fit_df: bool = True,
        df: float = 8.0,
        max_pairs: int = 200_000,
        seed: int = 0,
    ) -> None:
        if model != "auto" and model not in CORRELATION_MODELS:
            raise ValueError(
                f"unknown model {model!r}; expected 'auto' or one of "
                f"{sorted(CORRELATION_MODELS)}"
            )
        if distance not in ("haversine", "euclidean"):
            raise ValueError("distance must be 'haversine' or 'euclidean'")
        self.model = model
        self.distance = distance
        self.n_bins = n_bins
        self.max_distance = max_distance
        self.fit_df = fit_df
        self.df = df
        self.max_pairs = max_pairs
        self.seed = seed

        self.model_: str | None = None
        self.range_: float | None = None
        self.nugget_: float | None = None
        self.df_: float | None = None
        self.correlogram_: dict | None = None
        self.censored_: bool = False
        self.fit_error_: float | None = None
        self._rng = np.random.default_rng(seed)

    # -----------------------------------------------------------------
    def _dist(self, a, b=None):
        fn = haversine_matrix if self.distance == "haversine" else euclidean_matrix
        return fn(a, b)

    def fit(
        self,
        z: np.ndarray,
        coords: np.ndarray,
        censored: np.ndarray | None = None,
        df_tail: str = "lower",
    ) -> SpatialDependence:
        """
        z: array (n_time, n_locations) of values that are already marginally
            standard normal at every location -- the latent/normal-score
            representation produced upstream. Correlations are only
            interpretable on that scale.
        coords: array (n_locations, 2) of coordinates.
        censored: optional boolean array (n_time, n_locations) marking entries
            that sit on an atom -- dry days for rainfall, say. When given, the
            correlation is estimated from the censoring pattern (tetrachoric)
            rather than from the values, because the randomized transform that
            makes the marginal correct replaces every censored entry with
            independent noise and would otherwise dilute the correlation
            towards zero.
        df_tail: which tail identifies the degrees of freedom, ``"lower"`` or
            ``"upper"``. Use ``"upper"`` for a censored variable: the lower
            tail of rainfall *is* the censored region and carries no
            information about joint extremes.
        """
        z = np.asarray(z, dtype=np.float64)
        coords = np.atleast_2d(np.asarray(coords, dtype=np.float64))
        if z.ndim != 2:
            raise ValueError(f"z must be 2-D (time, locations), got shape {z.shape}")
        if z.shape[1] != len(coords):
            raise ValueError(
                f"z has {z.shape[1]} locations but {len(coords)} coordinates were given"
            )
        if z.shape[0] < 20:
            warnings.warn(
                f"only {z.shape[0]} time steps: the correlogram will be very noisy.",
                RuntimeWarning, stacklevel=2,
            )

        if censored is not None:
            censored = np.asarray(censored, dtype=bool)
            if censored.shape != z.shape:
                raise ValueError(
                    f"censored has shape {censored.shape}, expected {z.shape}"
                )
            if not censored.any():
                censored = None

        self.censored_ = censored is not None
        self.correlogram_ = (
            self._censored_correlogram(censored, coords) if censored is not None
            else self._empirical_correlogram(z, coords)
        )
        self._fit_correlation(self.correlogram_)
        if df_tail not in ("lower", "upper"):
            raise ValueError("df_tail must be 'lower' or 'upper'")
        self.df_ = (self._fit_df(z, coords, tail=df_tail) if self.fit_df
                    else float(self.df))
        return self

    # -----------------------------------------------------------------
    def _pair_indices(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """All location pairs, subsampled if there are too many."""
        i, j = np.triu_indices(n, k=1)
        if len(i) > self.max_pairs:
            keep = self._rng.choice(len(i), self.max_pairs, replace=False)
            i, j = i[keep], j[keep]
        return i, j

    def _empirical_correlogram(self, z: np.ndarray, coords: np.ndarray) -> dict:
        """Pairwise correlation binned by separation distance."""
        n = z.shape[1]
        i, j = self._pair_indices(n)
        d = self._dist(coords)[i, j]

        # Correlate pairs across time. Standardize once, then the correlation
        # of a pair is just the mean product.
        zs = (z - z.mean(0)) / np.where(z.std(0) > _EPS, z.std(0), 1.0)
        r = (zs[:, i] * zs[:, j]).mean(0)

        cutoff = self.max_distance if self.max_distance is not None else float(np.median(d))
        keep = d <= cutoff
        d, r = d[keep], r[keep]
        if d.size == 0:
            raise ValueError("no location pairs within max_distance; raise it")

        edges = np.linspace(0, d.max(), self.n_bins + 1)
        idx = np.clip(np.digitize(d, edges) - 1, 0, self.n_bins - 1)
        centers, means, counts = [], [], []
        for b in range(self.n_bins):
            m = idx == b
            if m.sum() >= 5:
                centers.append(float(d[m].mean()))
                means.append(float(r[m].mean()))
                counts.append(int(m.sum()))
        if len(centers) < 3:
            raise ValueError(
                "fewer than 3 usable distance bins; reduce n_bins or supply more locations"
            )
        return {
            "distance": np.array(centers),
            "correlation": np.array(means),
            "n_pairs": np.array(counts),
        }

    def _censored_correlogram(self, censored: np.ndarray, coords: np.ndarray) -> dict:
        """
        Correlogram estimated from the censoring pattern via tetrachoric
        correlation, pooled within distance bins.

        Pairs are pooled before inverting rather than inverted per pair: the
        inversion needs a bivariate normal CDF and a root find, which is far
        too slow for millions of pairs, and a single pair's co-occurrence rate
        is noisy anyway.
        """
        n = censored.shape[1]
        i, j = self._pair_indices(n)
        d = self._dist(coords)[i, j]
        cutoff = (self.max_distance if self.max_distance is not None
                  else float(np.median(d)))
        keep = d <= cutoff
        i, j, d = i[keep], j[keep], d[keep]
        if d.size == 0:
            raise ValueError("no location pairs within max_distance; raise it")

        p_cens = censored.mean(axis=0)                      # per location
        thresholds = norm.ppf(np.clip(p_cens, 1e-6, 1 - 1e-6))
        both = (censored[:, i] & censored[:, j]).mean(axis=0)

        edges = np.linspace(0, d.max(), self.n_bins + 1)
        idx = np.clip(np.digitize(d, edges) - 1, 0, self.n_bins - 1)
        centers, corrs, counts = [], [], []
        for b in range(self.n_bins):
            m = idx == b
            if m.sum() < 5:
                continue
            a1 = float(np.mean(thresholds[i[m]]))
            a2 = float(np.mean(thresholds[j[m]]))
            rho = _tetrachoric(float(both[m].mean()), a1, a2)
            centers.append(float(d[m].mean()))
            corrs.append(rho)
            counts.append(int(m.sum()))
        if len(centers) < 3:
            raise ValueError(
                "fewer than 3 usable distance bins; reduce n_bins or supply more locations"
            )
        return {
            "distance": np.array(centers),
            "correlation": np.array(corrs),
            "n_pairs": np.array(counts),
        }

    def _fit_correlation(self, cg: dict) -> None:
        """Weighted least squares of the correlation function on the correlogram."""
        d, r, w = cg["distance"], cg["correlation"], cg["n_pairs"].astype(float)
        w = w / w.sum()
        candidates = list(CORRELATION_MODELS) if self.model == "auto" else [self.model]

        best = None
        for name in candidates:
            fn = CORRELATION_MODELS[name]

            def sse(log_range, fn=fn):
                rng_ = float(np.exp(log_range))
                shape = fn(d, rng_)
                # Given the range, the optimal nugget is available in closed
                # form (weighted least squares on a linear model in nugget),
                # so only the range needs a numerical search.
                x = 1.0 - shape
                denom = float((w * x * x).sum())
                nug = 0.0 if denom < _EPS else float((w * x * (r - shape)).sum() / denom)
                nug = float(np.clip(nug, 0.0, 1.0))
                pred = nug + (1 - nug) * shape
                return float((w * (r - pred) ** 2).sum())

            lo, hi = np.log(max(d.min(), _EPS) / 10), np.log(d.max() * 10)
            res = minimize_scalar(sse, bounds=(lo, hi), method="bounded")
            rng_ = float(np.exp(res.x))
            shape = fn(d, rng_)
            x = 1.0 - shape
            denom = float((w * x * x).sum())
            nug = 0.0 if denom < _EPS else float((w * x * (r - shape)).sum() / denom)
            nug = float(np.clip(nug, 0.0, 1.0))
            if best is None or res.fun < best[0]:
                best = (float(res.fun), name, rng_, nug)

        self.fit_error_, self.model_, self.range_, self.nugget_ = best

    #: Degrees of freedom searched when fitting. The Gaussian limit is an
    #: explicit candidate rather than a boundary case, so "no tail dependence"
    #: is a conclusion the fit can actually reach.
    _DF_GRID = (2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0,
                100.0, np.inf)

    def _fit_df(self, z: np.ndarray, coords: np.ndarray, q: float = 0.02,
                tail: str = "lower") -> float:
        """
        Choose degrees of freedom by matching the observed co-exceedance rate
        as a function of distance.

        Two choices matter here. First, the comparison is made at the same
        finite threshold `q` on both sides -- see `_coexceedance` for why
        comparing against the asymptotic lambda silently fabricates tail
        dependence. Second, the match is made across the whole distance
        curve rather than at one pooled distance: a single number is weakly
        identified (at rho = 0.76, Gaussian and t with df = 5 differ by only
        about 0.06), whereas the shape of the decay with distance separates
        them much more clearly.
        """
        n = z.shape[1]
        i, j = self._pair_indices(n)
        d = self._dist(coords)[i, j]
        cutoff = (self.max_distance if self.max_distance is not None
                  else float(np.median(d)))
        keep = d <= cutoff
        i, j, d = i[keep], j[keep], d[keep]
        if len(i) < 20:
            return float(self.df)

        # Empirical co-exceedance per pair, then binned by distance. For a
        # censored variable the informative tail is the upper one; the lower
        # tail is the atom and says nothing about joint extremes.
        if tail == "upper":
            thresh = np.quantile(z, 1 - q, axis=0)
            beyond = z > thresh[None, :]
        else:
            thresh = np.quantile(z, q, axis=0)
            beyond = z < thresh[None, :]
        coex = (beyond[:, i] & beyond[:, j]).mean(0) / q

        edges = np.linspace(0, d.max(), self.n_bins + 1)
        idx = np.clip(np.digitize(d, edges) - 1, 0, self.n_bins - 1)
        centers, emp, weights = [], [], []
        for b in range(self.n_bins):
            m = idx == b
            if m.sum() >= 5:
                centers.append(float(d[m].mean()))
                emp.append(float(coex[m].mean()))
                weights.append(float(m.sum()))
        if len(centers) < 3:
            return float(self.df)

        centers = np.array(centers)
        emp = np.array(emp)
        w = np.array(weights)
        w = w / w.sum()
        rho = self.correlation(centers)

        best_df, best_err = float(self.df), np.inf
        for df in self._DF_GRID:
            pred = np.array([_coexceedance(r, df, q) for r in rho])
            err = float((w * (pred - emp) ** 2).sum())
            if err < best_err:
                best_err, best_df = err, float(df)
        self.df_fit_error_ = best_err
        self.coexceedance_ = {"distance": centers, "empirical": emp, "q": q}
        return best_df

    # -----------------------------------------------------------------
    def correlation(self, d: np.ndarray) -> np.ndarray:
        """Fitted correlation at separation distance(s) `d`."""
        self._check_fitted()
        shape = CORRELATION_MODELS[self.model_](np.asarray(d, dtype=np.float64), self.range_)
        return self.nugget_ + (1 - self.nugget_) * shape

    def correlation_matrix(self, coords: np.ndarray, jitter: float = 1e-8) -> np.ndarray:
        """
        Correlation matrix for an arbitrary set of locations.

        This is what makes an after-the-fact region query possible: the
        process is defined by distance, so any subset has a matrix.
        """
        self._check_fitted()
        coords = np.atleast_2d(np.asarray(coords, dtype=np.float64))
        C = self.correlation(self._dist(coords))
        np.fill_diagonal(C, 1.0)
        # A fitted correlation function is positive definite in theory but the
        # assembled matrix can be numerically indefinite; a small ridge keeps
        # the factorization stable.
        C[np.diag_indices_from(C)] += jitter
        return C

    def sample(self, coords: np.ndarray, n: int = 1000, seed: int | None = None
               ) -> np.ndarray:
        """
        Draw `n` joint realizations over the given locations, returned on the
        **standard normal scale** with t-copula dependence between locations.

        The construction is deliberate: draw from a multivariate t, map each
        coordinate through the t CDF to a uniform, then through the normal
        quantile function. The result has exactly N(0,1) margins -- which is
        what the upstream per-location model expects -- while the *dependence*
        is the t copula, so tail clustering survives.

        Returns: array (n, n_locations).
        """
        self._check_fitted()
        coords = np.atleast_2d(np.asarray(coords, dtype=np.float64))
        m = len(coords)
        if m > 20_000:
            warnings.warn(
                f"{m} locations: the Cholesky factorization is O(n^3) and will be slow. "
                "Consider querying a smaller region or coarsening the grid.",
                RuntimeWarning, stacklevel=2,
            )
        rng = np.random.default_rng(self.seed if seed is None else seed)
        C = self.correlation_matrix(coords)
        L = np.linalg.cholesky(C)

        w = rng.standard_normal((n, m)) @ L.T
        if np.isfinite(self.df_):
            # Multivariate t: a shared chi-square scaling across locations.
            # The fact that the SAME scale hits every location on a given draw
            # is precisely what creates tail dependence -- a small draw makes
            # everywhere extreme at once.
            g = rng.chisquare(self.df_, size=(n, 1)) / self.df_
            y = w / np.sqrt(g)
            u = student_t.cdf(y, self.df_)
        else:
            u = norm.cdf(w)
        return norm.ppf(np.clip(u, _EPS, 1 - _EPS))

    def tail_dependence(self, d: float) -> float:
        """
        Theoretical tail dependence between two locations `d` apart.

        Zero for a Gaussian field at every distance; positive and decaying
        with distance for a t field. Plotting this against distance is the
        clearest single summary of what the fitted field implies about joint
        extremes.
        """
        self._check_fitted()
        if not np.isfinite(self.df_):
            return 0.0
        return float(_t_tail_dependence(self.correlation(d), self.df_))

    @property
    def effective_range(self) -> float:
        """
        Distance at which correlation falls to 0.05 -- a more comparable
        summary than the raw range parameter, whose meaning differs between
        correlation shapes.
        """
        self._check_fitted()
        d = np.linspace(0, self.range_ * 20, 4000)
        r = self.correlation(d)
        below = np.flatnonzero(r <= 0.05)
        return float(d[below[0]]) if below.size else float(d[-1])

    def _check_fitted(self) -> None:
        if self.range_ is None:
            raise RuntimeError("SpatialDependence is not fitted yet; call fit() first.")

    def summary(self) -> dict:
        self._check_fitted()
        return {
            "model": self.model_,
            "range": self.range_,
            "effective_range": self.effective_range,
            "nugget": self.nugget_,
            "df": self.df_,
            "gaussian_limit": not np.isfinite(self.df_),
            "fit_error": self.fit_error_,
        }

    def __repr__(self) -> str:
        if self.range_ is None:
            return f"SpatialDependence(model={self.model!r}, unfitted)"
        df = "inf (Gaussian)" if not np.isfinite(self.df_) else f"{self.df_:.1f}"
        return (f"SpatialDependence(model={self.model_!r}, range={self.range_:.1f}, "
                f"nugget={self.nugget_:.3f}, df={df})")
