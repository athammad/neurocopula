"""
Metrics for scoring and comparing copula models.

These fall into three groups:

**Dependence summaries** -- `tail_dependence`, `tail_dependence_curve`,
`kendall_tau_matrix`, `spearman_rho_matrix`, `upper_lower_asymmetry`. These
describe a sample, so they apply equally to real data, flow samples, and
vine samples, which is what makes a like-for-like comparison possible.

**Two-sample distances** -- `energy_distance`, `mmd_rbf`,
`ks_statistic_per_column`. These ask "could these two samples have come from
the same distribution", which is the question you actually care about when
judging a generative fit.

**Errors against ground truth** -- `tail_dependence_error`,
`kendall_tau_error`. Only usable on synthetic data where the truth is known,
which is exactly why `neurocopula.datasets` exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import kendalltau, ks_2samp, spearmanr

__all__ = [
    "tail_dependence",
    "tail_dependence_curve",
    "kendall_tau_matrix",
    "spearman_rho_matrix",
    "upper_lower_asymmetry",
    "energy_distance",
    "mmd_rbf",
    "ks_statistic_per_column",
    "tail_dependence_error",
    "kendall_tau_error",
]


def _as_array(x) -> np.ndarray:
    if isinstance(x, (pd.DataFrame, pd.Series)):
        x = x.values
    return np.atleast_2d(np.asarray(x, dtype=np.float64))


# ---------------------------------------------------------------------
# dependence summaries
# ---------------------------------------------------------------------
def tail_dependence(x, y, q: float = 0.05, tail: str = "lower") -> float:
    """
    Empirical tail dependence coefficient between two 1-D samples:

        lambda_L(q) = P(Y < Q_q(Y) | X < Q_q(X))

    Being rank-based it is invariant to the marginals, so it compares
    directly across models and against theory.

    q: tail probability. The estimate rests on only ``q * n`` points, so a
        small `q` needs a large sample -- with n = 10,000 and q = 0.01 you
        are averaging 100 points and should expect a standard error near
        0.05.
    tail: ``"lower"`` or ``"upper"``.

    Returns: the coefficient in [0, 1], or NaN if the tail is empty.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if not 0 < q < 0.5:
        raise ValueError("q must be in (0, 0.5)")
    if tail == "lower":
        cond = x < np.quantile(x, q)
        joint = cond & (y < np.quantile(y, q))
    elif tail == "upper":
        cond = x > np.quantile(x, 1 - q)
        joint = cond & (y > np.quantile(y, 1 - q))
    else:
        raise ValueError("tail must be 'lower' or 'upper'")
    denom = cond.sum()
    return float(joint.sum() / denom) if denom else float("nan")


def tail_dependence_curve(
    x, y, quantiles=None, tail: str = "lower"
) -> pd.DataFrame:
    """
    `tail_dependence` across a range of `q`, as a DataFrame with columns
    ``q`` and ``lambda``.

    The shape of this curve is the diagnostic that matters. Asymptotic
    independence (a Gaussian copula) shows up as a curve sloping to zero;
    genuine tail dependence flattens to a positive plateau.
    """
    if quantiles is None:
        quantiles = np.geomspace(0.005, 0.25, 20)
    return pd.DataFrame(
        [{"q": float(q), "lambda": tail_dependence(x, y, q=q, tail=tail)}
         for q in quantiles]
    )


def kendall_tau_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise Kendall's tau. Rank-based, so it measures dependence with all
    marginal effects removed -- the right correlation measure for copula
    work, where Pearson's r would confound the two.
    """
    cols = list(df.columns)
    m = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            t = kendalltau(df[a].values, df[b].values).statistic
            m.loc[a, b] = m.loc[b, a] = t
    return m


def spearman_rho_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Spearman rank correlation, as a labelled DataFrame."""
    cols = list(df.columns)
    rho = np.asarray(spearmanr(df.values).statistic, dtype=float)
    if rho.ndim == 0:  # scipy returns a scalar for exactly two columns
        r = float(rho)
        rho = np.array([[1.0, r], [r, 1.0]])
    return pd.DataFrame(rho, index=cols, columns=cols)


def upper_lower_asymmetry(x, y, q: float = 0.05) -> float:
    """
    ``lambda_L(q) - lambda_U(q)``: how much more strongly a pair moves
    together on the way down than on the way up.

    Positive means crashes are more synchronised than rallies -- the
    signature of Clayton-like dependence. Any elliptical copula (Gaussian,
    Student-t) forces this to zero by construction, so a non-zero value in
    the data is direct evidence that an elliptical model is the wrong shape.
    """
    return tail_dependence(x, y, q, "lower") - tail_dependence(x, y, q, "upper")


# ---------------------------------------------------------------------
# two-sample distances
# ---------------------------------------------------------------------
def energy_distance(sample_a, sample_b, max_n: int = 2000, seed: int = 0) -> float:
    """
    Multivariate energy distance between two samples:

        E = 2 * E|A - B| - E|A - A'| - E|B - B'|

    Zero if and only if the two distributions are identical, so it is a
    genuine goodness-of-fit measure for a generative model -- unlike, say,
    comparing correlation matrices, which two very different joints can
    share. Lower is better.

    It is NOT scale-free: standardise or use copula-scale (uniform) samples
    before comparing across datasets. The cost is quadratic in sample size,
    so both samples are randomly subsampled to `max_n` rows.
    """
    a, b = _as_array(sample_a), _as_array(sample_b)
    if a.shape[1] != b.shape[1]:
        raise ValueError("both samples must have the same number of columns")
    rng = np.random.default_rng(seed)
    if len(a) > max_n:
        a = a[rng.choice(len(a), max_n, replace=False)]
    if len(b) > max_n:
        b = b[rng.choice(len(b), max_n, replace=False)]
    d_ab = cdist(a, b).mean()
    d_aa = cdist(a, a).mean()
    d_bb = cdist(b, b).mean()
    return float(2 * d_ab - d_aa - d_bb)


def mmd_rbf(sample_a, sample_b, bandwidth=None, max_n: int = 2000,
            seed: int = 0) -> float:
    """
    Squared maximum mean discrepancy with an RBF kernel -- a second,
    differently-shaped two-sample distance. Reporting it alongside
    `energy_distance` guards against a verdict that is an artifact of one
    particular distance.

    bandwidth: RBF bandwidth. None uses the median heuristic (the median
        pairwise distance of the pooled sample), which adapts to the data's
        scale and is the usual default.

    Returns: the squared MMD; lower is better, 0 means indistinguishable.
    """
    a, b = _as_array(sample_a), _as_array(sample_b)
    rng = np.random.default_rng(seed)
    if len(a) > max_n:
        a = a[rng.choice(len(a), max_n, replace=False)]
    if len(b) > max_n:
        b = b[rng.choice(len(b), max_n, replace=False)]
    if bandwidth is None:
        pooled = np.vstack([a, b])
        med = np.median(cdist(pooled, pooled))
        bandwidth = med if med > 0 else 1.0
    gamma = 1.0 / (2 * bandwidth ** 2)

    def k(p, q):
        return np.exp(-gamma * cdist(p, q, "sqeuclidean"))

    return float(k(a, a).mean() + k(b, b).mean() - 2 * k(a, b).mean())


def ks_statistic_per_column(sample_a, sample_b) -> pd.Series:
    """
    Two-sample Kolmogorov-Smirnov statistic per column -- a MARGINAL check
    only.

    Useful as a control: in copula mode the marginals are reproduced by
    construction, so these should be near zero, and anything else points at
    a bug rather than a modelling limitation. It says nothing at all about
    dependence, which is what `energy_distance` and the tail metrics are for.
    """
    a = sample_a if isinstance(sample_a, pd.DataFrame) else pd.DataFrame(sample_a)
    b = sample_b if isinstance(sample_b, pd.DataFrame) else pd.DataFrame(sample_b)
    return pd.Series(
        {c: float(ks_2samp(a[c].values, b[c].values).statistic) for c in a.columns},
        name="ks_statistic",
    )


# ---------------------------------------------------------------------
# errors against known truth
# ---------------------------------------------------------------------
def tail_dependence_error(
    sample, truth: dict[str, float], q: float = 0.02, col1: int = 0, col2: int = 1
) -> dict[str, float]:
    """
    Absolute error of a sample's empirical tail dependence against known
    theoretical values.

    sample: DataFrame or array; `col1`/`col2` select the pair by position.
    truth: ``{"lower": ..., "upper": ...}``, e.g. from
        `datasets.theoretical_tail_dependence`.
    q: tail probability at which to compare. This matters: theory is an
        asymptotic (q -> 0) statement, while any finite sample must be
        evaluated at a positive `q`, so even a perfect model shows some
        error. Keep `q` fixed across the models being compared and read the
        numbers relatively.

    Returns: dict with empirical values, truth, and absolute errors.
    """
    s = _as_array(sample)
    x, y = s[:, col1], s[:, col2]
    lo = tail_dependence(x, y, q, "lower")
    up = tail_dependence(x, y, q, "upper")
    return {
        "lambda_L": lo,
        "lambda_U": up,
        "lambda_L_true": truth["lower"],
        "lambda_U_true": truth["upper"],
        "lambda_L_abs_error": abs(lo - truth["lower"]),
        "lambda_U_abs_error": abs(up - truth["upper"]),
    }


def kendall_tau_error(sample, tau_true: float, col1: int = 0, col2: int = 1) -> float:
    """
    Absolute error of a sample's Kendall's tau against a known value.

    Unlike the tail metrics this is a whole-distribution summary, so almost
    any reasonable model gets it close. Treat it as a floor: a model that
    misses tau is broken, while matching tau proves very little on its own.
    """
    s = _as_array(sample)
    return float(abs(kendalltau(s[:, col1], s[:, col2]).statistic - tau_true))
