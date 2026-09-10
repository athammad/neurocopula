"""
Synthetic datasets with KNOWN dependence structure.

Every generator here has a closed-form tail dependence coefficient and
Kendall's tau, which is the point: it lets the tests and the benchmark suite
check what a fitted model recovers against ground truth, instead of merely
checking that it recovers something plausible.

Reference values used below (standard copula theory):

============  =========================  ==================================
copula        lambda_L / lambda_U        Kendall's tau
============  =========================  ==================================
Gaussian      0 / 0 (any |rho| < 1)      ``2/pi * arcsin(rho)``
Student-t     both ``2 * t_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho)))``  same as Gaussian
Clayton       ``2^(-1/theta)`` / 0       ``theta / (theta + 2)``
Gumbel        0 / ``2 - 2^(1/theta)``    ``1 - 1/theta``
============  =========================  ==================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t as student_t

__all__ = [
    "make_clayton",
    "make_gumbel",
    "make_gaussian_copula",
    "make_t_copula",
    "make_mixed_regime",
    "make_asset_panel",
    "theoretical_tail_dependence",
    "theoretical_kendall_tau",
]


# ---------------------------------------------------------------------
# closed-form reference values
# ---------------------------------------------------------------------
def theoretical_tail_dependence(family: str, **params) -> dict[str, float]:
    """
    Closed-form lower/upper tail dependence for a supported family.

    family: ``"clayton"`` (needs `theta`), ``"gumbel"`` (`theta`),
        ``"gaussian"`` (`rho`), or ``"t"`` (`rho`, `df`).

    Returns: ``{"lower": ..., "upper": ...}``.
    """
    family = family.lower()
    if family == "clayton":
        theta = params["theta"]
        return {"lower": 2.0 ** (-1.0 / theta) if theta > 0 else 0.0, "upper": 0.0}
    if family == "gumbel":
        theta = params["theta"]
        return {"lower": 0.0, "upper": 2.0 - 2.0 ** (1.0 / theta)}
    if family == "gaussian":
        return {"lower": 0.0, "upper": 0.0}
    if family == "t":
        rho, dof = params["rho"], params["df"]
        lam = 2 * student_t.cdf(
            -np.sqrt((dof + 1) * (1 - rho) / (1 + rho)), dof + 1
        )
        return {"lower": float(lam), "upper": float(lam)}
    raise ValueError(f"unknown family {family!r}")


def theoretical_kendall_tau(family: str, **params) -> float:
    """Closed-form Kendall's tau for a supported family."""
    family = family.lower()
    if family == "clayton":
        theta = params["theta"]
        return theta / (theta + 2.0)
    if family == "gumbel":
        return 1.0 - 1.0 / params["theta"]
    if family in ("gaussian", "t"):
        return float(2.0 / np.pi * np.arcsin(params["rho"]))
    raise ValueError(f"unknown family {family!r}")


# ---------------------------------------------------------------------
# bivariate archimedean copulas
# ---------------------------------------------------------------------
def _marginal(u: np.ndarray, marginals: str, df: int) -> np.ndarray:
    """Push uniform pseudo-observations through the requested marginal."""
    if marginals == "uniform":
        return u
    if marginals == "normal":
        return norm.ppf(u)
    if marginals == "t":
        return student_t.ppf(u, df)
    raise ValueError("marginals must be 'uniform', 'normal', or 't'")


def make_clayton(
    n: int = 2000, theta: float = 2.0, marginals: str = "t", df: int = 4,
    seed: int = 0, columns: tuple[str, str] = ("x1", "x2"),
) -> pd.DataFrame:
    """
    Bivariate Clayton copula: strong LOWER tail dependence, none in the
    upper tail. The canonical "assets crash together but rally
    independently" shape, and the case a Gaussian copula gets most wrong.

    Sampled by the standard conditional-inversion algorithm.

    theta: Clayton parameter, > 0. Larger means stronger lower-tail
        dependence; ``lambda_L = 2^(-1/theta)``.
    marginals: ``"t"`` (fat-tailed, default), ``"normal"``, or ``"uniform"``
        (copula scale, no marginal transform at all).
    df: degrees of freedom when ``marginals="t"``.
    """
    if theta <= 0:
        raise ValueError("Clayton theta must be > 0")
    rng = np.random.default_rng(seed)
    u1 = rng.uniform(size=n)
    w = rng.uniform(size=n)
    u2 = (w ** (-theta / (1 + theta)) - 1) * u1 ** (-theta) + 1
    u2 = u2 ** (-1 / theta)
    u = np.clip(np.column_stack([u1, u2]), 1e-9, 1 - 1e-9)
    return pd.DataFrame(_marginal(u, marginals, df), columns=list(columns))


def make_gumbel(
    n: int = 2000, theta: float = 2.0, marginals: str = "t", df: int = 4,
    seed: int = 0, columns: tuple[str, str] = ("x1", "x2"),
) -> pd.DataFrame:
    """
    Bivariate Gumbel copula: UPPER tail dependence, none in the lower tail
    -- the mirror image of Clayton, and the right shape for things that
    spike together (volatility, claim severities, peak loads).

    Sampled via the Archimedean frailty construction with a positive stable
    mixing variable.

    theta: Gumbel parameter, >= 1. ``lambda_U = 2 - 2^(1/theta)``.
    """
    if theta < 1:
        raise ValueError("Gumbel theta must be >= 1")
    rng = np.random.default_rng(seed)
    alpha = 1.0 / theta
    # Positive stable variable with index alpha (Chambers-Mallows-Stuck).
    u_unif = rng.uniform(size=n) * np.pi
    w = rng.exponential(size=n)
    s = (np.sin(alpha * u_unif) / np.sin(u_unif) ** (1 / alpha)) * (
        np.sin((1 - alpha) * u_unif) / w
    ) ** ((1 - alpha) / alpha)
    e = rng.exponential(size=(n, 2))
    u = np.exp(-((e / s[:, None]) ** alpha))
    u = np.clip(u, 1e-9, 1 - 1e-9)
    return pd.DataFrame(_marginal(u, marginals, df), columns=list(columns))


# ---------------------------------------------------------------------
# elliptical copulas, any dimension
# ---------------------------------------------------------------------
def _corr_from_spec(spec, d: int) -> np.ndarray:
    """Accept a scalar (equicorrelation) or a full matrix."""
    if np.isscalar(spec):
        R = np.full((d, d), float(spec))
        np.fill_diagonal(R, 1.0)
        return R
    R = np.asarray(spec, dtype=float)
    if R.shape != (d, d):
        raise ValueError(f"correlation matrix must be {(d, d)}, got {R.shape}")
    return R


def make_gaussian_copula(
    n: int = 2000, rho=0.7, d: int = 2, marginals: str = "t", df: int = 4,
    seed: int = 0, columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Gaussian copula in `d` dimensions, optionally with fat-tailed marginals.

    The instructive case: however high `rho` is, the tail dependence is
    exactly ZERO. Fat marginals make the data look tail-heavy while the
    joint tails stay asymptotically independent, which is precisely the
    trap this library exists to detect.

    rho: scalar equicorrelation, or a full (d, d) correlation matrix.
    """
    rng = np.random.default_rng(seed)
    R = _corr_from_spec(rho, d)
    z = rng.multivariate_normal(np.zeros(d), R, size=n)
    u = np.clip(norm.cdf(z), 1e-9, 1 - 1e-9)
    cols = columns or [f"x{i + 1}" for i in range(d)]
    return pd.DataFrame(_marginal(u, marginals, df), columns=cols)


def make_t_copula(
    n: int = 2000, rho=0.7, dof: int = 4, d: int = 2, marginals: str = "t",
    df: int = 4, seed: int = 0, columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Student-t copula in `d` dimensions: SYMMETRIC tail dependence in both
    tails, controlled by `dof`. Lower `dof` means fatter joint tails; as
    ``dof -> inf`` it converges to the Gaussian copula.

    The standard workhorse for multivariate financial returns, and the
    natural parametric baseline for a neural copula to be measured against.

    rho: scalar equicorrelation or a full (d, d) matrix.
    dof: degrees of freedom of the COPULA (distinct from `df`, the marginal
        degrees of freedom when ``marginals="t"``).
    """
    rng = np.random.default_rng(seed)
    R = _corr_from_spec(rho, d)
    z = rng.multivariate_normal(np.zeros(d), R, size=n)
    chi = rng.chisquare(dof, size=n)
    y = z / np.sqrt(chi / dof)[:, None]
    u = np.clip(student_t.cdf(y, dof), 1e-9, 1 - 1e-9)
    cols = columns or [f"x{i + 1}" for i in range(d)]
    return pd.DataFrame(_marginal(u, marginals, df), columns=cols)


# ---------------------------------------------------------------------
# harder structures
# ---------------------------------------------------------------------
def make_mixed_regime(
    n: int = 3000, rho_calm: float = 0.2, rho_crisis: float = 0.9,
    crisis_frac: float = 0.15, d: int = 2, seed: int = 0,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    A two-regime mixture: mostly a calm, weakly correlated Gaussian, with a
    `crisis_frac` minority drawn at much higher correlation and larger scale.

    The resulting copula is in no standard parametric family. Its dependence
    is asymmetric and concentrated in the joint tails, which is roughly what
    real markets look like and is hard for a vine of parametric pair copulas
    to capture -- a good stress test where a flow should genuinely win.

    Rows are shuffled, so the regime is a latent variable, not a time block.
    """
    rng = np.random.default_rng(seed)
    n_crisis = int(n * crisis_frac)
    n_calm = n - n_crisis
    calm = rng.multivariate_normal(np.zeros(d), _corr_from_spec(rho_calm, d), size=n_calm)
    crisis = rng.multivariate_normal(
        np.zeros(d), _corr_from_spec(rho_crisis, d), size=n_crisis
    ) * 2.5
    data = np.vstack([calm, crisis])
    rng.shuffle(data, axis=0)
    cols = columns or [f"x{i + 1}" for i in range(d)]
    return pd.DataFrame(data, columns=cols)


def make_asset_panel(
    n: int = 2500, seed: int = 0, dof: int = 5,
) -> pd.DataFrame:
    """
    A five-column stand-in for a small multi-asset book: two blocks of
    strongly related instruments, a weak cross-block link, and one nearly
    independent series, all with fat tails and t-copula tail dependence.

    Built as a t copula with a block correlation matrix, so it has known
    structure while still looking like something you would actually be
    handed. Used by the examples and the README.

    Columns: ``grain_A``, ``grain_B``, ``metal_A``, ``metal_B``, ``fx_A``.
    """
    R = np.array([
        [1.00, 0.75, 0.10, 0.05, 0.00],
        [0.75, 1.00, 0.05, 0.10, 0.00],
        [0.10, 0.05, 1.00, 0.65, 0.20],
        [0.05, 0.10, 0.65, 1.00, 0.20],
        [0.00, 0.00, 0.20, 0.20, 1.00],
    ])
    return make_t_copula(
        n=n, rho=R, dof=dof, d=5, marginals="t", df=dof, seed=seed,
        columns=["grain_A", "grain_B", "metal_A", "metal_B", "fx_A"],
    )
