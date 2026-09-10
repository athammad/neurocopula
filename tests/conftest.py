"""
Shared fixtures.

Model fits are session-scoped: fitting a flow takes seconds, and the tests
only ever read from the fitted models, so refitting per test would multiply
the suite's runtime for no added coverage. Any test that mutates a model
must build its own.
"""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")  # no display in CI; must be set before pyplot is imported

from neurocopula import NeuroCopula, datasets  # noqa: E402

# Small and fast on purpose. These sizes are enough to check that behaviour is
# correct, not that accuracy is high -- accuracy is the benchmark's job.
FAST_FIT = dict(epochs=120, patience=30, verbose=False)


@pytest.fixture(scope="session")
def clayton_df():
    """Bivariate Clayton: strong lower-tail dependence, none upper."""
    return datasets.make_clayton(n=1200, theta=2.0, seed=0)


@pytest.fixture(scope="session")
def gaussian_df():
    """Bivariate Gaussian copula: correlated but tail-INdependent."""
    return datasets.make_gaussian_copula(n=1200, rho=0.7, d=2, seed=0)


@pytest.fixture(scope="session")
def panel_df():
    """Five-column block-structured panel, for multivariate paths."""
    return datasets.make_asset_panel(n=800, seed=0)


@pytest.fixture(scope="session")
def fitted_copula(clayton_df):
    """A fitted copula-mode model (the library's default mode)."""
    return NeuroCopula(transforms=3, hidden_features=(32, 32), copula_mode=True,
                       seed=0).fit(clayton_df, **FAST_FIT)


@pytest.fixture(scope="session")
def fitted_joint(clayton_df):
    """A fitted full-joint model (copula_mode=False)."""
    return NeuroCopula(transforms=3, hidden_features=(32, 32), copula_mode=False,
                       seed=0).fit(clayton_df, **FAST_FIT)


@pytest.fixture(scope="session")
def fitted_panel(panel_df):
    """A fitted 5-D model in coupling sampling mode."""
    return NeuroCopula(transforms=3, hidden_features=(32, 32), copula_mode=True,
                       sampling_mode="coupling", seed=0).fit(panel_df, **FAST_FIT)


@pytest.fixture(scope="session")
def fitted_vine(clayton_df):
    """A fitted vine baseline, skipped when pyvinecopulib is unavailable."""
    pv = pytest.importorskip("pyvinecopulib")  # noqa: F841
    from neurocopula import VineCopula
    return VineCopula(family_set="parametric", seed=0).fit(clayton_df)
