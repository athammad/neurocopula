"""Tests for the metric functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neurocopula import datasets as ds
from neurocopula import metrics as M


@pytest.fixture(scope="module")
def clayton():
    return ds.make_clayton(n=8000, theta=2.0, seed=0)


@pytest.fixture(scope="module")
def independent():
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(size=(8000, 2)), columns=["x1", "x2"])


class TestTailDependence:
    def test_comonotone_pair_is_one(self):
        x = np.random.default_rng(0).normal(size=3000)
        assert M.tail_dependence(x, x, q=0.05) == pytest.approx(1.0, abs=1e-9)

    def test_independent_pair_is_near_q(self):
        # Under independence, P(Y in tail | X in tail) = P(Y in tail) = q.
        rng = np.random.default_rng(0)
        x, y = rng.normal(size=20000), rng.normal(size=20000)
        assert M.tail_dependence(x, y, q=0.05) == pytest.approx(0.05, abs=0.02)

    def test_invariant_to_monotone_marginal_transform(self, clayton):
        # Rank-based by construction; this is what lets the same metric be
        # applied to data and model samples with different marginals.
        raw = M.tail_dependence(clayton.x1, clayton.x2, q=0.05)
        shifted = M.tail_dependence(clayton.x1 * 3 + 7,
                                    np.exp(clayton.x2 / 4), q=0.05)
        assert raw == pytest.approx(shifted, abs=1e-9)

    def test_lower_and_upper_differ_for_clayton(self, clayton):
        assert (M.tail_dependence(clayton.x1, clayton.x2, 0.02, "lower")
                > M.tail_dependence(clayton.x1, clayton.x2, 0.02, "upper") + 0.4)

    def test_rejects_bad_arguments(self, clayton):
        with pytest.raises(ValueError, match="same length"):
            M.tail_dependence(clayton.x1, clayton.x2[:10])
        with pytest.raises(ValueError, match="q must be"):
            M.tail_dependence(clayton.x1, clayton.x2, q=0.9)
        with pytest.raises(ValueError, match="tail must be"):
            M.tail_dependence(clayton.x1, clayton.x2, tail="middle")


class TestTailDependenceCurve:
    def test_shape_and_range(self, clayton):
        c = M.tail_dependence_curve(clayton.x1, clayton.x2)
        assert list(c.columns) == ["q", "lambda"]
        assert ((c["lambda"] >= 0) & (c["lambda"] <= 1)).all()

    def test_clayton_curve_stays_high_into_the_tail(self, clayton):
        # A genuinely tail-dependent copula plateaus rather than decaying.
        c = M.tail_dependence_curve(clayton.x1, clayton.x2, quantiles=[0.01, 0.05, 0.2])
        assert c["lambda"].min() > 0.5

    def test_gaussian_curve_decays(self):
        g = ds.make_gaussian_copula(n=40000, rho=0.7, seed=0)
        c = M.tail_dependence_curve(g.x1, g.x2, quantiles=[0.005, 0.25])
        assert c["lambda"].iloc[0] < c["lambda"].iloc[-1]


class TestCorrelationMatrices:
    def test_kendall_matrix_is_symmetric_with_unit_diagonal(self, clayton):
        m = M.kendall_tau_matrix(clayton)
        assert np.allclose(m.values, m.values.T)
        assert np.allclose(np.diag(m.values), 1.0)

    def test_kendall_matrix_recovers_known_tau(self, clayton):
        assert M.kendall_tau_matrix(clayton).loc["x1", "x2"] == pytest.approx(0.5, abs=0.03)

    @pytest.mark.parametrize("d", [2, 3, 5])
    def test_spearman_matrix_shape(self, d):
        frame = ds.make_gaussian_copula(n=400, d=d, seed=0)
        m = M.spearman_rho_matrix(frame)
        assert m.shape == (d, d)
        assert np.allclose(np.diag(m.values), 1.0)

    def test_asymmetry_positive_for_clayton_zero_for_gaussian(self, clayton):
        g = ds.make_gaussian_copula(n=20000, rho=0.7, seed=0)
        assert M.upper_lower_asymmetry(clayton.x1, clayton.x2, q=0.05) > 0.3
        assert abs(M.upper_lower_asymmetry(g.x1, g.x2, q=0.05)) < 0.12


class TestTwoSampleDistances:
    def test_energy_distance_is_near_zero_for_same_distribution(self):
        a = ds.make_clayton(n=3000, seed=1)
        b = ds.make_clayton(n=3000, seed=2)
        assert M.energy_distance(a, b) < 0.02

    def test_energy_distance_detects_a_different_copula(self):
        a = ds.make_clayton(n=3000, seed=1)
        c = ds.make_gaussian_copula(n=3000, rho=0.7, seed=1)
        assert M.energy_distance(a, c) > M.energy_distance(a, ds.make_clayton(n=3000, seed=2))

    def test_energy_distance_of_a_sample_with_itself_is_zero(self):
        a = ds.make_clayton(n=800, seed=1)
        assert M.energy_distance(a, a) == pytest.approx(0.0, abs=1e-9)

    def test_mmd_behaves_like_energy_distance(self):
        a = ds.make_clayton(n=3000, seed=1)
        b = ds.make_clayton(n=3000, seed=2)
        c = ds.make_gaussian_copula(n=3000, rho=0.7, seed=1)
        assert M.mmd_rbf(a, b) < M.mmd_rbf(a, c)

    def test_mmd_is_non_negative(self):
        a = ds.make_clayton(n=600, seed=1)
        assert M.mmd_rbf(a, a) >= -1e-12

    def test_column_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="same number of columns"):
            M.energy_distance(ds.make_clayton(n=100), ds.make_gaussian_copula(n=100, d=3))

    def test_ks_statistic_is_small_for_same_marginals(self):
        a = ds.make_clayton(n=4000, seed=1)
        b = ds.make_clayton(n=4000, seed=2)
        assert (M.ks_statistic_per_column(a, b) < 0.05).all()

    def test_ks_statistic_detects_shifted_marginals(self):
        a = ds.make_clayton(n=4000, seed=1)
        assert (M.ks_statistic_per_column(a, a + 3.0) > 0.2).all()


class TestErrorsAgainstTruth:
    def test_tail_dependence_error_against_clayton_theory(self):
        d = ds.make_clayton(n=30000, theta=2.0, seed=0)
        truth = ds.theoretical_tail_dependence("clayton", theta=2.0)
        e = M.tail_dependence_error(d, truth, q=0.02)
        assert e["lambda_L_abs_error"] < 0.10
        assert e["lambda_U_abs_error"] < 0.10
        assert e["lambda_L_true"] == truth["lower"]

    def test_kendall_tau_error(self):
        d = ds.make_clayton(n=20000, theta=2.0, seed=0)
        assert M.kendall_tau_error(d, ds.theoretical_kendall_tau("clayton", theta=2.0)) < 0.02
