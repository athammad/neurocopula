"""Tests for the marginal transforms."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import kstest

from neurocopula.marginals import (
    EmpiricalCopulaTransform,
    StandardizeTransform,
    pseudo_observations,
)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    return np.column_stack([
        rng.standard_t(4, size=500) * 3 + 10,   # fat-tailed, shifted, scaled
        rng.exponential(2.0, size=500),          # skewed, positive only
        rng.normal(size=500),
    ])


class TestPseudoObservations:
    def test_strictly_inside_unit_interval(self, data):
        # The n+1 denominator exists precisely so norm.ppf never sees 0 or 1.
        u = pseudo_observations(data)
        assert u.shape == data.shape
        assert (u > 0).all() and (u < 1).all()

    def test_marginally_uniform(self, data):
        u = pseudo_observations(data)
        for j in range(u.shape[1]):
            assert kstest(u[:, j], "uniform").statistic < 0.05

    def test_preserves_ranks(self, data):
        # A copula transform must be strictly monotone per column, or it
        # would change the dependence structure it is supposed to isolate.
        u = pseudo_observations(data)
        for j in range(data.shape[1]):
            assert (np.argsort(data[:, j]) == np.argsort(u[:, j])).all()

    def test_rejects_1d(self):
        with pytest.raises(ValueError, match="2-D"):
            pseudo_observations(np.arange(10))


class TestStandardizeTransform:
    def test_roundtrip(self, data):
        t = StandardizeTransform().fit(data)
        assert np.allclose(t.inverse(t.forward(data)), data)

    def test_output_is_standardized(self, data):
        z = StandardizeTransform().fit(data).forward(data)
        assert np.allclose(z.mean(axis=0), 0, atol=1e-9)
        assert np.allclose(z.std(axis=0), 1, atol=1e-9)

    def test_log_det_jacobian_is_exact_and_constant(self, data):
        t = StandardizeTransform().fit(data)
        ldj = t.log_det_jacobian(data)
        assert np.allclose(ldj, ldj[0])
        assert np.isclose(ldj[0], -np.sum(np.log(data.std(axis=0))))

    def test_constant_column_does_not_divide_by_zero(self):
        d = np.column_stack([np.ones(50), np.arange(50.0)])
        z = StandardizeTransform().fit(d).forward(d)
        assert np.isfinite(z).all()

    def test_raises_before_fit(self, data):
        with pytest.raises(RuntimeError, match="not fitted"):
            StandardizeTransform().forward(data)


class TestEmpiricalCopulaTransform:
    def test_roundtrip(self, data):
        t = EmpiricalCopulaTransform().fit(data)
        assert np.allclose(t.inverse(t.forward(data)), data, atol=1e-8)

    def test_forward_is_marginally_standard_normal(self, data):
        # This is the whole point of copula mode: after the transform every
        # column is N(0,1), so the flow can only learn dependence.
        z = EmpiricalCopulaTransform().fit(data).forward(data)
        for j in range(z.shape[1]):
            assert kstest(z[:, j], "norm").statistic < 0.05

    def test_uniform_roundtrip(self, data):
        t = EmpiricalCopulaTransform().fit(data)
        assert np.allclose(t.from_uniform(t.to_uniform(data)), data, atol=1e-8)

    def test_extrapolation_is_clamped_to_observed_range(self, data):
        # An empirical CDF cannot extrapolate; the documented behaviour is a
        # clamp at the observed extremes, not a NaN or a runaway value.
        t = EmpiricalCopulaTransform().fit(data)
        far = np.full((1, data.shape[1]), 1e9)
        out = t.inverse(t.forward(far))
        assert np.all(out <= data.max(axis=0) + 1e-6)
        assert np.isfinite(out).all()

    def test_log_det_jacobian_finite(self, data):
        ldj = EmpiricalCopulaTransform().fit(data).log_det_jacobian(data)
        assert ldj.shape == (len(data),)
        assert np.isfinite(ldj).all()

    def test_density_none_disables_jacobian_with_actionable_message(self, data):
        t = EmpiricalCopulaTransform(marginal_density="none").fit(data)
        with pytest.raises(RuntimeError, match="copula_log_prob"):
            t.log_det_jacobian(data)

    def test_marginal_log_pdf_shape(self, data):
        t = EmpiricalCopulaTransform().fit(data)
        assert t.marginal_log_pdf(data).shape == data.shape

    def test_rejects_bad_density_option(self):
        with pytest.raises(ValueError, match="marginal_density"):
            EmpiricalCopulaTransform(marginal_density="spline")

    def test_n_features(self, data):
        assert EmpiricalCopulaTransform().fit(data).n_features == data.shape[1]
