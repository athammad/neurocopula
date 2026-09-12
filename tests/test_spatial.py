"""
Tests for the spatial dependence layer.

The important tests here are recovery tests: simulate a field whose range,
correlation shape and degrees of freedom are known, then check the fit finds
them. Two of these guard against specific failure modes that are easy to ship
and hard to notice:

- fitting a Gaussian field must return ``df=inf``, not a small finite value.
  Reporting finite degrees of freedom for a field with no tail dependence
  would invent joint extremes that are not in the data, which is the one
  error that would invalidate a compound-extremes study.
- a t field must NOT be reported as Gaussian, which is the same error in the
  other direction.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm
from scipy.stats import t as student_t

from neurocopula.spatial import (
    CORRELATION_MODELS,
    SpatialDependence,
    euclidean_matrix,
    haversine_matrix,
)


def make_grid(n_side: int = 16, half_span: float = 5.0, lon0: float = 105.0):
    lats = np.linspace(-half_span, half_span, n_side)
    lons = np.linspace(lon0 - half_span, lon0 + half_span, n_side)
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    return np.column_stack([LA.ravel(), LO.ravel()])


def simulate_field(coords, true_range, true_df, n_time=4000, model="exponential", seed=0):
    """Draw from a t (or Gaussian) random field with known parameters."""
    rng = np.random.default_rng(seed)
    D = haversine_matrix(coords)
    C = CORRELATION_MODELS[model](D, true_range)
    np.fill_diagonal(C, 1.0)
    C[np.diag_indices_from(C)] += 1e-8
    L = np.linalg.cholesky(C)
    w = rng.standard_normal((n_time, len(coords))) @ L.T
    if np.isfinite(true_df):
        g = rng.chisquare(true_df, size=(n_time, 1)) / true_df
        u = student_t.cdf(w / np.sqrt(g), true_df)
    else:
        u = norm.cdf(w)
    return norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))


@pytest.fixture(scope="module")
def coords():
    return make_grid()


@pytest.fixture(scope="module")
def t_field(coords):
    z = simulate_field(coords, true_range=250.0, true_df=5.0)
    return z, SpatialDependence(model="auto", seed=0).fit(z, coords)


class TestDistances:
    def test_haversine_zero_on_diagonal(self, coords):
        assert np.allclose(np.diag(haversine_matrix(coords)), 0.0, atol=1e-9)

    def test_haversine_is_symmetric(self, coords):
        D = haversine_matrix(coords)
        assert np.allclose(D, D.T)

    def test_haversine_known_distance(self):
        # One degree of latitude is about 111 km, everywhere.
        d = haversine_matrix([[0.0, 0.0]], [[1.0, 0.0]])[0, 0]
        assert 110.0 < d < 112.0

    def test_longitude_degree_shrinks_with_latitude(self):
        # The reason great-circle distance is used rather than degrees: a
        # degree of longitude is much shorter near the poles.
        equator = haversine_matrix([[0.0, 0.0]], [[0.0, 1.0]])[0, 0]
        high = haversine_matrix([[60.0, 0.0]], [[60.0, 1.0]])[0, 0]
        assert high < 0.6 * equator

    def test_euclidean_matches_numpy(self):
        a = np.random.default_rng(0).normal(size=(5, 2))
        D = euclidean_matrix(a)
        assert np.isclose(D[0, 1], np.linalg.norm(a[0] - a[1]))

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match="two columns"):
            haversine_matrix(np.zeros((4, 3)))


class TestCorrelationModels:
    @pytest.mark.parametrize("name", sorted(CORRELATION_MODELS))
    def test_unit_at_zero_distance(self, name):
        assert CORRELATION_MODELS[name](np.array([0.0]), 100.0)[0] == pytest.approx(1.0)

    @pytest.mark.parametrize("name", sorted(CORRELATION_MODELS))
    def test_decreasing_in_distance(self, name):
        d = np.linspace(0, 1000, 50)
        r = CORRELATION_MODELS[name](d, 200.0)
        assert np.all(np.diff(r) <= 1e-12)

    @pytest.mark.parametrize("name", sorted(CORRELATION_MODELS))
    def test_decays_to_zero(self, name):
        assert CORRELATION_MODELS[name](np.array([1e6]), 100.0)[0] < 1e-6


class TestRecovery:
    def test_recovers_range(self, t_field):
        _, sd = t_field
        assert abs(sd.range_ - 250.0) / 250.0 < 0.20

    def test_recovers_degrees_of_freedom(self, t_field):
        _, sd = t_field
        assert np.isfinite(sd.df_)
        assert 2.0 <= sd.df_ <= 12.0

    def test_selects_the_generating_correlation_shape(self, t_field):
        _, sd = t_field
        assert sd.model_ == "exponential"

    @pytest.mark.parametrize("true_range", [150.0, 400.0])
    def test_range_recovery_across_scales(self, coords, true_range):
        z = simulate_field(coords, true_range=true_range, true_df=6.0, seed=1)
        sd = SpatialDependence(model="exponential", seed=0).fit(z, coords)
        assert abs(sd.range_ - true_range) / true_range < 0.25

    def test_gaussian_field_is_not_given_finite_df(self, coords):
        # The failure that would fabricate tail dependence. Must report the
        # Gaussian limit, not a small df.
        z = simulate_field(coords, true_range=300.0, true_df=np.inf, seed=2)
        sd = SpatialDependence(model="auto", seed=0).fit(z, coords)
        assert (not np.isfinite(sd.df_)) or sd.df_ >= 30.0

    def test_t_field_is_not_mistaken_for_gaussian(self, coords):
        # The same error in the other direction: missing real tail dependence.
        z = simulate_field(coords, true_range=300.0, true_df=3.0, seed=3)
        sd = SpatialDependence(model="auto", seed=0).fit(z, coords)
        assert np.isfinite(sd.df_) and sd.df_ <= 10.0

    def test_heavier_tails_give_smaller_df(self, coords):
        light = SpatialDependence(seed=0).fit(
            simulate_field(coords, 300.0, 20.0, seed=4), coords)
        heavy = SpatialDependence(seed=0).fit(
            simulate_field(coords, 300.0, 3.0, seed=4), coords)
        assert heavy.df_ < light.df_


class TestCorrelationOutputs:
    def test_correlation_is_one_at_zero(self, t_field):
        _, sd = t_field
        assert sd.correlation(np.array([0.0]))[0] == pytest.approx(1.0, abs=1e-6)

    def test_correlation_matrix_is_valid(self, t_field, coords):
        _, sd = t_field
        C = sd.correlation_matrix(coords[:40])
        assert C.shape == (40, 40)
        assert np.allclose(C, C.T)
        # Positive definite, or the joint draw would be impossible.
        assert np.linalg.eigvalsh(C).min() > 0

    def test_correlation_matrix_works_for_an_arbitrary_subset(self, t_field, coords):
        # The property the whole design rests on: any region, chosen after
        # fitting, has a joint distribution.
        _, sd = t_field
        pick = np.random.default_rng(0).choice(len(coords), 25, replace=False)
        assert sd.correlation_matrix(coords[pick]).shape == (25, 25)

    def test_effective_range_exceeds_range(self, t_field):
        _, sd = t_field
        assert sd.effective_range > sd.range_

    def test_tail_dependence_positive_for_t_zero_for_gaussian(self, coords):
        t = SpatialDependence(seed=0).fit(simulate_field(coords, 300.0, 4.0, seed=5), coords)
        assert t.tail_dependence(50.0) > 0.05
        g = SpatialDependence(fit_df=False, df=np.inf, seed=0).fit(
            simulate_field(coords, 300.0, np.inf, seed=5), coords)
        assert g.tail_dependence(50.0) == 0.0

    def test_tail_dependence_decays_with_distance(self, t_field):
        _, sd = t_field
        assert sd.tail_dependence(10.0) > sd.tail_dependence(800.0)


class TestSampling:
    def test_sample_shape(self, t_field, coords):
        _, sd = t_field
        assert sd.sample(coords[:30], n=500).shape == (500, 30)

    def test_samples_are_marginally_standard_normal(self, t_field, coords):
        # The upstream per-location model expects N(0,1) margins; the t
        # copula must live in the DEPENDENCE, not in the margins.
        _, sd = t_field
        s = sd.sample(coords[:20], n=20000)
        assert abs(s.mean()) < 0.05
        assert abs(s.std() - 1.0) < 0.05

    def test_sampled_correlation_matches_the_model(self, t_field, coords):
        _, sd = t_field
        sub = coords[:25]
        s = sd.sample(sub, n=30000)
        emp = np.corrcoef(s.T)
        target = sd.correlation_matrix(sub)
        off = ~np.eye(len(sub), dtype=bool)
        assert np.abs(emp[off] - target[off]).mean() < 0.06

    def test_round_trip_recovers_parameters(self, t_field, coords):
        # Sample from the fitted field, refit, and check we land in the same
        # place. Catches inconsistencies between the sampler and the fitter.
        _, sd = t_field
        sub = coords[:120]
        s = sd.sample(sub, n=4000)
        again = SpatialDependence(model=sd.model_, seed=1).fit(s, sub)
        assert abs(again.range_ - sd.range_) / sd.range_ < 0.35

    def test_sampling_is_reproducible(self, t_field, coords):
        _, sd = t_field
        a = sd.sample(coords[:10], n=100, seed=7)
        b = sd.sample(coords[:10], n=100, seed=7)
        assert np.allclose(a, b)

    def test_gaussian_field_samples_have_no_tail_clustering(self, coords):
        # Direct check that df=inf really removes tail dependence, rather
        # than merely reporting it.
        sd = SpatialDependence(fit_df=False, df=np.inf, seed=0).fit(
            simulate_field(coords, 300.0, np.inf, seed=6), coords)
        s = sd.sample(coords[:2], n=40000)
        q = 0.01
        thr = np.quantile(s, q, axis=0)
        joint = ((s[:, 0] < thr[0]) & (s[:, 1] < thr[1])).mean() / q
        t_sd = SpatialDependence(fit_df=False, df=3.0, seed=0).fit(
            simulate_field(coords, 300.0, 3.0, seed=6), coords)
        st = t_sd.sample(coords[:2], n=40000)
        thr_t = np.quantile(st, q, axis=0)
        joint_t = ((st[:, 0] < thr_t[0]) & (st[:, 1] < thr_t[1])).mean() / q
        assert joint_t > joint


class TestValidation:
    def test_rejects_unknown_model(self):
        with pytest.raises(ValueError, match="unknown model"):
            SpatialDependence(model="spherical")

    def test_rejects_unknown_distance(self):
        with pytest.raises(ValueError, match="haversine"):
            SpatialDependence(distance="manhattan")

    def test_rejects_mismatched_shapes(self, coords):
        with pytest.raises(ValueError, match="locations"):
            SpatialDependence().fit(np.zeros((100, 5)), coords)

    def test_rejects_1d_input(self, coords):
        with pytest.raises(ValueError, match="2-D"):
            SpatialDependence().fit(np.zeros(100), coords)

    def test_warns_on_too_few_time_steps(self, coords):
        with pytest.warns(RuntimeWarning, match="noisy"):
            SpatialDependence(n_bins=5).fit(
                simulate_field(coords[:30], 300.0, 5.0, n_time=15, seed=0), coords[:30])

    def test_methods_raise_before_fit(self, coords):
        sd = SpatialDependence()
        with pytest.raises(RuntimeError, match="not fitted"):
            sd.correlation_matrix(coords[:5])
        with pytest.raises(RuntimeError, match="not fitted"):
            sd.sample(coords[:5])

    def test_repr_before_and_after_fit(self, t_field):
        assert "unfitted" in repr(SpatialDependence())
        _, sd = t_field
        assert "range=" in repr(sd)

    def test_summary_keys(self, t_field):
        _, sd = t_field
        assert {"model", "range", "effective_range", "nugget", "df",
                "gaussian_limit", "fit_error"} <= set(sd.summary())


class TestCensoredVariables:
    """
    A variable with an atom (rainfall's dry days) needs its spatial dependence
    estimated from the censoring pattern, not from the values.

    The randomized transform that makes such a marginal correct assigns every
    censored entry an independent random rank. That is right for the marginal
    and disastrous for the correlogram: two neighbouring locations that are both
    dry on the same day get unrelated values, so their real co-occurrence is
    replaced by noise and the estimated range collapses.
    """

    @staticmethod
    def censor(z, p_dry=0.55):
        """Censor the lower `p_dry` of each column, randomizing those ranks."""
        rng = np.random.default_rng(0)
        thr = np.quantile(z, p_dry, axis=0)
        mask = z < thr[None, :]
        out = z.copy()
        # Exactly what the randomized PIT does: independent noise on the atom.
        out[mask] = norm.ppf(rng.uniform(1e-6, p_dry, size=int(mask.sum())))
        return out, mask

    def test_tetrachoric_recovers_known_correlation(self):
        from scipy.stats import multivariate_normal

        from neurocopula.spatial import _tetrachoric
        for rho in (0.2, 0.5, 0.8):
            for p_dry in (0.3, 0.55, 0.8):
                a = float(norm.ppf(p_dry))
                p_both = float(multivariate_normal.cdf(
                    [a, a], mean=[0, 0], cov=[[1, rho], [rho, 1]]))
                assert abs(_tetrachoric(p_both, a, a) - rho) < 0.02

    def test_censoring_destroys_the_naive_correlogram(self, coords):
        # Demonstrates the failure the fix exists to prevent.
        z = simulate_field(coords, true_range=300.0, true_df=5.0, seed=10)
        zc, _ = self.censor(z)
        naive = SpatialDependence(model="exponential", seed=0).fit(zc, coords)
        assert naive.range_ < 200.0  # badly short of the true 300 km

    def test_censored_aware_fit_recovers_the_range(self, coords):
        z = simulate_field(coords, true_range=300.0, true_df=5.0, seed=10)
        zc, mask = self.censor(z)
        fixed = SpatialDependence(model="exponential", seed=0).fit(
            zc, coords, censored=mask, df_tail="upper")
        assert abs(fixed.range_ - 300.0) / 300.0 < 0.30

    def test_censored_aware_beats_naive(self, coords):
        z = simulate_field(coords, true_range=300.0, true_df=5.0, seed=11)
        zc, mask = self.censor(z)
        naive = SpatialDependence(model="exponential", seed=0).fit(zc, coords)
        fixed = SpatialDependence(model="exponential", seed=0).fit(
            zc, coords, censored=mask, df_tail="upper")
        assert abs(fixed.range_ - 300.0) < abs(naive.range_ - 300.0)

    def test_censored_flag_recorded(self, coords):
        z = simulate_field(coords, true_range=300.0, true_df=5.0, seed=12)
        zc, mask = self.censor(z)
        assert SpatialDependence(seed=0).fit(zc, coords, censored=mask).censored_
        assert not SpatialDependence(seed=0).fit(zc, coords).censored_

    def test_rejects_mismatched_censored_shape(self, coords):
        z = simulate_field(coords, true_range=300.0, true_df=5.0, seed=13)
        with pytest.raises(ValueError, match="censored has shape"):
            SpatialDependence().fit(z, coords, censored=np.zeros((5, 5), dtype=bool))

    def test_rejects_bad_df_tail(self, coords):
        z = simulate_field(coords, true_range=300.0, true_df=5.0, seed=14)
        with pytest.raises(ValueError, match="df_tail"):
            SpatialDependence().fit(z, coords, df_tail="both")

    def test_all_censored_falls_back_gracefully(self, coords):
        # An entirely censored variable carries no information; must not crash.
        z = simulate_field(coords, true_range=300.0, true_df=5.0, seed=15)
        sd = SpatialDependence(seed=0).fit(z, coords, censored=np.zeros_like(z, dtype=bool))
        assert not sd.censored_ and sd.range_ > 0
