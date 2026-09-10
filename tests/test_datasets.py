"""
Tests for the synthetic generators.

These matter more than they look: the benchmark's ground-truth claims rest
entirely on these generators actually producing the copulas they name. A
silent bug here would make every "error against truth" number meaningless,
so each generator is checked against its own closed-form theory.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import kendalltau, kstest

from neurocopula import datasets as ds
from neurocopula.metrics import tail_dependence

BIG = 20_000  # large enough that tail estimates at q = 0.01 are meaningful


class TestClosedForms:
    @pytest.mark.parametrize("theta,expected", [(1.0, 0.5), (2.0, 2 ** -0.5), (4.0, 2 ** -0.25)])
    def test_clayton_lambda(self, theta, expected):
        t = ds.theoretical_tail_dependence("clayton", theta=theta)
        assert np.isclose(t["lower"], expected)
        assert t["upper"] == 0.0

    @pytest.mark.parametrize("theta", [1.5, 2.0, 3.0])
    def test_gumbel_lambda(self, theta):
        t = ds.theoretical_tail_dependence("gumbel", theta=theta)
        assert np.isclose(t["upper"], 2 - 2 ** (1 / theta))
        assert t["lower"] == 0.0

    def test_gaussian_has_no_tail_dependence(self):
        # True for every rho < 1 -- the fact that motivates the library.
        t = ds.theoretical_tail_dependence("gaussian", rho=0.99)
        assert t["lower"] == 0.0 and t["upper"] == 0.0

    def test_t_copula_is_symmetric_and_positive(self):
        t = ds.theoretical_tail_dependence("t", rho=0.7, df=4)
        assert t["lower"] == t["upper"] > 0

    def test_t_copula_converges_to_gaussian_as_df_grows(self):
        lam = [ds.theoretical_tail_dependence("t", rho=0.7, df=n)["lower"]
               for n in (2, 8, 50, 500)]
        assert lam == sorted(lam, reverse=True)
        assert lam[-1] < 0.05

    def test_kendall_tau_formulas(self):
        assert np.isclose(ds.theoretical_kendall_tau("clayton", theta=2.0), 0.5)
        assert np.isclose(ds.theoretical_kendall_tau("gumbel", theta=2.0), 0.5)
        assert np.isclose(ds.theoretical_kendall_tau("gaussian", rho=0.0), 0.0)

    def test_unknown_family_raises(self):
        with pytest.raises(ValueError, match="unknown family"):
            ds.theoretical_tail_dependence("frank", theta=1.0)


class TestGeneratorsMatchTheory:
    def test_clayton_recovers_lower_tail_and_tau(self):
        d = ds.make_clayton(n=BIG, theta=2.0, seed=1)
        truth = ds.theoretical_tail_dependence("clayton", theta=2.0)
        assert abs(tail_dependence(d.x1, d.x2, q=0.01, tail="lower") - truth["lower"]) < 0.06
        assert tail_dependence(d.x1, d.x2, q=0.01, tail="upper") < 0.10
        assert abs(kendalltau(d.x1, d.x2).statistic - 0.5) < 0.02

    def test_gumbel_recovers_upper_tail_and_tau(self):
        d = ds.make_gumbel(n=BIG, theta=2.0, seed=1)
        truth = ds.theoretical_tail_dependence("gumbel", theta=2.0)
        assert abs(tail_dependence(d.x1, d.x2, q=0.01, tail="upper") - truth["upper"]) < 0.06
        assert tail_dependence(d.x1, d.x2, q=0.01, tail="lower") < 0.20
        assert abs(kendalltau(d.x1, d.x2).statistic - 0.5) < 0.02

    def test_clayton_and_gumbel_are_mirror_images(self):
        c = ds.make_clayton(n=BIG, theta=2.0, seed=1)
        g = ds.make_gumbel(n=BIG, theta=2.0, seed=1)
        c_asym = (tail_dependence(c.x1, c.x2, 0.02, "lower")
                  - tail_dependence(c.x1, c.x2, 0.02, "upper"))
        g_asym = (tail_dependence(g.x1, g.x2, 0.02, "lower")
                  - tail_dependence(g.x1, g.x2, 0.02, "upper"))
        assert c_asym > 0.3 and g_asym < -0.3

    def test_gaussian_tail_dependence_decays_toward_zero(self):
        # Not zero at any finite q, but it must shrink as q does. This is
        # exactly the behaviour a tail-dependence CURVE is meant to reveal.
        d = ds.make_gaussian_copula(n=BIG, rho=0.7, seed=1)
        far = tail_dependence(d.x1, d.x2, q=0.005, tail="lower")
        near = tail_dependence(d.x1, d.x2, q=0.20, tail="lower")
        assert far < near

    def test_t_copula_tail_dependence_is_symmetric(self):
        d = ds.make_t_copula(n=BIG, rho=0.7, dof=4, seed=1)
        lo = tail_dependence(d.x1, d.x2, q=0.02, tail="lower")
        up = tail_dependence(d.x1, d.x2, q=0.02, tail="upper")
        assert abs(lo - up) < 0.10
        assert lo > 0.25

    def test_t_copula_dominates_gaussian_in_the_tail(self):
        t = ds.make_t_copula(n=BIG, rho=0.7, dof=3, seed=1)
        g = ds.make_gaussian_copula(n=BIG, rho=0.7, seed=1)
        assert (tail_dependence(t.x1, t.x2, 0.01, "lower")
                > tail_dependence(g.x1, g.x2, 0.01, "lower") + 0.1)


class TestGeneratorMechanics:
    @pytest.mark.parametrize("gen", [ds.make_clayton, ds.make_gumbel])
    def test_uniform_marginals_are_uniform(self, gen):
        d = gen(n=4000, marginals="uniform", seed=2)
        for c in d.columns:
            assert kstest(d[c], "uniform").statistic < 0.04

    @pytest.mark.parametrize("gen", [ds.make_clayton, ds.make_gumbel])
    def test_normal_marginals_are_normal(self, gen):
        d = gen(n=4000, marginals="normal", seed=2)
        for c in d.columns:
            assert kstest(d[c], "norm").statistic < 0.04

    def test_reproducible_given_seed(self):
        assert ds.make_clayton(n=200, seed=7).equals(ds.make_clayton(n=200, seed=7))
        assert not ds.make_clayton(n=200, seed=7).equals(ds.make_clayton(n=200, seed=8))

    @pytest.mark.parametrize("d", [2, 3, 5])
    def test_elliptical_generators_scale_in_dimension(self, d):
        assert ds.make_gaussian_copula(n=100, d=d).shape == (100, d)
        assert ds.make_t_copula(n=100, d=d).shape == (100, d)

    def test_full_correlation_matrix_is_honoured(self):
        R = np.array([[1.0, 0.9], [0.9, 1.0]])
        d = ds.make_gaussian_copula(n=8000, rho=R, seed=3)
        assert kendalltau(d.x1, d.x2).statistic > 0.65

    def test_bad_correlation_shape_raises(self):
        with pytest.raises(ValueError, match="correlation matrix"):
            ds.make_gaussian_copula(n=50, rho=np.eye(3), d=2)

    def test_invalid_parameters_raise(self):
        with pytest.raises(ValueError, match="theta"):
            ds.make_clayton(n=50, theta=0.0)
        with pytest.raises(ValueError, match="theta"):
            ds.make_gumbel(n=50, theta=0.5)
        with pytest.raises(ValueError, match="marginals"):
            ds.make_clayton(n=50, marginals="cauchy")

    def test_mixed_regime_is_more_tail_dependent_than_its_calm_regime(self):
        # The mixture's crisis component is what creates joint-tail mass; if
        # this failed, the "hard case" in the benchmark would not be hard.
        d = ds.make_mixed_regime(n=BIG, seed=1)
        g = ds.make_gaussian_copula(n=BIG, rho=0.2, seed=1)
        assert (tail_dependence(d.x1, d.x2, 0.02, "lower")
                > tail_dependence(g.x1, g.x2, 0.02, "lower"))

    def test_asset_panel_has_expected_block_structure(self):
        d = ds.make_asset_panel(n=6000, seed=1)
        assert list(d.columns) == ["grain_A", "grain_B", "metal_A", "metal_B", "fx_A"]
        within = kendalltau(d.grain_A, d.grain_B).statistic
        across = kendalltau(d.grain_A, d.metal_A).statistic
        assert within > across + 0.3
