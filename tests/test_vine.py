"""
Tests for the vine-copula baseline.

Two things are being checked. First, that the wrapper is faithful -- it must
present exactly the `NeuroCopula` interface, or the benchmark is comparing
call conventions rather than models. Second, that `pyvinecopulib` is being
driven correctly, verified by checking it recovers the copula family it was
given (it should pick Clayton for Clayton data).
"""

from __future__ import annotations

import numpy as np
import pytest

from neurocopula import datasets as ds
from neurocopula.metrics import tail_dependence

pytestmark = pytest.mark.vine
pytest.importorskip("pyvinecopulib")

from neurocopula.vine import FAMILY_SETS, VineCopula  # noqa: E402


class TestConstruction:
    def test_rejects_unknown_family_set(self):
        with pytest.raises(ValueError, match="unknown family_set"):
            VineCopula(family_set="archimedean-only")

    @pytest.mark.parametrize("name", sorted(FAMILY_SETS))
    def test_named_family_sets_all_fit(self, name, clayton_df):
        v = VineCopula(family_set=name).fit(clayton_df)
        assert v.vine is not None

    def test_repr_before_and_after_fit(self, clayton_df):
        v = VineCopula()
        assert "unfitted" in repr(v)
        assert "npars" in repr(v.fit(clayton_df))

    def test_methods_raise_before_fit(self):
        v = VineCopula()
        with pytest.raises(RuntimeError, match="not fitted"):
            v.sample(5)
        with pytest.raises(RuntimeError, match="not fitted"):
            v.copula_log_prob(np.zeros((2, 2)))

    def test_rejects_non_dataframe(self):
        with pytest.raises(TypeError, match="DataFrame"):
            VineCopula().fit(np.zeros((50, 2)))

    def test_rejects_nan(self, clayton_df):
        bad = clayton_df.copy()
        bad.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            VineCopula().fit(bad)


class TestFamilySelection:
    def test_selects_clayton_for_clayton_data(self):
        # The strongest evidence the wrapper drives pyvinecopulib correctly.
        v = VineCopula(family_set="parametric").fit(ds.make_clayton(n=3000, theta=2.0, seed=0))
        assert v.pair_families()[0][0] == "clayton"

    def test_selects_gumbel_for_gumbel_data(self):
        v = VineCopula(family_set="parametric").fit(ds.make_gumbel(n=3000, theta=2.0, seed=0))
        assert v.pair_families()[0][0] in ("gumbel", "joe")

    def test_selects_gaussian_or_student_for_elliptical_data(self):
        v = VineCopula(family_set="parametric").fit(
            ds.make_gaussian_copula(n=3000, rho=0.7, seed=0))
        assert v.pair_families()[0][0] in ("gaussian", "student", "frank")

    def test_restricted_family_set_is_respected(self):
        v = VineCopula(family_set="gaussian").fit(ds.make_clayton(n=1500, seed=0))
        assert v.pair_families()[0][0] == "gaussian"

    def test_recovers_kendall_tau(self):
        v = VineCopula(family_set="parametric").fit(ds.make_clayton(n=4000, theta=2.0, seed=0))
        assert abs(v.taus()[0][0] - 0.5) < 0.05

    def test_five_dimensional_vine_has_expected_tree_count(self, panel_df):
        v = VineCopula().fit(panel_df)
        assert len(v.pair_families()) == 4  # d - 1 trees for d = 5
        assert v.fit_report_["n_features"] == 5

    def test_truncation_reduces_the_tree_count(self, panel_df):
        assert len(VineCopula(trunc_lvl=2).fit(panel_df).pair_families()) == 2

    def test_summary_is_a_readable_string(self, clayton_df):
        assert len(VineCopula().fit(clayton_df).summary()) > 0


class TestInterfaceParity:
    """Every method the benchmark calls on both models must behave the same."""

    def test_sample_shape_and_columns(self, fitted_vine, clayton_df):
        s = fitted_vine.sample(300)
        assert s.shape == (300, 2)
        assert list(s.columns) == list(clayton_df.columns)
        assert np.isfinite(s.values).all()

    def test_sample_accepts_and_ignores_batch_size(self, fitted_vine):
        # Present purely so it is drop-in interchangeable with NeuroCopula.
        assert len(fitted_vine.sample(100, batch_size=10)) == 100

    def test_sample_uniform_in_unit_square(self, fitted_vine):
        u = fitted_vine.sample_uniform(2000)
        assert ((u.values > 0) & (u.values < 1)).all()

    def test_copula_log_prob_shape(self, fitted_vine, clayton_df):
        lp = fitted_vine.copula_log_prob(clayton_df)
        assert lp.shape == (len(clayton_df),) and np.isfinite(lp).all()

    def test_log_prob_equals_copula_plus_marginal_jacobian(self, fitted_vine, clayton_df):
        lp = fitted_vine.log_prob(clayton_df)
        expected = (fitted_vine.copula_log_prob(clayton_df)
                    + fitted_vine.marginals_.log_det_jacobian(clayton_df.values))
        assert np.allclose(lp, expected)

    def test_score_matches_mean_log_prob(self, fitted_vine, clayton_df):
        assert fitted_vine.score(clayton_df, copula=True) == pytest.approx(
            float(np.mean(fitted_vine.copula_log_prob(clayton_df))))

    def test_positive_copula_loglik_on_dependent_data(self, fitted_vine, clayton_df):
        assert fitted_vine.score(clayton_df, copula=True) > 0.1

    def test_aic_and_bic_are_finite_and_ordered(self, fitted_vine, clayton_df):
        aic = fitted_vine.aic(clayton_df)
        bic = fitted_vine.bic(clayton_df)
        assert np.isfinite(aic) and np.isfinite(bic)
        assert bic > aic  # BIC penalises parameters more heavily for n > 7

    def test_tail_dependence_in_range(self, fitted_vine):
        assert 0.0 <= fitted_vine.tail_dependence("x1", "x2", q=0.05, n_mc=20000) <= 1.0

    def test_tail_dependence_curve_shape(self, fitted_vine):
        c = fitted_vine.tail_dependence_curve("x1", "x2", quantiles=[0.02, 0.1], n_mc=20000)
        assert list(c.columns) == ["q", "lambda"] and len(c) == 2

    def test_joint_exceedance_probability_is_a_probability(self, fitted_vine):
        assert 0.0 <= fitted_vine.joint_exceedance_probability(
            {"x1": ("<", -1.0), "x2": ("<", -1.0)}, n_mc=20000) <= 1.0

    def test_pseudo_observations(self, fitted_vine):
        u = fitted_vine.pseudo_observations()
        assert ((u.values > 0) & (u.values < 1)).all()


class TestStatisticalBehaviour:
    def test_recovers_clayton_lower_tail_dependence(self):
        v = VineCopula(family_set="parametric").fit(ds.make_clayton(n=4000, theta=2.0, seed=0))
        u = v.sample_uniform(60000)
        truth = ds.theoretical_tail_dependence("clayton", theta=2.0)["lower"]
        assert abs(tail_dependence(u.x1, u.x2, q=0.02, tail="lower") - truth) < 0.10

    def test_gaussian_family_set_cannot_produce_tail_asymmetry(self):
        # The point of the comparison: restricted to Gaussian pair copulas, a
        # vine CANNOT represent Clayton's asymmetry however much data it gets.
        v = VineCopula(family_set="gaussian").fit(ds.make_clayton(n=4000, theta=2.0, seed=0))
        u = v.sample_uniform(60000)
        lo = tail_dependence(u.x1, u.x2, q=0.02, tail="lower")
        up = tail_dependence(u.x1, u.x2, q=0.02, tail="upper")
        assert abs(lo - up) < 0.12

    def test_reproducible_simulation_given_seed(self, clayton_df):
        a = VineCopula(seed=3).fit(clayton_df).sample_uniform(500)
        b = VineCopula(seed=3).fit(clayton_df).sample_uniform(500)
        assert np.allclose(a.values, b.values)
