"""
Tests for the `NeuroCopula` estimator.

The statistical-quality tests here use loose tolerances on purpose: they are
checking that a small, fast fit learns the RIGHT KIND of structure, not that
it is accurate. Accuracy under a real training budget is what the benchmark
suite measures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from neurocopula import NeuroCopula
from neurocopula import datasets as ds
from neurocopula.metrics import tail_dependence

from .conftest import FAST_FIT


class TestConstruction:
    def test_rejects_bad_sampling_mode(self):
        with pytest.raises(ValueError, match="sampling_mode"):
            NeuroCopula(sampling_mode="parallel")

    def test_rejects_non_positive_transforms(self):
        with pytest.raises(ValueError, match="transforms"):
            NeuroCopula(transforms=0)

    def test_get_params_roundtrips_through_the_constructor(self):
        m = NeuroCopula(transforms=3, hidden_features=(16, 8), bins=6, seed=7)
        assert NeuroCopula(**m.get_params()).get_params() == m.get_params()

    def test_repr_says_unfitted_before_fit(self):
        assert "unfitted" in repr(NeuroCopula())

    def test_repr_says_fitted_after_fit(self, fitted_copula):
        assert "fitted on" in repr(fitted_copula)

    def test_queries_raise_before_fit(self):
        m = NeuroCopula()
        with pytest.raises(RuntimeError, match="not fitted"):
            m.sample(5)
        with pytest.raises(RuntimeError, match="not fitted"):
            m.log_prob(pd.DataFrame({"a": [1.0]}))


class TestFitValidation:
    def test_rejects_non_dataframe(self):
        with pytest.raises(TypeError, match="DataFrame"):
            NeuroCopula().fit(np.zeros((100, 2)))

    def test_rejects_unknown_split_method(self, clayton_df):
        with pytest.raises(ValueError, match="split_method"):
            NeuroCopula().fit(clayton_df, split_method="kfold", **FAST_FIT)

    def test_rejects_out_of_range_val_frac(self, clayton_df):
        with pytest.raises(ValueError, match="val_frac"):
            NeuroCopula().fit(clayton_df, val_frac=1.5, **FAST_FIT)

    def test_rejects_nan(self):
        df = ds.make_clayton(n=100, seed=0)
        df.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            NeuroCopula().fit(df, **FAST_FIT)

    def test_rejects_constant_column_with_a_clear_message(self):
        df = ds.make_clayton(n=100, seed=0)
        df["flat"] = 1.0
        with pytest.raises(ValueError, match="constant"):
            NeuroCopula().fit(df, **FAST_FIT)

    def test_warns_on_tiny_datasets(self):
        with pytest.warns(RuntimeWarning, match="overfit"):
            NeuroCopula(transforms=2, hidden_features=(8,)).fit(
                ds.make_clayton(n=30, seed=0), epochs=5, patience=3, verbose=False)


class TestFitMechanics:
    def test_fit_returns_self_for_chaining(self, clayton_df):
        m = NeuroCopula(transforms=2, hidden_features=(16, 16), seed=0)
        assert m.fit(clayton_df, **FAST_FIT) is m

    def test_fit_report_is_populated(self, fitted_copula):
        r = fitted_copula.fit_report_
        for key in ("n_train", "n_val", "best_epoch", "final_train_nll",
                    "final_val_nll", "overfit_gap", "n_parameters"):
            assert key in r
        assert r["n_train"] > r["n_val"] > 0

    def test_history_recorded_for_both_curves(self, fitted_copula):
        h = fitted_copula.history
        assert len(h["train"]) == len(h["val"]) > 0
        assert np.isfinite(h["train"]).all()

    def test_training_reduces_nll(self, fitted_copula):
        h = fitted_copula.history["train"]
        assert h[-1] < h[0]

    def test_best_epoch_is_within_history(self, fitted_copula):
        assert 0 <= fitted_copula.best_epoch < len(fitted_copula.history["val"])

    def test_chronological_split_holds_out_the_most_recent_rows(self, clayton_df):
        # A time-series split must not shuffle, or it cannot detect a regime
        # shift -- the whole reason it is the default.
        m = NeuroCopula(transforms=2, hidden_features=(16, 16), seed=0)
        m.fit(clayton_df, val_frac=0.2, split_method="chronological", **FAST_FIT)
        assert m.fit_report_["n_val"] == int(len(clayton_df) * 0.2)

    def test_embargo_shrinks_the_training_set(self, clayton_df):
        base = NeuroCopula(transforms=2, hidden_features=(16, 16), seed=0).fit(
            clayton_df, embargo=0, **FAST_FIT)
        emb = NeuroCopula(transforms=2, hidden_features=(16, 16), seed=0).fit(
            clayton_df, embargo=100, **FAST_FIT)
        assert emb.fit_report_["n_train"] == base.fit_report_["n_train"] - 100

    def test_refit_on_full_data_flag_is_recorded(self, clayton_df):
        m = NeuroCopula(transforms=2, hidden_features=(16, 16), seed=0).fit(
            clayton_df, refit_on_full_data=False, **FAST_FIT)
        assert m.fit_report_["refit_on_full_data"] is False
        assert "full_data_final_nll" not in m.fit_report_

    def test_refit_on_full_data_produces_a_usable_model(self, fitted_copula):
        assert fitted_copula.fit_report_["refit_on_full_data"] is True
        assert "full_data_final_nll" in fitted_copula.fit_report_
        assert len(fitted_copula.sample(10)) == 10

    def test_same_seed_gives_identical_fits(self, clayton_df):
        kw = dict(transforms=2, hidden_features=(16, 16), seed=42)
        a = NeuroCopula(**kw).fit(clayton_df, **FAST_FIT)
        b = NeuroCopula(**kw).fit(clayton_df, **FAST_FIT)
        assert np.allclose(a.history["train"], b.history["train"])

    def test_different_seed_gives_a_different_fit(self, clayton_df):
        a = NeuroCopula(transforms=2, hidden_features=(16, 16), seed=1).fit(clayton_df, **FAST_FIT)
        b = NeuroCopula(transforms=2, hidden_features=(16, 16), seed=2).fit(clayton_df, **FAST_FIT)
        assert not np.allclose(a.history["train"], b.history["train"])

    def test_minibatch_training_runs(self, clayton_df):
        m = NeuroCopula(transforms=2, hidden_features=(16, 16), seed=0).fit(
            clayton_df, batch_size=256, epochs=20, patience=10, verbose=False)
        assert np.isfinite(m.history["train"]).all()

    def test_random_split_runs(self, clayton_df):
        m = NeuroCopula(transforms=2, hidden_features=(16, 16), seed=0).fit(
            clayton_df, split_method="random", **FAST_FIT)
        assert m.flow is not None


class TestSampling:
    def test_sample_shape_and_columns(self, fitted_copula, clayton_df):
        s = fitted_copula.sample(250)
        assert s.shape == (250, 2)
        assert list(s.columns) == list(clayton_df.columns)
        assert np.isfinite(s.values).all()

    def test_batched_sampling_matches_requested_size(self, fitted_copula):
        assert len(fitted_copula.sample(2500, batch_size=500)) == 2500

    def test_sample_rejects_non_positive_n(self, fitted_copula):
        with pytest.raises(ValueError, match="n must be positive"):
            fitted_copula.sample(0)

    def test_sample_uniform_lies_in_the_unit_square(self, fitted_copula):
        u = fitted_copula.sample_uniform(1000)
        assert ((u.values > 0) & (u.values < 1)).all()

    def test_sample_uniform_is_marginally_uniform(self, fitted_copula):
        # Guaranteed by construction in copula mode; a failure here means the
        # probit/CDF pair has come apart.
        u = fitted_copula.sample_uniform(20000)
        for c in u.columns:
            assert abs(u[c].mean() - 0.5) < 0.02

    def test_sample_uniform_requires_copula_mode(self, fitted_joint):
        with pytest.raises(RuntimeError, match="copula_mode=True"):
            fitted_joint.sample_uniform(10)

    def test_pseudo_observations_of_training_data(self, fitted_copula):
        u = fitted_copula.pseudo_observations()
        assert ((u.values > 0) & (u.values < 1)).all()

    def test_coupling_mode_samples(self, fitted_panel):
        s = fitted_panel.sample(500)
        assert s.shape == (500, 5) and np.isfinite(s.values).all()

    def test_marginals_are_reproduced_in_copula_mode(self, fitted_copula, clayton_df):
        # Copula mode reproduces marginals exactly by construction, so the
        # sample's quantiles must track the data's closely.
        s = fitted_copula.sample(20000)
        for c in clayton_df.columns:
            for q in (0.05, 0.5, 0.95):
                assert abs(np.quantile(s[c], q) - np.quantile(clayton_df[c], q)) < 0.35


class TestDensity:
    def test_log_prob_shape_and_finiteness(self, fitted_copula, clayton_df):
        lp = fitted_copula.log_prob(clayton_df)
        assert lp.shape == (len(clayton_df),) and np.isfinite(lp).all()

    def test_log_prob_works_in_joint_mode(self, fitted_joint, clayton_df):
        assert np.isfinite(fitted_joint.log_prob(clayton_df)).all()

    def test_copula_log_prob_requires_copula_mode(self, fitted_joint, clayton_df):
        with pytest.raises(RuntimeError, match="copula_mode=True"):
            fitted_joint.copula_log_prob(clayton_df)

    def test_copula_log_prob_is_positive_on_average_for_dependent_data(
            self, fitted_copula, clayton_df):
        # A copula density integrates to 1 over the unit cube, so the mean log
        # copula density is 0 under independence and positive under genuine
        # dependence. Negative here would mean the model found no dependence.
        assert fitted_copula.score(clayton_df, copula=True) > 0.1

    def test_copula_log_prob_near_zero_for_independent_data(self):
        rng = np.random.default_rng(0)
        indep = pd.DataFrame(rng.normal(size=(1200, 2)), columns=["a", "b"])
        m = NeuroCopula(transforms=2, hidden_features=(16, 16), seed=0).fit(indep, **FAST_FIT)
        assert abs(m.score(indep, copula=True)) < 0.15

    def test_log_prob_accepts_a_numpy_array(self, fitted_copula, clayton_df):
        assert np.allclose(fitted_copula.log_prob(clayton_df.values),
                           fitted_copula.log_prob(clayton_df))

    def test_log_prob_reorders_columns_by_name(self, fitted_copula, clayton_df):
        swapped = clayton_df[["x2", "x1"]]
        assert np.allclose(fitted_copula.log_prob(swapped),
                           fitted_copula.log_prob(clayton_df))

    def test_missing_column_raises(self, fitted_copula, clayton_df):
        with pytest.raises(KeyError, match="missing columns"):
            fitted_copula.log_prob(clayton_df[["x1"]].rename(columns={"x1": "zzz"}))

    def test_wrong_width_array_raises(self, fitted_copula):
        with pytest.raises(ValueError, match="expected 2 columns"):
            fitted_copula.log_prob(np.zeros((5, 3)))

    def test_score_matches_mean_log_prob(self, fitted_copula, clayton_df):
        assert fitted_copula.score(clayton_df) == pytest.approx(
            float(np.mean(fitted_copula.log_prob(clayton_df))))

    def test_aic_penalises_parameters(self, fitted_copula, clayton_df):
        aic = fitted_copula.aic(clayton_df)
        k = sum(p.numel() for p in fitted_copula.flow.parameters())
        assert aic == pytest.approx(2 * k - 2 * fitted_copula.log_prob(clayton_df).sum())

    def test_bic_penalises_parameters_more_heavily_than_aic(self, fitted_copula, clayton_df):
        # k*log(n) > 2k for n > 7, so BIC must exceed AIC on any real dataset.
        assert fitted_copula.bic(clayton_df) > fitted_copula.aic(clayton_df)

    def test_marginal_density_none_disables_log_prob_but_not_copula_log_prob(
            self, clayton_df):
        m = NeuroCopula(transforms=2, hidden_features=(16, 16), seed=0,
                        marginal_density="none").fit(clayton_df, **FAST_FIT)
        assert np.isfinite(m.copula_log_prob(clayton_df)).all()
        with pytest.raises(RuntimeError, match="copula_log_prob"):
            m.log_prob(clayton_df)


class TestProbabilisticQueries:
    def test_joint_exceedance_probability_is_a_probability(self, fitted_copula):
        p = fitted_copula.joint_exceedance_probability(
            {"x1": ("<", -1.0), "x2": ("<", -1.0)}, n_mc=20000)
        assert 0.0 <= p <= 1.0

    def test_joint_exceedance_decreases_with_a_stricter_threshold(self, fitted_copula):
        loose = fitted_copula.joint_exceedance_probability({"x1": ("<", 0.0)}, n_mc=20000)
        tight = fitted_copula.joint_exceedance_probability({"x1": ("<", -2.0)}, n_mc=20000)
        assert tight < loose

    def test_adding_a_condition_cannot_increase_the_probability(self, fitted_copula):
        one = fitted_copula.joint_exceedance_probability({"x1": ("<", -1.0)}, n_mc=30000)
        two = fitted_copula.joint_exceedance_probability(
            {"x1": ("<", -1.0), "x2": ("<", -1.0)}, n_mc=30000)
        assert two <= one + 0.01

    @pytest.mark.parametrize("op", ["<", "<=", ">", ">="])
    def test_all_operators_accepted(self, fitted_copula, op):
        assert 0 <= fitted_copula.joint_exceedance_probability(
            {"x1": (op, 0.0)}, n_mc=5000) <= 1

    def test_unknown_operator_raises(self, fitted_copula):
        with pytest.raises(ValueError, match="unsupported operator"):
            fitted_copula.joint_exceedance_probability({"x1": ("==", 0.0)}, n_mc=1000)

    def test_unknown_column_raises(self, fitted_copula):
        with pytest.raises(KeyError, match="unknown column"):
            fitted_copula.joint_exceedance_probability({"nope": ("<", 0.0)}, n_mc=1000)

    def test_tail_dependence_in_range(self, fitted_copula):
        lam = fitted_copula.tail_dependence("x1", "x2", q=0.05, n_mc=20000)
        assert 0.0 <= lam <= 1.0

    @pytest.mark.slow
    def test_recovers_clayton_lower_tail_asymmetry(self, fitted_copula):
        # The headline statistical claim: the flow learns that this data
        # crashes together but does not rally together.
        lo = fitted_copula.tail_dependence("x1", "x2", q=0.05, tail="lower", n_mc=60000)
        up = fitted_copula.tail_dependence("x1", "x2", q=0.05, tail="upper", n_mc=60000)
        assert lo > 0.45
        assert lo > up + 0.3

    @pytest.mark.slow
    def test_learns_absence_of_tail_dependence_for_gaussian_data(self, gaussian_df):
        # The converse check, and the more important one: the flow must not
        # invent tail dependence that is not there.
        m = NeuroCopula(transforms=3, hidden_features=(32, 32), seed=0).fit(
            gaussian_df, **FAST_FIT)
        data_lam = tail_dependence(gaussian_df.x1, gaussian_df.x2, q=0.05, tail="lower")
        model_lam = m.tail_dependence("x1", "x2", q=0.05, tail="lower", n_mc=60000)
        assert abs(model_lam - data_lam) < 0.20

    def test_tail_dependence_rejects_bad_arguments(self, fitted_copula):
        with pytest.raises(ValueError, match="q must be"):
            fitted_copula.tail_dependence("x1", "x2", q=0.7)
        with pytest.raises(ValueError, match="tail must be"):
            fitted_copula.tail_dependence("x1", "x2", tail="sideways")
        with pytest.raises(KeyError, match="unknown column"):
            fitted_copula.tail_dependence("x1", "nope")

    def test_tail_dependence_curve_shape(self, fitted_copula):
        c = fitted_copula.tail_dependence_curve("x1", "x2", quantiles=[0.02, 0.1, 0.25],
                                                n_mc=20000)
        assert list(c.columns) == ["q", "lambda"] and len(c) == 3


class TestConditionalQueries:
    def test_conditional_distribution_weights_sum_to_one(self, fitted_copula):
        cd = fitted_copula.conditional_distribution("x2", {"x1": 0.0}, n_mc=20000)
        assert cd["weight"].sum() == pytest.approx(1.0)
        assert cd.attrs["effective_sample_size"] > 0

    def test_conditional_distribution_requires_a_condition(self, fitted_copula):
        with pytest.raises(ValueError, match="at least one column"):
            fitted_copula.conditional_distribution("x2", {}, n_mc=1000)

    def test_conditional_distribution_rejects_bad_tolerance(self, fitted_copula):
        with pytest.raises(ValueError, match="tol must be positive"):
            fitted_copula.conditional_distribution("x2", {"x1": 0.0}, tol=0.0, n_mc=1000)

    def test_larger_tolerance_keeps_more_effective_samples(self, fitted_copula):
        narrow = fitted_copula.conditional_distribution("x2", {"x1": 0.0}, tol=0.1, n_mc=20000)
        wide = fitted_copula.conditional_distribution("x2", {"x1": 0.0}, tol=0.6, n_mc=20000)
        assert (wide.attrs["effective_sample_size"]
                > narrow.attrs["effective_sample_size"])

    @pytest.mark.slow
    def test_conditioning_on_a_crash_raises_the_crash_probability(self, fitted_copula):
        # Tail dependence restated as a conditional probability. For Clayton
        # data the gap should be dramatic, not marginal.
        crash, _ = fitted_copula.conditional_probability(
            "x2", "<", -2.0, given={"x1": -3.0}, tol=0.3, n_mc=60000)
        calm, _ = fitted_copula.conditional_probability(
            "x2", "<", -2.0, given={"x1": 0.0}, tol=0.3, n_mc=60000)
        assert crash > calm + 0.2

    def test_conditional_probability_returns_probability_and_ess(self, fitted_copula):
        p, ess = fitted_copula.conditional_probability(
            "x2", "<", 0.0, given={"x1": 0.0}, n_mc=20000)
        assert 0.0 <= p <= 1.0 and ess > 0

    def test_conditional_quantiles_are_monotone(self, fitted_copula):
        q = fitted_copula.conditional_quantiles("x2", {"x1": -1.0},
                                                quantiles=(0.1, 0.5, 0.9), n_mc=20000)
        assert list(q.values) == sorted(q.values)

    def test_multiple_conditioning_columns(self, fitted_panel):
        cd = fitted_panel.conditional_distribution(
            "metal_A", {"grain_A": 0.0, "grain_B": 0.0}, tol=0.5, n_mc=20000)
        assert cd["weight"].sum() == pytest.approx(1.0)

    def test_warns_when_effective_sample_size_collapses(self, fitted_copula):
        with pytest.warns(RuntimeWarning, match="effective sample size"):
            fitted_copula.conditional_distribution("x2", {"x1": -8.0}, tol=0.01, n_mc=2000)


class TestPersistence:
    def test_save_and_load_roundtrip(self, fitted_copula, clayton_df, tmp_path):
        path = fitted_copula.save(tmp_path / "m.pt")
        loaded = NeuroCopula.load(path)
        assert np.allclose(loaded.log_prob(clayton_df), fitted_copula.log_prob(clayton_df))

    def test_loaded_model_keeps_metadata(self, fitted_copula, tmp_path):
        loaded = NeuroCopula.load(fitted_copula.save(tmp_path / "m.pt"))
        assert loaded.columns == fitted_copula.columns
        assert loaded.fit_report_ == fitted_copula.fit_report_
        assert loaded.get_params() == fitted_copula.get_params()

    def test_loaded_model_can_sample_and_query(self, fitted_copula, tmp_path):
        loaded = NeuroCopula.load(fitted_copula.save(tmp_path / "m.pt"))
        assert len(loaded.sample(50)) == 50
        assert 0 <= loaded.tail_dependence("x1", "x2", n_mc=5000) <= 1

    def test_save_requires_a_fitted_model(self, tmp_path):
        with pytest.raises(RuntimeError, match="not fitted"):
            NeuroCopula().save(tmp_path / "m.pt")


class TestBayesian:
    @pytest.fixture(scope="class")
    def bayes_model(self, clayton_df):
        return NeuroCopula(transforms=2, hidden_features=(16, 16), bayesian=True,
                           seed=0).fit(clayton_df, epochs=60, patience=20, verbose=False)

    def test_fit_produces_both_wrapper_and_concrete_flow(self, bayes_model):
        assert bayes_model.bayes is not None and bayes_model.flow is not None

    def test_uncertainty_report_fields(self, bayes_model):
        r = bayes_model.uncertainty_report(
            lambda m: m.tail_dependence("x1", "x2", q=0.1, n_mc=4000), n_posterior=5)
        assert set(r) == {"samples", "mean", "std", "q05", "q95"}
        assert len(r["samples"]) == 5
        assert r["q05"] <= r["mean"] <= r["q95"]

    def test_uncertainty_report_restores_the_original_flow(self, bayes_model):
        before = bayes_model.flow
        bayes_model.uncertainty_report(lambda m: m.score(m.sample(50)), n_posterior=3)
        assert bayes_model.flow is before

    def test_uncertainty_report_restores_flow_even_if_metric_raises(self, bayes_model):
        before = bayes_model.flow
        with pytest.raises(ValueError):
            bayes_model.uncertainty_report(
                lambda m: (_ for _ in ()).throw(ValueError("boom")), n_posterior=2)
        assert bayes_model.flow is before

    def test_uncertainty_report_requires_bayesian(self, fitted_copula):
        with pytest.raises(RuntimeError, match="bayesian=True"):
            fitted_copula.uncertainty_report(lambda m: 1.0, n_posterior=2)

    def test_bayesian_model_still_samples(self, bayes_model):
        assert len(bayes_model.sample(50)) == 50

    def test_bayesian_save_load_restores_the_posterior(self, bayes_model, tmp_path):
        # A Bayesian model saves the BayesianModel wrapper, not the sampled
        # flow, so loading must restore something you can still draw posterior
        # samples from -- otherwise uncertainty_report breaks after a reload.
        loaded = NeuroCopula.load(bayes_model.save(tmp_path / "b.pt"))
        assert loaded.bayes is not None
        assert loaded.bayesian is True
        assert len(loaded.sample(20)) == 20
        r = loaded.uncertainty_report(
            lambda m: m.tail_dependence("x1", "x2", q=0.1, n_mc=3000), n_posterior=3)
        assert set(r) == {"samples", "mean", "std", "q05", "q95"}


class TestMultivariate:
    def test_five_dimensional_fit(self, fitted_panel, panel_df):
        assert fitted_panel.fit_report_["n_features"] == 5
        assert fitted_panel.sample(100).shape == (100, 5)

    @pytest.mark.slow
    def test_recovers_block_dependence_structure(self, fitted_panel, panel_df):
        from neurocopula.metrics import kendall_tau_matrix
        model_tau = kendall_tau_matrix(fitted_panel.sample(20000))
        data_tau = kendall_tau_matrix(panel_df)
        # Within-block dependence must stay clearly above cross-block.
        assert model_tau.loc["grain_A", "grain_B"] > model_tau.loc["grain_A", "metal_A"] + 0.2
        err = np.abs((model_tau - data_tau).values[~np.eye(5, dtype=bool)])
        assert err.mean() < 0.12


class TestDevice:
    def test_defaults_to_an_available_device(self):
        m = NeuroCopula()
        assert m.device.type in ("cpu", "cuda")

    def test_explicit_cpu_is_honoured(self, clayton_df):
        m = NeuroCopula(transforms=2, hidden_features=(16, 16), device="cpu",
                        seed=0).fit(clayton_df, **FAST_FIT)
        assert m.device == torch.device("cpu")
        assert len(m.sample(10)) == 10
