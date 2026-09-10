"""
Tests for the benchmark harness.

These check the harness is measuring what it claims, which matters more than
the numbers themselves: a benchmark that leaks test data, or that scores the
two models on different quantities, would produce confident and wrong
conclusions. The cases run here are deliberately tiny -- correctness of the
machinery, not accuracy of the models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neurocopula import benchmark as B
from neurocopula import datasets as ds

pytest.importorskip("pyvinecopulib")

TINY = dict(epochs=40, n_mc=4000, test_frac=0.25)


@pytest.fixture(scope="module")
def tiny_case():
    return B._case(
        "tiny_clayton",
        lambda n, seed: ds.make_clayton(n=n, theta=2.0, seed=seed),
        family="clayton", params={"theta": 2.0},
        description="tiny case for testing the harness", n=600,
    )


@pytest.fixture(scope="module")
def tiny_result(tiny_case):
    return B.run_case(tiny_case, seed=0, vine_family_sets=("parametric",), **TINY)


class TestSplitting:
    def test_split_is_disjoint_and_complete(self):
        df = ds.make_clayton(n=500, seed=0)
        train, test = B._split(df, 0.25, seed=0)
        assert len(train) + len(test) == len(df)
        assert len(test) == 125

    def test_split_is_reproducible(self):
        df = ds.make_clayton(n=300, seed=0)
        a, _ = B._split(df, 0.2, seed=1)
        b, _ = B._split(df, 0.2, seed=1)
        assert a.equals(b)

    def test_split_actually_shuffles(self):
        df = ds.make_clayton(n=300, seed=0)
        train, _ = B._split(df, 0.2, seed=1)
        assert not np.allclose(train.values[:50], df.values[:50])

    def test_rows_are_not_shared_between_train_and_test(self):
        # The property the whole benchmark depends on: if test rows appeared
        # in training, every held-out score would be inflated.
        df = ds.make_clayton(n=400, seed=0)
        train, test = B._split(df, 0.25, seed=0)
        train_rows = {tuple(r) for r in train.values}
        assert not any(tuple(r) in train_rows for r in test.values)


class TestSuiteDefinition:
    def test_standard_suite_is_non_empty_and_uniquely_named(self):
        names = [c.name for c in B.STANDARD_SUITE]
        assert len(names) == len(set(names)) >= 6

    def test_every_case_has_a_description(self):
        assert all(c.description for c in B.STANDARD_SUITE)

    def test_every_generator_produces_the_requested_shape(self):
        for case in B.STANDARD_SUITE:
            df = case.generator(120, 0)
            assert isinstance(df, pd.DataFrame) and len(df) == 120

    def test_parametric_cases_carry_ground_truth(self):
        known = [c for c in B.STANDARD_SUITE if c.truth_tail is not None]
        assert len(known) >= 4
        for c in known:
            assert set(c.truth_tail) == {"lower", "upper"}
            assert c.truth_tau is not None

    def test_suite_covers_both_matched_and_mismatched_families(self):
        # The suite must contain cases the vine is expected to win as well as
        # cases the flow is expected to win, or it is not a fair benchmark.
        assert any(c.truth_tail is not None for c in B.STANDARD_SUITE)
        assert any("mixture" in c.description for c in B.STANDARD_SUITE)


class TestRunCase:
    def test_returns_one_row_per_model(self, tiny_result):
        assert len(tiny_result) == 2
        assert set(tiny_result["model"]) == {"NeuroCopula", "Vine (parametric)"}

    def test_reports_every_documented_metric(self, tiny_result):
        for col in ("case", "model", "test_copula_loglik", "lambda_L", "lambda_U",
                    "tau_mae", "energy_distance", "mmd", "fit_seconds",
                    "sample_seconds", "n_parameters"):
            assert col in tiny_result.columns

    def test_metrics_are_finite(self, tiny_result):
        numeric = tiny_result.select_dtypes("number")
        assert np.isfinite(numeric.values).all()

    def test_ground_truth_errors_present_when_truth_is_known(self, tiny_result):
        assert "lambda_L_err" in tiny_result.columns
        assert (tiny_result["lambda_L_err"] >= 0).all()

    def test_ground_truth_errors_absent_when_truth_is_unknown(self):
        case = B._case("no_truth", lambda n, seed: ds.make_mixed_regime(n=n, seed=seed),
                       description="no closed form", n=400)
        out = B.run_case(case, seed=0, vine_family_sets=("gaussian",), **TINY)
        assert "lambda_L_err" not in out.columns

    def test_tail_coefficients_are_probabilities(self, tiny_result):
        for col in ("lambda_L", "lambda_U"):
            assert ((tiny_result[col] >= 0) & (tiny_result[col] <= 1)).all()

    def test_parameter_counts_differ_by_orders_of_magnitude(self, tiny_result):
        # Context for reading AIC: a flow has thousands of parameters where a
        # bivariate vine has one or two.
        by_model = tiny_result.set_index("model")["n_parameters"]
        assert by_model["NeuroCopula"] > 100 * by_model["Vine (parametric)"]

    def test_both_models_are_scored_on_the_same_quantity(self, tiny_case):
        # Copula log-likelihood is exact for both and needs no density
        # estimate; if the two models were scored differently the comparison
        # would be meaningless.
        out = B.run_case(tiny_case, seed=0, vine_family_sets=("parametric",), **TINY)
        assert out["test_copula_loglik"].notna().all()

    def test_multiple_vine_family_sets_produce_multiple_rows(self, tiny_case):
        out = B.run_case(tiny_case, seed=0,
                         vine_family_sets=("gaussian", "parametric"), **TINY)
        assert len(out) == 3


class TestRunBenchmarkAndSummaries:
    @pytest.fixture(scope="class")
    def multi_seed(self, tiny_case):
        return B.run_benchmark([tiny_case], seeds=(0, 1),
                               vine_family_sets=("parametric",), verbose=False, **TINY)

    def test_seeds_are_recorded(self, multi_seed):
        assert set(multi_seed["seed"]) == {0, 1}
        assert len(multi_seed) == 4

    def test_different_seeds_give_different_numbers(self, multi_seed):
        flow = multi_seed[multi_seed["model"] == "NeuroCopula"]
        assert flow["test_copula_loglik"].nunique() == 2

    def test_summarize_reports_mean_and_std(self, multi_seed):
        s = B.summarize(multi_seed)
        assert ("test_copula_loglik", "mean") in s.columns
        assert ("test_copula_loglik", "std") in s.columns

    def test_summarize_has_one_row_per_case_and_model(self, multi_seed):
        assert len(B.summarize(multi_seed)) == 2

    def test_tail_table_includes_theoretical_columns(self, multi_seed, tiny_case):
        t = B.tail_dependence_table(multi_seed, cases=[tiny_case])
        assert {"lambda_L", "lambda_U", "lambda_L_true", "lambda_U_true"} <= set(t.columns)
        assert t["lambda_L_true"].iloc[0] == pytest.approx(2 ** -0.5)

    def test_tail_table_tolerates_cases_without_truth(self, multi_seed):
        t = B.tail_dependence_table(multi_seed, cases=[])
        assert t["lambda_L_true"].isna().all()
