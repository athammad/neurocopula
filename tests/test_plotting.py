"""
Tests for the plotting functions.

Plots are checked for the things that can be checked mechanically -- that a
figure is produced, that it has the expected number of axes, that a file is
written, that the documented guard rails fire. Whether a chart is READABLE is
not something a test can assert; that is what looking at the rendered output
during development is for.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytest

matplotlib.use("Agg")

from neurocopula import datasets as ds  # noqa: E402
from neurocopula import plotting as P
from neurocopula import theme


def titles(fig) -> list[str]:
    """
    Non-empty axis titles. The library's theme left-aligns titles, so they
    must be read with loc="left" -- the default loc="center" returns "".
    """
    return [t for t in (ax.get_title("left") for ax in fig.axes) if t]


@pytest.fixture(autouse=True)
def close_figures():
    """Close every figure after each test, so the suite does not leak memory."""
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def frames():
    return {
        "data": ds.make_clayton(n=800, marginals="uniform", seed=0),
        "NeuroCopula": ds.make_clayton(n=800, marginals="uniform", seed=1),
        "Vine": ds.make_gaussian_copula(n=800, rho=0.7, marginals="uniform", seed=2),
    }


class TestTheme:
    def test_palette_has_eight_distinct_slots(self):
        assert len(theme.CATEGORICAL) == 8
        assert len(set(theme.CATEGORICAL)) == 8

    def test_every_colour_is_a_hex_string(self):
        for c in theme.CATEGORICAL + list(theme.SERIES.values()):
            assert c.startswith("#") and len(c) == 7

    def test_named_entities_map_to_stable_colours(self):
        # Colour follows the entity, so "data" must be the same colour in
        # every figure regardless of which models appear beside it.
        assert theme.series_color("data") == theme.SERIES["data"]
        assert theme.series_color("data", 5) == theme.series_color("data", 0)

    def test_unknown_name_falls_back_to_a_categorical_slot(self):
        assert theme.series_color("something else", 1) == theme.CATEGORICAL[1]

    def test_fallback_slot_wraps_around(self):
        assert theme.series_color("x", 9) == theme.CATEGORICAL[1]

    def test_apply_theme_sets_rcparams(self):
        theme.apply_theme()
        assert matplotlib.rcParams["axes.spines.top"] is False

    def test_diverging_colormap_has_a_neutral_midpoint(self):
        # A diverging scale must read as "nothing" in the middle, so the
        # midpoint has to be near-gray: R, G and B close together.
        r, g, b, _ = theme.DIVERGING_CMAP(0.5)
        assert max(r, g, b) - min(r, g, b) < 0.06


class TestTrainingAndDiagnostics:
    def test_training_curves_returns_a_figure(self, fitted_copula):
        fig = P.plot_training_curves(fitted_copula)
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1

    def test_training_curves_can_draw_into_an_existing_axis(self, fitted_copula):
        _, ax = plt.subplots()
        assert P.plot_training_curves(fitted_copula, ax=ax) is ax

    def test_training_curves_legend_names_both_curves(self, fitted_copula):
        fig = P.plot_training_curves(fitted_copula)
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert any("train" in x for x in labels)
        assert any("validation" in x for x in labels)

    def test_diagnostics_without_data(self, fitted_copula):
        assert isinstance(P.plot_diagnostics(fitted_copula, n_samples=400), plt.Figure)

    def test_diagnostics_with_data(self, fitted_copula, clayton_df):
        assert isinstance(
            P.plot_diagnostics(fitted_copula, data=clayton_df, n_samples=400), plt.Figure)

    def test_diagnostics_on_five_dimensions(self, fitted_panel, panel_df):
        # The marginal row is capped at three slots; a 5-column model must
        # still lay out correctly rather than overflowing the grid.
        assert isinstance(
            P.plot_diagnostics(fitted_panel, data=panel_df, n_samples=400), plt.Figure)

    def test_diagnostics_writes_a_file(self, fitted_copula, tmp_path):
        path = tmp_path / "d.png"
        P.plot_diagnostics(fitted_copula, n_samples=300, path=path)
        assert path.exists() and path.stat().st_size > 5000


class TestDependencePlots:
    def test_copula_scatter_panel_count(self, frames):
        assert len(P.plot_copula_scatter(frames, "x1", "x2", n=300).axes) == 3

    def test_copula_scatter_rejects_more_than_three_series(self, frames):
        too_many = dict(frames)
        too_many["extra"] = frames["data"]
        with pytest.raises(ValueError, match="at most 3"):
            P.plot_copula_scatter(too_many, "x1", "x2")

    def test_pairwise_grid_is_square_in_the_column_count(self):
        panel = ds.make_asset_panel(n=300, seed=0)
        fig = P.plot_pairwise_grid({"data": panel}, n=200)
        assert len(fig.axes) == 25

    def test_dependence_heatmap(self, frames):
        assert isinstance(P.plot_dependence_heatmap(
            {k: frames[k] for k in ("data", "NeuroCopula")}), plt.Figure)

    def test_tau_error_heatmap(self, frames):
        fig = P.plot_tau_error_heatmap(frames["data"],
                                       {"NeuroCopula": frames["NeuroCopula"],
                                        "Vine": frames["Vine"]})
        assert isinstance(fig, plt.Figure)

    def test_tau_error_heatmap_titles_report_mean_absolute_error(self, frames):
        fig = P.plot_tau_error_heatmap(frames["data"], {"NeuroCopula": frames["NeuroCopula"]})
        assert any("mean abs error" in t for t in titles(fig))


class TestTailPlots:
    def test_tail_dependence_curve_uses_a_log_x_axis(self, frames):
        # The interesting behaviour is at small q, which a linear axis hides.
        fig = P.plot_tail_dependence_curve(frames, "x1", "x2",
                                           quantiles=[0.01, 0.05, 0.2])
        assert fig.axes[0].get_xscale() == "log"

    def test_tail_dependence_curve_draws_the_theoretical_line(self, frames):
        fig = P.plot_tail_dependence_curve(frames, "x1", "x2", theoretical=0.707,
                                           quantiles=[0.05, 0.2])
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert any("theoretical" in x for x in labels)

    def test_tail_dependence_curve_y_axis_is_the_unit_interval(self, frames):
        fig = P.plot_tail_dependence_curve(frames, "x1", "x2", quantiles=[0.05, 0.2])
        assert fig.axes[0].get_ylim() == (0.0, 1.0)

    def test_tail_concentration_has_two_panels(self, frames):
        assert len(P.plot_tail_concentration(frames, "x1", "x2").axes) == 2

    def test_exceedance_comparison_bars(self):
        r = pd.DataFrame({"NeuroCopula": [0.04, 0.002], "Vine": [0.05, 0.003]},
                         index=["both < -2", "both < -3"])
        fig = P.plot_exceedance_comparison(r)
        assert len(fig.axes[0].patches) == 4


class TestMarginalAndConditionalPlots:
    def test_marginal_comparison_has_two_rows_per_column(self, clayton_df):
        fig = P.plot_marginal_comparison(
            clayton_df, {"NeuroCopula": ds.make_clayton(n=500, seed=9)})
        assert len(fig.axes) == 4  # 2 columns x (density + Q-Q)

    def test_conditional_distribution_plot(self, fitted_copula):
        fig = P.plot_conditional_distribution(
            fitted_copula, "x2", [{"x1": 0.0}, {"x1": -2.0}], n_mc=8000)
        assert isinstance(fig, plt.Figure)

    def test_conditional_plot_legend_reports_effective_sample_size(self, fitted_copula):
        fig = P.plot_conditional_distribution(fitted_copula, "x2", [{"x1": 0.0}], n_mc=8000)
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert any("eff. n" in x for x in labels)

    def test_conditional_plot_rejects_too_many_conditions(self, fitted_copula):
        with pytest.raises(ValueError, match="at most 3"):
            P.plot_conditional_distribution(
                fitted_copula, "x2", [{"x1": v} for v in (-3, -2, -1, 0)], n_mc=2000)


class TestSummaryPlots:
    def test_uncertainty_plot(self):
        reports = {"lambda_L": {"mean": 0.6, "std": 0.05, "q05": 0.5, "q95": 0.7}}
        assert isinstance(P.plot_uncertainty(reports), plt.Figure)

    def test_benchmark_bars_one_panel_per_metric(self):
        r = pd.DataFrame({"model": ["NeuroCopula", "Vine"],
                          "loglik": [0.44, 0.40], "energy": [0.001, 0.002]})
        fig = P.plot_benchmark_bars(r, lower_is_better={"loglik": False, "energy": True})
        assert any("higher is better" in t for t in titles(fig))
        assert any("lower is better" in t for t in titles(fig))

    def test_benchmark_bars_ignores_non_numeric_columns(self):
        r = pd.DataFrame({"model": ["a", "b"], "note": ["x", "y"], "loglik": [0.4, 0.5]})
        fig = P.plot_benchmark_bars(r)
        assert titles(fig) == ["loglik"]

    def test_model_comparison_layout(self, fitted_copula, clayton_df):
        fig = P.plot_model_comparison(clayton_df, {"NeuroCopula": fitted_copula},
                                      "x1", "x2", n=500, theoretical=0.707)
        assert len(fig.axes) == 3  # data scatter + model scatter + curve

    def test_model_comparison_rejects_three_models(self, fitted_copula, clayton_df):
        with pytest.raises(ValueError, match="at most 2 models"):
            P.plot_model_comparison(clayton_df,
                                    {f"m{i}": fitted_copula for i in range(3)}, "x1", "x2")
