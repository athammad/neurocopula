"""
Tests for the spatial probabilistic layer and its persistence.

The tests that matter most are the ones checking that the aggregation modes
actually differ from each other. "Probability at a typical point" and
"probability somewhere in the region" are different questions, and a model
that has lost its spatial dependence would silently collapse them together --
returning the independence answer, which for a large region is catastrophically
wrong in a way that still looks like a plausible number.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm
from scipy.stats import t as student_t

from neurocopula.io import layer_to_dataset, load_layer, save_layer
from neurocopula.layer import SpatialProbabilisticLayer
from neurocopula.spatial import CORRELATION_MODELS, haversine_matrix

pytest.importorskip("xarray")

N_SIDE, N_TIME = 6, 600
TRUE_RANGE, TRUE_DF = 300.0, 4.0
FAST = dict(epochs=60, patience=20, verbose=False)


def make_coords(n_side=N_SIDE):
    lats = np.linspace(-5, 5, n_side)
    lons = np.linspace(100, 110, n_side)
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    return np.column_stack([LA.ravel(), LO.ravel()])


def make_data(coords, n_time=N_TIME, seed=0):
    """Three variables sharing a spatial t field, correlated within a cell."""
    m = len(coords)
    D = haversine_matrix(coords)
    C = CORRELATION_MODELS["exponential"](D, TRUE_RANGE)
    np.fill_diagonal(C, 1.0)
    C[np.diag_indices_from(C)] += 1e-8
    L = np.linalg.cholesky(C)

    def field(s):
        r = np.random.default_rng(s)
        w = r.standard_normal((n_time, m)) @ L.T
        g = r.chisquare(TRUE_DF, size=(n_time, 1)) / TRUE_DF
        return norm.ppf(np.clip(student_t.cdf(w / np.sqrt(g), TRUE_DF), 1e-10, 1 - 1e-10))

    e1, e2, e3 = field(seed + 1), field(seed + 2), field(seed + 3)
    a = 0.75
    z_rain, z_temp, z_wind = e1, a * e1 + np.sqrt(1 - a**2) * e2, a * e1 + np.sqrt(1 - a**2) * e3
    u = norm.cdf(z_rain)
    rain = np.where(u < 0.55, 0.0, 12 * (-np.log(np.clip(1 - (u - 0.55) / 0.45, 1e-12, 1))))
    temp = 28 + 3 * z_temp
    wind = 4 + 1.5 * np.exp(0.35 * z_wind)
    return np.stack([rain, temp, wind], axis=-1)


@pytest.fixture(scope="module")
def coords():
    return make_coords()


@pytest.fixture(scope="module")
def data(coords):
    return make_data(coords)


@pytest.fixture(scope="module")
def layer(data, coords):
    return SpatialProbabilisticLayer(
        variables=["rain", "temp", "wind"],
        marginal_specs={"rain": dict(zero_inflated=True, zero_threshold=0.05)},
        copula_scope="global", transforms=3, hidden_features=(32, 32), seed=0,
    ).fit(data, coords, **FAST)


class TestConstruction:
    def test_rejects_bad_scope(self):
        with pytest.raises(ValueError, match="copula_scope"):
            SpatialProbabilisticLayer(["a"], copula_scope="region")

    def test_repr_before_and_after_fit(self, layer):
        assert "unfitted" in repr(SpatialProbabilisticLayer(["a", "b"]))
        assert "cells=" in repr(layer)

    def test_queries_raise_before_fit(self):
        lay = SpatialProbabilisticLayer(["a"])
        with pytest.raises(RuntimeError, match="not fitted"):
            lay.sample()
        with pytest.raises(RuntimeError, match="not fitted"):
            lay.select()


class TestFitValidation:
    def test_rejects_2d_data(self, coords):
        with pytest.raises(ValueError, match="3-D"):
            SpatialProbabilisticLayer(["a"]).fit(np.zeros((10, len(coords))), coords)

    def test_rejects_variable_count_mismatch(self, coords):
        with pytest.raises(ValueError, match="variables"):
            SpatialProbabilisticLayer(["a", "b"]).fit(
                np.random.default_rng(0).normal(size=(50, len(coords), 3)), coords)

    def test_rejects_coord_count_mismatch(self, data, coords):
        with pytest.raises(ValueError, match="coordinates"):
            SpatialProbabilisticLayer(["rain", "temp", "wind"]).fit(data, coords[:5])

    def test_rejects_nan(self, data, coords):
        bad = data.copy()
        bad[0, 0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            SpatialProbabilisticLayer(["rain", "temp", "wind"]).fit(bad, coords)


class TestFitOutputs:
    def test_fit_report_populated(self, layer, coords):
        r = layer.fit_report_
        assert r["n_cells"] == len(coords)
        assert r["n_variables"] == 3
        assert set(r["spatial"]) == {"rain", "temp", "wind"}

    def test_spatial_fitted_per_variable(self, layer):
        for v in layer.variables:
            assert layer.spatial_[v].range_ > 0

    def test_marginals_are_per_cell(self, layer, coords):
        assert layer.marginals_.scope == "group"
        assert layer.marginals_.coverage_report()["n_groups"] == len(coords)

    def test_zero_inflation_detected_for_rain(self, layer):
        # ~55% dry by construction; the marginal must notice.
        p0 = layer.marginals_._set_for(0)["rain"].p_zero_
        assert 0.4 < p0 < 0.7

    def test_latent_round_trip(self, layer, data):
        # The composition depends on this being exact: data -> latent -> data.
        z = np.random.default_rng(0).normal(size=(50, 4, 3))
        e = layer._to_latent(z)
        back = layer._from_latent(e, np.arange(4))
        assert np.abs(back - z).max() < 1e-3


class TestSelection:
    def test_select_all_by_default(self, layer, coords):
        assert len(layer.select()) == len(coords)

    def test_select_bbox(self, layer):
        sel = layer.select(bbox=(-2, 2, 103, 107))
        assert 0 < len(sel) < layer.n_cells
        picked = layer.coords_[sel]
        assert picked[:, 0].min() >= -2 and picked[:, 0].max() <= 2

    def test_select_indices_and_mask_agree(self, layer):
        mask = np.zeros(layer.n_cells, dtype=bool)
        mask[[1, 4, 9]] = True
        assert np.array_equal(layer.select(mask=mask), layer.select(indices=[1, 4, 9]))

    def test_empty_bbox_raises(self, layer):
        with pytest.raises(ValueError, match="no locations"):
            layer.select(bbox=(80, 85, 0, 5))

    def test_rejects_multiple_selectors(self, layer):
        with pytest.raises(ValueError, match="at most one"):
            layer.select(bbox=(-2, 2, 103, 107), indices=[0])

    def test_rejects_wrong_length_mask(self, layer):
        with pytest.raises(ValueError, match="entries"):
            layer.select(mask=np.ones(3, dtype=bool))


class TestSampling:
    def test_sample_shape(self, layer):
        assert layer.sample(n=50).shape == (50, layer.n_cells, 3)

    def test_sample_subset_shape(self, layer):
        sel = layer.select(indices=[0, 3, 7])
        assert layer.sample(sel, n=40).shape == (40, 3, 3)

    def test_sample_is_finite(self, layer):
        assert np.isfinite(layer.sample(n=100)).all()

    def test_sample_reproducible(self, layer):
        assert np.allclose(layer.sample(n=50, seed=5), layer.sample(n=50, seed=5))

    def test_rain_samples_include_exact_zeros(self, layer):
        # Zero inflation must survive the whole round trip, or the simulated
        # rainfall would have no dry days at all.
        s = layer.sample(n=400, seed=1)
        assert (s[:, :, 0] == 0.0).mean() > 0.2

    def test_marginals_roughly_preserved(self, layer, data):
        s = layer.sample(n=1500, seed=2)
        for i in range(3):
            d_med = np.quantile(data[:, :, i], 0.5)
            s_med = np.quantile(s[:, :, i], 0.5)
            scale = max(abs(d_med), 1.0)
            assert abs(s_med - d_med) / scale < 0.5

    def test_near_pairs_more_correlated_than_far(self, layer):
        # The spatial layer doing its job at sampling time.
        D = haversine_matrix(layer.coords_)
        s = layer.sample(n=3000, seed=3)
        iu = np.triu_indices(layer.n_cells, k=1)
        # Thresholds taken from the grid's own distance distribution: absolute
        # km cut-offs silently match nothing when the test grid is rescaled.
        d_all = D[iu]
        near_cut, far_cut = np.quantile(d_all, 0.15), np.quantile(d_all, 0.85)
        near = [(i, j) for i, j in zip(*iu, strict=True) if D[i, j] <= near_cut][:20]
        far = [(i, j) for i, j in zip(*iu, strict=True) if D[i, j] >= far_cut][:20]
        assert near and far
        cn = np.mean([np.corrcoef(s[:, i, 1], s[:, j, 1])[0, 1] for i, j in near])
        cf = np.mean([np.corrcoef(s[:, i, 1], s[:, j, 1])[0, 1] for i, j in far])
        assert cn > cf + 0.1


class TestAggregationModes:
    CONDS = {"rain": (">", 20.0), "temp": (">", 30.0)}

    def test_pointwise_returns_one_value_per_cell(self, layer):
        sel = layer.select(bbox=(-3, 3, 102, 108))
        p = layer.exceedance_probability(self.CONDS, sel, mode="pointwise", n=4000)
        assert p.shape == (len(sel),)
        assert ((p >= 0) & (p <= 1)).all()

    def test_mean_equals_average_of_pointwise(self, layer):
        # 'mean' is exactly the average of per-cell probabilities -- true
        # regardless of dependence, by linearity of expectation.
        sel = layer.select(indices=list(range(8)))
        pw = layer.exceedance_probability(self.CONDS, sel, mode="pointwise", n=6000, seed=1)
        mn = layer.exceedance_probability(self.CONDS, sel, mode="mean", n=6000, seed=1)
        assert mn == pytest.approx(pw.mean(), abs=1e-9)

    def test_any_is_at_least_mean(self, layer):
        sel = layer.select(indices=list(range(10)))
        mn = layer.exceedance_probability(self.CONDS, sel, mode="mean", n=6000, seed=1)
        an = layer.exceedance_probability(self.CONDS, sel, mode="any", n=6000, seed=1)
        assert an >= mn - 1e-9

    def test_all_is_at_most_mean(self, layer):
        sel = layer.select(indices=list(range(10)))
        mn = layer.exceedance_probability(self.CONDS, sel, mode="mean", n=6000, seed=1)
        al = layer.exceedance_probability(self.CONDS, sel, mode="all", n=6000, seed=1)
        assert al <= mn + 1e-9

    def test_dependence_makes_any_far_below_the_independent_value(self, layer):
        # THE test. Under independence P(any) = 1-(1-p)^m, which saturates
        # towards 1 for a large region. With real spatial dependence the cells
        # fail together, so P(any) stays much lower. If this ever fails, the
        # spatial layer has stopped contributing.
        #
        # A RARE event is used deliberately: for a common one, 1-(1-p)^m is
        # near 1 whatever the dependence, so both numbers saturate and the
        # comparison loses its power. The rare case is also the one that
        # matters, being where an independence assumption does real damage.
        rare = {"rain": (">", 45.0), "temp": (">", 31.5)}
        sel = layer.select()
        mn = layer.exceedance_probability(rare, sel, mode="mean", n=20000, seed=1)
        an = layer.exceedance_probability(rare, sel, mode="any", n=20000, seed=1)
        independent = 1 - (1 - mn) ** len(sel)

        # Assert against Monte Carlo noise rather than an arbitrary ratio. How
        # LARGE the gap is depends on the region's size relative to the
        # correlation range: this test grid spans ~1100 km with a ~300 km
        # range, so it holds roughly a dozen effectively independent cells and
        # the overstatement is ~1.3x. Over a region small compared with the
        # range, every cell fails together and the factor is far larger.
        se = np.sqrt(max(an, 1e-6) * (1 - an) / 20000)
        assert independent > an + 4 * se

    def test_fraction_returns_per_draw_values(self, layer):
        sel = layer.select()
        f = layer.exceedance_probability(self.CONDS, sel, mode="fraction", n=2000, seed=1)
        assert f.shape == (2000,)
        assert ((f >= 0) & (f <= 1)).all()

    def test_fraction_has_heavy_upper_tail_under_dependence(self, layer):
        # Dependent cells occasionally fail en masse; independent ones never
        # would. This is the signature of the spatial layer in the output.
        sel = layer.select()
        f = layer.exceedance_probability(self.CONDS, sel, mode="fraction", n=4000, seed=1)
        assert f.max() > 5 * max(f.mean(), 1e-6)

    def test_count_matches_fraction(self, layer):
        sel = layer.select(indices=list(range(12)))
        c = layer.exceedance_probability(self.CONDS, sel, mode="count", n=1500, seed=4)
        f = layer.exceedance_probability(self.CONDS, sel, mode="fraction", n=1500, seed=4)
        assert np.allclose(c, f * len(sel))

    def test_stricter_threshold_lowers_probability(self, layer):
        sel = layer.select()
        loose = layer.exceedance_probability({"rain": (">", 5.0)}, sel, mode="mean", n=4000, seed=1)
        tight = layer.exceedance_probability({"rain": (">", 60.0)}, sel, mode="mean", n=4000, seed=1)
        assert tight < loose

    def test_adding_a_condition_cannot_increase_probability(self, layer):
        sel = layer.select()
        one = layer.exceedance_probability({"rain": (">", 20.0)}, sel, mode="mean", n=6000, seed=1)
        two = layer.exceedance_probability(
            {"rain": (">", 20.0), "temp": (">", 31.0)}, sel, mode="mean", n=6000, seed=1)
        assert two <= one + 1e-9

    def test_rejects_unknown_mode(self, layer):
        with pytest.raises(ValueError, match="mode must be"):
            layer.exceedance_probability(self.CONDS, mode="median", n=100)

    def test_rejects_unknown_variable(self, layer):
        with pytest.raises(KeyError, match="unknown variable"):
            layer.exceedance_probability({"snow": (">", 1.0)}, n=100)

    def test_rejects_unknown_operator(self, layer):
        with pytest.raises(ValueError, match="unsupported operator"):
            layer.exceedance_probability({"rain": ("==", 1.0)}, n=100)


class TestTailDependenceMap:
    def test_returns_one_value_per_variable(self, layer):
        td = layer.tail_dependence_map(0.0)
        assert set(td) == {"rain", "temp", "wind"}
        assert all(0.0 <= v <= 1.0 for v in td.values())

    def test_decays_with_distance(self, layer):
        near = layer.tail_dependence_map(10.0)
        far = layer.tail_dependence_map(2000.0)
        assert all(far[k] <= near[k] + 1e-9 for k in near)


class TestBayesian:
    @pytest.fixture(scope="class")
    def bayes_layer(self, data, coords):
        return SpatialProbabilisticLayer(
            variables=["rain", "temp", "wind"], copula_scope="global",
            transforms=2, hidden_features=(16, 16), bayesian=True, seed=0,
        ).fit(data, coords, epochs=40, patience=15, verbose=False)

    def test_posterior_returns_a_distribution(self, bayes_layer):
        r = bayes_layer.posterior_exceedance(
            {"rain": (">", 10.0)}, mode="mean", n=1500, n_posterior=4)
        assert set(r) == {"samples", "mean", "std", "q05", "q95"}
        assert len(r["samples"]) == 4
        assert r["q05"] <= r["mean"] <= r["q95"]

    def test_posterior_requires_bayesian(self, layer):
        with pytest.raises(RuntimeError, match="bayesian=True"):
            layer.posterior_exceedance({"rain": (">", 1.0)}, n_posterior=2)

    def test_posterior_rejects_array_modes(self, bayes_layer):
        with pytest.raises(ValueError, match="scalar mode"):
            bayes_layer.posterior_exceedance(
                {"rain": (">", 1.0)}, mode="pointwise", n_posterior=2)

    def test_posterior_restores_the_point_estimate_model(self, bayes_layer):
        before = {k: c.flow for k, c in bayes_layer.copulas_.items()}
        bayes_layer.posterior_exceedance({"rain": (">", 10.0)}, n=800, n_posterior=3)
        assert all(bayes_layer.copulas_[k].flow is before[k] for k in before)


class TestIO:
    def test_save_writes_both_artifacts(self, layer, tmp_path):
        paths = save_layer(layer, tmp_path / "lay", n=800)
        assert paths["bundle"].exists() and paths["netcdf"].exists()

    def test_round_trip_reproduces_samples(self, layer, tmp_path):
        save_layer(layer, tmp_path / "lay", netcdf=False)
        back = load_layer(tmp_path / "lay")
        assert np.allclose(layer.sample(n=200, seed=9), back.sample(n=200, seed=9))

    def test_round_trip_preserves_metadata(self, layer, tmp_path):
        save_layer(layer, tmp_path / "lay", netcdf=False)
        back = load_layer(tmp_path / "lay")
        assert back.variables == layer.variables
        assert back.copula_scope == layer.copula_scope
        assert np.allclose(back.coords_, layer.coords_)

    def test_dataset_is_gridded_when_coords_form_a_grid(self, layer):
        ds = layer_to_dataset(layer)
        assert set(ds.sizes) >= {"lat", "lon", "variable"}
        assert ds.sizes["lat"] == N_SIDE and ds.sizes["lon"] == N_SIDE

    def test_dataset_carries_spatial_parameters(self, layer):
        ds = layer_to_dataset(layer)
        for v in ("spatial_range", "spatial_nugget", "spatial_df", "tail_dependence_0km"):
            assert v in ds.data_vars

    def test_dataset_documents_that_there_is_no_time_axis(self, layer):
        ds = layer_to_dataset(layer)
        assert "time" not in ds.sizes
        assert "sample dimension" in ds.attrs["description"]

    def test_precomputed_field_is_a_probability_map(self, layer):
        ds = layer_to_dataset(layer, precompute={"compound": {"rain": (">", 15.0)}}, n=800)
        assert ds.compound.shape == (N_SIDE, N_SIDE)
        assert float(ds.compound.min()) >= 0.0 and float(ds.compound.max()) <= 1.0

    def test_netcdf_is_readable_by_xarray(self, layer, tmp_path):
        import xarray as xr
        paths = save_layer(layer, tmp_path / "lay", precompute={"c": {"rain": (">", 15.0)}}, n=600)
        ds = xr.open_dataset(paths["netcdf"])
        assert "c" in ds.data_vars
        assert ds.attrs["created_by"] == "neurocopula"
