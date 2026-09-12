"""
Tests for stratified (seasonal) layers.

The substantive test here is `test_seasonal_cycle_detects_the_wet_season`: the
whole reason to stratify is that a pooled layer averages the seasonal cycle away.
If stratifying did not recover a difference between a wet season and a dry one on
data built to have one, the class would be pure overhead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from scipy.stats import t as student_t

from neurocopula.seasonal import SEASONS, WMO_NORMAL, SeasonalLayers
from neurocopula.spatial import CORRELATION_MODELS, haversine_matrix

FAST = dict(transforms=2, hidden_features=(16, 16), seed=0)
FIT = dict(epochs=40, patience=15)


@pytest.fixture(scope="module")
def coords():
    lats = np.linspace(-5, 5, 4)
    lons = np.linspace(100, 110, 4)
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    return np.column_stack([LA.ravel(), LO.ravel()])


@pytest.fixture(scope="module")
def dated_data(coords):
    """Daily data 1988-2022 with a deliberately strong wet season in DJF."""
    m = len(coords)
    D = haversine_matrix(coords)
    C = CORRELATION_MODELS["exponential"](D, 300.0)
    np.fill_diagonal(C, 1.0)
    C[np.diag_indices_from(C)] += 1e-8
    L = np.linalg.cholesky(C)

    dates = pd.date_range("1988-01-01", "2022-12-31", freq="D")
    n = len(dates)
    rng = np.random.default_rng(0)
    w = rng.standard_normal((n, m)) @ L.T
    g = rng.chisquare(5, size=(n, 1)) / 5
    z = norm.ppf(np.clip(student_t.cdf(w / np.sqrt(g), 5), 1e-10, 1 - 1e-10))

    wet = np.isin(dates.month, [12, 1, 2]).astype(float)[:, None]
    u = norm.cdf(z)
    p_dry = 0.75 - 0.45 * wet          # DJF much wetter than the rest
    rain = np.where(u < p_dry, 0.0,
                    15 * (-np.log(np.clip(1 - (u - p_dry) / (1 - p_dry), 1e-12, 1))))
    temp = 28 + 3 * z + 1.5 * wet
    return np.stack([rain, temp], axis=-1), dates


@pytest.fixture(scope="module")
def seasonal(dated_data, coords):
    data, dates = dated_data
    return SeasonalLayers.fit(
        data, coords, dates, variables=["rain", "temp"], by="season",
        period=WMO_NORMAL,
        marginal_specs={"rain": dict(zero_inflated=True, zero_threshold=0.05)},
        verbose=False, **FAST, **FIT,
    )


class TestWindowing:
    def test_wmo_normal_is_a_30_year_window_ending_in_zero(self):
        lo, hi = WMO_NORMAL
        assert hi - lo + 1 == 30
        assert hi % 10 == 0

    def test_period_restricts_the_rows_used(self, dated_data, coords):
        data, dates = dated_data
        sl = SeasonalLayers.fit(
            data, coords, dates, variables=["rain", "temp"], by="annual",
            period=(2001, 2010), verbose=False, **FAST, **FIT)
        # 10 years of daily data, give or take leap days.
        assert 3600 < sl.n_days_["annual"] < 3700

    def test_period_none_keeps_everything(self, dated_data, coords):
        data, dates = dated_data
        sl = SeasonalLayers.fit(
            data, coords, dates, variables=["rain", "temp"], by="annual",
            period=None, verbose=False, **FAST, **FIT)
        assert sl.n_days_["annual"] == len(dates)

    def test_period_outside_the_record_raises(self, dated_data, coords):
        data, dates = dated_data
        with pytest.raises(ValueError, match="no data falls inside"):
            SeasonalLayers.fit(data, coords, dates, variables=["rain", "temp"],
                               period=(1800, 1810), verbose=False, **FAST, **FIT)


class TestStratification:
    def test_season_produces_four_strata(self, seasonal):
        assert set(seasonal.keys) == set(SEASONS)

    def test_month_produces_twelve_strata(self, dated_data, coords):
        data, dates = dated_data
        sl = SeasonalLayers.fit(
            data, coords, dates, variables=["rain", "temp"], by="month",
            period=(2011, 2020), verbose=False, **FAST, **FIT)
        assert set(sl.keys) == set(range(1, 13))

    def test_strata_partition_the_window(self, seasonal):
        # Every retained day lands in exactly one stratum.
        total = sum(seasonal.n_days_.values())
        assert 10_900 < total < 11_000        # ~30 years of days

    def test_rejects_unknown_stratification(self, dated_data, coords):
        data, dates = dated_data
        with pytest.raises(ValueError, match="by must be"):
            SeasonalLayers.fit(data, coords, dates, variables=["rain", "temp"],
                               by="weekly", verbose=False, **FAST, **FIT)

    def test_rejects_date_length_mismatch(self, dated_data, coords):
        data, dates = dated_data
        with pytest.raises(ValueError, match="dates for"):
            SeasonalLayers.fit(data, coords, dates[:10], variables=["rain", "temp"],
                               verbose=False, **FAST, **FIT)


class TestQuerying:
    CONDS = {"rain": (">", 20.0)}

    def test_seasonal_cycle_detects_the_wet_season(self, seasonal):
        # The point of the whole class. DJF was built to be far wetter; a
        # pooled layer would report one number for the whole year.
        cycle = seasonal.seasonal_cycle(self.CONDS, mode="mean", n=6000)
        assert cycle["DJF"] > 2 * cycle["JJA"]

    def test_seasonal_cycle_returns_one_value_per_stratum(self, seasonal):
        cycle = seasonal.seasonal_cycle(self.CONDS, mode="mean", n=2000)
        assert set(cycle.index) == set(SEASONS)

    def test_month_number_resolves_to_its_season(self, seasonal):
        a = seasonal.exceedance_probability(self.CONDS, when=1, mode="mean", n=3000, seed=1)
        b = seasonal.exceedance_probability(self.CONDS, when="DJF", mode="mean", n=3000, seed=1)
        assert a == pytest.approx(b)

    def test_ambiguous_stratum_raises(self, seasonal):
        with pytest.raises(ValueError, match="name a stratum"):
            seasonal.exceedance_probability(self.CONDS, n=500)

    def test_unknown_stratum_raises(self, seasonal):
        with pytest.raises(KeyError, match="no stratum"):
            seasonal.exceedance_probability(self.CONDS, when="XYZ", n=500)

    def test_annual_needs_no_stratum_name(self, dated_data, coords):
        data, dates = dated_data
        sl = SeasonalLayers.fit(
            data, coords, dates, variables=["rain", "temp"], by="annual",
            period=(2016, 2020), verbose=False, **FAST, **FIT)
        assert 0.0 <= sl.exceedance_probability(self.CONDS, n=2000) <= 1.0

    def test_sample_shape(self, seasonal):
        s = seasonal.sample(n=50, when="DJF")
        assert s.shape[0] == 50 and s.shape[2] == 2

    def test_select_is_shared_across_strata(self, seasonal):
        sel = seasonal.select(bbox=(-2, 2, 103, 107))
        assert len(sel) > 0
        p = seasonal.exceedance_probability(self.CONDS, cells=sel, when="DJF",
                                            mode="any", n=2000)
        assert 0.0 <= p <= 1.0

    def test_summary_lists_every_stratum(self, seasonal):
        s = seasonal.summary()
        assert len(s) == 4
        assert "n_days" in s.columns
        assert any(c.endswith("_range_km") for c in s.columns)


class TestPersistence:
    def test_round_trip(self, seasonal, tmp_path):
        seasonal.save(tmp_path / "strata", netcdf=False)
        back = SeasonalLayers.load(tmp_path / "strata")
        assert set(back.keys) == set(seasonal.keys)
        assert back.by == seasonal.by
        assert back.period == seasonal.period

    def test_round_trip_reproduces_samples(self, seasonal, tmp_path):
        seasonal.save(tmp_path / "strata", netcdf=False)
        back = SeasonalLayers.load(tmp_path / "strata")
        a = seasonal.sample(n=100, when="DJF", seed=3)
        b = back.sample(n=100, when="DJF", seed=3)
        assert np.allclose(a, b)

    def test_month_keys_survive_as_integers(self, dated_data, coords, tmp_path):
        data, dates = dated_data
        sl = SeasonalLayers.fit(
            data, coords, dates, variables=["rain", "temp"], by="month",
            period=(2016, 2020), verbose=False, **FAST, **FIT)
        sl.save(tmp_path / "m", netcdf=False)
        back = SeasonalLayers.load(tmp_path / "m")
        assert set(back.keys) == set(range(1, 13))
        assert 0.0 <= back.exceedance_probability({"rain": (">", 10.0)}, when=7, n=500) <= 1.0
