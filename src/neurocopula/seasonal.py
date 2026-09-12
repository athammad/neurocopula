"""
Stratified layers: one fitted layer per season or month, over a fixed climate
window.

This exists for a statistical reason, not for convenience. A copula assumes its
rows are i.i.d. draws from one joint distribution. Daily climate data breaks that
assumption twice over, and both breaks are handled here:

**Seasonality.** January and July are different climates. Pooling them produces a
two-humped mixture that describes neither, and the "dependence" estimated from it
is partly just the difference between the two seasons. Fitting January's layer on
January days only is the stationarisation step -- it does the job that an
ARMA-GARCH filter does in the standard vine-copula workflow for financial series.

**Long-term trend.** 1950 and 2025 are different climates too. Rather than
detrending, the data is restricted to a *climatological standard normal* window:
a 30-year period, by WMO convention the most recent one ending in a year ending in
zero, so 1991-2020 at present.

    World Meteorological Organization (2017). WMO Guidelines on the Calculation
    of Climate Normals, WMO-No. 1203. Geneva.
    https://library.wmo.int/viewer/55797/download?file=1203_en.pdf&type=pdf

Both choices deliberately keep the data in **physical units** throughout. The
alternative -- filtering to residuals or anomalies and converting back at query
time -- is equally valid and is what the finance workflow does, but it needs a
conversion layer wrapped around every query, and that layer is one more thing
that can be silently wrong. Stratifying instead means a threshold of 50 mm is
50 mm at every stage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .layer import SpatialProbabilisticLayer

__all__ = ["SeasonalLayers", "WMO_NORMAL", "SEASONS"]

#: The current WMO climatological standard normal period.
WMO_NORMAL = (1991, 2020)

#: Meteorological seasons, as month numbers.
SEASONS = {
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
}


class SeasonalLayers:
    """
    A dictionary of fitted layers, one per stratum, sharing a query interface.

    Use `SeasonalLayers.fit`; the constructor takes already-fitted layers.

    Attributes
    ----------
    layers:
        ``{key: SpatialProbabilisticLayer}``. Keys are month numbers (1-12),
        season codes (``"DJF"`` ...), or ``"annual"``.
    by:
        The stratification used.
    period:
        The (start_year, end_year) window the data was restricted to, inclusive.
    n_days_:
        Rows that went into each stratum -- worth checking, since a stratum with
        too few days will have a noisy fit.
    """

    def __init__(self, layers: dict, by: str, period, n_days: dict) -> None:
        self.layers = layers
        self.by = by
        self.period = period
        self.n_days_ = n_days

    # -----------------------------------------------------------------
    @classmethod
    def fit(
        cls,
        data: np.ndarray,
        coords: np.ndarray,
        dates,
        variables: Sequence[str],
        by: str = "season",
        period: tuple[int, int] | None = WMO_NORMAL,
        min_days: int = 300,
        epochs: int = 400,
        patience: int | None = 50,
        batch_size="auto",
        verbose: bool = True,
        **layer_kwargs,
    ) -> SeasonalLayers:
        """
        Fit one layer per stratum.

        data: array (n_time, n_cells, n_vars) in physical units.
        coords: array (n_cells, 2).
        dates: length n_time, anything `pandas.to_datetime` accepts.
        by: ``"season"`` (four layers), ``"month"`` (twelve), or ``"annual"``
            (one -- useful as a baseline, but it pools seasons and so violates
            the i.i.d. assumption this class exists to satisfy).
        period: ``(start_year, end_year)`` inclusive, or None for all years.
            Defaults to the current WMO standard normal, 1991-2020.
        min_days: warn if a stratum has fewer rows than this.
        epochs, patience, batch_size: training settings, forwarded to each
            stratum's `fit`.
        layer_kwargs: forwarded to each `SpatialProbabilisticLayer` constructor
            (marginal_specs, copula_scope, bayesian, ...).

        The strata are independent, so this loop is the natural place to
        parallelise on a cluster: each stratum is a separate job.
        """
        if by not in ("season", "month", "annual"):
            raise ValueError("by must be 'season', 'month' or 'annual'")
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 3:
            raise ValueError(f"data must be 3-D (time, cells, vars), got {data.shape}")
        idx = pd.DatetimeIndex(pd.to_datetime(dates))
        if len(idx) != len(data):
            raise ValueError(f"{len(idx)} dates for {len(data)} time steps")

        if period is not None:
            lo, hi = period
            in_window = (idx.year >= lo) & (idx.year <= hi)
            if not in_window.any():
                raise ValueError(
                    f"no data falls inside the period {lo}-{hi}; the record covers "
                    f"{idx.year.min()}-{idx.year.max()}"
                )
            if verbose:
                dropped = int((~in_window).sum())
                print(f"climate window {lo}-{hi}: keeping {int(in_window.sum())} days, "
                      f"dropping {dropped}")
            data, idx = data[in_window], idx[in_window]

        strata = cls._strata(idx, by)
        layers, n_days = {}, {}
        for key, mask in strata.items():
            n = int(mask.sum())
            n_days[key] = n
            if n == 0:
                continue
            if n < min_days:
                import warnings
                warnings.warn(
                    f"stratum {key!r} has only {n} days; its fit will be noisy.",
                    RuntimeWarning, stacklevel=2,
                )
            if verbose:
                print(f"\n=== fitting stratum {key!r}  ({n} days) ===")
            layers[key] = SpatialProbabilisticLayer(
                variables=variables, **layer_kwargs
            ).fit(data[mask], coords, epochs=epochs, patience=patience,
                  batch_size=batch_size, verbose=verbose)

        return cls(layers, by, period, n_days)

    @staticmethod
    def _strata(idx: pd.DatetimeIndex, by: str) -> dict:
        if by == "annual":
            return {"annual": np.ones(len(idx), dtype=bool)}
        if by == "month":
            return {m: np.asarray(idx.month == m) for m in range(1, 13)}
        return {name: np.asarray(np.isin(idx.month, months))
                for name, months in SEASONS.items()}

    # -----------------------------------------------------------------
    def _resolve(self, key=None):
        """Pick a stratum, accepting a month number, a season code, or None."""
        if key is None:
            if len(self.layers) == 1:
                return next(iter(self.layers.values()))
            raise ValueError(
                f"this is stratified by {self.by!r}; name a stratum, one of "
                f"{sorted(self.layers, key=str)}"
            )
        if key in self.layers:
            return self.layers[key]
        # Convenience: a month number resolves to its season when stratified by season.
        if self.by == "season" and isinstance(key, (int, np.integer)):
            for name, months in SEASONS.items():
                if key in months and name in self.layers:
                    return self.layers[name]
        raise KeyError(f"no stratum {key!r}; have {sorted(self.layers, key=str)}")

    # -----------------------------------------------------------------
    def select(self, *args, **kwargs) -> np.ndarray:
        """Select cells. The grid is shared, so any stratum can answer."""
        return next(iter(self.layers.values())).select(*args, **kwargs)

    def sample(self, cells=None, n: int = 1000, when=None, seed: int | None = None):
        """Draw joint realizations from one stratum. `when` names the stratum."""
        return self._resolve(when).sample(cells=cells, n=n, seed=seed)

    def exceedance_probability(
        self,
        conditions: Mapping[str, tuple[str, float]],
        cells=None,
        when=None,
        mode: str = "mean",
        n: int = 20_000,
        seed: int | None = None,
    ):
        """
        Exceedance probability within one stratum.

        `when` selects it: a month number, a season code, or None when there is
        only one. Everything else matches
        `SpatialProbabilisticLayer.exceedance_probability`.
        """
        return self._resolve(when).exceedance_probability(
            conditions, cells=cells, mode=mode, n=n, seed=seed)

    def posterior_exceedance(self, conditions, cells=None, when=None, **kwargs):
        """Bayesian credible interval on an exceedance probability."""
        return self._resolve(when).posterior_exceedance(conditions, cells=cells, **kwargs)

    def seasonal_cycle(
        self,
        conditions: Mapping[str, tuple[str, float]],
        cells=None,
        mode: str = "mean",
        n: int = 20_000,
    ) -> pd.Series:
        """
        The same question asked of every stratum, as a series.

        This is the payoff of stratifying: the seasonal cycle of compound-extreme
        risk, which a single pooled layer averages away entirely.
        """
        out = {
            key: self.layers[key].exceedance_probability(
                conditions, cells=cells, mode=mode, n=n)
            for key in self.layers
        }
        return pd.Series(out, name="probability").sort_index()

    # -----------------------------------------------------------------
    def to_dataset(self, precompute=None, n: int = 20_000):
        """
        Every stratum in **one** `xarray.Dataset`, with the stratum as a
        dimension.

        A NetCDF takes arbitrary dimensions, so there is no reason to split
        strata across files: they share a grid, and a single file is one thing
        to hand over, opens with ``.sel(season="DJF")``, and lets a reader
        difference two seasons without merging anything first.

        precompute: ``{field_name: conditions}`` evaluated for every stratum and
            stored as a (stratum, lat, lon) field.
        """
        import xarray as xr

        from .io import layer_to_dataset

        dim = "season" if self.by == "season" else ("month" if self.by == "month"
                                                    else "stratum")
        keys = self.keys
        parts = [layer_to_dataset(self.layers[k], precompute=precompute, n=n)
                 for k in keys]
        ds = xr.concat(parts, dim=dim)
        ds = ds.assign_coords({dim: keys})
        ds[dim].attrs["description"] = f"stratum ({self.by}); layers fitted independently"
        ds["n_days"] = (dim, [self.n_days_.get(k, 0) for k in keys])
        ds["n_days"].attrs["description"] = "observations used to fit this stratum"
        ds.attrs["stratified_by"] = self.by
        if self.period:
            ds.attrs["period"] = f"{self.period[0]}-{self.period[1]}"
            ds.attrs["period_note"] = (
                "fitted over a fixed window; the layer describes that period's "
                "climate, not a trend across the full record"
            )
        return ds

    def save(self, directory, netcdf: bool = True, single_netcdf: bool = True,
             precompute=None, n: int = 20_000, **kwargs) -> dict:
        """
        Save the strata under `directory`.

        Each stratum's fitted model goes to its own bundle, since those hold
        live objects. The NetCDF side is written as a **single** file spanning
        every stratum unless `single_netcdf` is False -- that is the artifact a
        reader actually opens, and splitting it gains nothing.
        """
        import json
        from pathlib import Path

        from .io import save_layer

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = {}
        for key, lay in self.layers.items():
            paths[key] = save_layer(lay, directory / f"{self.by}_{key}",
                                    netcdf=netcdf and not single_netcdf,
                                    precompute=precompute, n=n, **kwargs)
        if netcdf and single_netcdf:
            combined = directory / "layer.nc"
            self.to_dataset(precompute=precompute, n=n).to_netcdf(combined)
            paths["netcdf"] = combined

        meta = {"by": self.by, "period": self.period, "n_days": self.n_days_,
                "keys": list(self.layers)}
        (directory / "strata.json").write_text(json.dumps(meta, indent=2, default=str))
        return paths

    @classmethod
    def load(cls, directory) -> SeasonalLayers:
        """Reload a saved set of strata."""
        import json
        from pathlib import Path

        from .io import load_layer

        directory = Path(directory)
        meta = json.loads((directory / "strata.json").read_text())
        by = meta["by"]
        layers = {}
        for key in meta["keys"]:
            k = int(key) if by == "month" else key
            layers[k] = load_layer(directory / f"{by}_{key}")
        n_days = {(int(k) if by == "month" else k): v for k, v in meta["n_days"].items()}
        return cls(layers, by, tuple(meta["period"]) if meta["period"] else None, n_days)

    # -----------------------------------------------------------------
    @property
    def keys(self):
        return sorted(self.layers, key=str)

    def summary(self) -> pd.DataFrame:
        """Rows per stratum and the spatial parameters each one fitted."""
        rows = []
        for key, lay in self.layers.items():
            row = {"stratum": key, "n_days": self.n_days_.get(key)}
            for var, sp in lay.fit_report_["spatial"].items():
                row[f"{var}_range_km"] = round(sp["range"], 1)
                row[f"{var}_df"] = sp["df"]
            rows.append(row)
        return pd.DataFrame(rows).set_index("stratum")

    def __repr__(self) -> str:
        period = f"{self.period[0]}-{self.period[1]}" if self.period else "all years"
        return (f"SeasonalLayers(by={self.by!r}, period={period}, "
                f"strata={self.keys})")
