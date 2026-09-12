"""
Persisting a fitted layer, and exporting it as NetCDF.

Two artifacts, because they serve different purposes and one cannot do both.

**The model bundle** (`.pt`) holds the live object: flow weights, marginals,
spatial parameters. It is what you reload to *ask new questions*. It is a
pickle, so only load bundles you trust.

**The NetCDF** (`.nc`) holds what a NetCDF is good at: a grid of numbers,
self-describing, readable by anyone with `xarray`, `ncview`, R or GDAL, with no
Python objects and no dependency on this library. It cannot contain a live
flow, so it holds the grid, the fitted spatial parameters, per-location summary
statistics, and whatever probability fields you chose to precompute.

The division of labour follows from that: the NetCDF is for *looking*, the
bundle is for *asking*. A collaborator who only wants a map of compound-extreme
probability needs the NetCDF alone; one who wants to pose a new threshold
combination needs the bundle.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

__all__ = ["save_layer", "load_layer", "layer_to_dataset", "precompute_field"]


def save_layer(layer, path: str | Path, netcdf: bool = True,
               precompute: Mapping[str, Mapping] | None = None,
               n: int = 20_000) -> dict[str, Path]:
    """
    Save a fitted `SpatialProbabilisticLayer`.

    path:
        Base path without extension. Writes ``<path>.pt`` and, unless
        `netcdf` is False, ``<path>.nc``.
    precompute:
        Optional ``{field_name: conditions}`` to evaluate across the whole grid
        and store in the NetCDF, e.g.
        ``{"compound": {"rain": (">", 50), "temp": (">", 35)}}``.
        Each becomes a lat/lon field of probabilities that can be mapped
        without reloading the model.
    n:
        Monte Carlo draws per precomputed field.

    Returns: ``{"bundle": Path, "netcdf": Path}``.
    """
    layer._check_fitted()
    path = Path(path)
    out: dict[str, Path] = {}

    bundle = path.with_suffix(".pt")
    torch.save(
        {
            "format_version": 1,
            "variables": layer.variables,
            "coords": layer.coords_,
            "marginals": layer.marginals_,
            "copulas": layer.copulas_,
            "spatial": layer.spatial_,
            "config": {
                "marginal_specs": layer.marginal_specs,
                "copula_scope": layer.copula_scope,
                "min_cell_rows": layer.min_cell_rows,
                "max_copula_rows": layer.max_copula_rows,
                "transforms": layer.transforms,
                "hidden_features": layer.hidden_features,
                "sampling_mode": layer.sampling_mode,
                "bayesian": layer.bayesian,
                "spatial_model": layer.spatial_model,
                "distance": layer.distance,
                "seed": layer.seed,
            },
            "fit_report": layer.fit_report_,
        },
        bundle,
    )
    out["bundle"] = bundle

    if netcdf:
        ds = layer_to_dataset(layer, precompute=precompute, n=n)
        nc = path.with_suffix(".nc")
        ds.to_netcdf(nc)
        out["netcdf"] = nc
    return out


def load_layer(path: str | Path):
    """
    Reload a layer from its bundle. Only load bundles you trust -- this
    unpickles Python objects, with the usual caveat that applies to any
    `torch.save` artifact.
    """
    from .layer import SpatialProbabilisticLayer

    path = Path(path).with_suffix(".pt")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    layer = SpatialProbabilisticLayer(variables=blob["variables"], **blob["config"])
    layer.coords_ = blob["coords"]
    layer.marginals_ = blob["marginals"]
    layer.copulas_ = blob["copulas"]
    layer.spatial_ = blob["spatial"]
    layer.fit_report_ = blob["fit_report"]
    return layer


def _grid_shape(coords: np.ndarray):
    """
    Recover (lat, lon) axes if the locations form a regular grid.

    Worth the trouble: a NetCDF on a proper 2-D grid opens correctly in every
    GIS tool, whereas an unstructured point list does not. Falls back to a
    flat location axis when the coordinates are irregular.
    """
    lats = np.unique(coords[:, 0])
    lons = np.unique(coords[:, 1])
    if len(lats) * len(lons) != len(coords):
        return None, None
    order = np.lexsort((coords[:, 1], coords[:, 0]))
    if not np.array_equal(
        coords[order],
        np.column_stack([np.repeat(lats, len(lons)), np.tile(lons, len(lats))]),
    ):
        return None, None
    return lats, lons


def layer_to_dataset(layer, precompute: Mapping[str, Mapping] | None = None,
                     n: int = 20_000):
    """
    Build the inspectable `xarray.Dataset` for a fitted layer.

    Contains the grid, the fitted spatial parameters per variable, per-location
    marginal summaries, and any precomputed probability fields.
    """
    import xarray as xr

    layer._check_fitted()
    coords = layer.coords_
    lats, lons = _grid_shape(coords)
    gridded = lats is not None

    def reshape(v):
        return v.reshape(len(lats), len(lons)) if gridded else v

    dims = ("lat", "lon") if gridded else ("location",)
    base_coords = (
        {"lat": lats, "lon": lons} if gridded
        else {"location": np.arange(len(coords)),
              "cell_lat": ("location", coords[:, 0]),
              "cell_lon": ("location", coords[:, 1])}
    )

    data_vars: dict = {}

    # Per-location marginal summaries: enough to see the climate at a glance
    # without touching the model.
    for name in layer.variables:
        p_zero = np.zeros(len(coords))
        for s in range(len(coords)):
            mset = layer.marginals_._set_for(s)
            p_zero[s] = mset[name].p_zero_
        data_vars[f"{name}_p_zero"] = (dims, reshape(p_zero))

    # Fitted spatial parameters, one scalar set per variable.
    var_axis = list(layer.variables)
    for key in ("range", "effective_range", "nugget", "df"):
        vals = [layer.spatial_[v].summary()[key] for v in var_axis]
        data_vars[f"spatial_{key}"] = ("variable", np.asarray(vals, dtype=float))
    data_vars["spatial_model"] = (
        "variable", np.array([layer.spatial_[v].model_ for v in var_axis], dtype=object)
    )
    data_vars["tail_dependence_0km"] = (
        "variable",
        np.array([layer.spatial_[v].tail_dependence(0.0) for v in var_axis], dtype=float),
    )
    base_coords["variable"] = var_axis

    # Optional precomputed probability fields.
    if precompute:
        for field, conditions in precompute.items():
            p = layer.exceedance_probability(conditions, cells=None, mode="pointwise", n=n)
            data_vars[field] = (dims, reshape(p))

    ds = xr.Dataset(data_vars, coords=base_coords)
    ds.attrs.update({
        "title": "Spatial probabilistic layer",
        "description": (
            "Climatological joint distribution per location with a spatial "
            "dependence model. The time axis is the sample dimension and has "
            "been integrated over: this layer has no time coordinate and "
            "cannot answer questions about particular dates."
        ),
        "variables": ", ".join(layer.variables),
        "n_time_samples": int(layer.fit_report_["n_time"]),
        "n_cells": int(layer.fit_report_["n_cells"]),
        "copula_scope": layer.copula_scope,
        "bayesian": str(layer.bayesian),
        "distance_metric": layer.distance,
        "note_spatial_units": "range and effective_range are in km for haversine distance",
        "note_df": "spatial_df = inf means a Gaussian field, i.e. no tail dependence",
        "created_by": "neurocopula",
    })
    if gridded:
        ds.lat.attrs.update(units="degrees_north", standard_name="latitude")
        ds.lon.attrs.update(units="degrees_east", standard_name="longitude")
    return ds


def precompute_field(layer, conditions: Mapping[str, tuple[str, float]],
                     n: int = 20_000) -> np.ndarray:
    """
    Pointwise exceedance probability across every location, as a flat array.

    A thin wrapper kept separate so a caller can build several fields and
    assemble their own dataset.
    """
    return layer.exceedance_probability(conditions, cells=None, mode="pointwise", n=n)
