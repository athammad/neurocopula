"""
The spatial probabilistic layer: the queryable object.

This is where the two halves meet. Per-location marginals and a per-location
copula describe what happens *at* a point; a spatial process describes whether
points go wrong *together*. Combined, they answer questions about an arbitrary
region chosen long after the model was fitted.

How the two compose
-------------------

The order matters, and only one order is correct.

A normalizing flow already maps data to independent latent variables -- that is
what a flow *is*, and it is exactly a Rosenblatt transform. So the pipeline is:

    physical units
        -> per-location marginals            (uniform margins)
        -> probit                            (standard normal margins)
        -> flow's forward transform          (independent latents, per location)
        -> spatial process on the LATENTS    (correlation between locations)

and sampling runs the same chain backwards. Correlating the latents and then
pushing them back through the flow inherits the spatial dependence *and*
preserves the cross-variable copula at every location. Doing it the other way
round -- imposing spatial correlation on the observed scale -- would overwrite
whichever structure was applied first.

What a query costs
------------------

Because the spatial process is defined by distance, the joint distribution over
any subset of locations is available directly: pull those rows and columns and
factorize. Cost scales with the region asked for, not with the size of the
domain, so a 500-cell basin is cheap even when the full grid has hundreds of
thousands of cells.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from scipy.stats import norm

from .core import NeuroCopula
from .semiparametric import GroupedMarginalTransform
from .spatial import SpatialDependence

__all__ = ["SpatialProbabilisticLayer"]

_OPS = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal}


class SpatialProbabilisticLayer:
    """
    A fitted probabilistic layer over a grid.

    Parameters
    ----------
    variables:
        Variable names, in the order of the last axis of the data.
    marginal_specs:
        ``{variable: kwargs}`` for `SemiParametricMarginal`, e.g.
        ``{"precip": dict(zero_inflated=True, zero_threshold=0.1, tail="gpd")}``.
        Variables not listed get a plain empirical marginal.
    copula_scope:
        ``"cell"`` fits a separate cross-variable copula at every location --
        necessary if you want to ask *where* compound dependence is strongest,
        since a single shared copula makes that constant by construction. It is
        embarrassingly parallel but costs one flow fit per location.
        ``"global"`` (default) pools every location into one copula: far
        cheaper, and the right choice when the question is about levels rather
        than about spatial variation in dependence.
    min_cell_rows:
        Locations with fewer rows than this fall back to pooled marginals.
    max_copula_rows:
        Cap on rows used to fit the copula, subsampled at random above it.
        Pooling every location yields millions of rows to constrain a
        distribution with as many dimensions as there are variables -- usually
        three. Tens of thousands of rows already determine that far more
        tightly than any other approximation in the model, so the extra rows
        buy nothing and cost a great deal of training time. Raise it if the
        copula is genuinely high-dimensional.
    transforms, hidden_features, sampling_mode, bayesian:
        Passed through to `NeuroCopula` for the copula layer.
    spatial_model, distance:
        Passed through to `SpatialDependence` for the spatial layer.
    seed:
        Seed for every stochastic step.
    """

    def __init__(
        self,
        variables: Sequence[str],
        marginal_specs: dict[str, dict] | None = None,
        copula_scope: str = "global",
        min_cell_rows: int = 200,
        max_copula_rows: int = 100_000,
        transforms: int = 4,
        hidden_features: Sequence[int] = (64, 64),
        sampling_mode: str = "coupling",
        bayesian: bool = False,
        spatial_model: str = "auto",
        distance: str = "haversine",
        seed: int = 0,
    ) -> None:
        if copula_scope not in ("cell", "global"):
            raise ValueError("copula_scope must be 'cell' or 'global'")
        self.variables = list(variables)
        self.marginal_specs = dict(marginal_specs or {})
        self.copula_scope = copula_scope
        self.min_cell_rows = min_cell_rows
        self.max_copula_rows = max_copula_rows
        self.transforms = transforms
        self.hidden_features = tuple(hidden_features)
        self.sampling_mode = sampling_mode
        self.bayesian = bayesian
        self.spatial_model = spatial_model
        self.distance = distance
        self.seed = seed

        self.coords_: np.ndarray | None = None
        self.marginals_: GroupedMarginalTransform | None = None
        self.copulas_: dict[int | None, NeuroCopula] = {}
        self.spatial_: dict[str, SpatialDependence] = {}
        self.regions_ = None
        self.fit_report_: dict | None = None

    # =================================================================
    # fitting
    # =================================================================
    def fit(
        self,
        data: np.ndarray,
        coords: np.ndarray,
        epochs: int = 400,
        patience: int | None = 50,
        batch_size: int | None = "auto",
        verbose: bool = True,
    ) -> SpatialProbabilisticLayer:
        """
        data: array (n_time, n_cells, n_vars) in physical units.
        coords: array (n_cells, 2) of latitude, longitude (or projected x, y).
        batch_size: minibatch size for the copula fit. ``"auto"`` (default)
            picks 4096 once the pooled training set exceeds 50,000 rows and
            stays full-batch below that -- pooling every location easily
            reaches millions of rows, where full-batch steps are wasteful.

        The time axis is the SAMPLE dimension. It is integrated over, not
        indexed: the result is a climatology with no time axis, and queries
        about particular dates are not possible by design.
        """
        data = np.asarray(data, dtype=np.float64)
        coords = np.atleast_2d(np.asarray(coords, dtype=np.float64))
        if data.ndim != 3:
            raise ValueError(
                f"data must be 3-D (time, cells, variables), got shape {data.shape}"
            )
        n_time, n_cells, n_vars = data.shape
        if n_vars != len(self.variables):
            raise ValueError(f"data has {n_vars} variables but {len(self.variables)} names")
        if len(coords) != n_cells:
            raise ValueError(f"data has {n_cells} cells but {len(coords)} coordinates")
        if not np.isfinite(data).all():
            raise ValueError("data contains NaN or infinite values; clean or mask them first")

        self.coords_ = coords
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # --- marginals, always per location -------------------------------
        # Cheap, and it removes every spatial difference in level and spread
        # so the layers above only ever see dependence.
        if verbose:
            print(f"[1/3] fitting per-cell marginals for {n_cells} cells ...")
        flat = data.reshape(n_time * n_cells, n_vars)
        cell_of_row = np.tile(np.arange(n_cells), n_time)
        self.marginals_ = GroupedMarginalTransform(
            columns=self.variables, specs=self.marginal_specs, scope="group",
            min_group_size=self.min_cell_rows, seed=self.seed,
        ).fit(flat, cell_of_row)

        z_flat = self.marginals_.forward(flat, cell_of_row)
        z = z_flat.reshape(n_time, n_cells, n_vars)

        # --- copula layer --------------------------------------------------
        if verbose:
            scope = "one per cell" if self.copula_scope == "cell" else "one shared"
            print(f"[2/3] fitting cross-variable copula ({scope}) ...")
        self._fit_copulas(z, epochs, patience, batch_size, verbose)

        # --- latents, then the spatial layer --------------------------------
        e = self._to_latent(z)
        raw = data
        if verbose:
            print("[3/3] fitting spatial dependence on latents, per variable ...")
        for v, name in enumerate(self.variables):
            # A zero-inflated variable needs different treatment here. Its dry
            # days were given independent random ranks (correct for the
            # marginal), which erases exactly the co-occurrence information the
            # correlogram depends on -- so the correlation is recovered from
            # the dry/wet pattern instead. Its degrees of freedom also have to
            # come from the UPPER tail, since the lower tail is the atom.
            spec = self.marginal_specs.get(name, {})
            is_censored = bool(spec.get("zero_inflated", False))
            censored = None
            if is_censored:
                thr = spec.get("zero_threshold", 0.0)
                censored = raw[:, :, v] <= thr
            self.spatial_[name] = SpatialDependence(
                model=self.spatial_model, distance=self.distance, seed=self.seed,
            ).fit(
                e[:, :, v], coords, censored=censored,
                df_tail="upper" if is_censored else "lower",
            )
            if verbose:
                tag = "  [censored-aware]" if is_censored else ""
                print(f"      {name}: {self.spatial_[name]}{tag}")

        self.fit_report_ = {
            "n_time": n_time,
            "n_cells": n_cells,
            "n_variables": n_vars,
            "copula_scope": self.copula_scope,
            "marginal_coverage": self.marginals_.coverage_report(),
            "spatial": {k: v.summary() for k, v in self.spatial_.items()},
        }
        return self

    def _fit_copulas(self, z, epochs, patience, batch_size, verbose) -> None:
        import pandas as pd
        n_time, n_cells, _ = z.shape
        n_rows = (min(n_time * n_cells, self.max_copula_rows)
                  if self.copula_scope == "global" else n_time)
        if batch_size == "auto":
            batch_size = 4096 if n_rows > 50_000 else None

        def new_flow():
            return NeuroCopula(
                transforms=self.transforms, hidden_features=self.hidden_features,
                copula_mode=False,   # z is already Gaussianized by the marginals
                sampling_mode=self.sampling_mode, bayesian=self.bayesian,
                seed=self.seed,
            )

        if self.copula_scope == "global":
            pooled = z.reshape(-1, len(self.variables))
            if len(pooled) > self.max_copula_rows:
                rng = np.random.default_rng(self.seed)
                pooled = pooled[rng.choice(len(pooled), self.max_copula_rows, replace=False)]
                if verbose:
                    print(f"      subsampled to {self.max_copula_rows:,} rows "
                          f"(ample for a {len(self.variables)}-dimensional copula)")
            df = pd.DataFrame(pooled, columns=self.variables)
            self.copulas_[None] = new_flow().fit(
                df, epochs=epochs, patience=patience, batch_size=batch_size,
                split_method="random", verbose=False,
            )
        else:
            for s in range(n_cells):
                df = pd.DataFrame(z[:, s, :], columns=self.variables)
                self.copulas_[s] = new_flow().fit(
                    df, epochs=epochs, patience=patience, batch_size=batch_size,
                    split_method="random", verbose=False,
                )
                if verbose and (s + 1) % max(n_cells // 10, 1) == 0:
                    print(f"      {s + 1}/{n_cells} cells")

    def _copula_for(self, cell: int) -> NeuroCopula:
        return self.copulas_[None] if self.copula_scope == "global" else self.copulas_[cell]

    # -----------------------------------------------------------------
    @torch.no_grad()
    def _to_latent(self, z: np.ndarray) -> np.ndarray:
        """Gaussianized observations -> independent latents, per location."""
        n_time, n_cells, n_vars = z.shape
        out = np.empty_like(z)
        if self.copula_scope == "global":
            t = self.copulas_[None].flow().transform
            flat = torch.tensor(z.reshape(-1, n_vars), dtype=torch.float32)
            out = t(flat).numpy().reshape(n_time, n_cells, n_vars)
        else:
            for s in range(n_cells):
                t = self.copulas_[s].flow().transform
                out[:, s, :] = t(torch.tensor(z[:, s, :], dtype=torch.float32)).numpy()
        return out.astype(np.float64)

    @torch.no_grad()
    def _from_latent(self, e: np.ndarray, cells: np.ndarray) -> np.ndarray:
        """Independent latents -> Gaussianized observations, per location."""
        n, m, n_vars = e.shape
        out = np.empty_like(e)
        if self.copula_scope == "global":
            t = self.copulas_[None].flow().transform
            flat = torch.tensor(e.reshape(-1, n_vars), dtype=torch.float32)
            out = t.inv(flat).numpy().reshape(n, m, n_vars)
        else:
            for k, s in enumerate(cells):
                t = self.copulas_[int(s)].flow().transform
                out[:, k, :] = t.inv(torch.tensor(e[:, k, :], dtype=torch.float32)).numpy()
        return out.astype(np.float64)

    # =================================================================
    # selection
    # =================================================================
    def attach_regions(self, regions) -> SpatialProbabilisticLayer:
        """
        Attach a `regions.RegionSet` so `select` accepts region names.

        The assignment of cells to regions is pure geometry, so it is computed
        once and reused for every later query.
        """
        self.regions_ = regions
        return self

    def select(
        self,
        bbox: tuple[float, float, float, float] | None = None,
        indices: Sequence[int] | None = None,
        mask: np.ndarray | None = None,
        region: str | None = None,
        max_cells: int | None = None,
        seed: int = 0,
    ) -> np.ndarray:
        """
        Choose the locations a query applies to.

        bbox: ``(lat_min, lat_max, lon_min, lon_max)``.
        indices: explicit positional indices.
        mask: boolean array over all locations.
        region: name of a region, once `attach_regions` has been called.
        max_cells:
            Thin the selection to at most this many cells, chosen at random.
            Joint sampling factorizes an n x n correlation matrix, which is
            O(n^3) in time and O(n^2) in memory: a 15,000-cell country needs
            about 1.8 GB and several minutes, while 2,000 cells takes under a
            second. For an AREAL statistic -- a regional mean, an affected
            fraction, whether anywhere was hit -- a random subset of the
            region is an unbiased estimate of the whole, so thinning costs
            accuracy that is small next to the Monte Carlo error already
            present. Leave it None to use every cell.
        seed: seed for that thinning, so a query is reproducible.

        Returns: integer array of location indices. With no argument, every
        location is selected.
        """
        self._check_fitted()
        n = len(self.coords_)
        given = sum(x is not None for x in (bbox, indices, mask, region))
        if given > 1:
            raise ValueError("give at most one of bbox, indices, mask, region")

        if region is not None:
            if getattr(self, "regions_", None) is None:
                raise RuntimeError(
                    "no regions attached; call attach_regions(RegionSet(...)) first"
                )
            mask = self.regions_.mask(region, self.coords_)
        if indices is not None:
            return self._thin(np.asarray(indices, dtype=int), max_cells, seed)
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if len(mask) != n:
                raise ValueError(f"mask has {len(mask)} entries for {n} locations")
            return self._thin(np.flatnonzero(mask), max_cells, seed)
        if bbox is not None:
            lat_min, lat_max, lon_min, lon_max = bbox
            lat, lon = self.coords_[:, 0], self.coords_[:, 1]
            sel = np.flatnonzero(
                (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)
            )
            if sel.size == 0:
                raise ValueError("the bounding box selects no locations")
            return self._thin(sel, max_cells, seed)
        return self._thin(np.arange(n), max_cells, seed)

    @staticmethod
    def _thin(sel: np.ndarray, max_cells: int | None, seed: int) -> np.ndarray:
        """Randomly reduce a selection, keeping it sorted."""
        if max_cells is None or len(sel) <= max_cells:
            return sel
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(sel, max_cells, replace=False))

    # =================================================================
    # sampling and queries
    # =================================================================
    def sample(
        self, cells: Sequence[int] | None = None, n: int = 1000,
        seed: int | None = None,
    ) -> np.ndarray:
        """
        Draw `n` joint realizations over the selected locations, in physical
        units.

        Every draw is one coherent "day": the locations in it are dependent on
        each other through the spatial process, and the variables within each
        location are dependent through that location's copula.

        Returns: array (n, n_selected, n_variables).
        """
        self._check_fitted()
        cells = np.arange(len(self.coords_)) if cells is None else np.asarray(cells, dtype=int)
        coords = self.coords_[cells]
        rng_seed = self.seed if seed is None else seed

        # Spatial layer: one correlated field per variable, in latent space.
        e = np.empty((n, len(cells), len(self.variables)))
        for v, name in enumerate(self.variables):
            e[:, :, v] = self.spatial_[name].sample(coords, n=n, seed=rng_seed + v)

        # Copula layer: latents -> Gaussianized observations, per location.
        z = self._from_latent(e, cells)

        # Marginal layer: back to physical units, using each location's own
        # marginals.
        u = norm.cdf(z)
        out = np.empty_like(u)
        for k, s in enumerate(cells):
            groups = np.full(n, s)
            out[:, k, :] = self.marginals_.from_uniform(u[:, k, :], groups)
        return out

    def exceedance_probability(
        self,
        conditions: Mapping[str, tuple[str, float]],
        cells: Sequence[int] | None = None,
        mode: str = "pointwise",
        n: int = 20_000,
        seed: int | None = None,
    ):
        """
        Probability that all `conditions` hold simultaneously, aggregated over
        the selected region.

        conditions:
            ``{variable: (op, value)}`` with `op` in ``<``, ``<=``, ``>``,
            ``>=``. All conditions must hold *at the same location on the same
            draw* for that location to count as hit.
        mode:
            How to aggregate across the region. The choice is not cosmetic --
            these are different questions with different answers:

            - ``"pointwise"``: probability at each location separately.
              Returns one value per location. Needs no spatial dependence.
            - ``"mean"``: the average of the pointwise probabilities, i.e.
              *at a typical point in the region*. Also independent of spatial
              dependence, by linearity of expectation.
            - ``"any"``: probability that **at least one** location is hit.
              Depends strongly on spatial dependence -- under independence
              this tends to 1 for a large region, which is badly wrong.
            - ``"all"``: probability that **every** location is hit.
            - ``"fraction"``: the distribution of the *fraction* of the region
              hit. Returns the draws, so any summary can be taken from them.
            - ``"count"``: same, as counts of locations.

        Returns: a float, an array over locations (``"pointwise"``), or an
            array of per-draw values (``"fraction"``, ``"count"``).
        """
        self._check_fitted()
        valid = ("pointwise", "mean", "any", "all", "fraction", "count")
        if mode not in valid:
            raise ValueError(f"mode must be one of {valid}")
        cells = np.arange(len(self.coords_)) if cells is None else np.asarray(cells, dtype=int)

        x = self.sample(cells, n=n, seed=seed)             # (n, m, v)
        hit = np.ones(x.shape[:2], dtype=bool)             # (n, m)
        for var, (op, val) in conditions.items():
            if var not in self.variables:
                raise KeyError(f"unknown variable {var!r}; have {self.variables}")
            if op not in _OPS:
                raise ValueError(f"unsupported operator {op!r}; expected one of {sorted(_OPS)}")
            hit &= _OPS[op](x[:, :, self.variables.index(var)], val)

        if mode == "pointwise":
            return hit.mean(axis=0)
        if mode == "mean":
            return float(hit.mean())
        if mode == "any":
            return float(hit.any(axis=1).mean())
        if mode == "all":
            return float(hit.all(axis=1).mean())
        if mode == "fraction":
            return hit.mean(axis=1)
        return hit.sum(axis=1)

    def posterior_exceedance(
        self,
        conditions: Mapping[str, tuple[str, float]],
        cells: Sequence[int] | None = None,
        mode: str = "mean",
        n: int = 10_000,
        n_posterior: int = 20,
    ) -> dict:
        """
        `exceedance_probability` repeated over posterior draws of the copula,
        giving a distribution over the answer rather than a point estimate.

        Requires ``bayesian=True`` at construction. The spread reported here is
        epistemic -- how much the answer would move given a different sample of
        the same size -- and is the reason to prefer this over the point
        estimate when the number will be quoted.

        Returns: dict with ``samples``, ``mean``, ``std``, ``q05``, ``q95``.
        """
        self._check_fitted()
        if not self.bayesian:
            raise RuntimeError("posterior_exceedance requires bayesian=True at construction")
        if mode in ("pointwise", "fraction", "count"):
            raise ValueError(f"mode={mode!r} returns an array; use a scalar mode here")

        originals = {k: c.flow for k, c in self.copulas_.items()}
        vals = []
        try:
            for i in range(n_posterior):
                for c in self.copulas_.values():
                    c.flow = c.bayes.sample_model()
                vals.append(float(self.exceedance_probability(
                    conditions, cells=cells, mode=mode, n=n, seed=self.seed + i)))
        finally:
            for k, c in self.copulas_.items():
                c.flow = originals[k]

        arr = np.asarray(vals, dtype=float)
        return {
            "samples": arr, "mean": float(arr.mean()), "std": float(arr.std()),
            "q05": float(np.quantile(arr, 0.05)), "q95": float(np.quantile(arr, 0.95)),
        }

    def tail_dependence_map(self, distance_km: float = 0.0) -> dict[str, float]:
        """
        Fitted tail dependence between two locations `distance_km` apart, per
        variable.

        At distance 0 this is the limiting within-pair value; comparing across
        variables shows which of them cluster most strongly in space.
        """
        self._check_fitted()
        return {k: v.tail_dependence(distance_km) for k, v in self.spatial_.items()}

    # =================================================================
    def _check_fitted(self) -> None:
        if self.coords_ is None:
            raise RuntimeError("this layer is not fitted yet; call fit() first.")

    @property
    def n_cells(self) -> int:
        self._check_fitted()
        return len(self.coords_)

    def __repr__(self) -> str:
        if self.coords_ is None:
            return f"SpatialProbabilisticLayer({self.variables}, unfitted)"
        return (f"SpatialProbabilisticLayer({self.variables}, "
                f"cells={self.n_cells}, copula_scope={self.copula_scope!r}, "
                f"bayesian={self.bayesian})")
