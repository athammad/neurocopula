"""
Named regions: turning polygons into cell selections.

A fitted layer answers questions about a set of grid cells. Users think in
countries, provinces and river basins. This module bridges the two: give it a
set of polygons and a grid, and it returns the boolean masks that
`SpatialProbabilisticLayer.select` already accepts.

Masks are the right interface for this rather than, say, clipping the data:
the model is fitted once over the whole domain, and a region is chosen
afterwards purely by deciding which cells to draw jointly. Nothing is refitted
per region, and two overlapping regions cost no more than two selections.

Polygons can come from anywhere `geopandas` can read -- a shapefile of river
basins, a GeoJSON of provinces, an administrative boundary set. Country
outlines can be fetched automatically from Natural Earth.

For river basins, HydroBASINS (hydrosheds.org) is the usual source. Its
polygons are identified by an integer ``HYBAS_ID`` rather than a name, which
`from_file` handles::

    basins = RegionSet.from_file("hybas_as_lev05_v1c.shp", name_field="HYBAS_ID")
    basins = basins.clip_to((-12, 30, 90, 142))
    layer.attach_regions(basins)
    cells = layer.select(region=basins.names[0], max_cells=2000)

Level 3 gives continental basins, level 5 or 6 something closer to a major
river system, and level 12 individual catchments far smaller than a 0.1 degree
cell. Pick the level against the grid: a basin smaller than one cell captures
nothing, and `mask` will say so.

`geopandas` is an optional dependency::

    pip install neurocopula[regions]

Cell assignment is by **centroid containment**: a cell belongs to the region
whose polygon contains its centre. For cells much smaller than the regions this
is accurate and unambiguous. It does mean a coastal cell whose centre falls
just offshore is unassigned, and that a region far smaller than one cell may
capture no cells at all -- `mask` raises rather than silently returning an
empty selection in that case.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np

__all__ = ["RegionSet", "NATURAL_EARTH_URL", "SEA_COUNTRIES"]

#: Natural Earth admin-0 boundaries, 1:10m. Served from the project's GitHub
#: mirror, which is more reliable than naturalearthdata.com. 1:10m rather than
#: the coarser sets because at 0.1 degrees a cell is about 11 km, and 1:110m
#: coastlines would misassign a visible fraction of coastal cells.
NATURAL_EARTH_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
    "geojson/ne_10m_admin_0_countries.geojson"
)

#: The eleven Southeast Asian countries, as named in Natural Earth's NAME field.
SEA_COUNTRIES = [
    "Indonesia", "Malaysia", "Thailand", "Vietnam", "Philippines", "Myanmar",
    "Cambodia", "Laos", "Singapore", "Brunei", "Timor-Leste",
]


def _require_geopandas():
    try:
        import geopandas as gpd
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "regions needs geopandas, an optional dependency.\n"
            "Install it with:  pip install neurocopula[regions]"
        ) from exc
    return gpd


class RegionSet:
    """
    A collection of named polygons, and the masks they imply on a grid.

    Build one with `from_file` for your own boundaries, or `natural_earth`
    for country outlines.

    Parameters
    ----------
    frame:
        A `geopandas.GeoDataFrame` in geographic coordinates (EPSG:4326).
    name_field:
        Column holding each polygon's name.
    """

    def __init__(self, frame, name_field: str = "NAME") -> None:
        gpd = _require_geopandas()
        if not isinstance(frame, gpd.GeoDataFrame):
            raise TypeError("frame must be a geopandas.GeoDataFrame")
        if name_field not in frame.columns:
            raise KeyError(
                f"no column {name_field!r}; available: {list(frame.columns)[:12]}"
            )
        if frame.crs is not None and frame.crs.to_epsg() != 4326:
            # Grid coordinates are latitude/longitude, so the polygons must be
            # too, or containment tests silently compare different spaces.
            frame = frame.to_crs(epsg=4326)
        self.frame = frame
        self.name_field = name_field
        self._cache: dict[tuple, np.ndarray] = {}

    # -----------------------------------------------------------------
    @classmethod
    def from_file(cls, path, name_field: str = "NAME", **kwargs) -> RegionSet:
        """
        Load polygons from any file `geopandas` can read -- shapefile,
        GeoJSON, GeoPackage. Use this for river basins or administrative
        levels other than country.
        """
        gpd = _require_geopandas()
        return cls(gpd.read_file(path, **kwargs), name_field=name_field)

    @classmethod
    def natural_earth(cls, cache_dir=None, url: str = NATURAL_EARTH_URL,
                      name_field: str = "NAME") -> RegionSet:
        """
        Country outlines from Natural Earth, downloaded once and cached.

        cache_dir: where to keep the download. Defaults to
            ``~/.cache/neurocopula``.
        """
        gpd = _require_geopandas()
        cache_dir = Path(cache_dir or Path.home() / ".cache" / "neurocopula")
        cache_dir.mkdir(parents=True, exist_ok=True)
        name = hashlib.sha1(url.encode()).hexdigest()[:12] + ".geojson"
        path = cache_dir / name
        if not path.exists():
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(path)
        return cls(gpd.read_file(path), name_field=name_field)

    # -----------------------------------------------------------------
    @property
    def names(self) -> list:
        """
        Every region identifier, sorted.

        These are not always strings: HydroBASINS numbers its polygons, so the
        identifiers come back as integers and are used as such throughout.
        """
        return sorted(self.frame[self.name_field].dropna().unique().tolist())

    def subset(self, names) -> RegionSet:
        """Restrict to the named regions, keeping the same name field."""
        wanted = list(names)
        missing = [n for n in wanted if n not in set(self.frame[self.name_field])]
        if missing:
            raise KeyError(f"unknown region(s): {missing}")
        sub = self.frame[self.frame[self.name_field].isin(wanted)].copy()
        return RegionSet(sub, name_field=self.name_field)

    def clip_to(self, bbox) -> RegionSet:
        """
        Drop regions that do not intersect ``(lat_min, lat_max, lon_min,
        lon_max)``. Worth doing before `assign` on a large boundary set, since
        it removes most of the world from the containment test.
        """
        from shapely.geometry import box
        lat_min, lat_max, lon_min, lon_max = bbox
        window = box(lon_min, lat_min, lon_max, lat_max)
        keep = self.frame[self.frame.intersects(window)].copy()
        if keep.empty:
            raise ValueError("no regions intersect that bounding box")
        return RegionSet(keep, name_field=self.name_field)

    # -----------------------------------------------------------------
    def _points(self, coords):
        gpd = _require_geopandas()
        coords = np.atleast_2d(np.asarray(coords, dtype=np.float64))
        if coords.shape[1] != 2:
            raise ValueError("coords must have two columns: latitude, longitude")
        return gpd.GeoDataFrame(
            {"_i": np.arange(len(coords))},
            geometry=gpd.points_from_xy(coords[:, 1], coords[:, 0]),
            crs="EPSG:4326",
        )

    def assign(self, coords) -> np.ndarray:
        """
        The region containing each grid cell's centre.

        Returns: object array of names, with None where a cell falls in no
            region (ocean, or outside the boundary set).

        Computed once per grid and cached, since it depends only on geometry.
        """
        coords = np.atleast_2d(np.asarray(coords, dtype=np.float64))
        key = (coords.shape, float(coords[0, 0]), float(coords[-1, -1]),
               float(coords.sum()))
        if key in self._cache:
            return self._cache[key]

        gpd = _require_geopandas()
        points = self._points(coords)
        joined = gpd.sjoin(points, self.frame[[self.name_field, "geometry"]],
                           how="left", predicate="within")
        # A point on a shared border can match two polygons; keep the first.
        joined = joined.drop_duplicates(subset="_i").sort_values("_i")
        out = joined[self.name_field].to_numpy(dtype=object)
        out[[x is np.nan or x != x for x in out]] = None
        self._cache[key] = out
        return out

    def mask(self, name, coords) -> np.ndarray:
        """
        Boolean mask over the grid for one region, ready for
        `SpatialProbabilisticLayer.select(mask=...)`.

        Raises if the region selects no cells -- usually a region smaller than
        the grid spacing, or a name that exists but lies outside the domain.
        Silently returning an empty selection would surface much later as a
        confusing error from the sampler.
        """
        if name not in set(self.frame[self.name_field]):
            raise KeyError(f"unknown region {name!r}")
        m = self.assign(coords) == name
        if not m.any():
            raise ValueError(
                f"region {name!r} contains no grid cell centres -- it may be "
                "smaller than the grid spacing, or outside the fitted domain"
            )
        return m

    def masks(self, coords, names=None) -> dict[str, np.ndarray]:
        """
        Masks for every region that captures at least one cell.

        Regions with no cells are omitted rather than raising, so this can be
        used to sweep a large boundary set over a small domain.
        """
        assigned = self.assign(coords)
        wanted = (list(names) if names is not None
                  else self.frame[self.name_field].dropna().unique().tolist())
        # Iterate the frame's own values rather than np.unique on the object
        # array, which would coerce integer identifiers to float.
        out = {}
        for n in wanted:
            m = assigned == n
            if m.any():
                out[n] = m
        return out

    def cell_counts(self, coords) -> dict[str, int]:
        """
        Cells per region -- worth checking before querying: a region with only
        a handful of cells gives a noisy areal answer, and one with none at all
        cannot be queried.
        """
        assigned = self.assign(coords)
        out = {}
        for n in self.frame[self.name_field].dropna().unique().tolist():
            c = int((assigned == n).sum())
            if c:
                out[n] = c
        return out

    def __len__(self) -> int:
        return len(self.frame)

    def __repr__(self) -> str:
        return f"RegionSet({len(self.frame)} regions, name_field={self.name_field!r})"
