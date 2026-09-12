"""
Tests for region selection.

Built on synthetic polygons rather than real boundaries so the suite needs no
network. One test marked `network` exercises the Natural Earth download path.
"""

from __future__ import annotations

import numpy as np
import pytest

gpd = pytest.importorskip("geopandas")
from shapely.geometry import Polygon, box  # noqa: E402

from neurocopula.regions import SEA_COUNTRIES, RegionSet  # noqa: E402


@pytest.fixture
def frame():
    """Two adjacent squares and a tiny one smaller than the grid spacing."""
    return gpd.GeoDataFrame(
        {"NAME": ["West", "East", "Speck"]},
        geometry=[box(100, -5, 105, 5), box(105, -5, 110, 5),
                  Polygon([(120.01, 0.01), (120.02, 0.01),
                           (120.02, 0.02), (120.01, 0.02)])],
        crs="EPSG:4326",
    )


@pytest.fixture
def regions(frame):
    return RegionSet(frame)


@pytest.fixture
def coords():
    """A 1-degree grid spanning both squares and beyond."""
    lats = np.arange(-6, 7, 1.0)
    lons = np.arange(98, 125, 1.0)
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    return np.column_stack([LA.ravel(), LO.ravel()])


class TestConstruction:
    def test_rejects_non_geodataframe(self):
        with pytest.raises(TypeError, match="GeoDataFrame"):
            RegionSet({"NAME": ["a"]})

    def test_rejects_missing_name_field(self, frame):
        with pytest.raises(KeyError, match="no column"):
            RegionSet(frame, name_field="COUNTRY")

    def test_reprojects_to_wgs84(self, frame):
        # Grid coordinates are degrees; a projected frame must be converted or
        # containment silently compares different coordinate spaces.
        projected = frame.to_crs(epsg=3857)
        rs = RegionSet(projected)
        assert rs.frame.crs.to_epsg() == 4326

    def test_names_sorted(self, regions):
        assert regions.names == ["East", "Speck", "West"]

    def test_len_and_repr(self, regions):
        assert len(regions) == 3
        assert "3 regions" in repr(regions)


class TestAssignment:
    def test_assigns_cells_to_the_containing_polygon(self, regions, coords):
        assigned = regions.assign(coords)
        west = assigned == "West"
        assert west.sum() > 0
        # Every West cell must actually lie inside the West box.
        assert coords[west, 1].min() >= 100 and coords[west, 1].max() <= 105

    def test_cells_outside_every_polygon_are_none(self, regions, coords):
        assigned = regions.assign(coords)
        assert any(a is None for a in assigned)

    def test_assignment_is_cached(self, regions, coords):
        first = regions.assign(coords)
        assert regions.assign(coords) is first

    def test_cell_counts_match_assignment(self, regions, coords):
        counts = regions.cell_counts(coords)
        assigned = regions.assign(coords)
        for name, n in counts.items():
            assert (assigned == name).sum() == n

    def test_rejects_wrong_coordinate_shape(self, regions):
        with pytest.raises(ValueError, match="two columns"):
            regions.assign(np.zeros((10, 3)))


class TestMasks:
    def test_mask_is_boolean_over_the_grid(self, regions, coords):
        m = regions.mask("West", coords)
        assert m.dtype == bool and len(m) == len(coords) and m.any()

    def test_mask_matches_cell_counts(self, regions, coords):
        assert regions.mask("East", coords).sum() == regions.cell_counts(coords)["East"]

    def test_adjacent_regions_do_not_overlap(self, regions, coords):
        # A point on a shared border matches two polygons; the first wins, so
        # the masks must still be disjoint or areal statistics double-count.
        w, e = regions.mask("West", coords), regions.mask("East", coords)
        assert not (w & e).any()

    def test_unknown_region_raises(self, regions, coords):
        with pytest.raises(KeyError, match="unknown region"):
            regions.mask("Atlantis", coords)

    def test_region_smaller_than_the_grid_raises_clearly(self, regions, coords):
        # Silently returning an empty selection would surface much later as a
        # confusing error inside the sampler.
        with pytest.raises(ValueError, match="no grid cell centres"):
            regions.mask("Speck", coords)

    def test_masks_skips_empty_regions(self, regions, coords):
        out = regions.masks(coords)
        assert "Speck" not in out
        assert set(out) == {"West", "East"}


class TestSubsetting:
    def test_subset_keeps_only_named(self, regions):
        assert len(regions.subset(["West"])) == 1

    def test_subset_rejects_unknown(self, regions):
        with pytest.raises(KeyError, match="unknown region"):
            regions.subset(["Atlantis"])

    def test_clip_drops_non_intersecting(self, regions):
        clipped = regions.clip_to((-6, 6, 99, 106))
        assert "Speck" not in clipped.names

    def test_clip_with_no_overlap_raises(self, regions):
        with pytest.raises(ValueError, match="no regions intersect"):
            regions.clip_to((60, 70, 0, 10))


class TestSeaCountryList:
    def test_eleven_countries(self):
        assert len(SEA_COUNTRIES) == 11
        assert "Indonesia" in SEA_COUNTRIES and "Timor-Leste" in SEA_COUNTRIES


@pytest.mark.network
class TestNaturalEarth:
    def test_download_and_assign(self, tmp_path):
        rs = RegionSet.natural_earth(cache_dir=tmp_path)
        assert len(rs) > 200
        missing = [c for c in SEA_COUNTRIES if c not in set(rs.names)]
        assert not missing, f"Natural Earth is missing {missing}"

    def test_cached_on_second_call(self, tmp_path):
        RegionSet.natural_earth(cache_dir=tmp_path)
        files = list(tmp_path.glob("*.geojson"))
        assert len(files) == 1
        RegionSet.natural_earth(cache_dir=tmp_path)          # must not re-download
        assert list(tmp_path.glob("*.geojson")) == files
