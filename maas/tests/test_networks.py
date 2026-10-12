# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for `maas.networks` snapping and pyramid serialization."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pyogrio
import pytest
from shapely.geometry import LineString

from maas.networks import (
    FLOODHUB_PYRAMID_LOD,
    GEOGLOWS_LOD,
    GLOFAS_LOD,
    as_linkno,
    extract_level_features,
    glofas_cell_center,
    glofas_cell_polygon,
    is_geoglows_river_id,
    load_network_pyramid,
    lod_for_zoom,
    pyramid_signature,
    resolve_cross_network_click,
    save_network_pyramid,
    snap_geoglows_reach_from_network,
    snap_glofas_cell_from_network,
    trace_main_stem_chain,
)
from maas.tools.build_network_pyramids import build_all_pyramids

if TYPE_CHECKING:
    from pathlib import Path

_RING_VERTEX_COUNT: int = 5
_EXPECTED_ZOOM_4: int = 4
_EXPECTED_LINKNO: int = 720010511
_EXPECTED_SEV_RANK: int = 2
_EXPECTED_STREAM_ORDER: int = 8
_EXPECTED_CHAIN_LEN: int = 3
_EXPECTED_HYBAS_L12: int = 7120050010


class TestGridGeometryAndIdentifiers:
    """Tests GloFAS 0.05 deg grid geometry and reach ID validation."""

    def test_glofas_cell_center_and_polygon_bounds(self) -> None:
        """Verify 0.05-deg cell centers and closed 5-vertex polygons."""
        snapped_lat, snapped_lon = glofas_cell_center(38.6270, -90.1994)
        assert math.isclose(
            (snapped_lat - 0.025) / 0.05,
            round((snapped_lat - 0.025) / 0.05),
            abs_tol=1e-6,
        )
        assert math.isclose(
            (snapped_lon - 0.025) / 0.05,
            round((snapped_lon - 0.025) / 0.05),
            abs_tol=1e-6,
        )

        feat = glofas_cell_polygon(38.6270, -90.1994)
        assert feat['type'] == 'Feature'
        assert feat['geometry']['type'] == 'Polygon'
        ring = feat['geometry']['coordinates'][0]
        assert len(ring) == _RING_VERTEX_COUNT
        assert ring[0] == ring[-1]
        lons = [pt[0] for pt in ring]
        lats = [pt[1] for pt in ring]
        assert math.isclose(max(lons) - min(lons), 0.05, abs_tol=1e-4)
        assert math.isclose(max(lats) - min(lats), 0.05, abs_tol=1e-4)

    def test_geoglows_river_id_validation(self) -> None:
        """Verify 9-digit GEOGLOWS LINKNO validation and parsing."""
        assert is_geoglows_river_id(_EXPECTED_LINKNO) is True
        assert is_geoglows_river_id('720010511') is True
        assert is_geoglows_river_id(99999999) is False
        assert is_geoglows_river_id(None) is False
        assert as_linkno('720010511') == _EXPECTED_LINKNO
        assert as_linkno('HYRIV_123') is None


class TestPyramidSerializationAndSnapping:
    """Tests LOD binary pyramid save/load and upstream-area reach snapping."""

    def test_save_and_load_pyramid_and_extract_features(
        self, tmp_path: Path
    ) -> None:
        """Verify pyramid `.npz` round-trip and bounding-box extraction."""
        level = {
            'coords': np.array(
                [[-90.25, 38.60], [-90.20, 38.62], [-90.15, 38.65]],
                dtype=np.float32,
            ),
            'offsets': np.array([0, 3], dtype=np.int64),
            'bbox': np.array(
                [[-90.25, 38.60, -90.15, 38.65]], dtype=np.float32
            ),
            'amin': np.array([1750000.0], dtype=np.float32),
            'amax': np.array([1800000.0], dtype=np.float32),
        }
        sig = pyramid_signature(GLOFAS_LOD)
        npz_path = tmp_path / 'glofas_pyramid.npz'
        save_network_pyramid(npz_path, [level], sig)

        loaded = load_network_pyramid(npz_path, sig)
        assert loaded is not None
        assert len(loaded['levels']) == 1
        assert load_network_pyramid(npz_path, 'wrong_sig') is None

        lod_row = lod_for_zoom(GLOFAS_LOD, _EXPECTED_ZOOM_4)
        assert lod_row[0] == _EXPECTED_ZOOM_4

        feats = extract_level_features(
            loaded['levels'][0], -91.0, 38.0, -89.0, 39.0
        )
        assert len(feats) == 1
        assert feats[0]['properties']['upstream_area_km2'] == pytest.approx(
            1800000.0
        )

    def test_glofas_snapping_and_cross_network_click_resolution(self) -> None:
        """Verify GloFAS cell snapping by upstream area and click resolution."""
        row_main = round((90.0 - 38.625) / 0.05 - 0.5)
        col_main = round((-90.175 + 180.0) / 0.05 - 0.5)
        lin_main = row_main * 7200 + col_main
        lin_trib = row_main * 7200 + (col_main + 1)

        net = {
            'cell_lin': np.array([lin_main, lin_trib], dtype=np.int64),
            'cell_area': np.array([1800000.0, 250.0], dtype=np.float32),
        }
        snapped = snap_glofas_cell_from_network(
            net, 38.627, -90.180, target_area_km2=1750000.0
        )
        assert snapped is not None
        assert snapped['upstream_area_km2'] == pytest.approx(1800000.0)

        resolved = resolve_cross_network_click(
            38.627,
            -90.180,
            upstream_area_km2=1800000.0,
            snap_glofas_fn=lambda lat, lon, area, **kw: (
                snap_glofas_cell_from_network(net, lat, lon, area, **kw)
            ),
            snap_geoglows_fn=lambda _lat, _lon, _area, **_kw: {
                'river_id': _EXPECTED_LINKNO,
                'upstream_area_km2': 1795000.0,
                'offset_km': 0.4,
            },
            network='geoglows',
            river_id='720010511',
        )
        assert resolved is not None
        assert resolved['geoglows']['river_id'] == _EXPECTED_LINKNO
        assert resolved['glofas']['upstream_area_km2'] == pytest.approx(
            1800000.0
        )

    def test_geoglows_in_memory_snapping_and_floodhub_level_features(
        self, tmp_path: Path
    ) -> None:
        """Verify GEOGLOWS snapping and FloodHub level feature extraction."""
        gg_net = {
            'reach_lat': np.array([38.61, 38.63], dtype=np.float32),
            'reach_lon': np.array([-90.19, -90.18], dtype=np.float32),
            'reach_lon0': np.array([-90.20, -90.19], dtype=np.float32),
            'reach_lat0': np.array([38.60, 38.64], dtype=np.float32),
            'reach_lon1': np.array([-90.18, -90.17], dtype=np.float32),
            'reach_lat1': np.array([38.62, 38.62], dtype=np.float32),
            'reach_linkno': np.array(
                [720010510, _EXPECTED_LINKNO], dtype=np.int32
            ),
            'reach_area': np.array([450.0, 1795000.0], dtype=np.float32),
        }
        snapped = snap_geoglows_reach_from_network(
            gg_net, 38.627, -90.180, target_area_km2=1800000.0
        )
        assert snapped is not None
        assert snapped['river_id'] == _EXPECTED_LINKNO
        assert snapped['upstream_area_km2'] == pytest.approx(1795000.0)

        fh_level = {
            'coords': np.array(
                [[-90.25, 38.60], [-90.20, 38.62], [-90.15, 38.65]],
                dtype=np.float32,
            ),
            'offsets': np.array([0, 3], dtype=np.int64),
            'bbox': np.array(
                [[-90.25, 38.60, -90.15, 38.65]], dtype=np.float32
            ),
            'amin': np.array([1750000.0], dtype=np.float32),
            'amax': np.array([1800000.0], dtype=np.float32),
            'river_id': np.array([71234567], dtype=np.int32),
            'hybas_l12': np.array([7120012340], dtype=np.int64),
            'has_forecast': np.array([1], dtype=np.uint8),
            'stream_order': np.array([_EXPECTED_STREAM_ORDER], dtype=np.int8),
        }
        sig = 'v1:test'
        npz_path = tmp_path / 'fh_pyramid.npz'
        save_network_pyramid(npz_path, [fh_level], sig)
        loaded = load_network_pyramid(npz_path, sig)
        assert loaded is not None
        feats = extract_level_features(
            loaded['levels'][0],
            -91.0,
            38.0,
            -89.0,
            39.0,
            hybas_to_sev={7120012340: _EXPECTED_SEV_RANK},
        )
        assert len(feats) == 1
        props = feats[0]['properties']
        assert props['river_id'] == 'HYRIV_71234567'
        assert props['gauge_id'] == 'hybas_7120012340'
        assert props['has_forecast'] is True
        assert props['severity_rank'] == _EXPECTED_SEV_RANK
        assert props['stream_order'] == _EXPECTED_STREAM_ORDER

    def test_build_all_pyramids_persists_geoglows_and_floodhub_snap_arrays(
        self, tmp_path: Path
    ) -> None:
        """Verify `build_all_pyramids` saves reach arrays for snapping."""
        gg_dir = tmp_path / 'geoglows_v2'
        gg_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            gg_dir / 'geoglows_attrs_v1.npz',
            linkno=np.array([720010510, _EXPECTED_LINKNO], dtype=np.int32),
            area_km2=np.array([600.0, 1795000.0], dtype=np.float32),
            ds_linkno=np.array([_EXPECTED_LINKNO, 0], dtype=np.int32),
        )
        gg_gdf = gpd.GeoDataFrame(
            {'LINKNO': np.array([720010510, _EXPECTED_LINKNO], dtype=np.int32)},
            geometry=[
                LineString([[-90.20, 38.60], [-90.18, 38.62]]),
                LineString([[-90.19, 38.64], [-90.17, 38.62]]),
            ],
            crs='EPSG:4326',
        )
        pyogrio.write_dataframe(
            gg_gdf, str(gg_dir / 'global_streams_simplified.gpkg')
        )

        hr_dir = tmp_path / 'HydroRIVERS_v10_shp'
        hr_dir.mkdir(parents=True, exist_ok=True)
        hr_gdf = gpd.GeoDataFrame(
            {
                'HYRIV_ID': np.array([70000001], dtype=np.int32),
                'NEXT_DOWN': np.array([0], dtype=np.int32),
                'UPLAND_SKM': np.array([25000.0], dtype=np.float32),
                'ORD_STRA': np.array([7], dtype=np.int32),
                'HYBAS_L12': np.array([7120050010], dtype=np.int64),
            },
            geometry=[LineString([[-90.20, 38.64], [-90.17, 38.62]])],
            crs='EPSG:4326',
        )
        pyogrio.write_dataframe(hr_gdf, str(hr_dir / 'HydroRIVERS_v10.shp'))

        counts = build_all_pyramids(tmp_path, models=('geoglows', 'floodhub'))
        assert 'geoglows' in counts
        assert 'floodhub' in counts

        gg_loaded = load_network_pyramid(
            gg_dir / 'geoglows_network_v1.npz',
            pyramid_signature(GEOGLOWS_LOD),
        )
        assert gg_loaded is not None
        snapped = snap_geoglows_reach_from_network(
            gg_loaded, 38.627, -90.180, target_area_km2=1800000.0
        )
        assert snapped is not None
        assert snapped['river_id'] == _EXPECTED_LINKNO

        fh_loaded = load_network_pyramid(
            tmp_path / 'glofas_v4' / 'floodhub_network_v1.npz',
            pyramid_signature(FLOODHUB_PYRAMID_LOD),
        )
        assert fh_loaded is not None
        assert 'reach_hyriv_id' in fh_loaded
        assert int(fh_loaded['reach_hybas_l12'][0]) == _EXPECTED_HYBAS_L12
        fh_feats = extract_level_features(
            fh_loaded['levels'][0], -91.0, 38.0, -89.0, 39.0
        )
        assert len(fh_feats) == 1
        assert fh_feats[0]['properties']['gauge_id'] == 'hybas_7120050010'


class TestMainStemTracing:
    """Tests topological main-stem chain tracing across HydroRIVERS reaches."""

    def test_trace_main_stem_chain(self) -> None:
        """Verify main-stem chain traversal upstream and downstream."""
        reaches = [
            {
                'hyriv_id': 101,
                'next_down': 102,
                'main_riv': 103,
                'stream_order': 6,
                'upstream_area_km2': 50000.0,
                'mean_discharge_m3s': 1200.0,
                'geometry': LineString([[-90.30, 38.65], [-90.25, 38.63]]),
            },
            {
                'hyriv_id': 102,
                'next_down': 103,
                'main_riv': 103,
                'stream_order': 6,
                'upstream_area_km2': 52000.0,
                'mean_discharge_m3s': 1250.0,
                'geometry': LineString([[-90.25, 38.63], [-90.20, 38.61]]),
            },
            {
                'hyriv_id': 103,
                'next_down': 0,
                'main_riv': 103,
                'stream_order': 6,
                'upstream_area_km2': 55000.0,
                'mean_discharge_m3s': 1300.0,
                'geometry': LineString([[-90.20, 38.61], [-90.15, 38.59]]),
            },
        ]
        start, chain, snap_km = trace_main_stem_chain(reaches, 38.63, -90.25)
        assert start is not None
        assert len(chain) == _EXPECTED_CHAIN_LEN
        assert snap_km is not None
        assert snap_km < 1.0
