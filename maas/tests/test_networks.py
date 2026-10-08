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

"""Unit tests for `maas.networks` spatial grid snapping, pyramid serialization, and cross-network reach matching."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import LineString

from maas.networks import (
    GLOFAS_LOD,
    as_linkno,
    cama_cell_area_km2,
    cama_cell_id,
    cama_cell_polygon,
    extract_level_features,
    glofas_cell_center,
    glofas_cell_polygon,
    is_geoglows_river_id,
    load_network_pyramid,
    lod_for_zoom,
    pyramid_signature,
    resolve_cross_network_click,
    save_network_pyramid,
    snap_cama_cell,
    trace_main_stem_chain,
)
from maas.networks import (
    snap_glofas_cell_from_network as GLofasCellSnap,
)


class TestGridGeometryAndIdentifiers:
    """Tests GloFAS 0.05 deg and CaMa-Flood 0.25 deg grid geometry and reach ID validation."""

    def test_glofas_cell_center_and_polygon_bounds(self) -> None:
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
        assert len(ring) == 5
        assert ring[0] == ring[-1]
        lons = [pt[0] for pt in ring]
        lats = [pt[1] for pt in ring]
        assert math.isclose(max(lons) - min(lons), 0.05, abs_tol=1e-4)
        assert math.isclose(max(lats) - min(lats), 0.05, abs_tol=1e-4)

    def test_cama_cell_snapping_and_spherical_area(self) -> None:
        c_lat, c_lon = snap_cama_cell(35.6895, 139.6917)
        cid = cama_cell_id(c_lat, c_lon)
        assert cid.startswith('cama_025_')
        ring, bbox = cama_cell_polygon(c_lat, c_lon)
        assert len(ring) == 5
        assert math.isclose(
            bbox['max_lon'] - bbox['min_lon'], 0.25, abs_tol=1e-4
        )
        assert math.isclose(
            bbox['max_lat'] - bbox['min_lat'], 0.25, abs_tol=1e-4
        )

        equator_area = cama_cell_area_km2(0.125)
        high_lat_area = cama_cell_area_km2(60.125)
        assert equator_area > 700.0
        assert math.isclose(
            high_lat_area / equator_area,
            math.cos(math.radians(60.125)),
            rel_tol=1e-2,
        )

    def test_geoglows_river_id_validation(self) -> None:
        assert is_geoglows_river_id(720010511) is True
        assert is_geoglows_river_id('720010511') is True
        assert is_geoglows_river_id(99999999) is False
        assert is_geoglows_river_id(None) is False
        assert as_linkno('720010511') == 720010511
        assert as_linkno('HYRIV_123') is None


class TestPyramidSerializationAndSnapping:
    """Tests LOD binary pyramid save/load and upstream-area-consistent reach snapping."""

    def test_save_and_load_pyramid_and_extract_features(
        self, tmp_path: Path
    ) -> None:
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

        lod_row = lod_for_zoom(GLOFAS_LOD, 4)
        assert lod_row[0] == 4

        feats = extract_level_features(
            loaded['levels'][0], -91.0, 38.0, -89.0, 39.0
        )
        assert len(feats) == 1
        assert feats[0]['properties']['upstream_area_km2'] == pytest.approx(
            1800000.0
        )

    def test_glofas_snapping_and_cross_network_click_resolution(self) -> None:
        # Row/col corresponding to ~ (38.625, -90.175) on 0.05 deg grid
        row_main = round((90.0 - 38.625) / 0.05 - 0.5)
        col_main = round((-90.175 + 180.0) / 0.05 - 0.5)
        lin_main = row_main * 7200 + col_main
        lin_trib = row_main * 7200 + (col_main + 1)

        net = {
            'cell_lin': np.array([lin_main, lin_trib], dtype=np.int64),
            'cell_area': np.array([1800000.0, 250.0], dtype=np.float32),
        }
        snapped = GLofasCellSnap(
            net, 38.627, -90.180, target_area_km2=1750000.0
        )
        assert snapped is not None
        assert snapped['upstream_area_km2'] == pytest.approx(1800000.0)

        resolved = resolve_cross_network_click(
            38.627,
            -90.180,
            upstream_area_km2=1800000.0,
            snap_glofas_fn=lambda lat, lon, area, **kw: GLofasCellSnap(
                net, lat, lon, area, **kw
            ),
            snap_geoglows_fn=lambda lat, lon, area, **kw: {
                'river_id': 720010511,
                'upstream_area_km2': 1795000.0,
                'offset_km': 0.4,
            },
            network='geoglows',
            river_id='720010511',
        )
        assert resolved is not None
        assert resolved['geoglows']['river_id'] == 720010511
        assert resolved['glofas']['upstream_area_km2'] == pytest.approx(
            1800000.0
        )

    def test_geoglows_in_memory_snapping_and_floodhub_level_features(
        self, tmp_path: Path
    ) -> None:
        from maas.networks import snap_geoglows_reach_from_network

        gg_net = {
            'reach_lat': np.array([38.61, 38.63], dtype=np.float32),
            'reach_lon': np.array([-90.19, -90.18], dtype=np.float32),
            'reach_lon0': np.array([-90.20, -90.19], dtype=np.float32),
            'reach_lat0': np.array([38.60, 38.64], dtype=np.float32),
            'reach_lon1': np.array([-90.18, -90.17], dtype=np.float32),
            'reach_lat1': np.array([38.62, 38.62], dtype=np.float32),
            'reach_linkno': np.array([720010510, 720010511], dtype=np.int32),
            'reach_area': np.array([450.0, 1795000.0], dtype=np.float32),
        }
        snapped = snap_geoglows_reach_from_network(
            gg_net, 38.627, -90.180, target_area_km2=1800000.0
        )
        assert snapped is not None
        assert snapped['river_id'] == 720010511
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
            'stream_order': np.array([8], dtype=np.int8),
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
            hybas_to_sev={7120012340: 2},
        )
        assert len(feats) == 1
        props = feats[0]['properties']
        assert props['river_id'] == 'HYRIV_71234567'
        assert props['gauge_id'] == 'hybas_7120012340'
        assert props['has_forecast'] is True
        assert props['severity_rank'] == 2
        assert props['stream_order'] == 8


class TestMainStemTracing:
    """Tests topological main-stem chain tracing across HydroRIVERS reaches."""

    def test_trace_main_stem_chain(self) -> None:
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
        assert len(chain) == 3
        assert snap_km is not None and snap_km < 1.0
