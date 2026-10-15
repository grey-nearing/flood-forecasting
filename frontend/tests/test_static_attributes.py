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

"""Unit tests for the `frontend.static_attributes` adapter."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

import frontend.static_attributes as _static_attributes
from frontend.static_attributes import (
    DEFAULT_GCS_ERA5_CLIMATE_URI,
    DEFAULT_GCS_HYDROATLAS_URI,
    build_attribute_card_payload,
    build_map_layer_geojson,
    build_schema_payload,
    card_payload_from_cache_entry,
    get_attributes_extractor,
    reset_attributes_extractor,
    resolve_app_static_extractor_paths,
    sanitize_for_json,
    zoom_to_hydroatlas_level,
)
from multimet.static_extractor import CatchmentAttributes

_EXPECTED_SUBBASIN_COUNT = 2
_EXPECTED_MIN_SCHEMA_ATTRS = 10
_EXPECTED_TEMP_C = 15.0
_EXPECTED_SLOPE_DEG = 15.0
_EXPECTED_ELEV_M = 500.0
_EXPECTED_LAND_COVER_CLASS = 2
_EXPECTED_TOTAL_AREA_KM2 = 100.0


def build_synthetic_hydroatlas_fixtures(root: Path) -> tuple[Path, Path]:
    """Create synthetic HydroATLAS Level-12 GeoParquet and ERA5 fixtures.

    Args:
        root: Temporary directory root.

    Returns:
        Tuple of `(hydroatlas_dir, era5_climate_dir)`.
    """
    hydroatlas_dir = root / 'hydroatlas'
    subpoly_dir = hydroatlas_dir / 'subpolygons'
    subpoly_dir.mkdir(parents=True, exist_ok=True)

    hybas_ids = [7120000010, 7120000020]
    attr_df = pd.DataFrame(
        {
            'HYBAS_ID': hybas_ids,
            'NEXT_DOWN': [7120000020, 0],
            'SUB_AREA': [1200.0, 1100.0],
            'UP_AREA': [1200.0, 2300.0],
            'ele_mt_sav': [2600.0, 2400.0],
            'slp_dg_sav': [120.0, 95.0],
            'pre_mm_syr': [780.0, 720.0],
            'tmp_dc_syr': [45.0, 60.0],
            'ari_ix_sav': [85.0, 92.0],
            'cly_pc_sav': [18.0, 22.0],
            'snd_pc_sav': [48.0, 44.0],
            'slt_pc_sav': [34.0, 34.0],
            'for_pc_sse': [65.0, 55.0],
            'crp_pc_sse': [5.0, 10.0],
            'urb_pc_sse': [2.0, 4.0],
            'gwt_cm_sav': [180.0, 210.0],
            'swc_pc_syr': [52.0, 48.0],
            'inu_pc_smx': [2.0, 3.0],
            'glc_cl_smj': [1, 4],
            'clz_cl_smj': [14, 14],
            'lit_cl_smj': [5, 5],
            'dis_m3_pyr': [18.0, 35.0],
            'run_mm_syr': [310.0, 290.0],
        }
    )
    attr_df.to_parquet(
        hydroatlas_dir / 'hydro_atlas_lev12.parquet', index=False
    )

    subpoly_gdf = gpd.GeoDataFrame(
        {'HYBAS_ID': hybas_ids},
        geometry=[
            box(-106.6, 39.4, -106.1, 39.9),
            box(-106.1, 39.4, -105.7, 39.9),
        ],
        crs='EPSG:4326',
    )
    subpoly_gdf.to_parquet(subpoly_dir / 'hybas_na_lev12_v1c.geoparquet')

    era5_dir = root / 'era5_climate'
    era5_dir.mkdir(parents=True, exist_ok=True)
    climate_rows = [
        {
            'gauge_id': 'hybas_7120000010',
            'p_mean': 2.4,
            'pet_mean': 3.1,
            'pet_mean_ERA5_LAND': 2.8,
            'aridity': 1.29,
            'aridity_ERA5_LAND': 1.17,
            'frac_snow': 0.38,
            'moisture_index': -0.22,
            'moisture_index_ERA5_LAND': -0.15,
            'seasonality': 0.42,
            'seasonality_ERA5_LAND': 0.36,
            'high_prec_freq': 16.5,
            'high_prec_dur': 1.3,
            'low_prec_freq': 195.0,
            'low_prec_dur': 5.4,
        },
        {
            'gauge_id': 'hybas_7120000020',
            'p_mean': 2.1,
            'pet_mean': 3.3,
            'pet_mean_ERA5_LAND': 3.0,
            'aridity': 1.57,
            'aridity_ERA5_LAND': 1.43,
            'frac_snow': 0.32,
            'moisture_index': -0.36,
            'moisture_index_ERA5_LAND': -0.30,
            'seasonality': 0.46,
            'seasonality_ERA5_LAND': 0.40,
            'high_prec_freq': 14.2,
            'high_prec_dur': 1.2,
            'low_prec_freq': 210.0,
            'low_prec_dur': 6.1,
        },
    ]
    (era5_dir / 'na_climate_indices.txt').write_text(
        '\n'.join(json.dumps(r) for r in climate_rows) + '\n',
        encoding='utf-8',
    )
    return hydroatlas_dir, era5_dir


class StaticAttributesAdapterTest(unittest.TestCase):
    """Test suite for `frontend.static_attributes`."""

    def test_sanitize_for_json(self) -> None:
        """Verify NaN and Inf floats are converted to None."""
        assert sanitize_for_json(np.nan) is None
        assert sanitize_for_json(float('inf')) is None
        assert sanitize_for_json({'a': np.nan, 'b': 1.5}) == {
            'a': None,
            'b': 1.5,
        }

    def test_zoom_to_hydroatlas_level(self) -> None:
        """Verify slippy-map zoom maps to expected Pfafstetter level."""
        avail = [4, 5, 6, 8, 10, 12]
        assert zoom_to_hydroatlas_level(3, avail) == 4  # noqa: PLR2004
        assert zoom_to_hydroatlas_level(7, avail) == 8  # noqa: PLR2004
        assert zoom_to_hydroatlas_level(15, avail) == 12  # noqa: PLR2004

    def test_build_schema_payload(self) -> None:
        """Verify attribute registry schema payload structure."""
        payload = build_schema_payload()
        assert 'attributes' in payload
        assert 'attribute_list' in payload
        assert len(payload['attribute_list']) > _EXPECTED_MIN_SCHEMA_ATTRS
        assert payload['extractor_source'] == 'multimet.static_extractor'

    def test_build_attribute_card_payload(self) -> None:
        """Verify physical unit scaling and summary fields on card payload."""
        res = CatchmentAttributes(
            catchment_id='test_id',
            attributes={
                'tmp_dc_syr': 150,
                'pre_mm_syr': 1200,
                'ele_mt_sav': 500,
                'slp_dg_sav': 150,
                'for_pc_sse': 60,
                'glc_cl_smj': 2,
            },
            area_km2=_EXPECTED_TOTAL_AREA_KM2,
            area_fraction_used_for_aggregation=1.0,
            subbasin_ids=(123, 456),
            subbasin_weights_km2=(50.0, 50.0),
            min_overlap_threshold_km2=0.0,
            era5_source='hybas',
        )
        payload = build_attribute_card_payload(res)

        assert payload['status'] == 'success'
        assert payload['catchment_id'] == 'test_id'
        assert (
            payload['intersected_subbasins_count'] == _EXPECTED_SUBBASIN_COUNT
        )
        assert payload['flat_attributes']['tmp_dc_syr'] == _EXPECTED_TEMP_C
        assert payload['flat_attributes']['slp_dg_sav'] == _EXPECTED_SLOPE_DEG
        assert payload['flat_attributes']['ele_mt_sav'] == _EXPECTED_ELEV_M
        assert (
            payload['flat_attributes']['glc_cl_smj']
            == _EXPECTED_LAND_COVER_CLASS
        )

        summary = payload['summary']
        assert summary['elevation_mean_m'] == _EXPECTED_ELEV_M
        assert summary['total_area_km2'] == _EXPECTED_TOTAL_AREA_KM2
        assert summary['subbasin_ids'] == [123, 456]

    def test_card_payload_from_cache_entry_rejects_missing_metadata(
        self,
    ) -> None:
        """Verify cache entries without area, weights, or coverage fail fast."""
        with pytest.raises(ValueError, match='missing area_km2'):
            card_payload_from_cache_entry(
                'missing_area',
                {
                    'attributes': {'ele_mt_sav': 500.0},
                    'subbasin_ids': [123],
                    'subbasin_weights_km2': [50.0],
                    'area_fraction_used_for_aggregation': 1.0,
                },
            )
        with pytest.raises(ValueError, match='missing subbasin_weights_km2'):
            card_payload_from_cache_entry(
                'missing_weights',
                {
                    'attributes': {'ele_mt_sav': 500.0},
                    'area_km2': 50.0,
                    'subbasin_ids': [123],
                    'area_fraction_used_for_aggregation': 1.0,
                },
            )
        with pytest.raises(
            ValueError, match='missing area_fraction_used_for_aggregation'
        ):
            card_payload_from_cache_entry(
                'missing_fraction',
                {
                    'attributes': {'ele_mt_sav': 500.0},
                    'area_km2': 50.0,
                    'subbasin_ids': [123],
                    'subbasin_weights_km2': [50.0],
                },
            )

    def test_hermetic_extractor_and_map_layer_with_openhydronet_fixtures(
        self,
    ) -> None:
        """Verify extraction and map-layer GeoJSON using synthetic fixtures."""
        self.addCleanup(reset_attributes_extractor)
        with tempfile.TemporaryDirectory() as tmp_dir:
            hydroatlas_dir, era5_dir = build_synthetic_hydroatlas_fixtures(
                Path(tmp_dir)
            )
            with mock.patch.dict(
                os.environ,
                {
                    'HYDROATLAS_GDB_PATH': str(hydroatlas_dir),
                    'ERA5_CLIMATE_CACHE_DIR': str(era5_dir),
                },
            ):
                reset_attributes_extractor()
                extractor = get_attributes_extractor()
                poly = box(-106.2, 39.5, -106.0, 39.7)
                res = extractor.extract_attributes_for_polygon(
                    poly,
                    catchment_id='openhydronet_test_basin',
                    era5_source='hybas',
                )
                card = build_attribute_card_payload(res)
                assert card['status'] == 'success'
                assert card['flat_attributes']['ele_mt_sav'] > 0
                assert card['flat_attributes']['p_mean'] > 0

                fc_elev = build_map_layer_geojson(
                    extractor,
                    (-106.5, 39.4, -105.8, 39.9),
                    zoom=10,
                    attribute='ele_mt_sav',
                )
                assert fc_elev['properties']['count'] == (
                    _EXPECTED_SUBBASIN_COUNT
                )
                fc_clim = build_map_layer_geojson(
                    extractor,
                    (-106.5, 39.4, -105.8, 39.9),
                    zoom=10,
                    attribute='p_mean',
                )
                assert fc_clim['properties']['count'] == (
                    _EXPECTED_SUBBASIN_COUNT
                )

    def test_get_attributes_extractor_streams_from_gcs_when_no_local_gdb(
        self,
    ) -> None:
        """Verify extractor streams from GCS when local caches are absent."""
        self.addCleanup(reset_attributes_extractor)
        reset_attributes_extractor()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            hydroatlas_dir, era5_dir = build_synthetic_hydroatlas_fixtures(root)
            empty_dir = root / 'empty_local_cache'
            gcs_prefix = DEFAULT_GCS_HYDROATLAS_URI.replace('gs://', '').rstrip(
                '/'
            )
            era5_prefix = DEFAULT_GCS_ERA5_CLIMATE_URI.replace(
                'gs://', ''
            ).rstrip('/')

            class _FakeGCSFileSystem:
                def _resolve_local(self, path: str) -> Path:
                    clean = path.replace('gs://', '').rstrip('/')
                    if clean == era5_prefix:
                        return era5_dir
                    if clean.startswith(f'{era5_prefix}/'):
                        return era5_dir / clean[len(era5_prefix) + 1 :]
                    if clean == gcs_prefix:
                        return hydroatlas_dir
                    if clean.startswith(f'{gcs_prefix}/'):
                        return hydroatlas_dir / clean[len(gcs_prefix) + 1 :]
                    return empty_dir / clean

                def exists(self, path: str) -> bool:
                    return self._resolve_local(path).exists()

                def ls(self, path: str) -> list[str]:
                    local_target = self._resolve_local(path)
                    clean = path.replace('gs://', '').rstrip('/')
                    return [
                        f'{clean}/{p.name}'
                        for p in sorted(local_target.iterdir())
                    ]

                def open(self, path: str, mode: str = 'rb'):  # noqa: ANN202
                    return self._resolve_local(path).open(mode)

                def cat_file(self, path: str) -> bytes:
                    return self._resolve_local(path).read_bytes()

            env_without_overrides = {
                k: v
                for k, v in os.environ.items()
                if k
                not in {
                    'HYDROATLAS_GDB_PATH',
                    'ERA5_CLIMATE_CACHE_DIR',
                    'ERA5_GRIDDED_ZARR_URI',
                }
            }
            with (
                mock.patch.dict(os.environ, env_without_overrides, clear=True),
                mock.patch.object(
                    _static_attributes,
                    'OPENHYDRONET_HYDROATLAS_CACHE_DIR',
                    empty_dir / 'hydroatlas',
                ),
                mock.patch.object(
                    _static_attributes,
                    'OPENHYDRONET_DATA_CACHE_DIR',
                    empty_dir / 'data',
                ),
                mock.patch.object(
                    _static_attributes,
                    'OPENHYDRONET_ERA5_CLIMATE_CACHE_DIR',
                    empty_dir / 'era5_climate',
                ),
                mock.patch.object(
                    _static_attributes,
                    'SHARED_HYDROATLAS_DIR',
                    empty_dir / 'shared_hydroatlas',
                ),
                mock.patch.object(
                    _static_attributes,
                    'HYDRO_BASINS_DIR',
                    empty_dir / 'hydro_basins',
                ),
                mock.patch('gcsfs.GCSFileSystem', _FakeGCSFileSystem),
            ):
                paths = resolve_app_static_extractor_paths()
                assert paths == {
                    'gdb_path': None,
                    'era5_cache_dir': None,
                    'gridded_era5_uri': None,
                    'gcs_gdb_uri': DEFAULT_GCS_HYDROATLAS_URI,
                    'gcs_era5_climate_uri': DEFAULT_GCS_ERA5_CLIMATE_URI,
                }
                extractor = get_attributes_extractor()
                assert extractor.no_download is True
                assert extractor.gdb_path is None
                assert extractor.gcs_gdb_uri == DEFAULT_GCS_HYDROATLAS_URI
                assert (
                    extractor.gcs_era5_climate_uri
                    == DEFAULT_GCS_ERA5_CLIMATE_URI
                )
                poly = box(-106.2, 39.5, -106.0, 39.7)
                res_gcs = extractor.extract_attributes_for_polygon(
                    poly,
                    catchment_id='gcs_streamed_basin',
                    era5_source='hybas',
                )
                card_gcs = build_attribute_card_payload(res_gcs)
                assert card_gcs['status'] == 'success'
                assert card_gcs['flat_attributes']['ele_mt_sav'] > 0
                assert card_gcs['flat_attributes']['p_mean'] > 0

                # Verify local GDB + missing local ERA5 streams ERA5 from GCS
                # while keeping HydroATLAS on the local partitioned path.
                reset_attributes_extractor()
                with mock.patch.dict(
                    os.environ,
                    {'HYDROATLAS_GDB_PATH': str(hydroatlas_dir)},
                ):
                    extractor_mixed = get_attributes_extractor()
                    assert extractor_mixed.gdb_path == hydroatlas_dir
                    assert extractor_mixed.gcs_gdb_uri is None
                    assert extractor_mixed.no_download is True
                    assert extractor_mixed.era5_loader is not None
                    res_mixed = extractor_mixed.extract_attributes_for_polygon(
                        poly,
                        catchment_id='mixed_local_gdb_gcs_era5',
                        era5_source='hybas',
                    )
                    card_mixed = build_attribute_card_payload(res_mixed)
                    assert (
                        card_mixed['flat_attributes']
                        == card_gcs['flat_attributes']
                    )


if __name__ == '__main__':
    unittest.main()
