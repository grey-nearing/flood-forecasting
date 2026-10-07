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

"""Unit tests for ``benchmarks.catchment_delineation`` and ``benchmarks.tools.build_benchmark_dataset``."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from benchmarks.catchment_delineation import (
    _evaluate_single_basin,
    run_benchmark,
    summarize_results,
)
from benchmarks.tools import build_benchmark_dataset

pytestmark = pytest.mark.unit


@pytest.fixture
def synthetic_tile_dir(tmp_path: Path) -> Path:
    """Create a synthetic 5x5 degree D8 tile for n40w090 with a known river channel."""
    shape = (6000, 6000)
    grid = np.zeros(shape, dtype=np.uint8)
    grid[3000, 3000] = 64  # North
    grid[2999, 3000] = 64  # North
    grid[2998, 3000] = 1  # East
    grid[2998, 3001] = 1  # East (outlet)

    tile_path = tmp_path / 'n40w090.npy'
    np.save(tile_path, grid)
    return tmp_path


class TestCatchmentBenchmarkModule:
    def test_evaluate_single_basin_success(
        self, synthetic_tile_dir: Path
    ) -> None:
        row_dict = {
            'gauge_id': 'test_001',
            'continent': 'North America',
            'hemisphere': 'NW',
            'size_tier': '1_micro',
            'latitude': 35.0 + (6000 - 2998.5) * (5.0 / 6000),
            'longitude': -90.0 + 3001.5 * (5.0 / 6000),
            'reference_area_km2': 0.03,
            'geometry_wkt': Polygon(
                [
                    (-87.500, 37.499),
                    (-87.498, 37.499),
                    (-87.498, 37.502),
                    (-87.500, 37.502),
                ]
            ).wkt,
        }
        res, _ = _evaluate_single_basin(
            row_dict,
            tiles_dir=str(synthetic_tile_dir),
            gcs_uri=None,
            cache_dir=None,
            snap_window_cells=5,
            use_area_hint=True,
            save_geometries=False,
        )
        assert res['gauge_id'] == 'test_001'
        assert res['status'] == 'SUCCESS'
        assert res['del_area_km2'] > 0
        assert res['iou'] >= 0.0

    def test_evaluate_single_basin_out_of_coverage(
        self, synthetic_tile_dir: Path
    ) -> None:
        row_dict = {
            'gauge_id': 'arctic_001',
            'continent': 'North America',
            'hemisphere': 'NW',
            'size_tier': '2_small',
            'latitude': 65.0,
            'longitude': -140.0,
            'reference_area_km2': 150.0,
            'geometry_wkt': Polygon(
                [(-140.1, 64.9), (-139.9, 64.9), (-139.9, 65.1), (-140.1, 65.1)]
            ).wkt,
        }
        res, _ = _evaluate_single_basin(
            row_dict,
            tiles_dir=str(synthetic_tile_dir),
            gcs_uri=None,
            cache_dir=None,
            snap_window_cells=5,
            use_area_hint=True,
            save_geometries=False,
        )
        assert res['status'].startswith('OUT_OF_COVERAGE')
        assert np.isnan(res['iou'])
        assert np.isnan(res['iou_unconditional'])

    def test_evaluate_single_basin_area_hint_failure_and_report_penalization(
        self, synthetic_tile_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        row_fail = {
            'gauge_id': 'hint_fail_001',
            'continent': 'North America',
            'hemisphere': 'NW',
            'size_tier': '4_large',
            'latitude': 35.0 + (6000 - 2998.5) * (5.0 / 6000),
            'longitude': -90.0 + 3001.5 * (5.0 / 6000),
            'reference_area_km2': 5000.0,
            'geometry_wkt': Polygon(
                [(-87.6, 37.4), (-87.4, 37.4), (-87.4, 37.6), (-87.6, 37.6)]
            ).wkt,
        }
        res_fail, _ = _evaluate_single_basin(
            row_fail,
            tiles_dir=str(synthetic_tile_dir),
            gcs_uri=None,
            cache_dir=None,
            snap_window_cells=5,
            use_area_hint=True,
            save_geometries=True,
        )
        assert res_fail['status'].startswith('AREA_HINT_FAILURE')
        assert res_fail['iou_unconditional'] == 0.0
        assert res_fail['del_geometry_wkt'] is None

        row_ok = dict(row_fail, gauge_id='ok_001', reference_area_km2=0.03)
        res_ok, _ = _evaluate_single_basin(
            row_ok,
            tiles_dir=str(synthetic_tile_dir),
            gcs_uri=None,
            cache_dir=None,
            snap_window_cells=5,
            use_area_hint=True,
            save_geometries=True,
        )
        assert res_ok['status'] == 'SUCCESS'
        assert res_ok['del_geometry_wkt'] is not None

        df_res = pd.DataFrame([res_ok, res_fail])
        summarize_results(df_res, total_time=1.0)
        captured = capsys.readouterr().out
        assert 'UNCONDITIONAL IN-COVERAGE METRICS' in captured
        assert 'CONDITIONAL SUCCESS METRICS (status == SUCCESS only)' in captured
        assert 'Unconditional IoU Lower Tail : Min=' in captured

    def test_run_benchmark_and_export_redelineated_dataset(
        self, synthetic_tile_dir: Path, tmp_path: Path
    ) -> None:
        df_in = pd.DataFrame(
            [
                {
                    'gauge_id': 'ok_001',
                    'continent': 'North America',
                    'hemisphere': 'NW',
                    'size_tier': '1_micro',
                    'latitude': 35.0 + (6000 - 2998.5) * (5.0 / 6000),
                    'longitude': -90.0 + 3001.5 * (5.0 / 6000),
                    'ref_area_km2': 0.03,
                    'geometry_wkt': Polygon(
                        [
                            (-87.500, 37.499),
                            (-87.498, 37.499),
                            (-87.498, 37.502),
                            (-87.500, 37.502),
                        ]
                    ).wkt,
                },
                {
                    'gauge_id': 'fail_002',
                    'continent': 'North America',
                    'hemisphere': 'NW',
                    'size_tier': '4_large',
                    'latitude': 35.0 + (6000 - 2998.5) * (5.0 / 6000),
                    'longitude': -90.0 + 3001.5 * (5.0 / 6000),
                    'ref_area_km2': 5000.0,
                    'geometry_wkt': Polygon(
                        [(-87.6, 37.4), (-87.4, 37.4), (-87.4, 37.6), (-87.6, 37.6)]
                    ).wkt,
                },
            ]
        )
        dataset_path = tmp_path / 'bench_in.parquet'
        df_in.to_parquet(dataset_path, index=False)

        output_path = tmp_path / 'bench_out.parquet'
        export_path = tmp_path / 'redelineated.parquet'
        res_df = run_benchmark(
            dataset_path=dataset_path,
            tiles_dir=synthetic_tile_dir,
            workers=1,
            snap_window_cells=5,
            use_area_hint=True,
            output_path=output_path,
            save_geometries=True,
            export_redelineated_dataset_path=export_path,
        )
        assert len(res_df) == 2
        assert output_path.exists()
        assert export_path.exists()

        exported = pd.read_parquet(export_path)
        assert len(exported) == 2
        assert exported.loc[exported['gauge_id'] == 'ok_001', 'geometry_wkt'].iloc[0] is not None
        assert pd.isna(exported.loc[exported['gauge_id'] == 'fail_002', 'geometry_wkt'].iloc[0])
        assert exported.loc[exported['gauge_id'] == 'fail_002', 'delineation_status'].iloc[0].startswith('AREA_HINT_FAILURE')


class TestBuildBenchmarkDatasetCLI:
    def test_cli_requires_explicit_flags(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys, 'argv', ['build-benchmark-dataset'])
        with pytest.raises(SystemExit):
            build_benchmark_dataset.main()

    def test_missing_world_geojson_raises_file_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dummy_shp = tmp_path / 'test_basin_shapes.shp'
        dummy_shp.write_text('dummy')
        monkeypatch.setattr(
            sys,
            'argv',
            [
                'build-benchmark-dataset',
                '--shapes',
                str(dummy_shp),
                '--world-geojson',
                str(tmp_path / 'nonexistent_world.geojson'),
                '--output',
                str(tmp_path / 'out.parquet'),
            ],
        )
        with pytest.raises(FileNotFoundError, match='World GeoJSON file does not exist'):
            build_benchmark_dataset.main()

    def test_missing_shapes_raises_file_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        world_geojson = tmp_path / 'world.geojson'
        world_geojson.write_text(
            json.dumps({'type': 'FeatureCollection', 'features': []})
        )
        monkeypatch.setattr(
            sys,
            'argv',
            [
                'build-benchmark-dataset',
                '--shapes',
                str(tmp_path / 'nonexistent_dir'),
                '--world-geojson',
                str(world_geojson),
                '--output',
                str(tmp_path / 'out.parquet'),
            ],
        )
        with pytest.raises(FileNotFoundError, match='Shapefile path does not exist'):
            build_benchmark_dataset.main()
