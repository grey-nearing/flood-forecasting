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

"""Unit tests for ``benchmarks.static_extractor``."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from benchmarks.static_extractor import (
    compute_categorical_metrics,
    compute_continuous_metrics,
    run_benchmark,
)
from multimet.static_extractor import ATTRIBUTE_DEFINITIONS

pytestmark = pytest.mark.unit


@pytest.fixture
def synthetic_hydroatlas_layer(tmp_path: Path) -> Path:
    """Create a synthetic shapefile mimicking BasinATLAS Level 12 sub-basins."""
    polys = [
        box(-100.0, 40.0, -99.5, 40.5),
        box(-99.5, 40.0, -99.0, 40.5),
    ]
    data: dict[str, list[float | int]] = {
        'HYBAS_ID': [101, 102],
        'NEXT_DOWN': [102, 0],
        'SUB_AREA': [2500.0, 2500.0],
        'UP_AREA': [2500.0, 5000.0],
        'ele_mt_sav': [1.0, 3.0],
        'slp_dg_sav': [10.0, 30.0],
        'pre_mm_syr': [800.0, 1200.0],
        'tmp_dc_syr': [100.0, 200.0],
        'ari_ix_sav': [50.0, 150.0],
        'cly_pc_sav': [20.0, 40.0],
        'snd_pc_sav': [50.0, 30.0],
        'slt_pc_sav': [30.0, 30.0],
        'for_pc_sse': [60.0, 20.0],
        'crp_pc_sse': [10.0, 50.0],
        'urb_pc_sse': [5.0, 15.0],
        'gwt_cm_sav': [100.0, 300.0],
        'swc_pc_syr': [40.0, 60.0],
        'inu_pc_smx': [2.0, 8.0],
        'glc_cl_smj': [12, 16],
        'clz_cl_smj': [5, 5],
        'lit_cl_smj': [1, 3],
        'dis_m3_pyr': [15.0, 45.0],
        'run_mm_syr': [300.0, 500.0],
    }
    assert 'ele_mt_sav' in ATTRIBUTE_DEFINITIONS

    gdf = gpd.GeoDataFrame(data, geometry=polys, crs='EPSG:4326')
    shp_path = tmp_path / 'synthetic_basinatlas.shp'
    gdf.to_file(shp_path)
    return shp_path


def test_benchmark_module_runs_end_to_end_and_reports_unconditional_metrics(
    synthetic_hydroatlas_layer: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify benchmarks/static_extractor.py computes conditional and unconditional metrics."""
    basin_poly_1 = box(-100.0, 40.0, -99.5, 40.5)
    basin_poly_2 = box(-99.75, 40.0, -99.25, 40.5)
    df = pd.DataFrame(
        {
            'gauge_id': [
                'bench_ok_1',
                'bench_ok_2',
                'bench_none_geom',
                'bench_out_of_cov',
            ],
            'dataset': ['camels', 'camels', 'camels', 'camels'],
            'size_tier': ['2_small', '2_small', '2_small', '2_small'],
            'country': ['USA', 'USA', 'USA', 'USA'],
            'ref_area_km2': [2500.0, 2500.0, 2500.0, 2500.0],
            'geometry_wkt': [
                basin_poly_1.wkt,
                basin_poly_2.wkt,
                None,
                None,
            ],
            'delineation_status': [
                'SUCCESS',
                'SUCCESS',
                'AREA_HINT_FAILURE',
                'OUT_OF_COVERAGE: latitude >= 60',
            ],
            'ref_ele_mt_sav': [1.0, 2.0, 2.0, 2.0],
            'ref_clz_cl_smj': [5.0, 5.0, 5.0, 5.0],
        }
    )
    dataset_parquet = tmp_path / 'bench.parquet'
    df.to_parquet(dataset_parquet)

    out_dir = tmp_path / 'bench_results'
    attr_metrics_df, basin_metrics_df = run_benchmark(
        dataset_path=str(dataset_parquet),
        gdb_path=str(synthetic_hydroatlas_layer),
        era5_source='none',
        workers=1,
        output_dir=str(out_dir),
    )
    assert (out_dir / 'benchmark_attribute_metrics.csv').exists()
    assert (out_dir / 'benchmark_basin_metrics.csv').exists()
    assert (out_dir / 'benchmark_report.md').exists()
    assert len(basin_metrics_df) == 4

    ele_row = attr_metrics_df[attr_metrics_df['attribute'] == 'ele_mt_sav'].iloc[0]
    # 4 total basins, 1 OUT_OF_COVERAGE -> 3 in-coverage basins (2 valid, 1 NaN failure)
    assert int(ele_row['n']) == 2
    assert int(ele_row['pred_nan_when_ref_valid']) == 1
    assert np.isclose(ele_row['mae'], 0.0, atol=1e-4)
    assert np.isclose(
        ele_row['unconditional_within_1pct_pct'], 100.0 * 2.0 / 3.0, atol=1e-2
    )

    clz_row = attr_metrics_df[attr_metrics_df['attribute'] == 'clz_cl_smj'].iloc[0]
    assert np.isclose(clz_row['accuracy_pct'], 100.0)
    assert np.isclose(
        clz_row['unconditional_accuracy_pct'], 100.0 * 2.0 / 3.0, atol=1e-2
    )

    captured = capsys.readouterr().out
    assert 'In-Coverage Success Rate' in captured
    assert 'Unconditional Acc % (NaN=wrong)' in captured


def test_compute_continuous_and_categorical_metrics_penalizes_missing_predictions() -> None:
    """Verify compute_continuous_metrics and compute_categorical_metrics penalize NaN predictions."""
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([100.0, 200.0, 300.0, np.nan])
    cont = compute_continuous_metrics(y_true, y_pred)
    assert cont['n'] == 3
    assert cont['pred_nan_when_ref_valid'] == 1
    assert np.isclose(cont['pearson_r'], 1.0)
    assert np.isclose(cont['penalized_pearson_r'], 0.75)
    assert np.isclose(cont['unconditional_within_1pct_pct'], 75.0)

    cat = compute_categorical_metrics(
        np.array([1.0, 2.0, 1.0, 2.0]),
        np.array([1.0, 2.0, 1.0, np.nan]),
    )
    assert cat['n'] == 3
    assert cat['pred_nan_when_ref_valid'] == 1
    assert np.isclose(cat['accuracy_pct'], 100.0)
    assert np.isclose(cat['unconditional_accuracy_pct'], 75.0)
