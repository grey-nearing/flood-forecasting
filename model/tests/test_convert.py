# Copyright 2025 Google LLC
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

"""Unit tests for Caravan to Zarr conversion utilities."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from model.datautils.convert import (
    convert_caravan_attributes,
    convert_caravan_timeseries,
    convert_caravan_to_zarr,
)


@pytest.fixture
def mock_caravan_directory(tmp_path: Path):
    """Creates a mock Caravan NetCDF/CSV directory structure."""
    caravan_dir = tmp_path / 'mock_caravan'

    # Attributes
    attr_dir = caravan_dir / 'attributes' / 'camelsus'
    attr_dir.mkdir(parents=True, exist_ok=True)
    df_attr = pd.DataFrame({
        'gauge_id': ['camelsus_0101', 'camelsus_0102'],
        'area': [100.5, 250.0],
        'elevation': [500.0, 1200.0],
    })
    df_attr.to_csv(attr_dir / 'attributes_camelsus.csv', index=False)

    # Timeseries NetCDF
    ts_dir = caravan_dir / 'timeseries' / 'netcdf' / 'camelsus'
    ts_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range('2020-01-01', '2020-01-10', freq='D')

    for basin in ['camelsus_0101', 'camelsus_0102']:
        ds = xr.Dataset(
            {
                'streamflow': (
                    ('date',),
                    np.random.rand(len(dates)).astype(np.float32),
                ),
                'precipitation': (
                    ('date',),
                    np.random.rand(len(dates)).astype(np.float32),
                ),
            },
            coords={'date': dates},
        )
        ds.to_netcdf(ts_dir / f'{basin}.nc')
        ds.close()

    return caravan_dir


def test_convert_caravan_attributes(
    mock_caravan_directory: Path, tmp_path: Path
):
    output_zarr = tmp_path / 'attributes.zarr'
    ds = convert_caravan_attributes(
        mock_caravan_directory / 'attributes', output_zarr
    )

    assert output_zarr.exists()
    assert 'basin' in ds.coords
    assert set(ds.coords['basin'].values) == {'camelsus_0101', 'camelsus_0102'}
    assert 'area' in ds.data_vars
    assert 'elevation' in ds.data_vars
    assert ds['area'].dtype == np.float32


def test_convert_caravan_timeseries(
    mock_caravan_directory: Path, tmp_path: Path
):
    output_zarr = tmp_path / 'streamflow.zarr'
    ds = convert_caravan_timeseries(
        mock_caravan_directory / 'timeseries' / 'netcdf',
        output_zarr,
        variables=['streamflow'],
    )

    assert output_zarr.exists()
    assert 'basin' in ds.coords
    assert 'date' in ds.coords
    assert set(ds.coords['basin'].values) == {'camelsus_0101', 'camelsus_0102'}
    assert 'streamflow' in ds.data_vars
    assert 'precipitation' not in ds.data_vars
    assert ds['streamflow'].shape == (2, 10)


def test_convert_caravan_to_zarr_full(
    mock_caravan_directory: Path, tmp_path: Path
):
    output_dir = tmp_path / 'caravan_zarr'
    attr_ds, ts_ds = convert_caravan_to_zarr(mock_caravan_directory, output_dir)

    assert (output_dir / 'attributes.zarr').exists()
    assert (output_dir / 'streamflow.zarr').exists()
    assert set(attr_ds.coords['basin'].values) == {
        'camelsus_0101',
        'camelsus_0102',
    }
    assert set(ts_ds.coords['basin'].values) == {
        'camelsus_0101',
        'camelsus_0102',
    }


def test_convert_caravan_attributes_multi_table_and_subdatasets(tmp_path: Path):
    attr_root = tmp_path / 'attributes'
    sub1 = attr_root / 'camelsus'
    sub2 = attr_root / 'camelsgb'
    sub1.mkdir(parents=True)
    sub2.mkdir(parents=True)

    pd.DataFrame({
        'gauge_id': ['camelsus_01', 'camelsus_02'],
        'area': [100.0, 200.0],
    }).to_csv(sub1 / 'attributes_other.csv', index=False)
    pd.DataFrame({
        'gauge_id': ['camelsus_01', 'camelsus_02'],
        'p_mean': [3.1, 4.2],
    }).to_csv(sub1 / 'attributes_caravan.csv', index=False)

    pd.DataFrame({
        'gauge_id': ['camelsgb_01'],
        'area': [50.0],
        'p_mean': [2.5],
    }).to_csv(sub2 / 'attributes_all.csv', index=False)

    out_zarr = tmp_path / 'out_attributes.zarr'
    ds = convert_caravan_attributes(
        attr_root, out_zarr, subdatasets=['camelsus', 'camelsgb']
    )
    assert set(ds.coords['basin'].values) == {
        'camelsus_01',
        'camelsus_02',
        'camelsgb_01',
    }
    assert set(ds.data_vars) == {'area', 'p_mean'}


def test_convert_caravan_attributes_duplicate_columns_raises(tmp_path: Path):
    attr_root = tmp_path / 'dup_attributes' / 'camelsus'
    attr_root.mkdir(parents=True)

    pd.DataFrame({
        'gauge_id': ['camelsus_01'],
        'area': [100.0],
    }).to_csv(attr_root / 'table1.csv', index=False)
    pd.DataFrame({
        'gauge_id': ['camelsus_01'],
        'area': [999.0],
    }).to_csv(attr_root / 'table2.csv', index=False)

    with pytest.raises(ValueError, match='Duplicate attribute columns'):
        convert_caravan_attributes(
            tmp_path / 'dup_attributes', tmp_path / 'out.zarr'
        )


def test_convert_caravan_missing_dirs_raise_file_not_found(
    mock_caravan_directory: Path, tmp_path: Path
):
    with pytest.raises(FileNotFoundError, match='Attributes directory not found'):
        convert_caravan_attributes(
            tmp_path / 'nonexistent_attrs', tmp_path / 'out.zarr'
        )

    with pytest.raises(FileNotFoundError, match='Subdataset directory not found'):
        convert_caravan_attributes(
            mock_caravan_directory / 'attributes',
            tmp_path / 'out.zarr',
            subdatasets=['missing_subdataset'],
        )

    incomplete_caravan = tmp_path / 'incomplete_caravan'
    incomplete_caravan.mkdir()
    with pytest.raises(
        FileNotFoundError, match='Caravan attributes directory not found'
    ):
        convert_caravan_to_zarr(incomplete_caravan, tmp_path / 'out_dir')

    (incomplete_caravan / 'attributes').mkdir()
    with pytest.raises(
        FileNotFoundError, match='Caravan timeseries directory not found'
    ):
        convert_caravan_to_zarr(incomplete_caravan, tmp_path / 'out_dir')
