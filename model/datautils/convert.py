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

"""Utilities to convert legacy Caravan NetCDF/CSV into unified Zarr format."""

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr

LOGGER = logging.getLogger(__name__)


def convert_caravan_attributes(
    attributes_dir: Path | str,
    output_zarr_path: Path | str,
    subdatasets: Sequence[str] | None = None,
) -> xr.Dataset:
    """Converts Caravan attribute CSV files into a consolidated Zarr store.

    Parameters
    ----------
    attributes_dir : Path | str
        Directory containing Caravan attribute subdirectories or CSV files.
    output_zarr_path : Path | str
        Destination path for the attributes Zarr store.
    subdatasets : Sequence[str], optional
        Optional list of subdataset directory names to convert.

    Returns
    -------
    xr.Dataset
        The combined static attributes Dataset with dimension ('basin').
    """
    attributes_dir = Path(attributes_dir)
    output_zarr_path = Path(output_zarr_path)

    if not attributes_dir.exists():
        raise FileNotFoundError(
            f'Attributes directory not found: {attributes_dir}'
        )

    # Group attribute CSV files by subdataset directory
    if subdatasets:
        csv_groups: list[list[Path]] = []
        for sub in subdatasets:
            sub_dir = attributes_dir / sub
            if not sub_dir.is_dir():
                raise FileNotFoundError(
                    f'Subdataset directory not found: {sub_dir}'
                )
            sub_csvs = sorted(sub_dir.glob('*.csv'))
            if not sub_csvs:
                raise FileNotFoundError(
                    f'No attribute CSV files found in {sub_dir}'
                )
            csv_groups.append(sub_csvs)
    else:
        all_csvs = sorted(attributes_dir.glob('**/*.csv'))
        by_parent: dict[Path, list[Path]] = {}
        for csv_file in all_csvs:
            by_parent.setdefault(csv_file.parent, []).append(csv_file)
        csv_groups = list(by_parent.values())

    if not csv_groups:
        raise FileNotFoundError(
            f'No attribute CSV files found in {attributes_dir}'
        )

    subdataset_dfs: list[pd.DataFrame] = []
    for group_files in csv_groups:
        table_dfs: list[pd.DataFrame] = []
        seen_columns: set[str] = set()
        for csv_file in group_files:
            df = pd.read_csv(csv_file)
            if 'gauge_id' in df.columns:
                df = df.set_index('gauge_id')
            elif df.index.name != 'gauge_id' and 'basin' in df.columns:
                df = df.set_index('basin')
            df.index.name = 'basin'

            if df.columns.duplicated().any():
                dup_cols = sorted(set(df.columns[df.columns.duplicated()]))
                raise ValueError(
                    f'Duplicate attribute columns {dup_cols} in {csv_file}'
                )
            overlap = sorted(seen_columns.intersection(df.columns))
            if overlap:
                raise ValueError(
                    f'Duplicate attribute columns {overlap} across CSV files '
                    f'in {csv_file.parent}'
                )
            seen_columns.update(df.columns)

            num_cols = df.select_dtypes(include=[np.number]).columns
            df[num_cols] = df[num_cols].astype(np.float32)
            table_dfs.append(df)

        sub_df = pd.concat(table_dfs, axis=1)
        subdataset_dfs.append(sub_df)

    combined_df = pd.concat(subdataset_dfs, axis=0)
    combined_df.index.name = 'basin'
    num_cols = combined_df.select_dtypes(include=[np.number]).columns
    combined_df[num_cols] = combined_df[num_cols].astype(np.float32)

    ds = combined_df.to_xarray()

    # Save as Zarr
    output_zarr_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(output_zarr_path, mode='w', consolidated=True)
    LOGGER.info('Successfully wrote attributes Zarr to %s', output_zarr_path)
    return ds


def convert_caravan_timeseries(
    timeseries_dir: Path | str,
    output_zarr_path: Path | str,
    variables: Sequence[str] | None = None,
) -> xr.Dataset:
    """Converts Caravan timeseries NetCDF/CSV files into a single Zarr store.

    Parameters
    ----------
    timeseries_dir : Path | str
        Directory containing timeseries files (e.g. timeseries/netcdf/...).
    output_zarr_path : Path | str
        Destination path for the unified timeseries Zarr store.
    variables : Sequence[str], optional
        Variables to extract (e.g. ['streamflow']). If None, extracts all.

    Returns
    -------
    xr.Dataset
        The unified timeseries Dataset with dimensions ('basin', 'date').
    """
    timeseries_dir = Path(timeseries_dir)
    output_zarr_path = Path(output_zarr_path)

    if not timeseries_dir.exists():
        raise FileNotFoundError(
            f'Timeseries directory not found: {timeseries_dir}'
        )

    # Find all NC files first, then fallback to CSV files
    nc_files = sorted(list(timeseries_dir.glob('**/*.nc')))
    csv_files = (
        sorted(list(timeseries_dir.glob('**/*.csv'))) if not nc_files else []
    )
    files = nc_files if nc_files else csv_files

    if not files:
        raise FileNotFoundError(
            f'No timeseries files (.nc or .csv) found in {timeseries_dir}'
        )

    datasets = []
    basins = []

    for file_path in files:
        basin_id = file_path.stem
        if nc_files:
            ds = xr.open_dataset(file_path)
            if variables:
                missing_vars = sorted(set(variables) - set(ds.data_vars))
                if missing_vars:
                    raise ValueError(
                        f'Requested variables {missing_vars} not found in '
                        f'{file_path}.'
                    )
                ds = ds[list(variables)]
        else:
            df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
            if variables:
                missing_vars = sorted(set(variables) - set(df.columns))
                if missing_vars:
                    raise ValueError(
                        f'Requested variables {missing_vars} not found in '
                        f'{file_path}.'
                    )
                df = df[list(variables)]
            df = df.astype(np.float32)
            ds = df.to_xarray()

        # Cast floats to float32
        for v in ds.data_vars:
            if np.issubdtype(ds[v].dtype, np.floating):
                ds[v] = ds[v].astype(np.float32)

        datasets.append(ds)
        basins.append(basin_id)

    # Align dates across all basins
    combined_ds = xr.concat(
        datasets, dim=pd.Index(basins, name='basin'), join='outer'
    )

    # Save to Zarr
    output_zarr_path.parent.mkdir(parents=True, exist_ok=True)
    combined_ds = combined_ds.chunk('auto')
    combined_ds.to_zarr(output_zarr_path, mode='w', consolidated=True)
    LOGGER.info('Successfully wrote timeseries Zarr to %s', output_zarr_path)
    return combined_ds


def convert_caravan_to_zarr(
    caravan_dir: Path | str,
    output_dir: Path | str,
    variables: Sequence[str] | None = ('streamflow',),
) -> tuple[xr.Dataset, xr.Dataset]:
    """Converts a full Caravan directory (attributes and timeseries) to Zarr stores.

    Parameters
    ----------
    caravan_dir : Path | str
        Root directory of the Caravan dataset.
    output_dir : Path | str
        Destination directory for the converted Zarr stores.
    variables : Sequence[str], optional
        Variables to extract for timeseries. Default is ('streamflow',).

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset]
        (attributes_ds, timeseries_ds)
    """
    caravan_dir = Path(caravan_dir)
    output_dir = Path(output_dir)

    if not caravan_dir.exists():
        raise FileNotFoundError(f'Caravan directory not found: {caravan_dir}')

    attr_dir = caravan_dir / 'attributes'
    if not attr_dir.is_dir():
        raise FileNotFoundError(
            f'Caravan attributes directory not found: {attr_dir}'
        )

    ts_dir = caravan_dir / 'timeseries' / 'netcdf'
    if not ts_dir.is_dir():
        raise FileNotFoundError(
            f'Caravan timeseries directory not found: {ts_dir}'
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    attr_ds = convert_caravan_attributes(
        attr_dir, output_dir / 'attributes.zarr'
    )
    ts_ds = convert_caravan_timeseries(
        ts_dir, output_dir / 'streamflow.zarr', variables=variables
    )

    return attr_ds, ts_ds
