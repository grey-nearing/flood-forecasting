# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Historical weather data extractor and multi-basin Zarr archive generator."""

from __future__ import annotations

import hashlib
import logging
import math
import shutil
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fsspec
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry
import xarray as xr
from shapely.geometry.base import BaseGeometry

from frontend.config import HISTORICAL_DIR
from frontend.weather_sources import get_weather_source
from multimet.utils.spatial import slice_coordinates_by_bounds
from multimet.utils.zonal import (
    MIN_VALID_COVERAGE_FRACTION,
    ZonalWeightMatrix,
)
from utils.file_paths import (
    GITHUB_REPO_URL,
    HISTORICAL_MASTER_ZARR_NAME,
)

MultiPolygon = shapely.geometry.MultiPolygon
Polygon = shapely.geometry.Polygon
shape = shapely.geometry.shape

# Alias ZonalEngine to ZonalWeightMatrix for canonical zonal engine access.
ZonalEngine = ZonalWeightMatrix

logger = logging.getLogger(__name__)

_FORECAST_ARRAY_NDIM: int = 2

# Canonical variable aliases per weather source:
# maps requested variable name -> (canonical_output_name, candidate_source_vars)
_SOURCE_VARIABLE_ALIASES: dict[str, dict[str, tuple[str, tuple[str, ...]]]] = {
    'cpc': {
        'cpc_precipitation': (
            'cpc_precipitation',
            ('cpc_precipitation', 'precip', 'total_precipitation'),
        ),
        'total_precipitation': (
            'cpc_precipitation',
            ('cpc_precipitation', 'precip', 'total_precipitation'),
        ),
        'precip': (
            'cpc_precipitation',
            ('cpc_precipitation', 'precip', 'total_precipitation'),
        ),
    },
    'imerg': {
        'imerg_precipitation': (
            'imerg_precipitation',
            ('imerg_precipitation', 'precip', 'total_precipitation'),
        ),
        'total_precipitation': (
            'imerg_precipitation',
            ('imerg_precipitation', 'precip', 'total_precipitation'),
        ),
        'precip': (
            'imerg_precipitation',
            ('imerg_precipitation', 'precip', 'total_precipitation'),
        ),
    },
    'era5': {
        'era5land_total_precipitation': (
            'era5land_total_precipitation',
            ('era5land_total_precipitation', 'total_precipitation', 'tp'),
        ),
        'total_precipitation': (
            'era5land_total_precipitation',
            ('era5land_total_precipitation', 'total_precipitation', 'tp'),
        ),
        'tp': (
            'era5land_total_precipitation',
            ('era5land_total_precipitation', 'total_precipitation', 'tp'),
        ),
        'era5land_temperature_2m': (
            'era5land_temperature_2m',
            (
                'era5land_temperature_2m',
                'temperature_2m',
                '2m_temperature',
                't2m',
            ),
        ),
        '2m_temperature': (
            'era5land_temperature_2m',
            (
                'era5land_temperature_2m',
                'temperature_2m',
                '2m_temperature',
                't2m',
            ),
        ),
        'temperature_2m': (
            'era5land_temperature_2m',
            (
                'era5land_temperature_2m',
                'temperature_2m',
                '2m_temperature',
                't2m',
            ),
        ),
        't2m': (
            'era5land_temperature_2m',
            (
                'era5land_temperature_2m',
                'temperature_2m',
                '2m_temperature',
                't2m',
            ),
        ),
        'era5land_dewpoint_temperature_2m': (
            'era5land_dewpoint_temperature_2m',
            (
                'era5land_dewpoint_temperature_2m',
                'dewpoint_temperature_2m',
                'dewpoint_temperature',
                'd2m',
            ),
        ),
        'dewpoint_temperature': (
            'era5land_dewpoint_temperature_2m',
            (
                'era5land_dewpoint_temperature_2m',
                'dewpoint_temperature_2m',
                'dewpoint_temperature',
                'd2m',
            ),
        ),
        'dewpoint_temperature_2m': (
            'era5land_dewpoint_temperature_2m',
            (
                'era5land_dewpoint_temperature_2m',
                'dewpoint_temperature_2m',
                'dewpoint_temperature',
                'd2m',
            ),
        ),
        'd2m': (
            'era5land_dewpoint_temperature_2m',
            (
                'era5land_dewpoint_temperature_2m',
                'dewpoint_temperature_2m',
                'dewpoint_temperature',
                'd2m',
            ),
        ),
        'era5land_surface_pressure': (
            'era5land_surface_pressure',
            ('era5land_surface_pressure', 'surface_pressure', 'sp'),
        ),
        'surface_pressure': (
            'era5land_surface_pressure',
            ('era5land_surface_pressure', 'surface_pressure', 'sp'),
        ),
        'sp': (
            'era5land_surface_pressure',
            ('era5land_surface_pressure', 'surface_pressure', 'sp'),
        ),
        'era5land_surface_net_solar_radiation': (
            'era5land_surface_net_solar_radiation',
            (
                'era5land_surface_net_solar_radiation',
                'surface_net_solar_radiation',
                'surface_solar_radiation',
                'ssr',
            ),
        ),
        'surface_net_solar_radiation': (
            'era5land_surface_net_solar_radiation',
            (
                'era5land_surface_net_solar_radiation',
                'surface_net_solar_radiation',
                'surface_solar_radiation',
                'ssr',
            ),
        ),
        'surface_solar_radiation': (
            'era5land_surface_net_solar_radiation',
            (
                'era5land_surface_net_solar_radiation',
                'surface_net_solar_radiation',
                'surface_solar_radiation',
                'ssr',
            ),
        ),
        'ssr': (
            'era5land_surface_net_solar_radiation',
            (
                'era5land_surface_net_solar_radiation',
                'surface_net_solar_radiation',
                'surface_solar_radiation',
                'ssr',
            ),
        ),
        'era5land_surface_net_thermal_radiation': (
            'era5land_surface_net_thermal_radiation',
            (
                'era5land_surface_net_thermal_radiation',
                'surface_net_thermal_radiation',
                'str',
            ),
        ),
        'surface_net_thermal_radiation': (
            'era5land_surface_net_thermal_radiation',
            (
                'era5land_surface_net_thermal_radiation',
                'surface_net_thermal_radiation',
                'str',
            ),
        ),
        'str': (
            'era5land_surface_net_thermal_radiation',
            (
                'era5land_surface_net_thermal_radiation',
                'surface_net_thermal_radiation',
                'str',
            ),
        ),
        'era5land_potential_evaporation': (
            'era5land_potential_evaporation',
            ('era5land_potential_evaporation', 'potential_evaporation', 'pev'),
        ),
        'potential_evaporation': (
            'era5land_potential_evaporation',
            ('era5land_potential_evaporation', 'potential_evaporation', 'pev'),
        ),
        'pev': (
            'era5land_potential_evaporation',
            ('era5land_potential_evaporation', 'potential_evaporation', 'pev'),
        ),
    },
    'ifs': {
        'hres_total_precipitation': (
            'hres_total_precipitation',
            ('hres_total_precipitation', 'total_precipitation', 'tp'),
        ),
        'hres_temperature_2m': (
            'hres_temperature_2m',
            ('hres_temperature_2m', 'temperature_2m', '2m_temperature', 't2m'),
        ),
        'hres_surface_pressure': (
            'hres_surface_pressure',
            ('hres_surface_pressure', 'surface_pressure', 'sp'),
        ),
        'hres_surface_net_solar_radiation': (
            'hres_surface_net_solar_radiation',
            (
                'hres_surface_net_solar_radiation',
                'surface_net_solar_radiation',
                'ssr',
            ),
        ),
        'hres_surface_net_thermal_radiation': (
            'hres_surface_net_thermal_radiation',
            (
                'hres_surface_net_thermal_radiation',
                'surface_net_thermal_radiation',
                'str',
            ),
        ),
    },
    'graphcast': {
        'graphcast_total_precipitation': (
            'graphcast_total_precipitation',
            ('graphcast_total_precipitation', 'total_precipitation', 'tp'),
        ),
        'graphcast_temperature_2m': (
            'graphcast_temperature_2m',
            (
                'graphcast_temperature_2m',
                'temperature_2m',
                '2m_temperature',
                't2m',
            ),
        ),
    },
}
_SOURCE_VARIABLE_ALIASES['era5-land'] = _SOURCE_VARIABLE_ALIASES['era5']


def _canonical_units_for_variable(var_name: str) -> str:
    """Return canonical units string for a weather variable name."""
    lower = var_name.lower()
    if 'precip' in lower or 'evap' in lower or lower in {'tp', 'pev'}:
        return 'mm/day'
    if 'temp' in lower or lower in {'t2m', 'd2m', '2t'}:
        return 'degC'
    if 'pressure' in lower or lower == 'sp':
        return 'kPa'
    if (
        'radiation' in lower
        or 'solar' in lower
        or 'thermal' in lower
        or lower in {'ssr', 'str'}
    ):
        return 'W/m^2'
    return 'dimensionless'


def _verify_zarr_store_exists(archive_uri: str) -> bool:
    """Verify that a Zarr store exists and return whether it is consolidated.

    Args:
        archive_uri: Local directory path or remote URI (e.g. ``gs://...``).

    Returns:
        ``True`` if consolidated ``.zmetadata`` exists, ``False`` if the store
        exists without consolidated metadata.

    Raises:
        FileNotFoundError: If the Zarr store does not exist or lacks Zarr
          group metadata markers.
    """
    if archive_uri.startswith(('gs://', 's3://', 'http://', 'https://')):
        fs, fs_path = fsspec.core.url_to_fs(archive_uri)
        clean_path = fs_path.rstrip('/')
        if fs.exists(f'{clean_path}/.zmetadata'):
            return True
        if fs.exists(f'{clean_path}/.zgroup') or fs.exists(
            f'{clean_path}/zarr.json'
        ):
            return False
        msg = f'Remote Zarr archive not found at {archive_uri!r}.'
        raise FileNotFoundError(msg)

    local_path = Path(archive_uri)
    if not local_path.exists():
        msg = f'Local Zarr archive not found at {archive_uri!r}.'
        raise FileNotFoundError(msg)
    if (local_path / '.zmetadata').is_file():
        return True
    if (local_path / '.zgroup').is_file() or (
        local_path / 'zarr.json'
    ).is_file():
        return False
    msg = f'Path {archive_uri!r} is not a valid Zarr store.'
    raise FileNotFoundError(msg)


def _validate_positive_area_km2(
    props: Mapping[str, Any], basin_id: str
) -> float:
    """Validate and extract a finite positive catchment area in km^2."""
    raw_area = props.get('area_km2', props.get('area'))
    if (
        raw_area is None
        or isinstance(raw_area, bool)
        or not isinstance(raw_area, (int, float, np.number))
    ):
        msg = (
            f'Catchment {basin_id!r} is missing a numeric area_km2 property;'
            ' refusing to approximate area on a planar grid.'
        )
        raise ValueError(msg)
    area_val = float(raw_area)
    if not math.isfinite(area_val) or area_val <= 0.0:
        msg = (
            f'Catchment {basin_id!r} has invalid area_km2={area_val!r};'
            ' area_km2 must be a finite positive number.'
        )
        raise ValueError(msg)
    return round(area_val, 2)


def _compute_domain_coverage_fraction(
    geom: BaseGeometry,
    lats: np.ndarray,
    lons: np.ndarray,
    dlat: float,
    dlon: float,
) -> float:
    """Compute the fraction of ``geom``'s planar degree area inside the grid."""
    if geom.is_empty or geom.area <= 0.0 or len(lats) == 0 or len(lons) == 0:
        return 0.0
    half_lat = dlat / 2.0
    half_lon = dlon / 2.0
    grid_min_lat = float(np.min(lats)) - half_lat
    grid_max_lat = float(np.max(lats)) + half_lat
    grid_min_lon = float(np.min(lons)) - half_lon
    grid_max_lon = float(np.max(lons)) + half_lon
    grid_box = shapely.geometry.box(
        grid_min_lon, grid_min_lat, grid_max_lon, grid_max_lat
    )
    if grid_box.contains(geom):
        return 1.0
    if not grid_box.intersects(geom):
        return 0.0
    return float(
        np.clip(geom.intersection(grid_box).area / geom.area, 0.0, 1.0)
    )


class HistoricalZarrExtractor:
    """Extracts historical weather records for catchments from Zarr stores.

    Uses ``xarray.open_zarr`` and :data:`ZonalEngine`
    (:class:`multimet.utils.zonal.ZonalWeightMatrix`) to perform exact
    cosine-latitude-weighted spatial reductions with the >= 80% valid spatial
    coverage rule, and appends the extracted basin datasets into a unified
    multi-basin Zarr archive.
    """

    def __init__(
        self,
        output_dir: Path | str | None = None,
        archive_uris: Mapping[str, str | Path] | None = None,
    ) -> None:
        """Initialize the historical Zarr extractor.

        Args:
            output_dir: Directory where ``historical_training_master.zarr`` is
              stored. Defaults to ``HISTORICAL_DIR``.
            archive_uris: Optional mapping overriding source IDs (e.g.
              ``'era5'``, ``'cpc'``, ``'imerg'``) to local or cloud Zarr paths.
        """
        self.output_dir = Path(output_dir) if output_dir else HISTORICAL_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.master_zarr_path = self.output_dir / HISTORICAL_MASTER_ZARR_NAME
        self.archive_uris: dict[str, str] = {
            str(k): str(v) for k, v in (archive_uris or {}).items()
        }

    def extract_and_archive(  # noqa: PLR0913, PLR0915, PLR0917
        self,
        polygon_input: dict[str, Any] | BaseGeometry,
        basin_id: str | None = None,
        properties: dict[str, Any] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        weather_source: str = 'era5',
        freq: str = '1D',
        variables: list[str] | None = None,
        archive_uri: str | Path | None = None,
    ) -> dict[str, Any]:
        """Extract historical weather data for a polygon and append to Zarr.

        Args:
            polygon_input: GeoJSON Feature, Geometry dict, or Shapely geometry.
            basin_id: Optional explicit basin identifier.
            properties: Optional metadata properties dictionary (must contain
              positive ``area_km2`` here or inside ``polygon_input``).
            start_date: Start date string (``YYYY-MM-DD``).
            end_date: End date string (``YYYY-MM-DD``).
            weather_source: Registered dataset key (``'cpc'``, ``'imerg'``,
              ``'era5'``, ``'ifs'``, ``'graphcast'``, etc.).
            freq: Temporal frequency label (default ``'1D'``).
            variables: Optional list of variables to extract.
            archive_uri: Optional explicit Zarr store URI or path overriding the
              source's default ``zarr_path``.

        Returns:
            Dictionary containing master archive path, basin list, sizes, and
            preview timeseries.

        Raises:
            FileNotFoundError: If the target Zarr store does not exist.
            KeyError: If ``weather_source`` or a requested variable is unknown.
            ValueError: If ``area_km2`` is missing/non-positive or no timesteps
              match the requested date range.
        """
        geom, props, resolved_basin_id = self._parse_polygon_input(
            polygon_input, basin_id, properties
        )
        area_km2 = _validate_positive_area_km2(props, resolved_basin_id)

        centroid = geom.centroid
        outlet_info = props.get('outlet')
        if isinstance(outlet_info, Mapping):
            centroid_lat = float(outlet_info.get('latitude', centroid.y))
            centroid_lon = float(outlet_info.get('longitude', centroid.x))
        else:
            centroid_lat = float(centroid.y)
            centroid_lon = float(centroid.x)

        source_obj = get_weather_source(weather_source)
        resolved_uri = str(
            archive_uri
            or self.archive_uris.get(weather_source)
            or source_obj.zarr_path
            or ''
        )
        if not resolved_uri:
            msg = (
                f'No Zarr archive URI configured for weather source'
                f' {weather_source!r}.'
            )
            raise FileNotFoundError(msg)

        requested_vars = list(variables or source_obj.default_variables)
        if not requested_vars:
            msg = (
                f'No variables specified for weather source {weather_source!r}.'
            )
            raise ValueError(msg)

        extracted_arrays, time_index, lead_coords = (
            self._extract_polygon_from_zarr(
                geom=geom,
                basin_id=resolved_basin_id,
                weather_source=weather_source,
                archive_uri=resolved_uri,
                requested_vars=requested_vars,
                start_date=start_date,
                end_date=end_date,
            )
        )

        resolved_start_date = time_index[0].strftime('%Y-%m-%d')
        resolved_end_date = time_index[-1].strftime('%Y-%m-%d')
        n_times = len(time_index)
        canonical_vars = list(extracted_arrays.keys())

        data_vars: dict[str, Any] = {}
        for var_name, values_arr in extracted_arrays.items():
            units = _canonical_units_for_variable(var_name)
            if values_arr.ndim == _FORECAST_ARRAY_NDIM:
                # 3D forecast variable: (basin, date, lead_time)
                val_3d = np.expand_dims(values_arr.astype(np.float32), axis=0)
                dims = ['basin', 'date', 'lead_time']
                data_vars[var_name] = (
                    dims,
                    val_3d,
                    {
                        'units': units,
                        'long_name': var_name.replace('_', ' ').title(),
                        'description': (
                            f'Daily catchment-averaged {var_name} from'
                            f' {source_obj.name}'
                        ),
                        '_ARRAY_DIMENSIONS': dims,
                    },
                )
            else:
                # 2D nowcast/historical variable: (basin, date)
                val_2d = np.expand_dims(values_arr.astype(np.float32), axis=0)
                dims = ['basin', 'date']
                data_vars[var_name] = (
                    dims,
                    val_2d,
                    {
                        'units': units,
                        'long_name': var_name.replace('_', ' ').title(),
                        'description': (
                            f'Daily catchment-averaged {var_name} from'
                            f' {source_obj.name}'
                        ),
                        '_ARRAY_DIMENSIONS': dims,
                    },
                )

        data_vars['basin_area'] = (
            ['basin'],
            np.array([area_km2], dtype=np.float32),
            {
                'units': 'km2',
                'long_name': 'Catchment Contributing Drainage Area',
                '_ARRAY_DIMENSIONS': ['basin'],
            },
        )
        data_vars['area_km2'] = (
            ['basin'],
            np.array([area_km2], dtype=np.float32),
            {
                'units': 'km2',
                'long_name': 'Catchment Contributing Drainage Area',
                '_ARRAY_DIMENSIONS': ['basin'],
            },
        )
        data_vars['latitude'] = (
            ['basin'],
            np.array([round(centroid_lat, 4)], dtype=np.float32),
            {
                'units': 'degrees_north',
                'long_name': 'Catchment Outlet / Centroid Latitude',
                '_ARRAY_DIMENSIONS': ['basin'],
            },
        )
        data_vars['longitude'] = (
            ['basin'],
            np.array([round(centroid_lon, 4)], dtype=np.float32),
            {
                'units': 'degrees_east',
                'long_name': 'Catchment Outlet / Centroid Longitude',
                '_ARRAY_DIMENSIONS': ['basin'],
            },
        )

        coords: dict[str, Any] = {
            'basin': [resolved_basin_id],
            'basin_id': ('basin', [resolved_basin_id]),
            'date': time_index,
            'issue_time': ('date', time_index),
        }
        if lead_coords is not None:
            coords['lead_time'] = lead_coords

        new_basin_ds = xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs={
                'title': 'OpenHydroNet Historical Weather Archive',
                'institution': 'Google Research',
                'repository': GITHUB_REPO_URL,
                'source_dataset': source_obj.name,
                'source_id': weather_source,
                'temporal_resolution': freq,
                'spatial_aggregation': 'area_weighted_catchment_mean',
                'min_valid_coverage_fraction': MIN_VALID_COVERAGE_FRACTION,
                'frozen_archive': True,
                'last_updated': datetime.now(UTC).isoformat(),
            },
        )
        new_basin_ds.date.attrs['long_name'] = 'Date'
        if 'lead_time' in new_basin_ds.coords:
            new_basin_ds.lead_time.attrs['units'] = 'days'
            new_basin_ds.lead_time.attrs['long_name'] = 'Forecast Lead Time'

        combined_ds = self._append_to_master_store(
            new_basin_ds, resolved_basin_id
        )

        master_size_bytes = sum(
            f.stat().st_size
            for f in self.master_zarr_path.rglob('*')
            if f.is_file()
        )
        master_size_mb = round(master_size_bytes / (1024 * 1024), 2)

        preview_steps = min(60, n_times)
        first_var = canonical_vars[0]
        sec_var = canonical_vars[1] if len(canonical_vars) > 1 else first_var

        def _preview_slice(arr: np.ndarray) -> list[float]:
            flat_1d = arr[:, 0] if arr.ndim == _FORECAST_ARRAY_NDIM else arr
            return [
                round(float(v), 3) if np.isfinite(v) else float('nan')
                for v in flat_1d[:preview_steps]
            ]

        preview = {
            'timestamps': [
                t.strftime('%Y-%m-%d') for t in time_index[:preview_steps]
            ],
            'primary_var_name': first_var,
            'primary_var_values': _preview_slice(extracted_arrays[first_var]),
            'secondary_var_name': sec_var,
            'secondary_var_values': _preview_slice(extracted_arrays[sec_var]),
        }

        all_basins = [str(b) for b in combined_ds.basin.to_numpy()]
        lead_steps = (
            [int(lt) for lt in combined_ds.lead_time.to_numpy()]
            if 'lead_time' in combined_ds.coords
            else []
        )

        return {
            'status': 'success',
            'basin_id': resolved_basin_id,
            'catchment_id': resolved_basin_id,
            'weather_source': weather_source,
            'weather_source_name': source_obj.name,
            'start_date': resolved_start_date,
            'end_date': resolved_end_date,
            'master_zarr_path': str(self.master_zarr_path),
            'master_zarr_rel_path': str(
                self.master_zarr_path.relative_to(self.output_dir.parent)
            ),
            'single_zarr_path': str(self.master_zarr_path),
            'zarr_path': str(self.master_zarr_path),
            'zarr_rel_path': str(
                self.master_zarr_path.relative_to(self.output_dir.parent)
            ),
            'master_size_mb': master_size_mb,
            'size_mb': master_size_mb,
            'total_basins_in_master': len(all_basins),
            'all_basins': all_basins,
            'n_timesteps': n_times,
            'dimensions': list(combined_ds.dims),
            'dims': list(combined_ds.dims),
            'lead_time_steps': lead_steps,
            'variables': canonical_vars,
            'preview': preview,
        }

    def _extract_polygon_from_zarr(  # noqa: C901, PLR0912, PLR0913, PLR0915, PLR0917
        self,
        geom: BaseGeometry,
        basin_id: str,
        weather_source: str,
        archive_uri: str,
        requested_vars: list[str],
        start_date: str | None,
        end_date: str | None,
    ) -> tuple[dict[str, np.ndarray], pd.DatetimeIndex, np.ndarray | None]:
        """Open the gridded Zarr store and perform exact zonal reduction."""
        is_consolidated = _verify_zarr_store_exists(archive_uri)
        ds = xr.open_zarr(
            archive_uri,
            consolidated=is_consolidated,
            decode_timedelta=False,
        )

        time_dim = 'time' if 'time' in ds.dims else 'date'
        if time_dim not in ds.dims:
            msg = (
                f'Zarr store {archive_uri!r} is missing a time/date dimension;'
                f' found dims {list(ds.dims)}.'
            )
            raise ValueError(msg)

        lat_dim = 'latitude' if 'latitude' in ds.dims else 'lat'
        lon_dim = 'longitude' if 'longitude' in ds.dims else 'lon'
        if lat_dim not in ds.dims or lon_dim not in ds.dims:
            msg = (
                f'Zarr store {archive_uri!r} is missing latitude/longitude'
                f' dimensions; found dims {list(ds.dims)}.'
            )
            raise ValueError(msg)

        time_vals_all = pd.DatetimeIndex(
            pd.to_datetime(ds[time_dim].to_numpy())
        )
        if len(time_vals_all) == 0:
            msg = f'Zarr store {archive_uri!r} has an empty {time_dim!r} axis.'
            raise ValueError(msg)

        resolved_start = start_date or time_vals_all[0].strftime('%Y-%m-%d')
        resolved_end = end_date or time_vals_all[-1].strftime('%Y-%m-%d')
        start_ts = pd.Timestamp(resolved_start)
        end_ts = pd.Timestamp(resolved_end)
        if end_ts < start_ts:
            msg = (
                f'end_date ({resolved_end}) precedes start_date'
                f' ({resolved_start}).'
            )
            raise ValueError(msg)

        time_mask = (time_vals_all >= start_ts) & (time_vals_all <= end_ts)
        if not np.any(time_mask):
            msg = (
                f'No timesteps in {archive_uri!r} fall within'
                f' [{resolved_start}, {resolved_end}].'
            )
            raise ValueError(msg)

        ds_time = ds.isel({time_dim: np.where(time_mask)[0]})
        time_index = pd.DatetimeIndex(
            pd.to_datetime(ds_time[time_dim].to_numpy())
        )

        full_lats = np.asarray(ds_time[lat_dim].to_numpy(), dtype=np.float64)
        full_lons = np.asarray(ds_time[lon_dim].to_numpy(), dtype=np.float64)
        dlat = (
            abs(float(full_lats[1] - full_lats[0]))
            if len(full_lats) > 1
            else 0.1
        )
        dlon = (
            abs(float(full_lons[1] - full_lons[0]))
            if len(full_lons) > 1
            else 0.1
        )

        domain_cov = _compute_domain_coverage_fraction(
            geom, full_lats, full_lons, dlat, dlon
        )

        buffer_deg = max(dlat, dlon) * 2.0
        sub_lats, sub_lons, lat_slice, lon_slice = slice_coordinates_by_bounds(
            full_lats, full_lons, geom.bounds, buffer_degrees=buffer_deg
        )

        ds_sub = ds_time.isel({lat_dim: lat_slice, lon_dim: lon_slice})
        basins_gdf = gpd.GeoDataFrame(
            {'geometry': [geom]},
            index=pd.Index([basin_id], name='basin_id'),
            crs='EPSG:4326',
        )

        has_spatial_overlap = (
            domain_cov >= (MIN_VALID_COVERAGE_FRACTION - 1e-6)
            and len(sub_lats) > 0
            and len(sub_lons) > 0
        )
        zonal_matrix: ZonalWeightMatrix | None = None
        if has_spatial_overlap:
            zonal_matrix = ZonalEngine.from_geodataframe(
                basins_gdf=basins_gdf,
                lats=sub_lats,
                lons=sub_lons,
                cell_res_lat=dlat,
                cell_res_lon=dlon,
            )

        alias_map = _SOURCE_VARIABLE_ALIASES.get(weather_source, {})
        extracted: dict[str, np.ndarray] = {}
        lead_coords: np.ndarray | None = None

        for req_var in requested_vars:
            if req_var in alias_map:
                canonical_name, candidates = alias_map[req_var]
            else:
                canonical_name, candidates = req_var, (req_var,)

            matched_source_var: str | None = None
            for cand in candidates:
                if cand in ds_sub.data_vars:
                    matched_source_var = cand
                    break

            if matched_source_var is None:
                msg = (
                    f'Variable {req_var!r} (candidates {candidates}) not found'
                    f' in Zarr archive {archive_uri!r}; available variables:'
                    f' {sorted(ds_sub.data_vars)}.'
                )
                raise KeyError(msg)

            da = ds_sub[matched_source_var]
            if 'lead_time' in da.dims:
                da_ordered = da.transpose(
                    time_dim, 'lead_time', lat_dim, lon_dim
                )
                n_leads = da_ordered.sizes['lead_time']
                lead_coords = np.arange(1, n_leads + 1, dtype=np.int64)
                if not has_spatial_overlap or zonal_matrix is None:
                    extracted[canonical_name] = np.full(
                        (len(time_index), n_leads), np.nan, dtype=np.float32
                    )
                else:
                    grid_4d = np.asarray(
                        da_ordered.to_numpy(), dtype=np.float64
                    )
                    red_3d, _ = zonal_matrix.reduce_4d_with_coverage(grid_4d)
                    extracted[canonical_name] = red_3d[0]
            else:
                da_ordered = da.transpose(time_dim, lat_dim, lon_dim)
                if not has_spatial_overlap or zonal_matrix is None:
                    extracted[canonical_name] = np.full(
                        (len(time_index),), np.nan, dtype=np.float32
                    )
                else:
                    grid_3d = np.asarray(
                        da_ordered.to_numpy(), dtype=np.float64
                    )
                    red_2d, _ = zonal_matrix.reduce_3d_with_coverage(grid_3d)
                    extracted[canonical_name] = red_2d[0]

        return extracted, time_index, lead_coords

    def _append_to_master_store(  # noqa: C901, PLR0912
        self, new_ds: xr.Dataset, basin_id: str
    ) -> xr.Dataset:
        """Merge or append a new basin dataset into the master Zarr store."""
        if self.master_zarr_path.exists():
            is_consolidated = (self.master_zarr_path / '.zmetadata').is_file()
            existing_ds = xr.open_zarr(
                str(self.master_zarr_path),
                consolidated=is_consolidated,
                decode_timedelta=False,
            ).load()
            if 'time' in existing_ds.dims and 'date' not in existing_ds.dims:
                existing_ds = existing_ds.rename({'time': 'date'})

            existing_basins = [str(b) for b in existing_ds.basin.to_numpy()]
            other_basins = [b for b in existing_basins if b != str(basin_id)]
            if other_basins:
                existing_subset = existing_ds.sel(basin=other_basins)
                all_dates = pd.DatetimeIndex(
                    sorted(
                        set(existing_subset.date.to_numpy()).union(
                            set(new_ds.date.to_numpy())
                        )
                    )
                )
                ex_reindexed = existing_subset.reindex(date=all_dates)
                new_reindexed = new_ds.reindex(date=all_dates)
                combined = xr.concat(
                    [ex_reindexed, new_reindexed], dim='basin', join='outer'
                )
            else:
                combined = new_ds
        else:
            combined = new_ds

        combined.encoding.clear()
        for var_key in list(combined.variables):
            combined[var_key].encoding.clear()

        chunk_spec: dict[str, int] = {'basin': 1}
        if 'date' in combined.dims:
            chunk_spec['date'] = len(combined.date)
        if 'lead_time' in combined.dims:
            chunk_spec['lead_time'] = len(combined.lead_time)

        combined_chunked = combined.chunk(chunk_spec)

        encoding: dict[str, dict[str, Any]] = {}
        for var_name in combined_chunked.data_vars:
            dims = combined_chunked[var_name].dims
            if dims == ('basin', 'date', 'lead_time'):
                encoding[var_name] = {
                    'chunks': (
                        1,
                        len(combined_chunked.date),
                        len(combined_chunked.lead_time),
                    )
                }
            elif dims == ('basin', 'date'):
                encoding[var_name] = {'chunks': (1, len(combined_chunked.date))}
            elif dims == ('basin',):
                encoding[var_name] = {'chunks': (1,)}

        for coord_name in ('basin_id', 'area_km2', 'latitude', 'longitude'):
            if coord_name in combined_chunked.coords and combined_chunked[
                coord_name
            ].dims == ('basin',):
                encoding[coord_name] = {'chunks': (1,)}

        if self.master_zarr_path.exists():
            shutil.rmtree(self.master_zarr_path)

        combined_chunked.to_zarr(
            store=str(self.master_zarr_path),
            mode='w',
            consolidated=True,
            encoding=encoding or None,
        )
        return combined_chunked

    def extract_and_archive_batch(  # noqa: PLR0913, PLR0917
        self,
        features: list[dict[str, Any]],
        start_date: str | None = None,
        end_date: str | None = None,
        weather_source: str = 'era5',
        freq: str = '1D',
        variables: list[str] | None = None,
        archive_uri: str | Path | None = None,
    ) -> dict[str, Any]:
        """Extract historical weather data for a batch of basin features."""
        if not features:
            msg = 'No catchment features provided for historical batch.'
            raise ValueError(msg)

        results: list[dict[str, Any]] = []
        for feat in features:
            props = feat.get('properties', {})
            cid = (
                props.get('catchment_id')
                or props.get('gauge_id')
                or props.get('basin_id')
                or props.get('id')
            )
            res = self.extract_and_archive(
                polygon_input=feat,
                basin_id=str(cid) if cid is not None else None,
                properties=props,
                start_date=start_date,
                end_date=end_date,
                weather_source=weather_source,
                freq=freq,
                variables=variables,
                archive_uri=archive_uri,
            )
            results.append(res)

        master_size_bytes = sum(
            f.stat().st_size
            for f in self.master_zarr_path.rglob('*')
            if f.is_file()
        )
        master_size_mb = round(master_size_bytes / (1024 * 1024), 2)

        is_consolidated = (self.master_zarr_path / '.zmetadata').is_file()
        existing_ds = xr.open_zarr(
            str(self.master_zarr_path),
            consolidated=is_consolidated,
            decode_timedelta=False,
        )
        all_basins = [str(b) for b in existing_ds.basin.to_numpy()]
        last_res = results[-1]

        return {
            'status': 'success',
            'batch': True,
            'basins_extracted_count': len(results),
            'extracted_basins': [r['basin_id'] for r in results],
            'weather_source': weather_source,
            'master_zarr_path': str(self.master_zarr_path),
            'master_zarr_rel_path': str(
                self.master_zarr_path.relative_to(self.output_dir.parent)
            ),
            'zarr_path': str(self.master_zarr_path),
            'zarr_rel_path': str(
                self.master_zarr_path.relative_to(self.output_dir.parent)
            ),
            'master_size_mb': master_size_mb,
            'size_mb': master_size_mb,
            'total_basins_in_master': len(all_basins),
            'all_basins': all_basins,
            'n_timesteps': last_res.get('n_timesteps', 0),
            'dimensions': list(existing_ds.dims),
            'dims': list(existing_ds.dims),
            'variables': last_res.get('variables', []),
            'preview': last_res.get('preview', {}),
        }

    def _parse_polygon_input(
        self,
        polygon_input: dict[str, Any] | BaseGeometry,
        basin_id: str | None = None,
        properties: dict[str, Any] | None = None,
    ) -> tuple[BaseGeometry, dict[str, Any], str]:
        """Normalize polygon input and resolve a deterministic basin ID."""
        props: dict[str, Any] = dict(properties or {})

        if isinstance(polygon_input, dict):
            if polygon_input.get('type') == 'Feature':
                geom = shape(polygon_input['geometry'])
                props = {**polygon_input.get('properties', {}), **props}
            elif 'type' in polygon_input and (
                'coordinates' in polygon_input
                or polygon_input.get('type') in {'Polygon', 'MultiPolygon'}
            ):
                geom = shape(polygon_input)
            elif 'geometry' in polygon_input:
                geom = shape(polygon_input['geometry'])
                props = {**polygon_input.get('properties', {}), **props}
            else:
                geom = shape(polygon_input)
        elif isinstance(polygon_input, BaseGeometry):
            geom = polygon_input
        else:
            msg = f'Unsupported polygon input type: {type(polygon_input)}'
            raise TypeError(msg)

        if not isinstance(geom, (Polygon, MultiPolygon)):
            msg = (
                f'Geometry must be Polygon or MultiPolygon, got'
                f' {geom.geom_type}'
            )
            raise TypeError(msg)

        resolved_id = (
            basin_id
            or props.get('catchment_id')
            or props.get('basin_id')
            or props.get('gauge_id')
            or props.get('id')
            or props.get('name')
        )
        if not resolved_id:
            cent = geom.centroid
            wkt_hash = hashlib.sha256(geom.wkt.encode('utf-8')).hexdigest()[:8]
            resolved_id = f'custom_basin_{cent.y:.3f}_{cent.x:.3f}_{wkt_hash}'

        return geom, props, str(resolved_id)
