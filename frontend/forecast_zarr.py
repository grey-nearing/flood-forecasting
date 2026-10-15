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
"""Real-time weather forecast extractor for OpenHydroNet ML inference."""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import math
import shutil
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import fsspec
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry
import xarray as xr
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from frontend.config import FORECAST_DIR
from frontend.notifier import send_email_notification
from multimet.utils.spatial import slice_coordinates_by_bounds
from multimet.utils.zonal import (
    MIN_VALID_COVERAGE_FRACTION,
    ZonalWeightMatrix,
)
from utils.file_paths import (
    DEFAULT_LOCAL_SERVER_URL,
    FORECAST_LATEST_ZARR_NAME,
    GITHUB_FLOODHUB_CONFIG_URL,
    GITHUB_REPO_URL,
)

# Alias ZonalEngine to ZonalWeightMatrix for canonical zonal engine access.
ZonalEngine = ZonalWeightMatrix

logger = logging.getLogger(__name__)

HAS_DYNAMICAL_CATALOG = (
    importlib.util.find_spec('dynamical_catalog') is not None
)

_HOURS_PER_DAY: float = 24.0

# Feature definitions matching floodhub-settings-config.yml (forecast_inputs)
FORECAST_FEATURE_SPECS: dict[str, dict[str, str]] = {
    'hres_surface_net_solar_radiation': {
        'source_var': 'surface_net_solar_radiation',
        'model': 'hres',
        'units': 'W/m^2',
        'standard_name': 'surface_net_downward_shortwave_flux',
        'long_name': 'ECMWF HRES Daily Mean Surface Net Solar Radiation',
        'description': (
            'Daily mean surface net solar radiation flux from ECMWF HRES'
        ),
        'agg': 'mean',
    },
    'hres_surface_net_thermal_radiation': {
        'source_var': 'surface_net_thermal_radiation',
        'model': 'hres',
        'units': 'W/m^2',
        'standard_name': 'surface_net_Thermal_radiation_flux',
        'long_name': 'ECMWF HRES Daily Mean Surface Net Thermal Radiation',
        'description': (
            'Daily mean surface net thermal radiation flux from ECMWF HRES'
        ),
        'agg': 'mean',
    },
    'hres_surface_pressure': {
        'source_var': 'surface_pressure',
        'model': 'hres',
        'units': 'kPa',
        'standard_name': 'surface_air_pressure',
        'long_name': 'ECMWF HRES Daily Mean Surface Pressure',
        'description': (
            'Daily mean surface atmospheric pressure in kPa from ECMWF HRES'
        ),
        'agg': 'mean',
    },
    'hres_temperature_2m': {
        'source_var': 'temperature_2m',
        'model': 'hres',
        'units': 'degC',
        'standard_name': 'air_temperature',
        'long_name': 'ECMWF HRES Daily Mean 2m Temperature',
        'description': 'Daily mean 2m air temperature from ECMWF HRES',
        'agg': 'mean',
    },
    'hres_total_precipitation': {
        'source_var': 'total_precipitation',
        'model': 'hres',
        'units': 'mm/day',
        'standard_name': 'precipitation_flux',
        'long_name': 'ECMWF HRES Daily Total Precipitation',
        'description': 'Daily accumulated total precipitation from ECMWF HRES',
        'agg': 'sum_rate',
    },
    'graphcast_temperature_2m': {
        'source_var': 'temperature_2m',
        'model': 'graphcast',
        'units': 'degC',
        'standard_name': 'air_temperature',
        'long_name': 'GraphCast Daily Mean 2m Temperature',
        'description': 'Daily mean 2m air temperature from GraphCast',
        'agg': 'mean',
    },
    'graphcast_total_precipitation': {
        'source_var': 'total_precipitation',
        'model': 'graphcast',
        'units': 'mm/day',
        'standard_name': 'precipitation_flux',
        'long_name': 'GraphCast Daily Total Precipitation',
        'description': 'Daily accumulated total precipitation from GraphCast',
        'agg': 'sum_rate',
    },
}

# dynamical.org publishes neither real-time ECMWF IFS HRES nor GraphCast
# forecasts, so the operational stand-ins for the ``hres_*`` and
# ``graphcast_*`` model inputs are the IFS ENS control member
# (``ensemble_member=0``, selected in ``_resolve_init_time_and_slice``) and the
# deterministic AIFS run. These are exact ``dynamical_catalog.list()``
# identifiers; ``dynamical_catalog.open`` raises ``UnknownDatasetError`` for
# any other string, which is what broke ``GET /api/weather/forecast/info``.
DYNAMICAL_HRES_DATASET_ID = 'ecmwf-ifs-ens-forecast-15-day-0-25-degree'
DYNAMICAL_GRAPHCAST_DATASET_ID = 'ecmwf-aifs-single-forecast'
DYNAMICAL_DATASET_IDS: dict[str, str] = {
    'hres': DYNAMICAL_HRES_DATASET_ID,
    'graphcast': DYNAMICAL_GRAPHCAST_DATASET_ID,
}

# In-memory session cache for forecast datasets
_DYNAMICAL_CACHE: dict[str, xr.Dataset] = {}
_LAST_CACHE_TIME: float = 0.0
_CACHE_TTL_SECONDS: float = 600.0

ForecastSource = xr.Dataset | str | Path


def _describe_forecast_source(source: ForecastSource) -> str:
    """Return a human-readable label for an explicit forecast source."""
    if isinstance(source, xr.Dataset):
        return 'in-memory xarray.Dataset'
    return str(source)


def _open_zarr_or_dataset(
    source: ForecastSource,
) -> xr.Dataset:
    """Open a Zarr path or return an already-opened ``xarray.Dataset``."""
    if isinstance(source, xr.Dataset):
        return source
    uri = str(source)
    if uri.startswith(('gs://', 's3://', 'http://', 'https://')):
        fs, fs_path = fsspec.core.url_to_fs(uri)
        clean_path = fs_path.rstrip('/')
        if fs.exists(f'{clean_path}/.zmetadata'):
            is_consolidated = True
        elif fs.exists(f'{clean_path}/.zgroup') or fs.exists(
            f'{clean_path}/zarr.json'
        ):
            is_consolidated = False
        else:
            msg = f'Forecast Zarr store not found at {uri!r}.'
            raise FileNotFoundError(msg)
        return xr.open_zarr(
            uri, consolidated=is_consolidated, decode_timedelta=False
        )

    local_path = Path(uri)
    if not local_path.exists():
        msg = f'Local forecast Zarr store not found at {uri!r}.'
        raise FileNotFoundError(msg)
    if (local_path / '.zmetadata').is_file():
        is_consolidated = True
    elif (local_path / '.zgroup').is_file() or (
        local_path / 'zarr.json'
    ).is_file():
        is_consolidated = False
    else:
        msg = f'Path {uri!r} is not a valid Zarr store.'
        raise FileNotFoundError(msg)
    return xr.open_zarr(
        str(local_path), consolidated=is_consolidated, decode_timedelta=False
    )


def _get_dynamical_datasets() -> tuple[xr.Dataset, xr.Dataset]:
    """Retrieve open xarray datasets for the HRES and GraphCast stand-ins.

    Opens ``DYNAMICAL_HRES_DATASET_ID`` and ``DYNAMICAL_GRAPHCAST_DATASET_ID``
    from the dynamical.org catalog after verifying that both identifiers are
    actually published, so a renamed or retired upstream dataset fails with
    the list of available datasets instead of a bare ``UnknownDatasetError``.

    Raises:
        RuntimeError: If ``dynamical_catalog`` is not installed, or if a
          configured dataset ID is not published in the dynamical.org catalog.
    """
    global _LAST_CACHE_TIME  # noqa: PLW0603

    now = time.time()
    if (
        'hres' in _DYNAMICAL_CACHE
        and 'graphcast' in _DYNAMICAL_CACHE
        and (now - _LAST_CACHE_TIME) < _CACHE_TTL_SECONDS
    ):
        return _DYNAMICAL_CACHE['hres'], _DYNAMICAL_CACHE['graphcast']

    if not HAS_DYNAMICAL_CATALOG:
        msg = (
            'dynamical_catalog is not installed and no explicit forecast'
            ' datasets were provided; refusing to fabricate synthetic'
            ' forecasts.'
        )
        raise RuntimeError(msg)

    dynamical_catalog = importlib.import_module('dynamical_catalog')
    available_ids = set(dynamical_catalog.list())
    missing_ids = [
        dataset_id
        for dataset_id in DYNAMICAL_DATASET_IDS.values()
        if dataset_id not in available_ids
    ]
    if missing_ids:
        msg = (
            f'dynamical.org catalog does not publish {missing_ids};'
            f' available datasets: {sorted(available_ids)}.'
        )
        raise RuntimeError(msg)

    ds_hres = dynamical_catalog.open(DYNAMICAL_HRES_DATASET_ID, chunks=None)
    ds_graphcast = dynamical_catalog.open(
        DYNAMICAL_GRAPHCAST_DATASET_ID, chunks=None
    )
    _DYNAMICAL_CACHE['hres'] = ds_hres
    _DYNAMICAL_CACHE['graphcast'] = ds_graphcast
    _LAST_CACHE_TIME = now
    return ds_hres, ds_graphcast


def _validate_positive_area_km2(
    props: Mapping[str, Any], basin_id: str
) -> float:
    """Validate and return a finite positive catchment area in km^2."""
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


def _resolve_init_time_and_slice(
    ds: xr.Dataset,
) -> tuple[xr.Dataset, pd.Timestamp]:
    """Select the latest initialization time slice from a forecast dataset."""
    for init_dim in ('init_time', 'date', 'time', 'issue_time'):
        if init_dim in ds.dims:
            if ds.sizes[init_dim] == 0:
                msg = f'Forecast dataset has empty {init_dim!r} dimension.'
                raise ValueError(msg)
            latest_val = ds[init_dim].to_numpy()[-1]
            sliced = ds.isel({init_dim: -1})
            if 'ensemble_member' in sliced.dims:
                sliced = sliced.isel(ensemble_member=0)
            return sliced, pd.Timestamp(pd.to_datetime(latest_val))
        if init_dim in ds.coords:
            coord_vals = np.atleast_1d(ds.coords[init_dim].to_numpy())
            if len(coord_vals) > 0:
                sliced = (
                    ds.isel(ensemble_member=0)
                    if 'ensemble_member' in ds.dims
                    else ds
                )
                return sliced, pd.Timestamp(pd.to_datetime(coord_vals[-1]))

    msg = (
        'Forecast dataset is missing an initialization time coordinate'
        f' (expected init_time, date, or time; found {list(ds.coords)}).'
    )
    raise ValueError(msg)


def _resolve_lead_hours(ds_latest: xr.Dataset) -> np.ndarray:
    """Extract forecast lead times in hours from ``lead_time`` coordinate."""
    if 'lead_time' not in ds_latest.dims:
        msg = (
            'Forecast dataset is missing required lead_time dimension;'
            f' found dims {list(ds_latest.dims)}.'
        )
        raise ValueError(msg)

    lead_coord = ds_latest['lead_time']
    lead_vals = lead_coord.to_numpy()
    if len(lead_vals) == 0:
        msg = 'Forecast dataset has an empty lead_time axis.'
        raise ValueError(msg)

    if np.issubdtype(lead_vals.dtype, np.timedelta64):
        return np.asarray(
            pd.to_timedelta(lead_vals).total_seconds() / 3600.0,
            dtype=np.float64,
        )

    units = str(lead_coord.attrs.get('units', '')).lower()
    numeric_vals = np.asarray(lead_vals, dtype=np.float64)
    if units.startswith('hour') or units in {'h', 'hr', 'hrs'}:
        return numeric_vals
    if units.startswith('day') or units == 'd':
        return numeric_vals * _HOURS_PER_DAY
    # Default convention: 1-indexed integer lead days (1..15) vs hours (>24)
    if float(np.nanmax(numeric_vals)) <= _HOURS_PER_DAY and np.all(
        np.mod(numeric_vals, 1.0) == 0.0
    ):
        return numeric_vals * _HOURS_PER_DAY
    return numeric_vals


def _select_variable_dataarray(
    ds_sub: xr.Dataset,
    feature_name: str,
) -> tuple[xr.DataArray, str]:
    """Resolve a canonical forecast variable DataArray from a model dataset.

    Never maps downward radiation alone into ``surface_net_*_radiation``.
    """
    candidates_map: dict[str, tuple[str, ...]] = {
        'hres_surface_net_solar_radiation': (
            'hres_surface_net_solar_radiation',
            'surface_net_solar_radiation',
            'ssr',
        ),
        'hres_surface_net_thermal_radiation': (
            'hres_surface_net_thermal_radiation',
            'surface_net_thermal_radiation',
            'str',
        ),
        'hres_surface_pressure': (
            'hres_surface_pressure',
            'surface_pressure',
            'pressure_surface',
            'sp',
        ),
        'hres_temperature_2m': (
            'hres_temperature_2m',
            'temperature_2m',
            '2m_temperature',
            't2m',
        ),
        'hres_total_precipitation': (
            'hres_total_precipitation',
            'total_precipitation',
            'precipitation_surface',
            'tp',
        ),
        'graphcast_temperature_2m': (
            'graphcast_temperature_2m',
            'temperature_2m',
            '2m_temperature',
            't2m',
        ),
        'graphcast_total_precipitation': (
            'graphcast_total_precipitation',
            'total_precipitation',
            'precipitation_surface',
            'tp',
        ),
    }
    candidates = candidates_map.get(feature_name, (feature_name,))
    for cand in candidates:
        if cand in ds_sub.data_vars:
            return ds_sub[cand], cand

    # Check if net radiation can be computed from explicit downward - upward
    if feature_name == 'hres_surface_net_solar_radiation' and (
        'downward_short_wave_radiation_flux_surface' in ds_sub.data_vars
        and 'upward_short_wave_radiation_flux_surface' in ds_sub.data_vars
    ):
        net_da = (
            ds_sub['downward_short_wave_radiation_flux_surface']
            - ds_sub['upward_short_wave_radiation_flux_surface']
        )
        net_da.attrs['units'] = 'W/m^2'
        return net_da, 'computed_net_solar_radiation'

    if feature_name == 'hres_surface_net_thermal_radiation' and (
        'downward_long_wave_radiation_flux_surface' in ds_sub.data_vars
        and 'upward_long_wave_radiation_flux_surface' in ds_sub.data_vars
    ):
        net_da = (
            ds_sub['downward_long_wave_radiation_flux_surface']
            - ds_sub['upward_long_wave_radiation_flux_surface']
        )
        net_da.attrs['units'] = 'W/m^2'
        return net_da, 'computed_net_thermal_radiation'

    msg = (
        f'Required forecast variable for {feature_name!r} (candidates'
        f' {candidates}) not found in dataset; available variables:'
        f' {sorted(ds_sub.data_vars)}.'
    )
    raise KeyError(msg)


def _extract_model_features_for_basins(  # noqa: C901, PLR0912, PLR0915
    ds: xr.Dataset,
    basins_gdf: gpd.GeoDataFrame,
    geoms: list[BaseGeometry],
    feature_names: list[str],
    lead_time_days: np.ndarray,
) -> tuple[dict[str, np.ndarray], pd.Timestamp]:
    """Reduce gridded forecast fields for all basins using ZonalEngine.

    Args:
        ds: Gridded forecast dataset for a single model.
        basins_gdf: GeoDataFrame of catchment polygons indexed by ``basin_id``.
        geoms: List of catchment Shapely geometries in the same order.
        feature_names: Canonical feature names to extract from ``ds``.
        lead_time_days: 1-indexed array of forecast lead days ``[1..K]``.

    Returns:
        Tuple ``(feature_arrays, init_timestamp)`` where each entry in
        ``feature_arrays`` has shape ``(n_basins, len(lead_time_days))``.
    """
    ds_latest, init_ts = _resolve_init_time_and_slice(ds)
    lat_dim = 'latitude' if 'latitude' in ds_latest.dims else 'lat'
    lon_dim = 'longitude' if 'longitude' in ds_latest.dims else 'lon'
    if lat_dim not in ds_latest.dims or lon_dim not in ds_latest.dims:
        msg = (
            'Forecast dataset is missing latitude/longitude dimensions;'
            f' found {list(ds_latest.dims)}.'
        )
        raise ValueError(msg)

    full_lats = np.asarray(ds_latest[lat_dim].to_numpy(), dtype=np.float64)
    full_lons = np.asarray(ds_latest[lon_dim].to_numpy(), dtype=np.float64)
    dlat = (
        abs(float(full_lats[1] - full_lats[0])) if len(full_lats) > 1 else 0.25
    )
    dlon = (
        abs(float(full_lons[1] - full_lons[0])) if len(full_lons) > 1 else 0.25
    )

    all_minx = min(g.bounds[0] for g in geoms)
    all_miny = min(g.bounds[1] for g in geoms)
    all_maxx = max(g.bounds[2] for g in geoms)
    all_maxy = max(g.bounds[3] for g in geoms)
    union_bounds = (all_minx, all_miny, all_maxx, all_maxy)

    buffer_deg = max(dlat, dlon) * 2.0
    sub_lats, sub_lons, lat_slice, lon_slice = slice_coordinates_by_bounds(
        full_lats, full_lons, union_bounds, buffer_degrees=buffer_deg
    )
    ds_sub = ds_latest.isel({lat_dim: lat_slice, lon_dim: lon_slice})

    domain_valid_mask = np.array(
        [
            _compute_domain_coverage_fraction(
                g, full_lats, full_lons, dlat, dlon
            )
            >= (MIN_VALID_COVERAGE_FRACTION - 1e-6)
            for g in geoms
        ],
        dtype=bool,
    )

    zonal_matrix: ZonalWeightMatrix | None = None
    if len(sub_lats) > 0 and len(sub_lons) > 0 and np.any(domain_valid_mask):
        zonal_matrix = ZonalEngine.from_geodataframe(
            basins_gdf=basins_gdf,
            lats=sub_lats,
            lons=sub_lons,
            cell_res_lat=dlat,
            cell_res_lon=dlon,
        )

    lead_hours = _resolve_lead_hours(ds_sub)
    dt_sec = np.zeros(len(lead_hours), dtype=np.float64)
    if len(lead_hours) > 1:
        dt_sec[0] = lead_hours[0] * 3600.0
        dt_sec[1:] = np.diff(lead_hours) * 3600.0
    elif len(lead_hours) == 1:
        dt_sec[0] = max(lead_hours[0], _HOURS_PER_DAY) * 3600.0

    n_basins = len(geoms)
    n_days = len(lead_time_days)
    results: dict[str, np.ndarray] = {}

    for feat_name in feature_names:
        da, matched_var = _select_variable_dataarray(ds_sub, feat_name)
        da_ordered = da.transpose('lead_time', lat_dim, lon_dim)

        if zonal_matrix is None:
            results[feat_name] = np.full(
                (n_basins, n_days), np.nan, dtype=np.float32
            )
            continue

        grid_3d = np.asarray(da_ordered.to_numpy(), dtype=np.float64)
        step_means, _ = zonal_matrix.reduce_3d_with_coverage(grid_3d)
        step_means_f64 = step_means.astype(np.float64)
        step_means_f64[~domain_valid_mask, :] = np.nan

        var_units = str(da.attrs.get('units', '')).strip()
        is_precip = 'precipitation' in feat_name
        is_pressure = 'pressure' in feat_name
        is_temp = 'temperature' in feat_name

        if is_pressure:
            if var_units.lower() != 'kpa' and (
                matched_var in {'pressure_surface', 'surface_pressure', 'sp'}
                or var_units.lower() in {'pa', ''}
            ):
                step_means_f64 = step_means_f64 / 1000.0
        elif is_temp and var_units in {'K', 'kelvin', 'Kelvin'}:
            step_means_f64 = step_means_f64 - 273.15

        is_precip_rate = is_precip and (
            matched_var == 'precipitation_surface'
            or var_units.lower() in {'kg m-2 s-1', 'kg/m2/s', 'mm/s'}
        )
        if is_precip_rate:
            # Propagate NaNs naturally without np.nan_to_num
            step_means_f64 = step_means_f64 * dt_sec[np.newaxis, :]

        daily_out = np.full((n_basins, n_days), np.nan, dtype=np.float32)
        for day_idx, d in enumerate(lead_time_days):
            h_start = _HOURS_PER_DAY * (int(d) - 1)
            h_end = _HOURS_PER_DAY * int(d)
            step_idx = [
                i for i, h in enumerate(lead_hours) if h_start < h <= h_end
            ]
            if not step_idx:
                # Missing forecast steps must remain NaN
                continue

            sub_steps = step_means_f64[:, step_idx]
            if is_precip:
                if is_precip_rate or len(step_idx) == 1:
                    daily_out[:, day_idx] = np.sum(sub_steps, axis=1).astype(
                        np.float32
                    )
                else:
                    daily_out[:, day_idx] = np.mean(sub_steps, axis=1).astype(
                        np.float32
                    )
            else:
                daily_out[:, day_idx] = np.mean(sub_steps, axis=1).astype(
                    np.float32
                )

        results[feat_name] = daily_out

    return results, init_ts


class ForecastZarrExtractor:
    """Extracts ECMWF HRES and GraphCast forecasts into Zarr stores."""

    def __init__(
        self,
        output_dir: Path | str | None = None,
        datasets: Mapping[str, xr.Dataset | str | Path] | None = None,
        hres_zarr_uri: str | Path | None = None,
        graphcast_zarr_uri: str | Path | None = None,
    ) -> None:
        """Initialize the forecast Zarr extractor.

        Args:
            output_dir: Directory where ``forecast_latest.zarr`` is written.
            datasets: Optional mapping with ``'hres'`` and ``'graphcast'`` keys
              pointing to ``xarray.Dataset`` instances or Zarr store paths.
            hres_zarr_uri: Optional explicit path/URI for the HRES Zarr store.
            graphcast_zarr_uri: Optional explicit path/URI for the GraphCast
              Zarr store.
        """
        self.output_dir = Path(output_dir) if output_dir else FORECAST_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._datasets: dict[str, xr.Dataset | str | Path] = dict(
            datasets or {}
        )
        if hres_zarr_uri is not None:
            self._datasets['hres'] = hres_zarr_uri
        if graphcast_zarr_uri is not None:
            self._datasets['graphcast'] = graphcast_zarr_uri

    def _explicit_sources(
        self,
    ) -> tuple[ForecastSource, ForecastSource] | None:
        """Return explicit ``(hres, graphcast)`` sources, or ``None``.

        Raises:
            ValueError: If only one of the two sources was supplied; a single
              explicit store is never silently paired with a catalog dataset.
        """
        hres_src = self._datasets.get('hres', self._datasets.get('ifs'))
        gc_src = self._datasets.get('graphcast')
        if hres_src is None and gc_src is None:
            return None
        if hres_src is None or gc_src is None:
            msg = (
                'Provide both the hres and graphcast forecast sources, or'
                ' neither to use the dynamical.org catalog; got hres='
                f'{hres_src!r}, graphcast={gc_src!r}.'
            )
            raise ValueError(msg)
        return hres_src, gc_src

    def _resolve_datasets(self) -> tuple[xr.Dataset, xr.Dataset]:
        """Return opened ``(ds_hres, ds_graphcast)`` datasets."""
        explicit = self._explicit_sources()
        if explicit is None:
            return _get_dynamical_datasets()
        hres_src, gc_src = explicit
        return (
            _open_zarr_or_dataset(hres_src),
            _open_zarr_or_dataset(gc_src),
        )

    def get_latest_issue_info(self) -> dict[str, Any]:
        """Query the forecast datasets for the latest initialization time.

        ``source_datasets`` names the upstream store behind each model-input
        family (``hres_*`` / ``graphcast_*``) so API consumers can see which
        dynamical.org catalog entries (or explicit Zarr stores) are in use.
        """
        explicit = self._explicit_sources()
        if explicit is None:
            source = 'dynamical.org'
            provider = 'ECMWF (IFS ENS control member & AIFS) via dynamical.org'
            source_datasets = dict(DYNAMICAL_DATASET_IDS)
        else:
            source = 'explicit'
            provider = 'user-supplied forecast stores'
            source_datasets = {
                'hres': _describe_forecast_source(explicit[0]),
                'graphcast': _describe_forecast_source(explicit[1]),
            }
        ds_hres, ds_graphcast = self._resolve_datasets()
        _, latest_hres_time = _resolve_init_time_and_slice(ds_hres)
        _, latest_gc_time = _resolve_init_time_and_slice(ds_graphcast)
        latest_dt = min(latest_hres_time, latest_gc_time)
        return {
            'status': 'online',
            'source': source,
            'source_datasets': source_datasets,
            'provider': provider,
            'hres_latest_init_time': latest_hres_time.isoformat(),
            'graphcast_latest_init_time': latest_gc_time.isoformat(),
            'latest_issue_date': latest_dt.strftime('%Y-%m-%d'),
            'latest_issue_time': latest_dt.isoformat(),
            'models': [
                'ECMWF IFS HRES',
                'GraphCast',
            ],
            'features': list(FORECAST_FEATURE_SPECS.keys()),
            'max_horizon_days': 15,
            'temporal_resolution': 'Daily (1D)',
        }

    def _parse_polygon_input(
        self,
        polygon_input: dict[str, Any] | BaseGeometry,
        basin_id: str | None = None,
        properties: dict[str, Any] | None = None,
    ) -> tuple[BaseGeometry, dict[str, Any], str]:
        """Parse and normalize polygon geometry, properties, and basin ID."""
        props: dict[str, Any] = dict(properties or {})

        if isinstance(polygon_input, dict):
            if polygon_input.get('type') == 'Feature':
                geom = shape(polygon_input.get('geometry', {}))
                props = {**polygon_input.get('properties', {}), **props}
            elif 'coordinates' in polygon_input:
                geom = shape(polygon_input)
            elif polygon_input.get('type') == 'FeatureCollection':
                feats = polygon_input.get('features', [])
                if not feats:
                    msg = 'Empty FeatureCollection provided'
                    raise ValueError(msg)
                geom = shape(feats[0].get('geometry', {}))
                props = {**feats[0].get('properties', {}), **props}
            else:
                msg = 'Invalid GeoJSON dictionary structure'
                raise ValueError(msg)
        elif isinstance(polygon_input, BaseGeometry):
            geom = polygon_input
        else:
            msg = (
                f'Unsupported polygon geometry input type:'
                f' {type(polygon_input)}'
            )
            raise TypeError(msg)

        wkt_hash = hashlib.sha256(geom.wkt.encode('utf-8')).hexdigest()[:8]
        resolved_id = str(
            basin_id
            or props.get('catchment_id')
            or props.get('gauge_id')
            or props.get('basin_id')
            or props.get('id')
            or f'basin_{wkt_hash}'
        )
        props['catchment_id'] = resolved_id
        return geom, props, resolved_id

    def fetch_and_archive(  # noqa: PLR0913, PLR0917
        self,
        catchment_feature: dict[str, Any] | BaseGeometry,
        basin_id: str | None = None,
        properties: dict[str, Any] | None = None,
        horizon_days: int = 10,
        horizon_hours: int | None = None,
        step_interval_hours: int | None = None,
        model: str = 'all',
    ) -> dict[str, Any]:
        """Fetch ECMWF HRES and GraphCast forecasts for a single catchment."""
        del step_interval_hours, model
        geom, props, resolved_basin_id = self._parse_polygon_input(
            catchment_feature, basin_id, properties
        )
        if horizon_hours is not None:
            horizon_days = max(1, int(horizon_hours) // 24)
        horizon_days = max(1, min(int(horizon_days), 15))

        batch_res = self.fetch_and_archive_batch(
            features=[
                {
                    'type': 'Feature',
                    'geometry': shapely.geometry.mapping(geom),
                    'properties': {
                        **props,
                        'catchment_id': resolved_basin_id,
                    },
                }
            ],
            horizon_days=horizon_days,
        )

        if not batch_res.get('results'):
            msg = 'Failed to extract real-time forecast data.'
            raise RuntimeError(msg)

        single_result = dict(batch_res['results'][0])
        single_result.update(
            {
                'master_zarr_path': batch_res['master_zarr_path'],
                'master_zarr_rel_path': batch_res['master_zarr_rel_path'],
                'total_basins_in_master': batch_res['total_basins_in_master'],
            }
        )
        return single_result

    def fetch_and_archive_batch(  # noqa: PLR0915
        self,
        features: list[dict[str, Any]],
        horizon_days: int = 10,
        notify_email: str = '',
        server_url: str = DEFAULT_LOCAL_SERVER_URL,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, Any]:
        """Fetch ECMWF HRES and GraphCast forecasts for multiple catchments."""
        if not features:
            msg = 'No catchment features provided for forecast batch.'
            raise ValueError(msg)

        horizon_days = max(1, min(int(horizon_days), 15))
        lead_time_days = np.arange(1, horizon_days + 1, dtype=np.int64)

        parsed_basins: list[
            tuple[BaseGeometry, dict[str, Any], str, float]
        ] = []
        for feat in features:
            g, p, cid = self._parse_polygon_input(feat)
            area_km2 = _validate_positive_area_km2(p, cid)
            parsed_basins.append((g, p, cid, area_km2))

        ds_hres, ds_graphcast = self._resolve_datasets()

        geoms = [g for g, _, _, _ in parsed_basins]
        basin_ids = [cid for _, _, cid, _ in parsed_basins]
        basins_gdf = gpd.GeoDataFrame(
            {'geometry': geoms},
            index=pd.Index(basin_ids, name='basin_id'),
            crs='EPSG:4326',
        )

        hres_features = [
            'hres_surface_net_solar_radiation',
            'hres_surface_net_thermal_radiation',
            'hres_surface_pressure',
            'hres_temperature_2m',
            'hres_total_precipitation',
        ]
        gc_features = [
            'graphcast_temperature_2m',
            'graphcast_total_precipitation',
        ]

        hres_extracted, hres_init_ts = _extract_model_features_for_basins(
            ds=ds_hres,
            basins_gdf=basins_gdf,
            geoms=geoms,
            feature_names=hres_features,
            lead_time_days=lead_time_days,
        )
        gc_extracted, gc_init_ts = _extract_model_features_for_basins(
            ds=ds_graphcast,
            basins_gdf=basins_gdf,
            geoms=geoms,
            feature_names=gc_features,
            lead_time_days=lead_time_days,
        )

        init_ts = min(hres_init_ts, gc_init_ts)
        init_date = pd.Timestamp(init_ts.strftime('%Y-%m-%d'))
        init_time_iso = init_ts.isoformat()

        # Lead k (1 <= k <= K) has valid date D + (k - 1) per §2.2
        valid_dates = [
            (init_date + pd.Timedelta(days=int(d) - 1)).strftime('%Y-%m-%d')
            for d in lead_time_days
        ]

        individual_results: list[dict[str, Any]] = []
        combined_datasets: list[xr.Dataset] = []

        for idx, (geom, props, cid, area_km2) in enumerate(parsed_basins):
            if progress_callback is not None:
                progress_callback(
                    idx + 1,
                    len(parsed_basins),
                    (
                        f'Extracting forecast for {cid}'
                        f' ({idx + 1}/{len(parsed_basins)})'
                    ),
                )

            cent = geom.centroid
            outlet_info = props.get('outlet')
            if isinstance(outlet_info, Mapping):
                centroid_lat = float(outlet_info.get('latitude', cent.y))
                centroid_lon = float(outlet_info.get('longitude', cent.x))
            else:
                centroid_lat = float(cent.y)
                centroid_lon = float(cent.x)

            daily_hres_solar = hres_extracted[
                'hres_surface_net_solar_radiation'
            ][idx]
            daily_hres_thermal = hres_extracted[
                'hres_surface_net_thermal_radiation'
            ][idx]
            daily_hres_sp = hres_extracted['hres_surface_pressure'][idx]
            daily_hres_t2m = hres_extracted['hres_temperature_2m'][idx]
            daily_hres_tp = hres_extracted['hres_total_precipitation'][idx]

            daily_gc_t2m = gc_extracted['graphcast_temperature_2m'][idx]
            daily_gc_tp = gc_extracted['graphcast_total_precipitation'][idx]

            arr_hres_solar = daily_hres_solar.astype(np.float32)[None, None, :]
            arr_hres_thermal = daily_hres_thermal.astype(np.float32)[
                None, None, :
            ]
            arr_hres_sp = daily_hres_sp.astype(np.float32)[None, None, :]
            arr_hres_t2m = daily_hres_t2m.astype(np.float32)[None, None, :]
            arr_hres_tp = daily_hres_tp.astype(np.float32)[None, None, :]

            arr_gc_t2m = daily_gc_t2m.astype(np.float32)[None, None, :]
            arr_gc_tp = daily_gc_tp.astype(np.float32)[None, None, :]

            dims_3d = ['basin', 'date', 'lead_time']
            basin_ds = xr.Dataset(
                data_vars={
                    'hres_surface_net_solar_radiation': (
                        dims_3d,
                        arr_hres_solar,
                        {
                            'units': 'W/m^2',
                            'standard_name': (
                                'surface_net_downward_shortwave_flux'
                            ),
                            'long_name': (
                                'ECMWF HRES Daily Mean Surface Net Solar'
                                ' Radiation'
                            ),
                            'description': (
                                'Daily mean surface net short-wave radiation'
                                ' flux from ECMWF HRES'
                            ),
                            '_ARRAY_DIMENSIONS': dims_3d,
                        },
                    ),
                    'hres_surface_net_thermal_radiation': (
                        dims_3d,
                        arr_hres_thermal,
                        {
                            'units': 'W/m^2',
                            'standard_name': (
                                'surface_net_thermal_radiation_flux'
                            ),
                            'long_name': (
                                'ECMWF HRES Daily Mean Surface Net Thermal'
                                ' Radiation'
                            ),
                            'description': (
                                'Daily mean surface net long-wave radiation'
                                ' flux from ECMWF HRES'
                            ),
                            '_ARRAY_DIMENSIONS': dims_3d,
                        },
                    ),
                    'hres_surface_pressure': (
                        dims_3d,
                        arr_hres_sp,
                        {
                            'units': 'kPa',
                            'standard_name': 'surface_air_pressure',
                            'long_name': (
                                'ECMWF HRES Daily Mean Surface Pressure'
                            ),
                            'description': (
                                'Daily mean surface atmospheric pressure in'
                                ' kPa from ECMWF HRES'
                            ),
                            '_ARRAY_DIMENSIONS': dims_3d,
                        },
                    ),
                    'hres_temperature_2m': (
                        dims_3d,
                        arr_hres_t2m,
                        {
                            'units': 'degC',
                            'standard_name': 'air_temperature',
                            'long_name': (
                                'ECMWF HRES Daily Mean 2m Temperature'
                            ),
                            'description': (
                                'Daily mean 2m air temperature from ECMWF HRES'
                            ),
                            '_ARRAY_DIMENSIONS': dims_3d,
                        },
                    ),
                    'hres_total_precipitation': (
                        dims_3d,
                        arr_hres_tp,
                        {
                            'units': 'mm/day',
                            'standard_name': 'precipitation_flux',
                            'long_name': (
                                'ECMWF HRES Daily Total Precipitation'
                            ),
                            'description': (
                                'Daily accumulated total precipitation from'
                                ' ECMWF HRES'
                            ),
                            '_ARRAY_DIMENSIONS': dims_3d,
                        },
                    ),
                    'graphcast_temperature_2m': (
                        dims_3d,
                        arr_gc_t2m,
                        {
                            'units': 'degC',
                            'standard_name': 'air_temperature',
                            'long_name': (
                                'GraphCast Daily Mean 2m Temperature'
                            ),
                            'description': (
                                'Daily mean 2m air temperature from GraphCast'
                            ),
                            '_ARRAY_DIMENSIONS': dims_3d,
                        },
                    ),
                    'graphcast_total_precipitation': (
                        dims_3d,
                        arr_gc_tp,
                        {
                            'units': 'mm/day',
                            'standard_name': 'precipitation_flux',
                            'long_name': (
                                'GraphCast Daily Total Precipitation'
                            ),
                            'description': (
                                'Daily accumulated total precipitation from'
                                ' GraphCast'
                            ),
                            '_ARRAY_DIMENSIONS': dims_3d,
                        },
                    ),
                    'area_km2': (
                        ['basin'],
                        np.array([area_km2], dtype=np.float32),
                        {
                            'units': 'km2',
                            'long_name': 'Contributing Drainage Area',
                            '_ARRAY_DIMENSIONS': ['basin'],
                        },
                    ),
                    'latitude': (
                        ['basin'],
                        np.array([centroid_lat], dtype=np.float32),
                        {
                            'units': 'degrees_north',
                            'long_name': 'Catchment Outlet Latitude',
                            '_ARRAY_DIMENSIONS': ['basin'],
                        },
                    ),
                    'longitude': (
                        ['basin'],
                        np.array([centroid_lon], dtype=np.float32),
                        {
                            'units': 'degrees_east',
                            'long_name': 'Catchment Outlet Longitude',
                            '_ARRAY_DIMENSIONS': ['basin'],
                        },
                    ),
                },
                coords={
                    'basin': [cid],
                    'basin_id': ('basin', [cid]),
                    'date': pd.DatetimeIndex([init_date]),
                    'issue_time': ('date', pd.DatetimeIndex([init_date])),
                    'lead_time': lead_time_days,
                },
                attrs={
                    'title': (
                        f'Real-Time Weather Forecast Inference Store - {cid}'
                    ),
                    'source_platform': 'dynamical.org cloud-optimized Zarr',
                    'models': 'ECMWF IFS HRES & GraphCast',
                    'initialization_time': init_time_iso,
                    'issue_date': init_date.strftime('%Y-%m-%d'),
                    'catchment_id': cid,
                    'temporal_resolution': '1D',
                    'horizon_days': int(horizon_days),
                    'min_valid_coverage_fraction': MIN_VALID_COVERAGE_FRACTION,
                    'purpose': 'OpenHydroNet Real-Time Flood Inference',
                    'repository': GITHUB_REPO_URL,
                    'config_reference': GITHUB_FLOODHUB_CONFIG_URL,
                },
            )
            basin_ds.lead_time.attrs['units'] = 'days'
            basin_ds.lead_time.attrs['long_name'] = 'Forecast Lead Time'
            basin_ds.date.attrs['long_name'] = 'Forecast Issue Time'

            def _round_list(arr: np.ndarray, decimals: int = 2) -> list[float]:
                return [
                    round(float(v), decimals)
                    if np.isfinite(v)
                    else float('nan')
                    for v in arr
                ]

            meteogram = {
                'lead_steps_days': [int(d) for d in lead_time_days],
                'valid_times': valid_dates,
                'hres': {
                    'precipitation_mm': _round_list(daily_hres_tp, 2),
                    'accumulated_precipitation_mm': _round_list(
                        np.cumsum(daily_hres_tp), 2
                    ),
                    'temperature_c': _round_list(daily_hres_t2m, 2),
                    'surface_pressure_kpa': _round_list(daily_hres_sp, 2),
                    'net_solar_radiation_wm2': _round_list(daily_hres_solar, 1),
                    'net_thermal_radiation_wm2': _round_list(
                        daily_hres_thermal, 1
                    ),
                },
                'graphcast': {
                    'precipitation_mm': _round_list(daily_gc_tp, 2),
                    'accumulated_precipitation_mm': _round_list(
                        np.cumsum(daily_gc_tp), 2
                    ),
                    'temperature_c': _round_list(daily_gc_t2m, 2),
                },
                'step_precipitation_mm': _round_list(daily_hres_tp, 2),
                'temperature_c': _round_list(daily_hres_t2m, 2),
            }

            individual_results.append(
                {
                    'status': 'success',
                    'catchment_id': cid,
                    'name': str(props.get('name') or cid),
                    'area_km2': area_km2,
                    'init_time': init_time_iso,
                    'issue_date': init_date.strftime('%Y-%m-%d'),
                    'horizon_days': horizon_days,
                    'dimensions': ['basin', 'date', 'lead_time'],
                    'features_extracted': list(FORECAST_FEATURE_SPECS.keys()),
                    'meteogram': meteogram,
                }
            )

            combined_datasets.append(basin_ds)

        unified_forecast_path = self.output_dir / FORECAST_LATEST_ZARR_NAME
        master_ds = self._merge_to_master_forecast(
            combined_datasets, unified_forecast_path
        )

        size_bytes = sum(
            f.stat().st_size
            for f in unified_forecast_path.rglob('*')
            if f.is_file()
        )
        size_mb = round(size_bytes / (1024 * 1024), 3)
        size_kb = round(size_bytes / 1024, 2)
        total_basins = len(master_ds.basin.to_numpy())

        rel_path = (
            str(unified_forecast_path.relative_to(self.output_dir.parent))
            if self.output_dir.parent.exists()
            else str(unified_forecast_path)
        )

        for r in individual_results:
            r['zarr_path'] = str(unified_forecast_path)
            r['zarr_rel_path'] = rel_path
            r['size_kb'] = size_kb

        if notify_email:
            self._send_notification_email(
                notify_email=notify_email,
                server_url=server_url,
                basins_count=len(individual_results),
                init_time=init_time_iso,
                master_path=str(unified_forecast_path),
                master_size_mb=size_mb,
            )

        return {
            'status': 'success',
            'batch': len(individual_results) > 1,
            'count': len(individual_results),
            'init_time': init_time_iso,
            'issue_date': init_date.strftime('%Y-%m-%d'),
            'horizon_days': horizon_days,
            'zarr_path': str(unified_forecast_path),
            'zarr_rel_path': rel_path,
            'master_zarr_path': str(unified_forecast_path),
            'master_zarr_rel_path': rel_path,
            'size_kb': size_kb,
            'master_size_mb': size_mb,
            'total_basins_in_master': total_basins,
            'features': list(FORECAST_FEATURE_SPECS.keys()),
            'results': individual_results,
        }

    def _merge_to_master_forecast(  # noqa: C901, PLR0912
        self, new_datasets: list[xr.Dataset], master_path: Path
    ) -> xr.Dataset:
        """Merge multi-catchment datasets into the unified forecast Zarr store.

        Only retains existing basins from ``master_path`` when their forecast
        issue ``date`` and ``lead_time`` coordinates match ``incoming_batch``
        exactly, preventing stale forecast issue dates from being mixed.
        """
        if not new_datasets:
            if master_path.exists():
                is_consolidated = (master_path / '.zmetadata').is_file()
                return xr.open_zarr(
                    str(master_path),
                    consolidated=is_consolidated,
                    decode_timedelta=False,
                )
            msg = 'No datasets to merge into forecast store.'
            raise ValueError(msg)

        incoming_batch = (
            new_datasets[0]
            if len(new_datasets) == 1
            else xr.concat(new_datasets, dim='basin')
        )

        if master_path.exists():
            is_consolidated = (master_path / '.zmetadata').is_file()
            existing_ds = xr.open_zarr(
                str(master_path),
                consolidated=is_consolidated,
                decode_timedelta=False,
            ).load()
            same_dates = np.array_equal(
                existing_ds.date.to_numpy(), incoming_batch.date.to_numpy()
            )
            same_leads = np.array_equal(
                existing_ds.lead_time.to_numpy(),
                incoming_batch.lead_time.to_numpy(),
            )
            incoming_basins = {str(b) for b in incoming_batch.basin.to_numpy()}
            existing_basins = [
                str(b)
                for b in existing_ds.basin.to_numpy()
                if str(b) not in incoming_basins
            ]

            if same_dates and same_leads and existing_basins:
                existing_subset = existing_ds.sel(basin=existing_basins)
                combined = xr.concat(
                    [existing_subset, incoming_batch], dim='basin'
                )
            else:
                combined = incoming_batch
        else:
            combined = incoming_batch

        combined.encoding.clear()
        for var_key in list(combined.variables):
            combined[var_key].encoding.clear()

        chunk_spec: dict[str, int] = {'basin': 1}
        if 'date' in combined.dims:
            chunk_spec['date'] = 1
        if 'lead_time' in combined.dims:
            chunk_spec['lead_time'] = len(combined.lead_time)

        combined_chunked = combined.chunk(chunk_spec)

        encoding: dict[str, dict[str, Any]] = {}
        for var_name in combined_chunked.data_vars:
            dims = combined_chunked[var_name].dims
            if dims == ('basin', 'date', 'lead_time'):
                encoding[var_name] = {
                    'chunks': (1, 1, len(combined_chunked.lead_time))
                }
            elif dims == ('basin',):
                encoding[var_name] = {'chunks': (1,)}

        for coord_name in ('basin_id', 'area_km2', 'latitude', 'longitude'):
            if coord_name in combined_chunked.coords and combined_chunked[
                coord_name
            ].dims == ('basin',):
                encoding[coord_name] = {'chunks': (1,)}

        if master_path.exists():
            shutil.rmtree(master_path)

        combined_chunked.to_zarr(
            store=str(master_path),
            mode='w',
            consolidated=True,
            encoding=encoding or None,
        )
        return combined_chunked

    def _send_notification_email(  # noqa: PLR0913, PLR0917
        self,
        notify_email: str,
        server_url: str,
        basins_count: int,
        init_time: str,
        master_path: str,
        master_size_mb: float,
    ) -> None:
        """Send email notification upon forecast extraction completion."""
        subject = (
            f'[OpenHydroNet] Real-Time Forecast Extracted'
            f' ({basins_count} Basins - {init_time[:10]})'
        )
        text_body = (
            'OpenHydroNet - Real-Time Forecast Extraction Complete\n'
            '============================================================\n\n'
            'Real-time dynamic weather forecasts from dynamical.org'
            ' (ECMWF IFS HRES & GraphCast)\n'
            'have been extracted and formatted into OpenHydroNet inference'
            ' Zarr archives.\n\n'
            'Details:\n'
            '--------\n'
            f'Issue Time: {init_time}\n'
            f'Basins Extracted: {basins_count}\n'
            'Features Included:\n'
            '  - hres_surface_net_solar_radiation\n'
            '  - hres_surface_net_thermal_radiation\n'
            '  - hres_surface_pressure\n'
            '  - hres_temperature_2m\n'
            '  - hres_total_precipitation\n'
            '  - graphcast_temperature_2m\n'
            '  - graphcast_total_precipitation\n\n'
            'Storage Location:\n'
            '-----------------\n'
            f'Master Forecast Zarr: {master_path} ({master_size_mb} MB)\n\n'
            'Web Dashboard:\n'
            '--------------\n'
            f'{server_url}\n\n'
            'Best regards,\n'
            'OpenHydroNet Team\n'
        )
        send_email_notification(
            to_email=notify_email,
            subject=subject,
            text_body=text_body,
            html_body=f'<pre>{text_body}</pre>',
        )
