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

"""Configuration, schemas, constants, and unit helpers for the `maas` package."""

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.file_paths import (
    FLOODHUB_BASE_URL as FLOODHUB_BASE_URL,
)
from utils.file_paths import (
    GEOGLOWS_BASE_URL as GEOGLOWS_BASE_URL,
)
from utils.file_paths import (
    GLOFAS_BASE_URL as GLOFAS_BASE_URL,
)
from utils.file_paths import (
    OPEN_METEO_ELEVATION_URL as OPEN_METEO_ELEVATION_URL,
)

PROVIDERS: tuple[str, ...] = (
    'floodhub',
    'glofas',
    'geoglows',
)

DEFAULT_MAAS_MODELS: tuple[str, ...] = PROVIDERS

MAAS_MODEL_NAMES: dict[str, str] = {
    'floodhub': 'Google FloodHub',
    'glofas': 'Copernicus GloFAS v4',
    'geoglows': 'GEOGLOWS ECMWF v2',
}

MODEL_ALIASES: dict[str, str] = {
    'floodhub': 'floodhub',
    'google_floodhub': 'floodhub',
    'glofas': 'glofas',
    'copernicus_glofas': 'glofas',
    'geoglows': 'geoglows',
}

NETWORK_LABELS: dict[str, str] = {
    'floodhub': 'HydroSHEDS HydroRIVERS reaches',
    'glofas': 'GloFAS v4 LISFLOOD 0.05° river grid',
    'geoglows': 'GEOGLOWS v2 TDX-Hydro streams',
}

GLOFAS_RES_DEG = 0.05
GLOFAS_NLAT = 3600
GLOFAS_NLON = 7200
GLOFAS_MIN_AREA_KM2 = 100.0
GEOGLOWS_MIN_AREA_KM2 = 100.0

EARTH_RADIUS_KM = 6371.0088
CFS_TO_CMS = 0.028316846592
CMS_TO_CFS = 1.0 / CFS_TO_CMS

RETURN_PERIOD_YEARS: tuple[int, ...] = (2, 5, 10, 20, 25, 50, 100)
CANONICAL_RETURN_PERIODS: tuple[int, ...] = (2, 5, 10, 20, 50, 100)

PROVIDER_SCHEMAS: dict[str, Mapping[str, Any]] = {
    'floodhub': {
        'id': 'floodhub',
        'model_key': 'google_floodhub',
        'name': 'Google FloodHub',
        'network': 'HydroSHEDS HydroRIVERS / HydroBASINS',
        'horizon_days': 7,
        'central_field': 'discharge',
        'quantiles': (),
        'default_unit': 'm3/s',
    },
    'glofas': {
        'id': 'glofas',
        'model_key': 'copernicus_glofas',
        'name': 'Copernicus GloFAS v4',
        'network': 'LISFLOOD 0.05° grid',
        'horizon_days': 15,
        'central_field': 'discharge_median',
        'quantiles': (
            'discharge_min',
            'discharge_p25',
            'discharge_median',
            'discharge_mean',
            'discharge_p75',
            'discharge_max',
        ),
        'default_unit': 'm3/s',
    },
    'geoglows': {
        'id': 'geoglows',
        'model_key': 'geoglows',
        'name': 'GEOGLOWS ECMWF v2',
        'network': 'TDX-Hydro streams',
        'horizon_days': 15,
        'central_field': 'flow_med',
        'quantiles': (
            'flow_min',
            'flow_25p',
            'flow_med',
            'flow_avg',
            'flow_75p',
            'flow_max',
            'high_res',
        ),
        'default_unit': 'm3/s',
    },
}

_FLOAT_RE = re.compile(r'^[+-]?(?:(?:\d+\.?\d*)|(?:\.\d+))(?:[eE][+-]?\d+)?$')
_INT_RE = re.compile(r'^[+-]?\d+$')


@dataclass(frozen=True)
class MaaSConfig:
    """Explicit configuration for `MaaSEngine` and provider clients.

    Attributes:
        cache_dir: Directory for SQLite caches and derived indices.
        river_networks_dir: Directory containing static river network datasets.
        floodhub_api_key: API key for Google FloodHub v1 REST endpoints.
        floodhub_base_url: Base URL for Google FloodHub API.
        geoglows_base_url: Base URL for GEOGLOWS v2 REST API.
        glofas_base_url: Base URL for Open-Meteo GloFAS flood API.
        http_timeout_s: Default HTTP request timeout in seconds.
    """

    cache_dir: Path
    river_networks_dir: Path
    floodhub_api_key: str = ''
    floodhub_base_url: str = FLOODHUB_BASE_URL
    geoglows_base_url: str = GEOGLOWS_BASE_URL
    glofas_base_url: str = GLOFAS_BASE_URL
    http_timeout_s: float = 12.0

    def __post_init__(self) -> None:
        if self.cache_dir is None or str(self.cache_dir).strip() == '':
            raise ValueError('MaaSConfig requires a non-empty `cache_dir`.')
        if (
            self.river_networks_dir is None
            or str(self.river_networks_dir).strip() == ''
        ):
            raise ValueError(
                'MaaSConfig requires a non-empty `river_networks_dir`.'
            )
        if not isinstance(self.cache_dir, Path):
            object.__setattr__(self, 'cache_dir', Path(self.cache_dir))
        if not isinstance(self.river_networks_dir, Path):
            object.__setattr__(
                self, 'river_networks_dir', Path(self.river_networks_dir)
            )
        if self.http_timeout_s <= 0:
            raise ValueError('`http_timeout_s` must be positive.')


def parse_finite_float(val: Any) -> float | None:
    """Parse a finite float without exception masking, or return `None`."""
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        fval = float(val)
        return fval if math.isfinite(fval) else None
    if hasattr(val, 'item') and callable(val.item):
        scalar = val.item()
        if isinstance(scalar, (int, float)) and not isinstance(scalar, bool):
            fval = float(scalar)
            return fval if math.isfinite(fval) else None
    if isinstance(val, str):
        text = val.strip()
        if not text or not _FLOAT_RE.match(text):
            return None
        fval = float(text)
        return fval if math.isfinite(fval) else None
    return None


def parse_float_or_default(val: Any, default: float = 0.0) -> float:
    """Parse a finite float, returning `default` if `val` is not finite."""
    parsed = parse_finite_float(val)
    return parsed if parsed is not None else default


def parse_float_or_nan(val: Any) -> float:
    """Parse a finite float, returning `float('nan')` for missing/invalid data."""
    parsed = parse_finite_float(val)
    return parsed if parsed is not None else float('nan')


def parse_int(val: Any) -> int | None:
    """Parse an integer value without exception masking, or return `None`."""
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        if math.isfinite(val) and val.is_integer():
            return int(val)
        return None
    if hasattr(val, 'item') and callable(val.item):
        scalar = val.item()
        if isinstance(scalar, int) and not isinstance(scalar, bool):
            return int(scalar)
    if isinstance(val, str):
        text = val.strip()
        if _INT_RE.match(text):
            return int(text)
    return None


def convert_discharge_units(
    value: float,
    from_unit: str,
    to_unit: str = 'm3/s',
) -> float:
    """Convert volumetric river discharge between cubic meters and cubic feet per second.

    Args:
        value: Discharge value (may be `NaN`).
        from_unit: Source unit identifier (`'m3/s'`, `'cms'`, `'CUBIC_METERS_PER_SECOND'`,
            `'ft3/s'`, `'cfs'`, `'CUBIC_FEET_PER_SECOND'`).
        to_unit: Target unit identifier.

    Returns:
        Converted discharge value (`NaN` input yields `NaN` output).

    Raises:
        ValueError: If `from_unit` or `to_unit` is not a recognized discharge unit.
    """
    cms_tokens = {'m3/s', 'm³/s', 'cms', 'cubic_meters_per_second'}
    cfs_tokens = {'ft3/s', 'ft³/s', 'cfs', 'cubic_feet_per_second'}
    src = from_unit.strip().lower()
    dst = to_unit.strip().lower()
    if src not in cms_tokens and src not in cfs_tokens:
        raise ValueError(f'Unsupported source discharge unit: {from_unit!r}')
    if dst not in cms_tokens and dst not in cfs_tokens:
        raise ValueError(f'Unsupported target discharge unit: {to_unit!r}')
    if not math.isfinite(value):
        return float('nan')
    if (src in cms_tokens and dst in cms_tokens) or (
        src in cfs_tokens and dst in cfs_tokens
    ):
        return float(value)
    if src in cfs_tokens and dst in cms_tokens:
        return float(value) * CFS_TO_CMS
    return float(value) * CMS_TO_CFS


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers between two WGS84 coordinates."""
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = (
        math.sin(dp / 2.0) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def normalize_requested_models(
    requested: Sequence[str] | None,
) -> list[str]:
    """Normalize a list of model names/aliases to canonical `PROVIDERS` IDs."""
    out: list[str] = []
    for raw in requested or ():
        token = (
            str(raw)
            .strip()
            .lower()
            .replace('-', '_')
            .replace(' ', '_')
            .replace("'", '')
        )
        canonical = MODEL_ALIASES.get(token)
        if canonical and canonical not in out:
            out.append(canonical)
    return out or list(DEFAULT_MAAS_MODELS)
