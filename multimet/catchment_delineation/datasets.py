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

"""Scientific DEM dataset descriptors for multi-DEM catchment delineation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DemDataset:
    """Scientific metadata and geographic coverage bounds for a D8 DEM 
        dataset."""

    id: str
    name: str
    citation: str
    resolution_label: str
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    source: str
    public_tiles_uri: str | None = None


HYDROSHEDS_90M = DemDataset(
    id='hydrosheds_90m',
    name='HydroSHEDS 90m Conditioned DEM (3 arc-sec)',
    citation='Lehner et al. (2008), HydroSHEDS Technical Documentation',
    resolution_label='90m (3 arc-second)',
    min_lat=-56.0,
    max_lat=60.0,
    min_lon=-180.0,
    max_lon=180.0,
    source='hydrosheds:v1',
    public_tiles_uri='gs://open-multimet/ancillary-data/dems/tiles_5deg',
)

MERIT_HYDRO_90M = DemDataset(
    id='merit_hydro_90m',
    name='MERIT-Hydro 90m DEM (3 arc-sec)',
    citation='Yamazaki et al. (2019), Water Resources Research',
    resolution_label='90m (3 arc-second)',
    min_lat=-60.0,
    max_lat=90.0,
    min_lon=-180.0,
    max_lon=180.0,
    source='earth_engine:MERIT/Hydro/v1_0_1',
    public_tiles_uri=None,
)

DEM_DATASETS: dict[str, DemDataset] = {
    HYDROSHEDS_90M.id: HYDROSHEDS_90M,
    MERIT_HYDRO_90M.id: MERIT_HYDRO_90M,
}

DEM_ALIASES: dict[str, str] = {
    'hydrosheds_90m': 'hydrosheds_90m',
    'hydrosheds': 'hydrosheds_90m',
    'hydroatlas': 'hydrosheds_90m',
    'merit_hydro_90m': 'merit_hydro_90m',
    'merit-hydro': 'merit_hydro_90m',
    'merit_hydro': 'merit_hydro_90m',
    'merit': 'merit_hydro_90m',
}


def resolve_dem_dataset(alias: str | DemDataset) -> DemDataset:
    """Resolve a DEM dataset descriptor or string alias without default 
        fallbacks."""
    if isinstance(alias, DemDataset):
        return alias
    if not isinstance(alias, str) or not alias.strip():
        raise ValueError(
            f'An explicit DEM dataset identifier is required; got {alias!r}.'
        )
    key = alias.strip().lower()
    canonical = DEM_ALIASES.get(key, key)
    if canonical not in DEM_DATASETS:
        raise ValueError(
            f'Unknown DEM dataset {alias!r}. '
            f'Supported datasets: {sorted(DEM_DATASETS.keys())}'
        )
    return DEM_DATASETS[canonical]
