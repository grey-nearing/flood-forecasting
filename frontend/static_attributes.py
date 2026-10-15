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

"""Frontend adapter for `multimet.static_extractor`."""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import shapely.geometry

from frontend.config import HYDRO_BASINS_DIR, SHARED_HYDROATLAS_DIR
from multimet.static_extractor import (
    ATTRIBUTE_REGISTRY,
    CARAVAN_CLIMATE_ALIASES,
    CatchmentAttributes,
    StaticAttributesExtractor,
    UnsupportedHydroATLASLevelError,
    get_attribute_definition,
    to_physical_units,
)
from multimet.static_extractor.config import (
    DEFAULT_GCS_ERA5_CLIMATE_URI,
    DEFAULT_GCS_GRIDDED_ERA5_URI,
    DEFAULT_GCS_HYDROATLAS_URI,
)
from utils.file_paths import (
    BASIN_ATLAS_GDB_NAME,
    HYDRO_ATLAS_LEV12_PARQUET_NAME,
    OPENHYDRONET_DATA_CACHE_DIR,
    OPENHYDRONET_ERA5_CLIMATE_CACHE_DIR,
    OPENHYDRONET_HYDROATLAS_CACHE_DIR,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

MAX_MAP_FEATURES: int = 2500
_FINEST_HYDROATLAS_LEVEL: int = 12

ZOOM_TO_LEVEL: dict[int, int] = {
    0: 4,
    1: 4,
    2: 4,
    3: 4,
    4: 5,
    5: 6,
    6: 7,
    7: 8,
    8: 10,
    9: 12,
    10: 12,
    11: 12,
    12: 12,
    13: 12,
    14: 12,
    15: 12,
    16: 12,
    17: 12,
    18: 12,
    19: 12,
    20: 12,
}

LEVEL_SIMPLIFY_TOL: dict[int, float] = {
    1: 0.015,
    2: 0.015,
    3: 0.015,
    4: 0.015,
    5: 0.015,
    6: 0.006,
    7: 0.006,
    8: 0.002,
    9: 0.002,
    10: 0.0005,
    11: 0.0005,
    12: 0.0005,
}


def resolve_app_static_extractor_paths() -> dict[str, Any]:
    """Resolve local and GCS paths for `StaticAttributesExtractor`.

    Checks `HYDROATLAS_GDB_PATH`, `ERA5_CLIMATE_CACHE_DIR`, and
    `ERA5_GRIDDED_ZARR_URI` environment variables first, then probes standard
    local cache directories. Never returns a non-existent default fallback path.

    Returns:
        Dictionary of resolved paths and GCS URIs for
        `StaticAttributesExtractor`.
    """
    resolved_gdb: Path | None = None
    env_gdb = os.environ.get('HYDROATLAS_GDB_PATH', '').strip()
    if env_gdb:
        resolved_gdb = Path(env_gdb)
    else:
        gdb_candidates = (
            OPENHYDRONET_HYDROATLAS_CACHE_DIR / BASIN_ATLAS_GDB_NAME,
            OPENHYDRONET_HYDROATLAS_CACHE_DIR,
            OPENHYDRONET_DATA_CACHE_DIR / 'basin_atlas' / BASIN_ATLAS_GDB_NAME,
            OPENHYDRONET_DATA_CACHE_DIR / 'basin_atlas',
            SHARED_HYDROATLAS_DIR / BASIN_ATLAS_GDB_NAME,
            SHARED_HYDROATLAS_DIR,
            HYDRO_BASINS_DIR / 'basin_atlas' / BASIN_ATLAS_GDB_NAME,
        )
        for cand in gdb_candidates:
            if not cand.exists():
                continue
            if cand.name.endswith('.gdb') and any(cand.glob('*.gdbtable')):
                resolved_gdb = cand
                break
            if (cand / HYDRO_ATLAS_LEV12_PARQUET_NAME).is_file() and (
                cand / 'subpolygons'
            ).is_dir():
                resolved_gdb = cand
                break

    resolved_era5: Path | None = None
    env_era5 = os.environ.get('ERA5_CLIMATE_CACHE_DIR', '').strip()
    if env_era5:
        resolved_era5 = Path(env_era5)
    else:
        era5_candidates = (
            OPENHYDRONET_ERA5_CLIMATE_CACHE_DIR,
            SHARED_HYDROATLAS_DIR / 'era5_climate',
        )
        for cand in era5_candidates:
            if cand.is_dir() and any(cand.glob('*_climate_indices.txt')):
                resolved_era5 = cand
                break

    resolved_gridded = os.environ.get('ERA5_GRIDDED_ZARR_URI') or None

    return {
        'gdb_path': resolved_gdb,
        'era5_cache_dir': resolved_era5,
        'gridded_era5_uri': resolved_gridded,
        'gcs_gdb_uri': DEFAULT_GCS_HYDROATLAS_URI,
        'gcs_era5_climate_uri': DEFAULT_GCS_ERA5_CLIMATE_URI,
    }


_extractor_instance: StaticAttributesExtractor | None = None


def reset_attributes_extractor() -> None:
    """Clear the cached `StaticAttributesExtractor` singleton."""
    global _extractor_instance  # noqa: PLW0603
    _extractor_instance = None


def get_attributes_extractor() -> StaticAttributesExtractor:
    """Return a lazily initialized `StaticAttributesExtractor` singleton."""
    global _extractor_instance  # noqa: PLW0603
    if _extractor_instance is None:
        paths = resolve_app_static_extractor_paths()
        local_gdb = paths['gdb_path']
        local_era5 = paths['era5_cache_dir']
        _extractor_instance = StaticAttributesExtractor(
            gdb_path=local_gdb,
            era5_cache_dir=local_era5,
            gridded_era5_uri=paths['gridded_era5_uri'],
            gcs_gdb_uri=(
                None if local_gdb is not None else paths['gcs_gdb_uri']
            ),
            gcs_era5_climate_uri=paths['gcs_era5_climate_uri'],
            no_download=(local_gdb is None or local_era5 is None),
        )
    return _extractor_instance


def zoom_to_hydroatlas_level(zoom: int, available: Sequence[int]) -> int:
    """Map a slippy-map zoom level to the best available BasinATLAS level.

    Args:
        zoom: Slippy-map zoom integer (`0..20`).
        available: Sequence of available Pfafstetter levels (`1..12`).

    Returns:
        Selected Pfafstetter level integer.
    """
    target = ZOOM_TO_LEVEL.get(zoom, _FINEST_HYDROATLAS_LEVEL)
    valid = [lvl for lvl in available if lvl <= target]
    if valid:
        return max(valid)
    return min(available) if available else _FINEST_HYDROATLAS_LEVEL


def sanitize_for_json(obj: Any) -> Any:  # noqa: ANN401
    """Recursively convert `NaN`/`Inf` floats and NumPy scalars for JSON.

    Args:
        obj: Arbitrary nested Python/NumPy value.

    Returns:
        JSON-serializable value with `NaN` and `Inf` converted to `None`.
    """
    out: Any = obj
    if obj is None or isinstance(obj, bool):
        out = obj
    elif isinstance(obj, (np.integer, int)):
        out = int(obj)
    elif isinstance(obj, (np.floating, float)):
        fv = float(obj)
        out = None if (math.isnan(fv) or math.isinf(fv)) else fv
    elif isinstance(obj, dict):
        out = {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        out = [sanitize_for_json(item) for item in obj]
    return out


def _first_present[T](*values: T | None) -> T | None:
    """Return the first value that is not `None`."""
    for val in values:
        if val is not None:
            return val
    return None


def card_payload_from_cache_entry(
    catchment_id: str,
    entry: dict[str, Any],
) -> dict[str, Any]:
    """Construct the UI attribute card payload from a persisted cache entry.

    Validates that `area_km2`, `subbasin_weights_km2`, and
    `area_fraction_used_for_aggregation` are explicitly present rather than
    fabricating `0.0` weights or `1.0` coverage fractions.

    Args:
        catchment_id: Unique catchment identifier.
        entry: Persisted dictionary from `extracted_attributes.json`.

    Returns:
        Formatted UI attribute card payload dictionary.

    Raises:
        ValueError: If `area_km2`, `subbasin_weights_km2`, or
            `area_fraction_used_for_aggregation` is missing or invalid.
    """
    attrs = (
        entry['attributes']
        if 'attributes' in entry
        else entry.get('raw_attributes', entry)
    )
    summary = entry.get('summary') or {}

    raw_area = _first_present(
        entry.get('area_km2'),
        summary.get('total_area_km2'),
        attrs.get('basin_area') if isinstance(attrs, dict) else None,
    )
    if raw_area is None:
        raise ValueError(
            f'Cache entry for {catchment_id!r} is missing area_km2.'
        )
    area_km2 = float(raw_area)
    if not math.isfinite(area_km2) or area_km2 <= 0.0:
        raise ValueError(
            f'Cache entry for {catchment_id!r} has invalid area_km2={area_km2}.'
        )

    raw_ids = _first_present(
        entry.get('subbasin_ids'),
        summary.get('subbasin_ids'),
        (),
    )
    subbasin_ids = tuple(int(x) for x in raw_ids)

    raw_weights = _first_present(
        entry.get('subbasin_weights_km2'),
        summary.get('subbasin_weights_km2'),
    )
    if subbasin_ids:
        if raw_weights is None or len(raw_weights) != len(subbasin_ids):
            raise ValueError(
                f'Cache entry for {catchment_id!r} is missing '
                'subbasin_weights_km2 matching subbasin_ids.'
            )
        subbasin_weights_km2 = tuple(float(w) for w in raw_weights)
    else:
        subbasin_weights_km2 = ()

    raw_fraction = _first_present(
        entry.get('area_fraction_used_for_aggregation'),
        summary.get('area_fraction_used_for_aggregation'),
        (
            attrs.get('area_fraction_used_for_aggregation')
            if isinstance(attrs, dict)
            else None
        ),
    )
    if raw_fraction is None or not math.isfinite(float(raw_fraction)):
        raise ValueError(
            f'Cache entry for {catchment_id!r} is missing '
            'area_fraction_used_for_aggregation.'
        )
    area_fraction = float(raw_fraction)

    res = CatchmentAttributes(
        catchment_id=catchment_id,
        attributes=attrs,
        subbasin_ids=subbasin_ids,
        subbasin_weights_km2=subbasin_weights_km2,
        area_km2=area_km2,
        area_fraction_used_for_aggregation=area_fraction,
    )
    return build_attribute_card_payload(res)


def build_attribute_card_payload(res: CatchmentAttributes) -> dict[str, Any]:
    """Format a `CatchmentAttributes` result into the UI card payload.

    Args:
        res: Extracted `CatchmentAttributes` object.

    Returns:
        JSON-safe dictionary containing categorized physical-unit attributes,
        summary metrics, and aggregation provenance.
    """
    physical = res.physical_units(strict=False)

    processed_attributes: dict[str, float | int] = {}
    categories_dict: dict[str, list[dict[str, Any]]] = {
        'Topography': [],
        'Climate': [],
        'Soils': [],
        'Land Cover': [],
        'Hydrology': [],
        'Anthropogenic': [],
    }

    for key, val in physical.items():
        defn = get_attribute_definition(key)
        if not defn:
            continue

        unit = defn.physical_unit
        if math.isnan(val) or math.isinf(val):
            scaled_val: float | int = val
        elif unit in ('m', 'mm/yr', 'people', 'M m³'):
            scaled_val = round(val, 1) if unit != 'people' else round(val)
        elif unit.startswith('class'):
            scaled_val = round(val)
        else:
            scaled_val = round(val, 3)

        processed_attributes[key] = scaled_val

        categories_dict.setdefault(defn.category, []).append(
            {
                'key': key,
                'name': defn.name,
                'value': scaled_val,
                'unit': defn.physical_unit,
                'description': defn.description,
                'category': defn.category,
            }
        )

    summary = {
        'catchment_id': res.catchment_id,
        'elevation_mean_m': processed_attributes.get('ele_mt_sav', np.nan),
        'slope_mean_deg': processed_attributes.get('slp_dg_sav', np.nan),
        'annual_precip_mm': processed_attributes.get('pre_mm_syr', np.nan),
        'annual_temp_c': processed_attributes.get('tmp_dc_syr', np.nan),
        'aridity_index': processed_attributes.get('ari_ix_sav', np.nan),
        'era5_p_mean_mm_day': processed_attributes.get('p_mean', np.nan),
        'era5_pet_mean_mm_day': processed_attributes.get(
            'pet_mean_ERA5_LAND', np.nan
        ),
        'era5_fao_pet_mean_mm_day': processed_attributes.get(
            'pet_mean_FAO_PM', np.nan
        ),
        'era5_aridity': processed_attributes.get('aridity_ERA5_LAND', np.nan),
        'era5_fao_aridity': processed_attributes.get('aridity_FAO_PM', np.nan),
        'era5_frac_snow_pc': processed_attributes.get('frac_snow', np.nan),
        'forest_fraction_pc': processed_attributes.get('for_pc_sse', np.nan),
        'cropland_fraction_pc': processed_attributes.get('crp_pc_sse', np.nan),
        'urban_fraction_pc': processed_attributes.get('urb_pc_sse', np.nan),
        'dominant_land_cover_class': physical.get('glc_cl_smj', np.nan),
        'soil_clay_pc': processed_attributes.get('cly_pc_sav', np.nan),
        'soil_sand_pc': processed_attributes.get('snd_pc_sav', np.nan),
        'soil_silt_pc': processed_attributes.get('slt_pc_sav', np.nan),
        'groundwater_table_depth_cm': processed_attributes.get(
            'gwt_cm_sav', np.nan
        ),
        'soil_water_content_pc': processed_attributes.get('swc_pc_syr', np.nan),
        'inundation_max_pc': processed_attributes.get('inu_pc_smx', np.nan),
        'total_area_km2': round(res.area_km2, 2),
        'area_fraction_used_for_aggregation': float(
            res.area_fraction_used_for_aggregation
        ),
        'intersected_subbasins': len(res.subbasin_ids),
        'subbasin_ids': list(res.subbasin_ids),
        'subbasin_weights_km2': list(res.subbasin_weights_km2),
    }

    return sanitize_for_json(
        {
            'status': 'success',
            'catchment_id': res.catchment_id,
            'attributes': dict(res.attributes),
            'raw_attributes': dict(res.attributes),
            'flat_attributes': processed_attributes,
            'summary': summary,
            'categories': categories_dict,
            'processed_attributes': processed_attributes,
            'intersected_subbasins_count': len(res.subbasin_ids),
            'subbasin_ids': list(res.subbasin_ids),
            'subbasin_weights_km2': list(res.subbasin_weights_km2),
            'area_km2': round(res.area_km2, 2),
            'total_area_km2': round(res.area_km2, 2),
            'area_fraction_used_for_aggregation': float(
                res.area_fraction_used_for_aggregation
            ),
        }
    )


def build_map_layer_geojson(
    extractor: StaticAttributesExtractor,
    bbox: tuple[float, float, float, float],
    *,
    zoom: int,
    attribute: str,
    level: int | None = None,
) -> dict[str, Any]:
    """Build a GeoJSON `FeatureCollection` of BasinATLAS sub-basins in `bbox`.

    Args:
        extractor: Initialized `StaticAttributesExtractor`.
        bbox: Bounding box `(min_lon, min_lat, max_lon, max_lat)`.
        zoom: Slippy-map zoom level used to select the Pfafstetter level when
            `level` is `None`.
        attribute: Attribute key from `ATTRIBUTE_REGISTRY`.
        level: Optional explicit Pfafstetter level override (`1..12`).

    Returns:
        JSON-safe GeoJSON `FeatureCollection` with physical-unit values and
        summary statistics (`min_value`, `mean_value`, `max_value`).
    """
    avail = extractor.available_levels
    target_level = (
        level if level is not None else zoom_to_hydroatlas_level(zoom, avail)
    )

    defn = get_attribute_definition(attribute)
    if defn and defn.source == 'era5_climate':
        if level is not None and level != _FINEST_HYDROATLAS_LEVEL:
            raise UnsupportedHydroATLASLevelError(
                f"Attribute '{attribute}' (era5_climate) is only available "
                f'at HydroATLAS level {_FINEST_HYDROATLAS_LEVEL}'
            )
        target_level = _FINEST_HYDROATLAS_LEVEL

    label = defn.name if defn else attribute
    unit = defn.physical_unit if defn else ''
    category = defn.category if defn else ''
    desc = defn.description if defn else ''

    cols = ['HYBAS_ID', 'SUB_AREA', 'UP_AREA']
    if (
        attribute in ATTRIBUTE_REGISTRY
        and attribute not in cols
        and defn
        and defn.source == 'hydroatlas'
    ):
        cols.append(attribute)

    gdf = extractor.read_subbasins(bbox, level=target_level, columns=cols)

    truncated = False
    if len(gdf) > MAX_MAP_FEATURES:
        gdf = gdf.head(MAX_MAP_FEATURES).copy()
        truncated = True

    if (
        target_level == _FINEST_HYDROATLAS_LEVEL
        and defn
        and defn.source == 'era5_climate'
        and len(gdf) > 0
        and extractor.era5_loader is not None
    ):
        hybas_ids = [int(h) for h in gdf['HYBAS_ID'].tolist()]
        clim_df = extractor.read_subbasin_climate_indices(hybas_ids)
        gdf = gdf.merge(clim_df, on='HYBAS_ID', how='left')

    features: list[dict[str, Any]] = []
    valid_vals: list[float] = []
    simplify_tol = LEVEL_SIMPLIFY_TOL.get(target_level, 0.0)

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        geom_simplified = (
            geom.simplify(simplify_tol, preserve_topology=True)
            if simplify_tol > 0
            else geom
        )

        hybas_id = int(row.get('HYBAS_ID', 0))
        sub_area = round(float(row.get('SUB_AREA', 0.0)), 2)
        up_area = round(float(row.get('UP_AREA', 0.0)), 2)

        row_attrs = {attribute: row.get(attribute)} if attribute in row else {}
        phys = to_physical_units(row_attrs, strict=False)
        raw_scaled = phys.get(attribute)

        if (
            raw_scaled is not None
            and not math.isnan(raw_scaled)
            and not math.isinf(raw_scaled)
        ):
            scaled_val: float | None = round(float(raw_scaled), 4)
            valid_vals.append(scaled_val)
        else:
            scaled_val = None

        features.append(
            {
                'type': 'Feature',
                'geometry': shapely.geometry.mapping(geom_simplified),
                'properties': {
                    'HYBAS_ID': hybas_id,
                    'SUB_AREA': sub_area,
                    'UP_AREA': up_area,
                    'sub_area_km2': sub_area,
                    'up_area_km2': up_area,
                    'attribute': attribute,
                    attribute: scaled_val,
                    'value': scaled_val,
                    'unit': unit,
                    'level': target_level,
                },
            }
        )

    min_v = round(float(min(valid_vals)), 4) if valid_vals else None
    max_v = round(float(max(valid_vals)), 4) if valid_vals else None
    mean_v = round(float(np.mean(valid_vals)), 4) if valid_vals else None

    props = {
        'attribute': attribute,
        'attribute_key': attribute,
        'label': label,
        'unit': unit,
        'category': category,
        'description': desc,
        'level': target_level,
        'layer': f'BasinATLAS_v10_lev{target_level:02d}',
        'count': len(features),
        'min_value': min_v,
        'mean_value': mean_v,
        'max_value': max_v,
        'truncated': truncated,
    }

    return sanitize_for_json(
        {
            'status': 'success',
            'type': 'FeatureCollection',
            'features': features,
            'properties': props,
        }
    )


def build_schema_payload() -> dict[str, Any]:
    """Build the UI schema payload listing registered HydroATLAS attributes.

    Returns:
        Dictionary containing `attributes` map, `attribute_list`, and
        `extractor_source`.
    """
    attr_list = [
        {
            'key': v.key,
            'name': v.name,
            'label': v.name,
            'category': v.category,
            'unit': v.physical_unit,
            'desc': v.description,
            'description': v.description,
        }
        for v in ATTRIBUTE_REGISTRY.values()
    ]
    attr_dict = {a['key']: a for a in attr_list}
    return {
        'attributes': attr_dict,
        'attribute_list': attr_list,
        'extractor_source': 'multimet.static_extractor',
    }


__all__ = [
    'ATTRIBUTE_REGISTRY',
    'CARAVAN_CLIMATE_ALIASES',
    'DEFAULT_GCS_ERA5_CLIMATE_URI',
    'DEFAULT_GCS_GRIDDED_ERA5_URI',
    'DEFAULT_GCS_HYDROATLAS_URI',
    'LEVEL_SIMPLIFY_TOL',
    'MAX_MAP_FEATURES',
    'ZOOM_TO_LEVEL',
    'CatchmentAttributes',
    'StaticAttributesExtractor',
    'UnsupportedHydroATLASLevelError',
    'build_attribute_card_payload',
    'build_map_layer_geojson',
    'build_schema_payload',
    'card_payload_from_cache_entry',
    'get_attribute_definition',
    'get_attributes_extractor',
    'reset_attributes_extractor',
    'resolve_app_static_extractor_paths',
    'sanitize_for_json',
    'to_physical_units',
    'zoom_to_hydroatlas_level',
]
