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

"""Vector hydrography I/O for HydroRIVERS, HydroBASINS, and MERIT-Basins."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyogrio
import pyogrio.raw
import shapely

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from shapely.geometry.base import BaseGeometry

_MIN_PFAF_STEM_PARTS: int = 3
_BBOX_COORDS_LEN: int = 4
_MIN_COMID_LEN: int = 2


@dataclass(frozen=True)
class Partition:
    """Spatial partition descriptor for a hydrography shapefile."""

    path: Path
    bounds: tuple[float, float, float, float]
    name: str
    pfaf_code: str | None = None


@dataclass(frozen=True)
class Reach:
    """Normalized river network reach record."""

    reach_id: int
    dataset: str
    stream_order: int
    upstream_area_km2: float
    next_down: int
    length_km: float
    geometry: BaseGeometry
    extra: Mapping[str, float | int]


@dataclass(frozen=True)
class UnitCatchment:
    """Normalized unit catchment polygon record."""

    unit_id: int
    next_down: int
    area_km2: float
    geometry: BaseGeometry
    main_basin_id: int | None = None


def _extract_pfaf_code(stem: str) -> str | None:
    """Extract Pfafstetter code from a MERIT-Basins filename stem if present."""
    parts = stem.split('_')
    if len(parts) >= _MIN_PFAF_STEM_PARTS and parts[1].lower() == 'pfaf':
        code = parts[2].replace('MERIT', '').replace('.', '').strip()
        if code.isdigit():
            return code
    return None


def _bboxes_intersect(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    """Return whether two (min_lon, min_lat, max_lon, max_lat) boxes overlap."""
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def discover_partitions(
    directory: str | Path,
    glob_pattern: str = '*.shp',
) -> tuple[Partition, ...]:
    """Discover shapefile partitions and their bounds via pyogrio.read_info."""
    base_dir = Path(directory).expanduser().resolve()
    if not base_dir.is_dir():
        raise FileNotFoundError(
            f'Hydrography directory does not exist: {base_dir}'
        )

    l2_dir = base_dir / 'pfaf_level_02'
    l1_dir = base_dir / 'pfaf_level_01'
    if l2_dir.is_dir() and any(l2_dir.glob(glob_pattern)):
        shp_paths = sorted(l2_dir.glob(glob_pattern))
    elif l1_dir.is_dir() and any(l1_dir.glob(glob_pattern)):
        shp_paths = sorted(l1_dir.glob(glob_pattern))
    else:
        shp_paths = sorted(base_dir.glob(glob_pattern))
        if not shp_paths:
            all_matches = sorted(base_dir.rglob(glob_pattern))
            l2_matches = [
                p for p in all_matches if 'pfaf_level_02' in p.parts
            ]
            shp_paths = l2_matches or all_matches

    partitions: list[Partition] = []
    for shp_path in shp_paths:
        info = pyogrio.read_info(str(shp_path))
        raw_bounds = info.get('total_bounds')
        if raw_bounds is None or len(raw_bounds) != _BBOX_COORDS_LEN:
            continue
        b0, b1, b2, b3 = (
            float(raw_bounds[0]),
            float(raw_bounds[1]),
            float(raw_bounds[2]),
            float(raw_bounds[3]),
        )
        if not (
            math.isfinite(b0)
            and math.isfinite(b1)
            and math.isfinite(b2)
            and math.isfinite(b3)
        ):
            continue
        partitions.append(
            Partition(
                path=shp_path,
                bounds=(b0, b1, b2, b3),
                name=shp_path.name,
                pfaf_code=_extract_pfaf_code(shp_path.stem),
            )
        )
    return tuple(partitions)


def merit_partitions_for_level1(
    partitions: Sequence[Partition],
    pfaf1: int | str,
) -> tuple[Partition, ...]:
    """Return MERIT-Basins partitions sharing a level-1 Pfafstetter digit."""
    prefix = str(pfaf1).strip()[:1]
    if not prefix or not prefix.isdigit():
        raise ValueError(f'Invalid level-1 Pfafstetter digit: {pfaf1!r}')
    matched = tuple(
        p
        for p in partitions
        if p.pfaf_code is not None and p.pfaf_code.startswith(prefix)
    )
    if not matched:
        raise LookupError(
            f'No MERIT-Basins partitions found for level-1 Pfafstetter {prefix}'
        )
    return matched


def merit_partition_for_comid(
    partitions: Sequence[Partition],
    comid: int,
) -> Partition:
    """Locate the MERIT-Basins partition containing a given COMID."""
    comid_str = str(int(comid))
    if len(comid_str) < _MIN_COMID_LEN:
        raise LookupError(f'Invalid MERIT COMID: {comid}')
    pfaf2 = comid_str[:2]
    pfaf1 = comid_str[:1]
    for p in partitions:
        if p.pfaf_code == pfaf2:
            return p
    for p in partitions:
        if p.pfaf_code == pfaf1:
            return p
    raise LookupError(
        f'No MERIT-Basins partition found for COMID {comid} '
        f'(checked Pfafstetter prefixes {pfaf2!r} and {pfaf1!r}).'
    )


def read_hydrorivers(
    shp_path: str | Path,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    min_stream_order: int | None = None,
    min_upstream_area_km2: float | None = None,
) -> list[Reach]:
    """Read HydroRIVERS reaches using vectorized pyogrio.raw.read + shapely."""
    resolved = Path(shp_path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f'HydroRIVERS shapefile does not exist: {resolved}'
        )

    clauses: list[str] = []
    if min_stream_order is not None:
        clauses.append(f'ORD_STRA >= {int(min_stream_order)}')
    if min_upstream_area_km2 is not None:
        clauses.append(f'UPLAND_SKM >= {float(min_upstream_area_km2)}')
    where_clause = ' AND '.join(clauses) if clauses else None

    meta, _, wkb, field_arrays = pyogrio.raw.read(
        str(resolved),
        bbox=bbox,
        where=where_clause,
    )
    if wkb is None or len(wkb) == 0:
        return []

    geoms = shapely.from_wkb(wkb)
    cols = dict(zip(meta['fields'], field_arrays, strict=False))
    n_rows = len(geoms)

    hyriv_ids = cols['HYRIV_ID']
    orders = (
        cols['ORD_STRA']
        if 'ORD_STRA' in cols
        else np.ones(n_rows, dtype=np.int32)
    )
    up_areas = (
        cols['UPLAND_SKM']
        if 'UPLAND_SKM' in cols
        else np.zeros(n_rows, dtype=np.float64)
    )
    next_downs = (
        cols['NEXT_DOWN']
        if 'NEXT_DOWN' in cols
        else np.zeros(n_rows, dtype=np.int64)
    )
    lengths = (
        cols['LENGTH_KM']
        if 'LENGTH_KM' in cols
        else np.zeros(n_rows, dtype=np.float64)
    )
    ord_clas = cols.get('ORD_CLAS')
    hybas_l12 = cols.get('HYBAS_L12')
    catch_skm = cols.get('CATCH_SKM')
    dis_av_cms = cols.get('DIS_AV_CMS')
    dist_dn_km = cols.get('DIST_DN_KM')
    main_riv = cols.get('MAIN_RIV')

    reaches: list[Reach] = []
    for i in range(n_rows):
        geom = geoms[i]
        if geom is None or geom.is_empty:
            continue
        order_val = int(orders[i])
        extra: dict[str, float | int] = {
            'river_class': (
                int(ord_clas[i]) if ord_clas is not None else order_val
            ),
            'hydrobasins_unit': (
                int(hybas_l12[i]) if hybas_l12 is not None else 0
            ),
            'local_catchment_km2': (
                round(float(catch_skm[i]), 2) if catch_skm is not None else 0.0
            ),
            'mean_discharge_m3s': (
                round(float(dis_av_cms[i]), 2)
                if dis_av_cms is not None
                else 0.0
            ),
            'dist_to_ocean_km': (
                round(float(dist_dn_km[i]), 1)
                if dist_dn_km is not None
                else 0.0
            ),
            'main_river': int(main_riv[i]) if main_riv is not None else 0,
        }
        reaches.append(
            Reach(
                reach_id=int(hyriv_ids[i]),
                dataset='hydroatlas',
                stream_order=order_val,
                upstream_area_km2=round(float(up_areas[i]), 1),
                next_down=int(next_downs[i]),
                length_km=round(float(lengths[i]), 2),
                geometry=geom,
                extra=extra,
            )
        )
    return reaches


def read_merit_basins_rivers(
    partitions_or_dir: Sequence[Partition] | str | Path,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    min_stream_order: int | None = None,
    min_upstream_area_km2: float | None = None,
) -> list[Reach]:
    """Read MERIT-Basins river reaches across intersecting partitions."""
    if isinstance(partitions_or_dir, (str, Path)):
        partitions: Sequence[Partition] = discover_partitions(
            partitions_or_dir, 'riv_pfaf_*.shp'
        )
    else:
        partitions = partitions_or_dir

    clauses: list[str] = []
    if min_stream_order is not None:
        clauses.append(f'"order" >= {int(min_stream_order)}')
    if min_upstream_area_km2 is not None:
        clauses.append(f'uparea >= {float(min_upstream_area_km2)}')
    where_clause = ' AND '.join(clauses) if clauses else None

    reaches: list[Reach] = []
    for part in partitions:
        if bbox is not None and not _bboxes_intersect(bbox, part.bounds):
            continue
        if not part.path.is_file():
            raise FileNotFoundError(
                f'MERIT-Basins river partition does not exist: {part.path}'
            )
        meta, _, wkb, field_arrays = pyogrio.raw.read(
            str(part.path),
            bbox=bbox,
            where=where_clause,
        )
        if wkb is None or len(wkb) == 0:
            continue
        geoms = shapely.from_wkb(wkb)
        cols = dict(zip(meta['fields'], field_arrays, strict=False))
        n_rows = len(geoms)

        comids = cols['COMID']
        orders = (
            cols['order']
            if 'order' in cols
            else np.ones(n_rows, dtype=np.int32)
        )
        up_areas = (
            cols['uparea']
            if 'uparea' in cols
            else np.zeros(n_rows, dtype=np.float64)
        )
        next_downs = (
            cols['NextDownID']
            if 'NextDownID' in cols
            else np.zeros(n_rows, dtype=np.int64)
        )
        lengths = (
            cols['lengthkm']
            if 'lengthkm' in cols
            else np.zeros(n_rows, dtype=np.float64)
        )
        sinuosities = cols.get('sinuosity')
        slopes = cols.get('slope')

        for i in range(n_rows):
            geom = geoms[i]
            if geom is None or geom.is_empty:
                continue
            extra: dict[str, float | int] = {
                'sinuosity': (
                    round(float(sinuosities[i]), 2)
                    if sinuosities is not None
                    else 1.0
                ),
                'slope': (
                    round(float(slopes[i]), 4) if slopes is not None else 0.0
                ),
            }
            reaches.append(
                Reach(
                    reach_id=int(comids[i]),
                    dataset='merit-hydro',
                    stream_order=int(orders[i]),
                    upstream_area_km2=round(float(up_areas[i]), 1),
                    next_down=int(next_downs[i]),
                    length_km=round(float(lengths[i]), 2),
                    geometry=geom,
                    extra=extra,
                )
            )
    return reaches
