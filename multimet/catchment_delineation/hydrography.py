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

"""Domain models for vector river networks and unit-catchment topologies."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pyogrio.raw
import shapely
from shapely.geometry import Point

from multimet.catchment_delineation.delineator import CatchmentCoverageError
from multimet.utils.hydrography import (
    Partition,
    Reach,
    UnitCatchment,
    discover_partitions,
    merit_partition_for_comid,
    merit_partitions_for_level1,
    read_hydrorivers,
    read_merit_basins_rivers,
)

_KM_PER_DEGREE: float = 111.0
_METERS_PER_DEGREE: float = 111000.0


class RiverSnapError(ValueError):
    """Raised when no vector river reach is found within the snap tolerance."""


@dataclass(frozen=True)
class ReachSnap:
    """Result of snapping a coordinate pair onto a vector river reach."""

    reach: Reach
    lat: float
    lon: float
    distance_m: float


class RiverNetwork:
    """Vector river network supporting spatial queries and pour-point snapping."""

    def __init__(
        self,
        dataset: str,
        *,
        hydrorivers_shp: Path | None = None,
        merit_partitions: tuple[Partition, ...] | None = None,
    ) -> None:
        self.dataset = dataset
        self._hydrorivers_shp = hydrorivers_shp
        self._merit_partitions = merit_partitions

    @classmethod
    def from_hydrorivers(cls, shp_path: str | Path) -> RiverNetwork:
        """Construct a RiverNetwork backed by a HydroRIVERS shapefile."""
        resolved = Path(shp_path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(
                f'HydroRIVERS shapefile does not exist: {resolved}'
            )
        return cls('hydroatlas', hydrorivers_shp=resolved)

    @classmethod
    def from_merit_basins(cls, merit_basins_dir: str | Path) -> RiverNetwork:
        """Construct a RiverNetwork backed by MERIT-Basins riv_pfaf_* shapefiles."""
        partitions = discover_partitions(merit_basins_dir, 'riv_pfaf_*.shp')
        if not partitions:
            raise FileNotFoundError(
                'No MERIT-Basins river shapefiles (riv_pfaf_*.shp) found in '
                f'{merit_basins_dir}'
            )
        return cls('merit-hydro', merit_partitions=partitions)

    @property
    def partitions(self) -> tuple[Partition, ...]:
        """Return spatial partitions for partitioned networks (MERIT-Basins)."""
        return self._merit_partitions or ()

    def query_reaches(
        self,
        bbox: tuple[float, float, float, float] | None = None,
        *,
        min_stream_order: int | None = None,
        min_upstream_area_km2: float | None = None,
        simplify_tolerance_deg: float | None = None,
    ) -> list[Reach]:
        """Query river reaches intersecting bbox and meeting stream order/area filters."""
        if self.dataset == 'hydroatlas':
            assert self._hydrorivers_shp is not None
            reaches = read_hydrorivers(
                self._hydrorivers_shp,
                bbox=bbox,
                min_stream_order=min_stream_order,
                min_upstream_area_km2=min_upstream_area_km2,
            )
        elif self.dataset == 'merit-hydro':
            assert self._merit_partitions is not None
            reaches = read_merit_basins_rivers(
                self._merit_partitions,
                bbox=bbox,
                min_stream_order=min_stream_order,
                min_upstream_area_km2=min_upstream_area_km2,
            )
        else:
            raise ValueError(f'Unsupported river network dataset: {self.dataset}')

        if simplify_tolerance_deg is not None and simplify_tolerance_deg > 0.0:
            reaches = [
                replace(
                    r, geometry=r.geometry.simplify(simplify_tolerance_deg)
                )
                for r in reaches
            ]
        return reaches

    def snap_to_reach(
        self,
        lat: float,
        lon: float,
        *,
        search_radius_km: float = 5.0,
        max_distance_m: float = 250.0,
        area_weight: float = 0.25,
    ) -> ReachSnap:
        """Snap (lat, lon) onto the nearest river reach or raise RiverSnapError."""
        if not (math.isfinite(lat) and math.isfinite(lon)):
            raise ValueError(
                f'Coordinates ({lat}, {lon}) must be finite numbers.'
            )
        if search_radius_km <= 0.0 or max_distance_m <= 0.0:
            raise ValueError(
                'search_radius_km and max_distance_m must be positive.'
            )

        deg_radius = search_radius_km / _KM_PER_DEGREE
        search_bbox = (
            lon - deg_radius,
            lat - deg_radius,
            lon + deg_radius,
            lat + deg_radius,
        )
        candidates = self.query_reaches(bbox=search_bbox)
        if not candidates:
            raise RiverSnapError(
                f'No river reaches found within {search_radius_km:.2f} km '
                f'search box around ({lat:.4f}, {lon:.4f}).'
            )

        click_pt = Point(lon, lat)
        best_reach: Reach | None = None
        best_proj_pt: Point | None = None
        min_score = float('inf')

        for reach in candidates:
            geom = reach.geometry
            if geom is None or geom.is_empty:
                continue
            dist_deg = float(geom.distance(click_pt))
            score = dist_deg / (
                1.0
                + math.log10(max(1.0, float(reach.upstream_area_km2)))
                * area_weight
            )
            if score < min_score:
                min_score = score
                best_reach = reach
                proj_dist = geom.project(click_pt)
                best_proj_pt = geom.interpolate(proj_dist)

        if best_reach is None or best_proj_pt is None:
            raise RiverSnapError(
                f'No valid river geometry found near ({lat:.4f}, {lon:.4f}).'
            )

        dist_m = float(best_proj_pt.distance(click_pt) * _METERS_PER_DEGREE)
        if dist_m > max_distance_m:
            raise RiverSnapError(
                f'Nearest river reach ({best_reach.reach_id}) is {dist_m:.1f} m '
                f'away from ({lat:.4f}, {lon:.4f}), exceeding '
                f'max_distance_m={max_distance_m:.1f} m.'
            )

        return ReachSnap(
            reach=best_reach,
            lat=float(best_proj_pt.y),
            lon=float(best_proj_pt.x),
            distance_m=dist_m,
        )


@runtime_checkable
class UnitCatchmentLayer(Protocol):
    """Protocol for partitioned vector unit-catchment topologies."""

    dataset: str

    def locate_unit(self, lat: float, lon: float) -> int:
        """Return the unit_id of the polygon containing (lat, lon) or raise LookupError."""
        ...

    def get_unit(self, unit_id: int) -> UnitCatchment:
        """Return a single UnitCatchment by ID or raise LookupError."""
        ...

    def upstream_units(
        self, unit_id: int, *, max_units: int | None = None
    ) -> list[UnitCatchment]:
        """Return all upstream UnitCatchments (including unit_id) or raise on limit."""
        ...


class HydroBasinsLayer:
    """HydroBASINS Level-12 unit-catchment topological layer."""

    dataset: str = 'hydroatlas'

    def __init__(self, hydrobasins_dir: str | Path) -> None:
        self.hydrobasins_dir = Path(hydrobasins_dir).expanduser().resolve()
        self.partitions: tuple[Partition, ...] = discover_partitions(
            self.hydrobasins_dir, '*.shp'
        )
        if not self.partitions:
            raise FileNotFoundError(
                f'No HydroBASINS shapefiles found in {self.hydrobasins_dir}'
            )

    def _read_units_from_partition(
        self,
        part: Partition,
        *,
        bbox: tuple[float, float, float, float] | None = None,
        where: str | None = None,
    ) -> list[UnitCatchment]:
        meta, _, wkb, field_arrays = pyogrio.raw.read(
            str(part.path), bbox=bbox, where=where
        )
        if wkb is None or len(wkb) == 0:
            return []
        geoms = shapely.from_wkb(wkb)
        cols = dict(zip(meta['fields'], field_arrays))
        n_rows = len(geoms)
        hybas_ids = cols['HYBAS_ID']
        next_downs = (
            cols['NEXT_DOWN']
            if 'NEXT_DOWN' in cols
            else np.zeros(n_rows, dtype=np.int64)
        )
        sub_areas = (
            cols['SUB_AREA']
            if 'SUB_AREA' in cols
            else np.zeros(n_rows, dtype=np.float64)
        )
        main_bas = cols.get('MAIN_BAS')

        units: list[UnitCatchment] = []
        for i in range(n_rows):
            geom = geoms[i]
            if geom is None or geom.is_empty:
                continue
            units.append(
                UnitCatchment(
                    unit_id=int(hybas_ids[i]),
                    next_down=int(next_downs[i]),
                    area_km2=float(sub_areas[i]),
                    geometry=geom,
                    main_basin_id=(
                        int(main_bas[i]) if main_bas is not None else None
                    ),
                )
            )
        return units

    def locate_unit(self, lat: float, lon: float) -> int:
        """Locate the HydroBASINS unit polygon containing (lat, lon)."""
        pt = Point(lon, lat)
        for pad in (0.08, 0.5):
            bbox = (lon - pad, lat - pad, lon + pad, lat + pad)
            for part in self.partitions:
                b0, b1, b2, b3 = part.bounds
                if lon < b0 or lon > b2 or lat < b1 or lat > b3:
                    continue
                units = self._read_units_from_partition(part, bbox=bbox)
                for u in units:
                    if u.geometry.contains(pt) or u.geometry.touches(pt):
                        return u.unit_id
        raise LookupError(
            f'No HydroBASINS unit catchment polygon contains ({lat:.4f}, {lon:.4f}).'
        )

    def _find_unit_and_partition(
        self, unit_id: int, *, hint_lat: float | None = None, hint_lon: float | None = None
    ) -> tuple[UnitCatchment, Partition]:
        ordered_parts = list(self.partitions)
        if hint_lat is not None and hint_lon is not None:
            ordered_parts.sort(
                key=lambda p: not (
                    p.bounds[0] <= hint_lon <= p.bounds[2]
                    and p.bounds[1] <= hint_lat <= p.bounds[3]
                )
            )
        where_clause = f'HYBAS_ID = {int(unit_id)}'
        for part in ordered_parts:
            units = self._read_units_from_partition(part, where=where_clause)
            if units:
                return units[0], part
        raise LookupError(f'HydroBASINS unit_id {unit_id} not found.')

    def get_unit(self, unit_id: int) -> UnitCatchment:
        """Return a single HydroBASINS unit catchment by HYBAS_ID."""
        unit, _ = self._find_unit_and_partition(unit_id)
        return unit

    def upstream_units(
        self, unit_id: int, *, max_units: int | None = None
    ) -> list[UnitCatchment]:
        """Traverse all upstream HydroBASINS L12 units within the same MAIN_BAS."""
        outlet_unit, part = self._find_unit_and_partition(unit_id)
        where_clause = (
            f'MAIN_BAS = {outlet_unit.main_basin_id}'
            if outlet_unit.main_basin_id is not None
            else None
        )
        basin_units = self._read_units_from_partition(part, where=where_clause)
        by_id: dict[int, UnitCatchment] = {u.unit_id: u for u in basin_units}
        if outlet_unit.unit_id not in by_id:
            by_id[outlet_unit.unit_id] = outlet_unit

        down_to_up: dict[int, list[int]] = {}
        for u in by_id.values():
            down_to_up.setdefault(u.next_down, []).append(u.unit_id)

        visited: set[int] = set()
        queue: deque[int] = deque([int(unit_id)])
        while queue:
            cur = queue.popleft()
            if cur in visited:
                continue
            if max_units is not None and len(visited) >= max_units:
                raise CatchmentCoverageError(
                    f'HydroBASINS upstream traversal from {unit_id} exceeded '
                    f'max_units={max_units}. Delineation aborted to prevent '
                    'returning a truncated catchment.'
                )
            visited.add(cur)
            for up_id in down_to_up.get(cur, ()):
                if up_id not in visited:
                    queue.append(up_id)

        return [by_id[uid] for uid in visited if uid in by_id]


class MeritBasinsLayer:
    """MERIT-Basins unit-catchment topological layer with cross-partition traversal."""

    dataset: str = 'merit-hydro'

    def __init__(self, merit_basins_dir: str | Path) -> None:
        self.merit_basins_dir = Path(merit_basins_dir).expanduser().resolve()
        self.riv_partitions: tuple[Partition, ...] = discover_partitions(
            self.merit_basins_dir, 'riv_pfaf_*.shp'
        )
        self.cat_partitions: tuple[Partition, ...] = discover_partitions(
            self.merit_basins_dir, 'cat_pfaf_*.shp'
        )
        if not self.riv_partitions or not self.cat_partitions:
            raise FileNotFoundError(
                'MERIT-Basins directory must contain both riv_pfaf_*.shp and '
                f'cat_pfaf_*.shp partitions: {self.merit_basins_dir}'
            )

    def locate_unit(self, lat: float, lon: float) -> int:
        """Locate the MERIT-Basins unit catchment COMID containing (lat, lon)."""
        pt = Point(lon, lat)
        for pad in (0.15, 0.5):
            bbox = (lon - pad, lat - pad, lon + pad, lat + pad)
            for part in self.cat_partitions:
                b0, b1, b2, b3 = part.bounds
                if lon < b0 or lon > b2 or lat < b1 or lat > b3:
                    continue
                meta, _, wkb, field_arrays = pyogrio.raw.read(
                    str(part.path), bbox=bbox
                )
                if wkb is None or len(wkb) == 0:
                    continue
                geoms = shapely.from_wkb(wkb)
                cols = dict(zip(meta['fields'], field_arrays))
                comids = cols['COMID']
                for i, geom in enumerate(geoms):
                    if geom is not None and not geom.is_empty:
                        if geom.contains(pt) or geom.touches(pt):
                            return int(comids[i])
        raise LookupError(
            f'No MERIT-Basins unit catchment polygon contains ({lat:.4f}, {lon:.4f}).'
        )

    def get_unit(self, unit_id: int) -> UnitCatchment:
        """Return a single MERIT-Basins UnitCatchment by COMID."""
        cat_part = merit_partition_for_comid(self.cat_partitions, unit_id)
        meta, _, wkb, field_arrays = pyogrio.raw.read(
            str(cat_part.path), where=f'COMID = {int(unit_id)}'
        )
        if wkb is None or len(wkb) == 0:
            raise LookupError(
                f'MERIT-Basins COMID {unit_id} not found in {cat_part.path.name}'
            )
        geoms = shapely.from_wkb(wkb)
        cols = dict(zip(meta['fields'], field_arrays))
        area_arr = cols.get('unitarea')
        if area_arr is None:
            area_arr = cols.get('uparea', np.zeros(len(geoms), dtype=np.float64))
        return UnitCatchment(
            unit_id=int(unit_id),
            next_down=0,
            area_km2=float(area_arr[0]),
            geometry=geoms[0],
        )

    def upstream_units(
        self, unit_id: int, *, max_units: int | None = None
    ) -> list[UnitCatchment]:
        """Traverse upstream COMIDs across all sibling Pfafstetter partitions."""
        comid_str = str(int(unit_id))
        if len(comid_str) < 2:
            raise LookupError(f'Invalid MERIT-Basins COMID: {unit_id}')
        pfaf2 = int(comid_str[:2])
        # Even Pfafstetter level-2 codes (2, 4, 6, 8) are self-contained tributary
        # basins; odd codes (1, 3, 5, 7, 9) are mainstem interbasins whose upstream
        # network spans sibling level-2 partitions within the level-1 continent.
        if pfaf2 % 2 == 0:
            try:
                riv_parts: tuple[Partition, ...] = (
                    merit_partition_for_comid(self.riv_partitions, unit_id),
                )
            except LookupError:
                riv_parts = merit_partitions_for_level1(
                    self.riv_partitions, comid_str[0]
                )
        else:
            riv_parts = merit_partitions_for_level1(
                self.riv_partitions, comid_str[0]
            )

        down_to_up: dict[int, list[int]] = {}
        next_down_by_id: dict[int, int] = {}
        for r_part in riv_parts:
            meta, _, _, field_arrays = pyogrio.raw.read(
                str(r_part.path),
                columns=['COMID', 'NextDownID'],
                read_geometry=False,
            )
            cols = dict(zip(meta['fields'], field_arrays))
            comids = cols['COMID']
            next_downs = cols['NextDownID']
            for i in range(len(comids)):
                cid = int(comids[i])
                nid = int(next_downs[i])
                next_down_by_id[cid] = nid
                down_to_up.setdefault(nid, []).append(cid)

        if int(unit_id) not in next_down_by_id:
            raise LookupError(
                f'MERIT-Basins COMID {unit_id} not found in river topology.'
            )

        visited: set[int] = set()
        queue: deque[int] = deque([int(unit_id)])
        while queue:
            cur = queue.popleft()
            if cur in visited:
                continue
            if max_units is not None and len(visited) >= max_units:
                raise CatchmentCoverageError(
                    f'MERIT-Basins upstream traversal from {unit_id} exceeded '
                    f'max_units={max_units}. Delineation aborted to prevent '
                    'returning a truncated catchment.'
                )
            visited.add(cur)
            for up_id in down_to_up.get(cur, ()):
                if up_id not in visited:
                    queue.append(up_id)

        # Group visited COMIDs by their target cat_pfaf partition
        part_to_comids: dict[Path, set[int]] = {}
        part_by_path: dict[Path, Partition] = {}
        for cid in visited:
            c_part = merit_partition_for_comid(self.cat_partitions, cid)
            part_by_path[c_part.path] = c_part
            part_to_comids.setdefault(c_part.path, set()).add(cid)

        units: list[UnitCatchment] = []
        for path, target_cids in part_to_comids.items():
            meta, _, wkb, field_arrays = pyogrio.raw.read(str(path))
            if wkb is None or len(wkb) == 0:
                continue
            cols = dict(zip(meta['fields'], field_arrays))
            comids_arr = cols['COMID']
            mask = np.isin(comids_arr, list(target_cids))
            if not np.any(mask):
                continue
            idxs = np.flatnonzero(mask)
            sub_geoms = shapely.from_wkb(wkb[idxs])
            area_col = cols.get('unitarea')
            if area_col is None:
                area_col = cols.get(
                    'uparea', np.zeros(len(comids_arr), dtype=np.float64)
                )
            for j, row_idx in enumerate(idxs):
                geom = sub_geoms[j]
                if geom is None or geom.is_empty:
                    continue
                cid = int(comids_arr[row_idx])
                units.append(
                    UnitCatchment(
                        unit_id=cid,
                        next_down=next_down_by_id.get(cid, 0),
                        area_km2=float(area_col[row_idx]),
                        geometry=geom,
                    )
                )

        if not units:
            raise LookupError(
                f'No MERIT-Basins unit catchment geometries found for COMID {unit_id}.'
            )
        return units
