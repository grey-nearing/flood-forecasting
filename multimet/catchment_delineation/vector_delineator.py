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

"""Vector unit-catchment delineation (HydroBASINS Level 12 & MERIT-Basins)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
    mapping,
    shape,
)
from shapely.ops import unary_union

from multimet.utils.geometry import geodesic_area_km2

if TYPE_CHECKING:
    from multimet.catchment_delineation.hydrography import UnitCatchmentLayer

_MIN_REACH_VERTICES: int = 2
_SEGMENT_DISTANCE_EPSILON: float = 1e-6
_NORM_EPSILON: float = 1e-9
_MIN_HALF_PLANE_DIAG_DEG: float = 0.2
_HALF_PLANE_DIAG_MULTIPLIER: float = 4.0


def _extract_polygonal(geom: object) -> Polygon | MultiPolygon:
    """Extract Polygon or MultiPolygon from a Shapely geometry result."""
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    if isinstance(geom, GeometryCollection):
        polys = [
            g
            for g in geom.geoms
            if isinstance(g, (Polygon, MultiPolygon)) and not g.is_empty
        ]
        if not polys:
            return Polygon()
        unioned = unary_union(polys)
        if isinstance(unioned, (Polygon, MultiPolygon)):
            return unioned
    return Polygon()


def _get_target_line(reach_line: object, snapped_pt: Point) -> LineString:
    """Validate and resolve a reach geometry to a single target LineString."""
    if reach_line is None:
        raise ValueError(
            'reach_line is required for exact pour-point clipping.'
        )
    if isinstance(reach_line, dict):
        reach_line = shape(reach_line)
    if isinstance(reach_line, MultiLineString):
        if reach_line.is_empty:
            raise ValueError('reach_line must not be empty.')
        lines = [
            ln
            for ln in reach_line.geoms
            if not ln.is_empty and len(ln.coords) >= _MIN_REACH_VERTICES
        ]
        if not lines:
            raise ValueError(
                'MultiLineString reach_line contains no valid segments.'
            )
        return min(lines, key=lambda ln: ln.distance(snapped_pt))
    if isinstance(reach_line, LineString):
        if reach_line.is_empty:
            raise ValueError('reach_line must not be empty.')
        return reach_line
    raise ValueError(
        'Expected LineString or MultiLineString, '
        f'got {type(reach_line).__name__}.'
    )


def _get_segment_pts(
    target_line: LineString, snapped_pt: Point
) -> tuple[np.ndarray, np.ndarray]:
    """Return start and end 2D vertices of the segment nearest snapped_pt."""
    coords = list(target_line.coords)
    if len(coords) < _MIN_REACH_VERTICES:
        raise ValueError('reach_line must have at least 2 vertices.')
    proj_dist = target_line.project(snapped_pt)
    cum_len = 0.0
    seg_idx = 0
    for i in range(len(coords) - 1):
        p1_pt = Point(coords[i])
        p2_pt = Point(coords[i + 1])
        seg_len = p1_pt.distance(p2_pt)
        if cum_len + seg_len >= proj_dist - _SEGMENT_DISTANCE_EPSILON:
            seg_idx = i
            break
        cum_len += seg_len
    return (
        np.array(coords[seg_idx][:2], dtype=float),
        np.array(coords[seg_idx + 1][:2], dtype=float),
    )


def _create_half_plane(
    p1: np.ndarray,
    p2: np.ndarray,
    snapped_pt: Point,
    bounds: tuple[float, float, float, float],
) -> Polygon:
    """Construct the upstream orthogonal half-plane polygon at snapped_pt."""
    v = p2 - p1
    v_norm_len = float(np.linalg.norm(v))
    if v_norm_len < _NORM_EPSILON:
        raise ValueError('Degenerate zero-length reach segment at pour point.')
    v_norm = v / v_norm_len
    perp = np.array([-v_norm[1], v_norm[0]])
    diag = (
        max(
            _MIN_HALF_PLANE_DIAG_DEG,
            math.hypot(bounds[2] - bounds[0], bounds[3] - bounds[1]),
        )
        * _HALF_PLANE_DIAG_MULTIPLIER
    )
    p0 = np.array([snapped_pt.x, snapped_pt.y], dtype=float)
    return Polygon(
        [
            p0 - diag * perp,
            p0 + diag * perp,
            p0 + diag * perp - diag * v_norm,
            p0 - diag * perp - diag * v_norm,
            p0 - diag * perp,
        ]
    )


def clip_unit_catchment_to_pour_point(
    unit_polygon: Polygon | MultiPolygon,
    reach_line: LineString | MultiLineString | dict[str, object],
    pour_lat: float,
    pour_lon: float,
) -> Polygon | MultiPolygon:
    """Clip a unit catchment polygon to the upstream side of a pour point.

    Constructs an orthogonal cross-section plane at the projection of
    ``(pour_lon, pour_lat)`` onto ``reach_line`` and intersects ``unit_polygon``
    with the upstream half-plane.

    Args:
        unit_polygon: Local unit catchment (`Polygon` or `MultiPolygon`).
        reach_line: River reach centerline (`LineString`, `MultiLineString`,
            or GeoJSON dict).
        pour_lat: Pour-point latitude in decimal degrees.
        pour_lon: Pour-point longitude in decimal degrees.

    Returns:
        Clipped upstream `Polygon` or `MultiPolygon`.

    Raises:
        ValueError: If `unit_polygon` or `reach_line` is empty/degenerate
            or if the clipped intersection is empty.
    """
    if unit_polygon is None or unit_polygon.is_empty:
        raise ValueError(
            'unit_polygon must be a non-empty Polygon or MultiPolygon.'
        )

    snapped_pt = Point(pour_lon, pour_lat)
    target_line = _get_target_line(reach_line, snapped_pt)
    p1, p2 = _get_segment_pts(target_line, snapped_pt)
    half_plane = _create_half_plane(p1, p2, snapped_pt, unit_polygon.bounds)

    clipped = _extract_polygonal(unit_polygon.intersection(half_plane))
    if clipped.is_empty or clipped.area <= 0:
        raise ValueError(
            f'Clipping unit catchment at ({pour_lat:.5f}, {pour_lon:.5f}) '
            'produced an empty geometry.'
        )
    return clipped


@dataclass(frozen=True)
class VectorCatchment:
    """Result of a vector unit-catchment delineation."""

    geometry: Polygon | MultiPolygon
    area_km2: float
    unit_ids: list[int]
    outlet_unit_id: int
    dataset: str
    mode: str

    @property
    def bbox(self) -> dict[str, float]:
        """Return bounding box coordinates rounded to 5 decimal places."""
        bounds = self.geometry.bounds
        return {
            'min_lon': round(float(bounds[0]), 5),
            'min_lat': round(float(bounds[1]), 5),
            'max_lon': round(float(bounds[2]), 5),
            'max_lat': round(float(bounds[3]), 5),
        }

    @property
    def delineation_method(self) -> str:
        """Return human-readable label for the vector delineation mode."""
        if self.dataset == 'merit-hydro':
            if self.mode == 'exact_pour_point':
                return 'MERIT-Basins Exact Pour-Point Drainage Basin'
            if self.mode == 'unit_catchment':
                return 'MERIT-Basins Official Unit Catchment Polygon'
            return 'MERIT-Basins Official Unit Ridgeline Watershed'
        if self.mode == 'exact_pour_point':
            return 'Exact Pour-Point Drainage Basin (On-The-Fly Delineation)'
        if self.mode == 'unit_catchment':
            return 'HydroBASINS Level 12 Unit Catchment Polygon'
        return 'HydroBASINS Level 12 Official Ridgeline Polygon'

    def to_feature(
        self,
        *,
        catchment_id: str | None = None,
        outlet: dict[str, object] | None = None,
        extra_properties: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Serialize the delineated vector catchment to a GeoJSON Feature."""
        cid = catchment_id or f'catchment_{self.dataset}_{self.outlet_unit_id}'
        props: dict[str, object] = {
            'catchment_id': cid,
            'dataset': self.dataset,
            'area_km2': self.area_km2,
            'bbox': self.bbox,
            'outlet_unit_id': self.outlet_unit_id,
            'upstream_reaches_count': len(self.unit_ids),
            'delineation_method': self.delineation_method,
            'delineation_mode': self.mode,
        }
        if outlet is not None:
            props['outlet'] = outlet
        if extra_properties:
            props.update(extra_properties)
        return {
            'type': 'Feature',
            'properties': props,
            'geometry': mapping(self.geometry),
        }


class UnitCatchmentDelineator:
    """Delineates catchments from vector unit-catchment layers."""

    def __init__(self, layer: UnitCatchmentLayer) -> None:
        """Initialize the delineator with a unit-catchment topological layer."""
        self.layer = layer

    def delineate_ridgeline(
        self,
        unit_id: int,
        *,
        max_units: int | None = None,
    ) -> VectorCatchment:
        """Dissolve all unit catchments upstream of `unit_id`."""
        units = self.layer.upstream_units(unit_id, max_units=max_units)
        if not units:
            raise ValueError(f'Unit catchment {unit_id} not found in layer.')
        geoms = [
            u.geometry
            for u in units
            if u.geometry is not None and not u.geometry.is_empty
        ]
        if not geoms:
            raise ValueError(
                f'No valid geometries found upstream of unit {unit_id}.'
            )
        dissolved = _extract_polygonal(unary_union(geoms))
        total_sub_area = sum(float(u.area_km2) for u in units if u.area_km2 > 0)
        if total_sub_area > 0:
            area_km2 = round(total_sub_area, 1)
        else:
            area_km2 = round(float(geodesic_area_km2(dissolved)), 1)
        return VectorCatchment(
            geometry=dissolved,
            area_km2=area_km2,
            unit_ids=[u.unit_id for u in units],
            outlet_unit_id=int(unit_id),
            dataset=self.layer.dataset,
            mode='official_ridgeline',
        )

    def delineate_exact_pour_point(
        self,
        unit_id: int,
        pour_lat: float,
        pour_lon: float,
        reach_geometry: LineString | MultiLineString | dict[str, object],
        *,
        max_units: int | None = None,
    ) -> VectorCatchment:
        """Dissolve upstream tributaries and clip outlet unit at pour point."""
        units = self.layer.upstream_units(unit_id, max_units=max_units)
        if not units:
            raise ValueError(f'Unit catchment {unit_id} not found in layer.')
        outlet_unit = next((u for u in units if u.unit_id == unit_id), None)
        if outlet_unit is None or outlet_unit.geometry is None:
            raise ValueError(
                f'Outlet unit catchment {unit_id} has no geometry.'
            )

        outlet_poly = _extract_polygonal(outlet_unit.geometry)
        clipped_local = clip_unit_catchment_to_pour_point(
            outlet_poly,
            reach_geometry,
            pour_lat,
            pour_lon,
        )
        trib_geoms = [
            u.geometry
            for u in units
            if u.unit_id != unit_id
            and u.geometry is not None
            and not u.geometry.is_empty
        ]
        dissolved = _extract_polygonal(
            unary_union([*trib_geoms, clipped_local])
        )
        area_km2 = round(float(geodesic_area_km2(dissolved)), 1)
        return VectorCatchment(
            geometry=dissolved,
            area_km2=area_km2,
            unit_ids=[u.unit_id for u in units],
            outlet_unit_id=int(unit_id),
            dataset=self.layer.dataset,
            mode='exact_pour_point',
        )

    def delineate_unit_catchment(self, unit_id: int) -> VectorCatchment:
        """Return the single local unit catchment polygon for `unit_id`."""
        unit = self.layer.get_unit(unit_id)
        if unit is None or unit.geometry is None or unit.geometry.is_empty:
            raise ValueError(f'Unit catchment {unit_id} not found in layer.')
        poly = _extract_polygonal(unit.geometry)
        area_km2 = (
            round(float(unit.area_km2), 1)
            if unit.area_km2 > 0
            else round(float(geodesic_area_km2(poly)), 1)
        )
        return VectorCatchment(
            geometry=poly,
            area_km2=area_km2,
            unit_ids=[unit.unit_id],
            outlet_unit_id=unit.unit_id,
            dataset=self.layer.dataset,
            mode='unit_catchment',
        )
