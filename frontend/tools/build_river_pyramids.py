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

"""CLI utility to build river network pyramids for the web UI and MaaS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from shapely.geometry import mapping

from frontend.river_indexer import reach_to_ui_feature
from maas.tools.build_network_pyramids import build_all_pyramids
from multimet.catchment_delineation.hydrography import RiverNetwork

if TYPE_CHECKING:
    from collections.abc import Sequence

_HYDROATLAS_Z2_MIN_ORDER: int = 8
_HYDROATLAS_Z4_MIN_ORDER: int = 7
_MERIT_Z2_MIN_ORDER: int = 7
_MERIT_Z4_MIN_ORDER: int = 6
_Z2_SIMPLIFY_TOL_DEG: float = 0.02
_Z4_SIMPLIFY_TOL_DEG: float = 0.008


def build_pyramid_geojson(
    network: RiverNetwork,
    *,
    min_stream_order: int,
    simplify_tolerance_deg: float,
) -> dict[str, object]:
    """Extract reaches with `stream_order >= min_stream_order` and simplify."""
    reaches = network.query_reaches(
        bbox=(-180.0, -85.0, 180.0, 85.0),
        min_stream_order=min_stream_order,
    )
    features = []
    for reach in reaches:
        feat = reach_to_ui_feature(reach)
        if simplify_tolerance_deg > 0 and reach.geometry is not None:
            simp = reach.geometry.simplify(
                simplify_tolerance_deg, preserve_topology=True
            )
            if not simp.is_empty:
                feat['geometry'] = mapping(simp)
        features.append(feat)
    return {'type': 'FeatureCollection', 'features': features}


def main(argv: Sequence[str] | None = None) -> int:
    """Build UI GeoJSON and MaaS `.npz` river pyramids."""
    parser = argparse.ArgumentParser(
        description=(
            'Build z2/z4 GeoJSON and MaaS .npz river pyramids for OpenHydroNet.'
        ),
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Explicit output directory for pyramid files.',
    )
    parser.add_argument(
        '--river-networks-dir',
        type=Path,
        default=None,
        help='Optional directory containing glofas_v4/ and geoglows_v2/.',
    )
    parser.add_argument(
        '--models',
        type=str,
        default='glofas,geoglows',
        help='Comma-separated MaaS models when --river-networks-dir is set.',
    )
    parser.add_argument(
        '--hydrorivers-shp',
        type=Path,
        default=None,
        help='Optional path to HydroRIVERS_v10.shp.',
    )
    parser.add_argument(
        '--merit-rivers-dir',
        type=Path,
        default=None,
        help='Optional path to MERIT-Basins river shapefiles directory.',
    )
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.river_networks_dir is not None:
        models = [m.strip() for m in args.models.split(',') if m.strip()]
        build_all_pyramids(
            river_networks_dir=args.river_networks_dir,
            output_dir=args.output_dir,
            models=models,
        )

    if args.hydrorivers_shp is not None:
        net = RiverNetwork.from_hydrorivers(args.hydrorivers_shp)
        z2 = build_pyramid_geojson(
            net,
            min_stream_order=_HYDROATLAS_Z2_MIN_ORDER,
            simplify_tolerance_deg=_Z2_SIMPLIFY_TOL_DEG,
        )
        z4 = build_pyramid_geojson(
            net,
            min_stream_order=_HYDROATLAS_Z4_MIN_ORDER,
            simplify_tolerance_deg=_Z4_SIMPLIFY_TOL_DEG,
        )
        (args.output_dir / 'global_rivers_fast_z2.geojson').write_text(
            json.dumps(z2), encoding='utf-8'
        )
        (args.output_dir / 'global_rivers_fast_z4.geojson').write_text(
            json.dumps(z4), encoding='utf-8'
        )

    if args.merit_rivers_dir is not None:
        net = RiverNetwork.from_merit_basins(args.merit_rivers_dir)
        z2 = build_pyramid_geojson(
            net,
            min_stream_order=_MERIT_Z2_MIN_ORDER,
            simplify_tolerance_deg=_Z2_SIMPLIFY_TOL_DEG,
        )
        z4 = build_pyramid_geojson(
            net,
            min_stream_order=_MERIT_Z4_MIN_ORDER,
            simplify_tolerance_deg=_Z4_SIMPLIFY_TOL_DEG,
        )
        (args.output_dir / 'merit_rivers_fast_z2.geojson').write_text(
            json.dumps(z2), encoding='utf-8'
        )
        (args.output_dir / 'merit_rivers_fast_z4.geojson').write_text(
            json.dumps(z4), encoding='utf-8'
        )

    return 0


if __name__ == '__main__':
    sys.exit(main())
