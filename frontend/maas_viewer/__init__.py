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

"""Frontend visualization subpackage (`frontend.maas_viewer`) for Models-as-a-Service."""

from frontend.maas_viewer.consensus import (
    build_aligned_timeline,
    build_consensus_row,
    build_flood_summary,
    reach_exceedance_summary,
    spread_confidence,
)
from frontend.maas_viewer.inundation import (
    buffer_reach_corridor,
    camaflood_unit_feature,
    chain_length_km,
    channel_half_width_m,
    depth_color,
    emulate_camaflood_physics,
    geojson_feature,
    polygonal_only,
    route_floodplain_excess,
)
from frontend.maas_viewer.viewer import MaaSViewer

__all__ = [
    'MaaSViewer',
    'buffer_reach_corridor',
    'build_aligned_timeline',
    'build_consensus_row',
    'build_flood_summary',
    'camaflood_unit_feature',
    'chain_length_km',
    'channel_half_width_m',
    'depth_color',
    'emulate_camaflood_physics',
    'geojson_feature',
    'polygonal_only',
    'reach_exceedance_summary',
    'route_floodplain_excess',
    'spread_confidence',
]
