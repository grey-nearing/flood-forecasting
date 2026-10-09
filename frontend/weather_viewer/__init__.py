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

"""Frontend Weather Viewer subpackage for tile, frame, colormap, and wind rendering."""

from frontend.weather_viewer.colormaps import (
    classify_values,
    colorize_indexed,
    colorize_rgba,
    encode_indexed_png,
    encode_rgba_png,
    make_png_bytes,
    PRESSURE_LEVELS,
    RAIN_ACCUM_CLASSES,
    RAIN_RATE_CLASSES,
    render_colorbar_lut_png,
    SUPPORTED_VARIABLES,
    TEMP_LEVELS,
)
from frontend.weather_viewer.tiles import (
    clear_frame_cache,
    compute_frame_index,
    empty_frame,
    evaluate_tile_field,
    frame_coordinates,
    FRAME_SIZE,
    FRAME_VARIABLES,
    FRAME_VERSION,
    MERCATOR_MAX_LAT,
    render_raster_tile,
    render_weather_frame,
    tile_coordinates,
    TILE_VERSION,
    transparent_tile,
)
from frontend.weather_viewer.viewer import (
    WeatherViewer,
    WeatherViewerEngine,
)
from frontend.weather_viewer.wind import (
    compute_wind_speed_and_direction,
    extract_wind_vectors,
    format_wind_vectors,
)

__all__ = [
    "FRAME_SIZE",
    "FRAME_VARIABLES",
    "FRAME_VERSION",
    "MERCATOR_MAX_LAT",
    "PRESSURE_LEVELS",
    "RAIN_ACCUM_CLASSES",
    "RAIN_RATE_CLASSES",
    "SUPPORTED_VARIABLES",
    "TEMP_LEVELS",
    "TILE_VERSION",
    "WeatherViewer",
    "WeatherViewerEngine",
    "classify_values",
    "clear_frame_cache",
    "colorize_indexed",
    "colorize_rgba",
    "compute_frame_index",
    "compute_wind_speed_and_direction",
    "empty_frame",
    "encode_indexed_png",
    "encode_rgba_png",
    "evaluate_tile_field",
    "extract_wind_vectors",
    "format_wind_vectors",
    "frame_coordinates",
    "make_png_bytes",
    "render_colorbar_lut_png",
    "render_raster_tile",
    "render_weather_frame",
    "tile_coordinates",
    "transparent_tile",
]
