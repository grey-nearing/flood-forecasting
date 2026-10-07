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

"""High-level WeatherViewer coordinating WeatherDataFetcher with tile/colormap renderers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from frontend.weather_viewer.colormaps import render_colorbar_lut_png
from frontend.weather_viewer.tiles import (
    compute_frame_index,
    FRAME_VERSION,
    render_raster_tile,
    render_weather_frame,
    TILE_VERSION,
)
from frontend.weather_viewer.wind import extract_wind_vectors
from multimet.weather_fetcher.config import SUPPORTED_MODELS
from multimet.weather_fetcher.fetcher import WeatherDataFetcher


class WeatherViewer:
  """Coordinates backend `WeatherDataFetcher` with frontend tile and frame renderers."""

  def __init__(self, data_dir: Union[str, Path, WeatherDataFetcher]):
    if isinstance(data_dir, WeatherDataFetcher):
      self.fetcher = data_dir
    else:
      self.fetcher = WeatherDataFetcher(data_dir)

  @property
  def data_dir(self) -> Path:
    return self.fetcher.data_dir

  @property
  def arrays(self) -> Dict[str, Any]:
    return self.fetcher.arrays

  @property
  def stream_info(self) -> Dict[str, Dict[str, Any]]:
    return self.fetcher.stream_info

  def reload_if_changed(self) -> bool:
    """Reloads memory-mapped streams if a newer run was swapped into data_dir."""
    return self.fetcher.reload_if_changed()

  def get_model_info(self, model_key: str) -> Dict[str, Any]:
    """Returns metadata and availability status for `model_key`."""
    info = self.fetcher.get_model_info(model_key)
    return {
        **info,
        "tile_version": TILE_VERSION,
        "frame_version": FRAME_VERSION,
    }

  def get_all_models_info(self) -> List[Dict[str, Any]]:
    """Returns metadata and availability status for all supported models."""
    return [
        {**info, **self.get_model_info(key)}
        for key, info in SUPPORTED_MODELS.items()
    ]

  def get_sync_status(self) -> Dict[str, Any]:
    """Returns synchronization status from `<data_dir>/sync_status.json`."""
    return self.fetcher.get_sync_status()

  def render_tile(
      self,
      model_key: str,
      var_key: str,
      step_idx: int,
      z: int,
      x: int,
      y: int,
      bilinear: bool = True,
  ) -> bytes:
    """Renders a 256x256 Web Mercator PNG tile for the specified layer."""
    return render_raster_tile(
        self.fetcher.arrays,
        self.fetcher.stream_info,
        model_key,
        var_key,
        step_idx,
        z,
        x,
        y,
        bilinear=bilinear,
    )

  def get_frame_index(self, model_key: str, var_key: str) -> Dict[str, Any]:
    """Returns animation frame index metadata for `model_key` and `var_key`."""
    return compute_frame_index(self.fetcher.stream_info, model_key, var_key)

  def render_frame(
      self, model_key: str, var_key: str, step_idx: int
  ) -> bytes:
    """Renders a whole-world indexed PNG animation frame."""
    return render_weather_frame(
        self.fetcher.arrays,
        self.fetcher.stream_info,
        model_key,
        var_key,
        step_idx,
    )

  def render_colorbar(
      self, var_key: str, width: int = 256, height: int = 16
  ) -> bytes:
    """Renders a horizontal colorbar LUT PNG for `var_key`."""
    return render_colorbar_lut_png(var_key, width=width, height=height)

  def get_wind_vectors(
      self,
      model_key: str = "ecmwf_ifs",
      step_idx: int = 0,
      subsample: int = 2,
      bbox: Optional[Tuple[float, float, float, float]] = None,
      bilinear: bool = False,
  ) -> Dict[str, Any]:
    """Extracts subsampled U/V wind vectors for streamline rendering."""
    return extract_wind_vectors(
        self.fetcher.arrays,
        self.fetcher.stream_info,
        model_key=model_key,
        step_idx=step_idx,
        subsample=subsample,
        bbox=bbox,
        bilinear=bilinear,
    )

  def probe_point(
      self,
      lat: float,
      lon: float,
      models: Optional[Sequence[str]] = None,
  ) -> Dict[str, Any]:
    """Extracts 10-day multi-model meteogram time series at `(lat, lon)`."""
    return self.fetcher.fetch_point_timeseries(lat=lat, lon=lon, models=models)

  def summarize_catchment(
      self,
      geojson_feature: Mapping[str, Any],
      step_idx: int = 0,
      model_key: str = "ecmwf_ifs",
  ) -> Dict[str, Any]:
    """Computes catchment-averaged precipitation and temperature summary."""
    return self.fetcher.fetch_catchment_summary(
        geojson_feature=geojson_feature,
        step_idx=step_idx,
        model_key=model_key,
    )


WeatherViewerEngine = WeatherViewer
