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
"""High-level WeatherViewer coordinating WeatherDataFetcher with renderers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from frontend.weather_viewer.colormaps import render_colorbar_lut_png
from frontend.weather_viewer.tiles import (
    FRAME_VERSION,
    TILE_VERSION,
    compute_frame_index,
    render_raster_tile,
    render_weather_frame,
)
from frontend.weather_viewer.wind import extract_wind_vectors
from multimet.weather_fetcher.config import SUPPORTED_MODELS
from multimet.weather_fetcher.fetcher import WeatherDataFetcher

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path


class WeatherViewer:
    """Coordinates `WeatherDataFetcher` with frontend tile/frame renderers."""

    def __init__(self, data_dir: str | Path | WeatherDataFetcher) -> None:
        """Initialize the viewer from a data directory or fetcher instance."""
        if isinstance(data_dir, WeatherDataFetcher):
            self.fetcher = data_dir
        else:
            self.fetcher = WeatherDataFetcher(data_dir)

    @property
    def data_dir(self) -> Path:
        """Return the directory containing the loaded weather streams."""
        return self.fetcher.data_dir

    @property
    def arrays(self) -> dict[str, Any]:
        """Return memory-mapped stream arrays."""
        return self.fetcher.arrays

    @property
    def stream_info(self) -> dict[str, dict[str, Any]]:
        """Return stream metadata dictionary."""
        return self.fetcher.stream_info

    def reload_if_changed(self) -> bool:
        """Reload memory-mapped streams if a newer run was swapped in."""
        return self.fetcher.reload_if_changed()

    def get_model_info(self, model_key: str) -> dict[str, Any]:
        """Return metadata and availability status for `model_key`."""
        info = self.fetcher.get_model_info(model_key)
        return {
            **info,
            'tile_version': TILE_VERSION,
            'frame_version': FRAME_VERSION,
        }

    def get_all_models_info(self) -> list[dict[str, Any]]:
        """Return metadata and availability status for all supported models."""
        return [
            {**info, **self.get_model_info(key)}
            for key, info in SUPPORTED_MODELS.items()
        ]

    def get_sync_status(self) -> dict[str, Any]:
        """Return synchronization status from `<data_dir>/sync_status.json`."""
        return self.fetcher.get_sync_status()

    def render_tile(  # noqa: PLR0913, PLR0917
        self,
        model_key: str,
        var_key: str,
        step_idx: int,
        z: int,
        x: int,
        y: int,
        *,
        bilinear: bool = True,
    ) -> bytes:
        """Render a 256x256 Web Mercator PNG tile for the specified layer."""
        arrays, stream_info = self.fetcher.snapshot()
        return render_raster_tile(
            arrays,
            stream_info,
            model_key,
            var_key,
            step_idx,
            z,
            x,
            y,
            bilinear=bilinear,
        )

    def get_frame_index(self, model_key: str, var_key: str) -> dict[str, Any]:
        """Return animation frame index metadata for `model_key`/`var_key`."""
        _, stream_info = self.fetcher.snapshot()
        return compute_frame_index(stream_info, model_key, var_key)

    def render_frame(
        self, model_key: str, var_key: str, step_idx: int
    ) -> bytes:
        """Render a whole-world indexed PNG animation frame."""
        arrays, stream_info = self.fetcher.snapshot()
        return render_weather_frame(
            arrays,
            stream_info,
            model_key,
            var_key,
            step_idx,
        )

    def render_colorbar(
        self, var_key: str, width: int = 256, height: int = 16
    ) -> bytes:
        """Render a horizontal colorbar LUT PNG for `var_key`."""
        return render_colorbar_lut_png(var_key, width=width, height=height)

    def get_wind_vectors(
        self,
        model_key: str = 'ecmwf_ifs',
        step_idx: int = 0,
        subsample: int = 2,
        bbox: tuple[float, float, float, float] | None = None,
        *,
        bilinear: bool = False,
    ) -> dict[str, Any]:
        """Extract subsampled U/V wind vectors for streamline rendering."""
        arrays, stream_info = self.fetcher.snapshot()
        return extract_wind_vectors(
            arrays,
            stream_info,
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
        models: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Extract 10-day multi-model meteogram time series at `(lat, lon)`."""
        return self.fetcher.fetch_point_timeseries(
            lat=lat, lon=lon, models=models
        )

    def summarize_catchment(
        self,
        geojson_feature: Mapping[str, Any],
        step_idx: int = 0,
        model_key: str = 'ecmwf_ifs',
    ) -> dict[str, Any]:
        """Compute catchment-averaged precipitation and temperature summary."""
        return self.fetcher.fetch_catchment_summary(
            geojson_feature=geojson_feature,
            step_idx=step_idx,
            model_key=model_key,
        )


WeatherViewerEngine = WeatherViewer
