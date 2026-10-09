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

"""RGBA colormaps, PNG encoding, and colorbar LUT rendering for Weather Viewer."""

from __future__ import annotations

import struct
from typing import Sequence, Tuple, Union
import zlib

import numpy as np

from multimet.weather_fetcher.config import SUPPORTED_VARIABLES

TEMP_LEVELS: int = 128
PRESSURE_LEVELS: int = 51

RAIN_RATE_CLASSES: Tuple[Tuple[float, Tuple[int, int, int, int]], ...] = (
    (0.1, (125, 211, 252, 150)),
    (0.5, (59, 130, 246, 185)),
    (2.0, (34, 197, 94, 205)),
    (5.0, (234, 179, 8, 220)),
    (10.0, (239, 68, 68, 235)),
    (20.0, (192, 38, 211, 245)),
)

RAIN_ACCUM_CLASSES: Tuple[Tuple[float, Tuple[int, int, int, int]], ...] = (
    (1.0, (125, 211, 252, 140)),
    (5.0, (59, 130, 246, 175)),
    (10.0, (34, 197, 94, 195)),
    (25.0, (234, 179, 8, 215)),
    (50.0, (239, 68, 68, 230)),
    (100.0, (192, 38, 211, 245)),
)


def _png_chunk(tag: bytes, data: bytes) -> bytes:
  return (
      struct.pack("!I", len(data))
      + tag
      + data
      + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)
  )


def make_png_bytes(
    width: int, height: int, raw_pixels: Union[bytes, bytearray]
) -> bytes:
  """Encodes raw RGBA scanlines (with filter byte per row) into PNG bytes."""
  header = b"\x89PNG\r\n\x1a\n"
  ihdr = _png_chunk(
      b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 6, 0, 0, 0)
  )
  idat = _png_chunk(b"IDAT", zlib.compress(bytes(raw_pixels), 1))
  iend = _png_chunk(b"IEND", b"")
  return header + ihdr + idat + iend


def encode_rgba_png(rgba: np.ndarray) -> bytes:
  """Encodes an `(H, W, 4)` uint8 RGBA array as a PNG byte stream."""
  height, width, _ = rgba.shape
  raw = np.zeros((height, 1 + width * 4), dtype=np.uint8)
  raw[:, 1:] = rgba.reshape(height, width * 4)
  return make_png_bytes(width, height, raw.tobytes())


def encode_indexed_png(idx: np.ndarray, palette: np.ndarray) -> bytes:
  """Encodes an 8-bit palette PNG with per-entry alpha (tRNS)."""
  height, width = idx.shape
  raw = np.zeros((height, width + 1), dtype=np.uint8)
  raw[:, 1:] = idx
  return (
      b"\x89PNG\r\n\x1a\n"
      + _png_chunk(
          b"IHDR", struct.pack("!IIBBBBB", width, height, 8, 3, 0, 0, 0)
      )
      + _png_chunk(b"PLTE", palette[:, :3].astype(np.uint8).tobytes())
      + _png_chunk(b"tRNS", palette[:, 3].astype(np.uint8).tobytes())
      + _png_chunk(b"IDAT", zlib.compress(raw.tobytes(), 6))
      + _png_chunk(b"IEND", b"")
  )


def classify_values(
    values: np.ndarray,
    classes: Sequence[Tuple[float, Tuple[int, int, int, int]]],
) -> np.ndarray:
  """Maps 2D scalar values into discrete RGBA color classes."""
  rgba = np.zeros(values.shape + (4,), dtype=np.uint8)
  for lower, color in classes:
    rgba[values >= lower] = color
  return rgba


def colorize_rgba(var_key: str, values: np.ndarray) -> np.ndarray:
  """Maps 2D physical field values to an `(H, W, 4)` RGBA uint8 array."""
  finite = np.isfinite(values)
  v = np.where(finite, values, 0.0)
  if var_key == "precipitation":
    rgba = classify_values(v, RAIN_RATE_CLASSES)
  elif var_key == "accumulated_precip":
    rgba = classify_values(v, RAIN_ACCUM_CLASSES)
  elif var_key == "temperature":
    norm = np.clip((v + 35.0) / 80.0, 0.0, 1.0)
    rgba = np.empty(v.shape + (4,), dtype=np.uint8)
    for channel, centre in ((0, 3), (1, 2), (2, 1)):
      rgba[..., channel] = (
          255 * np.clip(1.5 - np.abs(norm * 4 - centre), 0.0, 1.0)
      ).astype(np.uint8)
    rgba[..., 3] = 180
  elif var_key == "pressure":
    rem = np.abs(np.mod(v, 4.0))
    isobar = (rem < 0.3) | (rem > 3.7)
    norm = np.clip((v - 980.0) / 50.0, 0.0, 1.0)
    rgba = np.empty(v.shape + (4,), dtype=np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = 14, 165, 233
    rgba[..., 3] = (30 + norm * 50).astype(np.uint8)
    rgba[isobar] = (255, 255, 255, 220)
  elif var_key == "wind":
    norm = np.clip(v / 40.0, 0.0, 1.0)
    rgba = np.empty(v.shape + (4,), dtype=np.uint8)
    for channel, centre in ((0, 3), (1, 2), (2, 1)):
      rgba[..., channel] = (
          255 * np.clip(1.5 - np.abs(norm * 4 - centre), 0.0, 1.0)
      ).astype(np.uint8)
    rgba[..., 3] = 180
  else:
    rgba = np.zeros(v.shape + (4,), dtype=np.uint8)
  rgba[~finite] = 0
  return rgba


def _box_smooth(values: np.ndarray) -> np.ndarray:
  padded = np.pad(values, 1, mode="edge")
  h, w = values.shape
  total = np.zeros_like(values, dtype=np.float32)
  for dy in range(3):
    for dx in range(3):
      total += padded[dy : dy + h, dx : dx + w]
  return total / 9.0


def colorize_indexed(
    var_key: str, values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
  """Returns palette indices (uint8) and RGBA palette (index 0 is transparent)."""
  finite = np.isfinite(values)
  v = np.where(finite, values, 0.0).astype(np.float32)
  if var_key in ("precipitation", "accumulated_precip"):
    classes = (
        RAIN_RATE_CLASSES if var_key == "precipitation" else RAIN_ACCUM_CLASSES
    )
    bounds = np.array([lower for lower, _ in classes], dtype=np.float32)
    idx = np.searchsorted(bounds, v, side="right").astype(np.uint8)
    palette = np.array(
        [(0, 0, 0, 0)] + [color for _, color in classes], dtype=np.uint8
    )
  elif var_key == "temperature":
    n = TEMP_LEVELS
    norm = np.clip((v + 35.0) / 80.0, 0.0, 1.0)
    idx = (1 + np.rint(norm * (n - 1))).astype(np.uint8)
    levels = np.linspace(0.0, 1.0, n)
    palette = np.zeros((n + 1, 4), dtype=np.uint8)
    for channel, centre in ((0, 3), (1, 2), (2, 1)):
      palette[1:, channel] = (
          255 * np.clip(1.5 - np.abs(levels * 4 - centre), 0.0, 1.0)
      ).astype(np.uint8)
    palette[1:, 3] = 180
  elif var_key == "pressure":
    n = PRESSURE_LEVELS
    v_press = np.where(finite, values, 1013.25).astype(np.float32)
    smooth = _box_smooth(v_press)
    norm = np.clip((smooth - 980.0) / 50.0, 0.0, 1.0)
    idx = (1 + np.rint(norm * (n - 1))).astype(np.uint8)
    band = np.floor(smooth / 4.0)
    edge = np.zeros(band.shape, dtype=bool)
    edge[:-1, :] |= band[:-1, :] != band[1:, :]
    edge[:, :-1] |= band[:, :-1] != band[:, 1:]
    eroded_finite = finite.copy()
    eroded_finite[:-1, :] &= finite[1:, :]
    eroded_finite[1:, :] &= finite[:-1, :]
    eroded_finite[:, :-1] &= finite[:, 1:]
    eroded_finite[:, 1:] &= finite[:, :-1]
    edge &= eroded_finite
    idx[edge] = n + 1
    levels = np.linspace(0.0, 1.0, n)
    palette = np.zeros((n + 2, 4), dtype=np.uint8)
    palette[1 : n + 1, 0] = 14
    palette[1 : n + 1, 1] = 165
    palette[1 : n + 1, 2] = 233
    palette[1 : n + 1, 3] = (30 + levels * 50).astype(np.uint8)
    palette[n + 1] = (255, 255, 255, 220)
  else:
    idx = np.zeros(v.shape, dtype=np.uint8)
    palette = np.zeros((1, 4), dtype=np.uint8)
  idx[~finite] = 0
  return idx, palette


def render_colorbar_lut_png(
    var_key: str, width: int = 256, height: int = 16
) -> bytes:
  """Renders a horizontal colorbar LUT PNG for a supported weather variable."""
  if var_key not in SUPPORTED_VARIABLES:
    raise ValueError(
        f"Unsupported variable {var_key!r}. Supported: {list(SUPPORTED_VARIABLES)}"
    )
  var_cfg = SUPPORTED_VARIABLES[var_key]
  vmin = float(var_cfg["min"])
  vmax = float(var_cfg["max"])
  ramp = np.linspace(vmin, vmax, width, dtype=np.float32)[None, :]
  grid = np.repeat(ramp, height, axis=0)
  rgba = colorize_rgba(var_key, grid)
  return encode_rgba_png(rgba)
