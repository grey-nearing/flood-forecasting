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

"""Shared fixtures for the gridded archive builder tests.

Every fixture here produces *synthetic* data on the local filesystem. The unit
and integration suites never touch NOAA PSL, NASA, or GCS, so they are hermetic
and safe to run in CI.

The one exception is ``test_canary.py``, which deliberately talks to those live
services. Those tests are skipped unless ``--run-canary`` is passed; see
:func:`pytest_collection_modifyitems` below.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

_CANARY_FLAG = "--run-canary"


def pytest_addoption(parser: pytest.Parser) -> None:
  """Registers the opt-in flag for live upstream canaries."""
  parser.addoption(
      _CANARY_FLAG,
      action="store_true",
      default=False,
      help=(
          "Run canaries against live third-party feeds (NOAA PSL, NASA CMR). "
          "Requires network access."
      ),
  )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
  """Skips canary tests unless they were explicitly requested."""
  if config.getoption(_CANARY_FLAG):
    return

  skip_canary = pytest.mark.skip(
      reason=f"live upstream canary; pass {_CANARY_FLAG} to run"
  )
  for item in items:
    if "canary" in item.keywords:
      item.add_marker(skip_canary)


PSL_LATS = np.linspace(89.75, -89.75, 360, dtype=np.float32)
PSL_LONS = np.linspace(0.25, 359.75, 720, dtype=np.float32)
PSL_MISSING_VALUE = -9.96921e36


def make_psl_precip_array(
    dates: pd.DatetimeIndex,
    *,
    fill_value: float = 1.0,
    missing_cells: Iterable[tuple[int, int, int]] = (),
) -> np.ndarray:
  """Builds a synthetic NOAA PSL ``precip`` array."""
  data = np.empty((len(dates), len(PSL_LATS), len(PSL_LONS)), dtype=np.float32)
  for i in range(len(dates)):
    data[i] = fill_value + i
  for index in missing_cells:
    data[index] = PSL_MISSING_VALUE
  return data


def write_psl_netcdf(
    directory: Path,
    year: int,
    dates: pd.DatetimeIndex,
    *,
    fill_value: float = 1.0,
    missing_cells: Iterable[tuple[int, int, int]] = (),
) -> Path:
  """Writes a synthetic ``precip.{year}.nc`` file in NOAA PSL layout."""
  data = make_psl_precip_array(
      dates, fill_value=fill_value, missing_cells=missing_cells
  )
  dataset = xr.Dataset(
      data_vars={"precip": (["time", "lat", "lon"], data)},
      coords={"time": dates, "lat": PSL_LATS, "lon": PSL_LONS},
  )
  path = directory / f"precip.{year}.nc"
  dataset.to_netcdf(path)
  dataset.close()
  return path


@pytest.fixture
def psl_cache(tmp_path: Path) -> Path:
  """Directory used as the builder's NetCDF download cache."""
  cache = tmp_path / "psl_cache"
  cache.mkdir()
  return cache


@pytest.fixture
def write_psl_year(psl_cache: Path) -> Callable[..., Path]:
  """Factory that drops a synthetic NOAA PSL year into the download cache."""

  def _write(
      year: int,
      start: str,
      end: str,
      *,
      fill_value: float = 1.0,
      missing_cells: Iterable[tuple[int, int, int]] = (),
  ) -> Path:
    dates = pd.date_range(start, end, freq="D")
    return write_psl_netcdf(
        psl_cache,
        year,
        dates,
        fill_value=fill_value,
        missing_cells=missing_cells,
    )

  return _write
