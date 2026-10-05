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

"""Canaries for the upstream feeds the gridded archive builders depend on."""

from __future__ import annotations

import datetime
from pathlib import Path
import urllib.request

from multimet import build_cpc_archive as cpc_module
from multimet import build_imerg_archive as imerg_module
import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.canary

MIN_FINITE_FRACTION = 0.05


def _assert_cpc_field_is_live(values: np.ndarray) -> None:
  """Asserts an extracted CPC precipitation field contains plausible data."""
  finite = float(np.isfinite(values).mean())
  assert finite >= MIN_FINITE_FRACTION, (
      f"cpc_precipitation: only {finite:.1%} of cells are finite. The upstream "
      "source is probably returning nothing."
  )
  finite_values = values[np.isfinite(values)]
  observed_low = float(finite_values.min())
  observed_high = float(finite_values.max())
  assert 0.0 <= observed_low and observed_high <= 2000.0, (
      f"cpc_precipitation: observed range [{observed_low:.4g}, "
      f"{observed_high:.4g}] falls outside plausible bounds [0, 2000] mm/day."
  )


class TestNoaaPslCpc:
  """Canaries for the NOAA PSL CPC precipitation feed."""

  def test_yearly_file_is_published(self) -> None:
    year = datetime.date.today().year - 1
    url = cpc_module.NOAA_PSL_URL_TEMPLATE.format(year=year)
    request = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": "OpenMultiMet/1.1"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
      assert response.status == 200
      size = int(response.headers.get("Content-Length", 0))

    assert size > 1024 * 1024, f"{url} is only {size} bytes."

  def test_yearly_netcdf_still_matches_expected_layout(
      self, tmp_path: Path
  ) -> None:
    year = datetime.date.today().year - 1
    path = cpc_module.ensure_psl_cpc_netcdf(year, cache_dir=str(tmp_path))

    dataset = cpc_module.process_cpc_netcdf_to_dataset(
        path,
        target_start_date=pd.Timestamp(f"{year}-06-01"),
        target_end_date=pd.Timestamp(f"{year}-06-03"),
    )
    assert dataset is not None, "no dates survived the filter"

    assert cpc_module.CPC_VARIABLE in dataset
    assert dataset.sizes["latitude"] == len(cpc_module.CPC_LATS)
    assert dataset.sizes["longitude"] == len(cpc_module.CPC_LONS)

    latitudes = dataset["latitude"].values
    longitudes = dataset["longitude"].values
    assert np.all(np.diff(latitudes) > 0), "latitude is no longer ascending"
    assert longitudes.min() >= -180.0 and longitudes.max() < 180.0

    _assert_cpc_field_is_live(dataset[cpc_module.CPC_VARIABLE].values)


class TestNasaCmrImerg:
  """Canaries for the NASA Earthdata CMR IMERG V07 granule discovery feed."""

  def test_half_hourly_cmr_returns_48_granules(self) -> None:
    urls = imerg_module.query_cmr_granules(
        imerg_module.IMERG_HHR_SHORT_NAME, pd.Timestamp("2024-06-01")
    )
    assert len(urls) == 48
    assert all(url.endswith((".RT-H5", ".HDF5")) for url in urls)

  def test_daily_cmr_returns_granule(self) -> None:
    urls = imerg_module.query_cmr_granules(
        imerg_module.IMERG_DAILY_SHORT_NAME, pd.Timestamp("2024-06-01")
    )
    assert len(urls) == 1
    assert urls[0].endswith(".nc4")
