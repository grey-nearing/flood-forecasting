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

"""Canonical benchmark suite for OpenHydroNet (`flood-forecasting`) components.

Modules:
- `benchmarks.catchment_delineation` (`benchmark-catchment`): Global 90m (`3-arcsec`)
  D8 watershed polygon delineation and pour-point snapping benchmark.
- `benchmarks.static_extractor` (`benchmark-static-extractor`): HydroATLAS Level 12
  and Caravan static watershed attribute extraction benchmark.
- `benchmarks.gridded_archive_builders` (`benchmark-gridded-archive`): Gridded Zarr
  weather archive builder parity benchmark (`CPC`, `IMERG`).
- `benchmarks.timeseries_extractors` (`benchmark-timeseries-extractor`): MultiMet
  catchment-averaged forcing timeseries reconstruction benchmark (`CPC`, `IMERG`, `HRES`).
- `benchmarks.return_periods` (`benchmark-return-periods`): USGS Bulletin 17C
  (`MGBT` and `EMA` LP-III) Caravan benchmark against USGS Fortran `peakfq` & R `MGBT`.
- `benchmarks.model` (`benchmark-model`): Core deep learning forecasting model
  (`MeanEmbeddingForecastLSTM`, `HandoffForecastLSTM`), forcing sensitivity, and
  cold-start vs. hot-start state handoff benchmark.
"""
