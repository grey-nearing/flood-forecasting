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

"""Offline unit tests for `maas.glofas` reference series and Zarr extraction."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from maas.config import GLOFAS_RES_DEG
from maas.glofas import (
    DEFAULT_GLOFAS_REFERENCE,
    GLOFAS_REFERENCE_FIELDS,
    describe_glofas_reference,
    extract_glofas_zarr_series,
    glofas_reference_field,
    normalize_glofas_reference,
)
from maas.networks import glofas_cell_center

if TYPE_CHECKING:
    from pathlib import Path

_LAT = 38.6270
_LON = -90.1994
_N_DAYS = 4
_N_STEPS = 3
_N_MEMBERS = 2
_DAY_SCALE = 1000.0
_STEP_SCALE = 100.0
_MEMBER_SCALE = 10.0


def _write_glofas_archive(
    path: Path,
    *,
    extra_dims: dict[str, int] | None = None,
    as_netcdf: bool = False,
) -> None:
    """Write a small synthetic GloFAS archive around the LISFLOOD cell.

    The target-cell value is `1000 * day + 100 * step + 10 * number` (using
    `0` for absent dimensions) so tests can verify exactly which slice was
    selected. Neighbouring cells stay `0.0`.
    """
    cell_lat, cell_lon = glofas_cell_center(_LAT, _LON, GLOFAS_RES_DEG)
    lats = cell_lat + GLOFAS_RES_DEG * np.arange(-1, 2)
    lons = cell_lon + GLOFAS_RES_DEG * np.arange(-1, 2)
    times = pd.date_range('2020-01-01', periods=_N_DAYS, freq='D')
    extras = dict(extra_dims or {})
    dims = ['time', *extras, 'latitude', 'longitude']
    shape = [_N_DAYS, *extras.values(), 3, 3]
    data = np.zeros(shape, dtype=np.float32)
    scales = {'step': _STEP_SCALE, 'number': _MEMBER_SCALE}
    for flat_idx in np.ndindex(*shape[:-2]):
        day, *extra_idx = flat_idx
        value = _DAY_SCALE * (day + 1)
        for name, idx in zip(extras, extra_idx, strict=True):
            value += scales.get(name, 0.0) * idx
        data[(*flat_idx, 1, 1)] = value
    coords: dict[str, object] = {
        'time': times,
        'latitude': lats,
        'longitude': lons,
    }
    for name, size in extras.items():
        coords[name] = np.arange(size)
    ds = xr.Dataset({'dis24': (dims, data)}, coords=coords)
    if as_netcdf:
        ds.to_netcdf(path)
    else:
        ds.to_zarr(path, mode='w')


@pytest.mark.unit
class TestGlofasReferenceSeries:
    """Tests the native GloFAS reference-series selection helpers."""

    def test_reference_fields_and_default(self) -> None:
        """Verify the three native series, the default, and normalization."""
        assert GLOFAS_REFERENCE_FIELDS == {
            'control': 'discharge_deterministic',
            'median': 'discharge_median',
            'mean': 'discharge_mean',
        }
        assert DEFAULT_GLOFAS_REFERENCE == 'control'
        assert normalize_glofas_reference(None) == 'control'
        assert normalize_glofas_reference(' Median ') == 'median'
        assert glofas_reference_field('MEAN') == 'discharge_mean'

    def test_describe_reference_lists_all_options(self) -> None:
        """Verify the payload descriptor exposes key, field, label, options."""
        desc = describe_glofas_reference('median')
        assert desc['key'] == 'median'
        assert desc['field'] == 'discharge_median'
        assert desc['label'] == 'Ensemble Median'
        assert [o['key'] for o in desc['options']] == [
            'control',
            'median',
            'mean',
        ]
        assert {o['label'] for o in desc['options']} == {
            'Control (Unperturbed)',
            'Ensemble Median',
            'Ensemble Mean',
        }

    def test_unknown_reference_raises_without_fallback(self) -> None:
        """Verify unknown or non-string references fail loudly."""
        with pytest.raises(ValueError, match='Unsupported GloFAS reference'):
            normalize_glofas_reference('p75')
        with pytest.raises(ValueError, match='Unsupported GloFAS reference'):
            glofas_reference_field('')
        with pytest.raises(TypeError, match='must be a string'):
            normalize_glofas_reference(1)  # type: ignore[arg-type]


@pytest.mark.unit
class TestExtractGlofasZarrSeries:
    """Tests `step` / `number` dimension handling in Zarr point extraction."""

    def test_time_only_archive_extracts_cell_series(
        self, tmp_path: Path
    ) -> None:
        """Verify a plain `(time, lat, lon)` archive yields the cell series."""
        _write_glofas_archive(tmp_path / 'glofas.zarr')
        records = extract_glofas_zarr_series(
            tmp_path / 'glofas.zarr', _LAT, _LON
        )
        assert [r['time'] for r in records] == [
            '2020-01-01',
            '2020-01-02',
            '2020-01-03',
            '2020-01-04',
        ]
        assert [r['discharge'] for r in records] == [
            1000.0,
            2000.0,
            3000.0,
            4000.0,
        ]

    def test_step_dimension_requires_explicit_index(
        self, tmp_path: Path
    ) -> None:
        """Verify a `step` dimension is never selected implicitly."""
        _write_glofas_archive(
            tmp_path / 'fc.zarr', extra_dims={'step': _N_STEPS}
        )
        with pytest.raises(ValueError, match=r'`step` dimension.*`step_index`'):
            extract_glofas_zarr_series(tmp_path / 'fc.zarr', _LAT, _LON)
        records = extract_glofas_zarr_series(
            tmp_path / 'fc.zarr', _LAT, _LON, step_index=2
        )
        assert len(records) == _N_DAYS
        assert [r['discharge'] for r in records] == [
            1200.0,
            2200.0,
            3200.0,
            4200.0,
        ]
        with pytest.raises(ValueError, match='out of range'):
            extract_glofas_zarr_series(
                tmp_path / 'fc.zarr', _LAT, _LON, step_index=_N_STEPS
            )

    def test_number_dimension_requires_explicit_member(
        self, tmp_path: Path
    ) -> None:
        """Verify a `number` dimension is never averaged or picked silently."""
        _write_glofas_archive(
            tmp_path / 'ens.nc',
            extra_dims={'number': _N_MEMBERS},
            as_netcdf=True,
        )
        with pytest.raises(
            ValueError, match=r'`number` dimension.*`member_index`'
        ):
            extract_glofas_zarr_series(tmp_path / 'ens.nc', _LAT, _LON)
        records = extract_glofas_zarr_series(
            tmp_path / 'ens.nc', _LAT, _LON, member_index=1
        )
        assert [r['discharge'] for r in records] == [
            1010.0,
            2010.0,
            3010.0,
            4010.0,
        ]

    def test_step_and_number_select_exact_trajectory(
        self, tmp_path: Path
    ) -> None:
        """Verify one `(step, number)` trajectory is returned, 1-D in time."""
        _write_glofas_archive(
            tmp_path / 'ens_fc.zarr',
            extra_dims={'step': _N_STEPS, 'number': _N_MEMBERS},
        )
        records = extract_glofas_zarr_series(
            tmp_path / 'ens_fc.zarr',
            _LAT,
            _LON,
            step_index=1,
            member_index=1,
        )
        assert len(records) == _N_DAYS
        assert [r['discharge'] for r in records] == [
            1110.0,
            2110.0,
            3110.0,
            4110.0,
        ]

    def test_unexpected_dimension_raises(self, tmp_path: Path) -> None:
        """Verify dimensions other than `time`/`step`/`number` are rejected."""
        _write_glofas_archive(tmp_path / 'odd.zarr', extra_dims={'surface': 1})
        with pytest.raises(
            ValueError, match=r"Unexpected dimensions \('time', 'surface'\)"
        ):
            extract_glofas_zarr_series(tmp_path / 'odd.zarr', _LAT, _LON)

    def test_missing_timestep_stays_nan(self, tmp_path: Path) -> None:
        """Verify a missing timestep is returned as `NaN`, never filled."""
        _write_glofas_archive(tmp_path / 'gap.zarr')
        with xr.open_zarr(tmp_path / 'gap.zarr') as ds:
            patched = ds['dis24'].to_numpy().copy()
            dims = ds['dis24'].dims
            coords = {k: v.to_numpy() for k, v in ds.coords.items()}
        patched[2, 1, 1] = np.nan
        xr.Dataset({'dis24': (dims, patched)}, coords=coords).to_zarr(
            tmp_path / 'gap_nan.zarr', mode='w'
        )
        records = extract_glofas_zarr_series(
            tmp_path / 'gap_nan.zarr', _LAT, _LON
        )
        assert math.isnan(records[2]['discharge'])
        assert records[3]['discharge'] == 4 * _DAY_SCALE
