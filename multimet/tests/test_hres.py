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
"""Tests for HRES extractor."""

import struct
from unittest import mock

import pytest

from multimet.timeseries_extractors.hres import decode_grib2_message


@pytest.mark.unit
@mock.patch(
    'multimet.timeseries_extractors.hres._has_module', return_value=False
)
def test_decode_grib2_message_invalid_section5(
    mock_has_module: mock.MagicMock,  # noqa: ARG001
) -> None:
    """Verify section 5 template validation."""
    grib_header = b'GRIB' + b'\x00' * 12
    sec5 = struct.pack('>IB', 11, 5) + b'\x00' * 4 + b'\x00\x05'
    end_sec = b'7777'
    raw_bytes = grib_header + sec5 + end_sec
    with pytest.raises(
        ValueError, match='Unsupported GRIB2 Section 5 template: 5'
    ):
        decode_grib2_message(raw_bytes, (1, 1), param='t2m')


@pytest.mark.unit
@mock.patch(
    'multimet.timeseries_extractors.hres._has_module', return_value=False
)
def test_decode_grib2_message_invalid_section6(
    mock_has_module: mock.MagicMock,  # noqa: ARG001
) -> None:
    """Verify section 6 bitmap validation."""
    grib_header = b'GRIB' + b'\x00' * 12
    sec6 = struct.pack('>IB', 6, 6) + b'\x00'
    end_sec = b'7777'
    raw_bytes = grib_header + sec6 + end_sec
    with pytest.raises(
        ValueError, match='Unsupported GRIB2 Section 6 bitmap indicator: 0'
    ):
        decode_grib2_message(raw_bytes, (1, 1), param='t2m')
