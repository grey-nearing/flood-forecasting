import pytest
import struct
from unittest.mock import patch
from multimet.timeseries_extractors.hres import decode_grib2_message

@patch("multimet.timeseries_extractors.hres._has_module", return_value=False)
def test_decode_grib2_message_invalid_section5(mock_has_module):
    grib_header = b"GRIB" + b"\x00"*12
    sec5 = struct.pack(">IB", 11, 5) + b"\x00"*4 + b"\x00\x05"
    end_sec = b"7777"
    raw_bytes = grib_header + sec5 + end_sec
    with pytest.raises(ValueError, match="Unsupported GRIB2 Section 5 template: 5"):
        decode_grib2_message(raw_bytes, (1, 1), param="t2m")

@patch("multimet.timeseries_extractors.hres._has_module", return_value=False)
def test_decode_grib2_message_invalid_section6(mock_has_module):
    grib_header = b"GRIB" + b"\x00"*12
    sec6 = struct.pack(">IB", 6, 6) + b"\x00"
    end_sec = b"7777"
    raw_bytes = grib_header + sec6 + end_sec
    with pytest.raises(ValueError, match="Unsupported GRIB2 Section 6 bitmap indicator: 0"):
        decode_grib2_message(raw_bytes, (1, 1), param="t2m")

