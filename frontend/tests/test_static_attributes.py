from pathlib import Path
import tempfile
import json
import math
try:
    from absl.testing import absltest
except ImportError:
    import unittest as absltest
import numpy as np
import pandas as pd
import shapely.geometry

from frontend.static_attributes import (
    get_attributes_extractor,
    build_attribute_card_payload,
    build_map_layer_geojson,
    build_schema_payload,
    sanitize_for_json,
    zoom_to_hydroatlas_level
)
from multimet.static_extractor import CatchmentAttributes

class StaticAttributesAdapterTest(absltest.TestCase):

    def test_sanitize_for_json(self):
        self.assertIsNone(sanitize_for_json(np.nan))
        self.assertIsNone(sanitize_for_json(float('inf')))
        self.assertEqual(sanitize_for_json({"a": np.nan, "b": 1.5}), {"a": None, "b": 1.5})

    def test_zoom_to_hydroatlas_level(self):
        avail = [4, 5, 6, 8, 10, 12]
        self.assertEqual(zoom_to_hydroatlas_level(3, avail), 4)
        self.assertEqual(zoom_to_hydroatlas_level(7, avail), 8)
        self.assertEqual(zoom_to_hydroatlas_level(15, avail), 12)

    def test_build_schema_payload(self):
        payload = build_schema_payload()
        self.assertIn("attributes", payload)
        self.assertIn("attribute_list", payload)
        self.assertTrue(len(payload["attribute_list"]) > 10)
        self.assertEqual(payload["extractor_source"], "multimet.static_extractor")
        
    def test_build_attribute_card_payload(self):
        # Create a dummy CatchmentAttributes
        res = CatchmentAttributes(
            catchment_id="test_id",
            attributes={"tmp_dc_syr": 150, "pre_mm_syr": 1200, "ele_mt_sav": 500, "slp_dg_sav": 150, "for_pc_sse": 60, "glc_cl_smj": 2},
            area_km2=100.0,
            area_fraction_used_for_aggregation=1.0,
            subbasin_ids=(123, 456),
            subbasin_weights_km2=(50.0, 50.0),
            min_overlap_threshold_km2=0.0,
            era5_source="hybas"
        )
        payload = build_attribute_card_payload(res)
        
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["catchment_id"], "test_id")
        self.assertEqual(payload["intersected_subbasins_count"], 2)
        
        # Check rounding and scaled units
        self.assertEqual(payload["flat_attributes"]["tmp_dc_syr"], 15.0)
        self.assertEqual(payload["flat_attributes"]["slp_dg_sav"], 15.0)
        self.assertEqual(payload["flat_attributes"]["ele_mt_sav"], 500.0)
        self.assertEqual(payload["flat_attributes"]["glc_cl_smj"], 2)
        
        # Check summary
        summary = payload["summary"]
        self.assertEqual(summary["elevation_mean_m"], 500.0)
        self.assertEqual(summary["total_area_km2"], 100.0)
        self.assertEqual(summary["subbasin_ids"], [123, 456])

if __name__ == "__main__":
    absltest.main()
