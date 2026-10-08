"""Unit and integration tests for the Catchment Delineator Tab, Dedicated Login Tab, and 1:1 DEM <-> River Network Delineation."""

from html.parser import HTMLParser
import json
from pathlib import Path
import threading
import unittest
import urllib.request

from frontend.config import (
    HYDRO_DATASETS,
    STATIC_DIR,
    ensure_flood_forecasting_on_sys_path,
    resolve_dem_tiles_dir,
    resolve_hydro_dataset_id,
)
from frontend.delineator import HydroDelineator, get_dem_delineator
from frontend.profile_manager import get_profile_manager
from frontend.server import (
    EarthkitHydroHandler,
    ThreadingHTTPServer,
)
from multimet.catchment_delineation import DemDelineator
# Importing the tests package points profiles at a temp folder (never real accounts).
from frontend import tests as _isolated_profiles  # pylint: disable=unused-import


class _DOMInspector(HTMLParser):
  """Lightweight HTML parser that records elements by id and tag hierarchy."""

  _VOID_TAGS = {
      "area",
      "base",
      "br",
      "col",
      "embed",
      "hr",
      "img",
      "input",
      "link",
      "meta",
      "param",
      "source",
      "track",
      "wbr",
  }

  def __init__(self):
    super().__init__()
    self.by_id = {}
    self.options_by_select_id = {}
    self.tab_buttons = []
    self._current_select_id = None
    self._current_option_attrs = None
    self._current_option_text = []
    self._stack = []

  def handle_starttag(self, tag, attrs):
    attr_dict = dict(attrs)
    el_id = attr_dict.get("id")
    classes = set((attr_dict.get("class") or "").split())
    ancestor_hidden = any("hidden" in item["classes"] for item in self._stack)
    info = {
        "tag": tag,
        "attrs": attr_dict,
        "classes": classes,
        "hidden_or_ancestor_hidden": ("hidden" in classes) or ancestor_hidden,
    }
    if tag not in self._VOID_TAGS:
      self._stack.append(info)
    if el_id:
      self.by_id[el_id] = info
    if tag == "button" and "data-tab" in attr_dict:
      self.tab_buttons.append(attr_dict)
    if tag == "select" and el_id:
      self._current_select_id = el_id
      self.options_by_select_id.setdefault(el_id, [])
    elif tag == "option" and self._current_select_id:
      self._current_option_attrs = attr_dict
      self._current_option_text = []

  def handle_data(self, data):
    if self._current_option_attrs is not None:
      self._current_option_text.append(data)

  def handle_endtag(self, tag):
    if tag == "option" and self._current_option_attrs is not None:
      text = "".join(self._current_option_text).strip()
      self.options_by_select_id[self._current_select_id].append({
          "attrs": self._current_option_attrs,
          "text": text,
      })
      self._current_option_attrs = None
      self._current_option_text = []
    elif tag == "select":
      self._current_select_id = None
    for i in range(len(self._stack) - 1, -1, -1):
      if self._stack[i]["tag"] == tag:
        del self._stack[i:]
        break


class CatchmentDelineatorWorkplanTest(unittest.TestCase):
  """Tests all 4 tasks of the Catchment Delineator Workplan."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    index_path = STATIC_DIR / "index.html"
    cls.html_text = index_path.read_text(encoding="utf-8")
    cls.dom = _DOMInspector()
    cls.dom.feed(cls.html_text)

    cls.server = ThreadingHTTPServer(("127.0.0.1", 0), EarthkitHydroHandler)
    cls.port = cls.server.server_port
    cls.base_url = f"http://127.0.0.1:{cls.port}"
    cls.server_thread = threading.Thread(
        target=cls.server.serve_forever, daemon=True
    )
    cls.server_thread.start()

  @classmethod
  def tearDownClass(cls):
    cls.server.shutdown()
    cls.server.server_close()
    super().tearDownClass()

  def _get(self, path: str):
    req = urllib.request.Request(f"{self.base_url}{path}", method="GET")
    with urllib.request.urlopen(req, timeout=30) as resp:
      return resp.status, json.loads(resp.read().decode("utf-8"))

  def _post(self, path: str, payload: dict):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{self.base_url}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
      return resp.status, json.loads(resp.read().decode("utf-8"))

  def test_task1_top_of_page_clutter_removed_and_legacy_ids_preserved(self):
    """Verifies <nav id='mainTabBar'> is the top visible element and legacy header IDs are hidden."""
    self.assertIn("mainTabBar", self.dom.by_id)
    self.assertFalse(
        self.dom.by_id["mainTabBar"]["hidden_or_ancestor_hidden"],
        "#mainTabBar must be visible at the top of the viewport",
    )

    # Legacy header IDs must exist in DOM for zero-regression compatibility but be hidden
    for legacy_id in (
        "activeBasinHeaderContainer",
        "riverStatusCount",
        "backgroundJobStatusPill",
        "openArchivesBtn",
        "profileDropdownBtn",
    ):
      self.assertIn(legacy_id, self.dom.by_id, f"Missing #{legacy_id} in DOM")
      self.assertTrue(
          self.dom.by_id[legacy_id]["hidden_or_ancestor_hidden"],
          f"#{legacy_id} should be hidden inside the collapsed header",
      )

    # Top-right map legend overlay (#legendBox) is hidden per user request
    self.assertIn("legendBox", self.dom.by_id)
    self.assertTrue(
        self.dom.by_id["legendBox"]["hidden_or_ancestor_hidden"]
    )

  def test_task2_login_merged_into_account_tab_and_section2_profile_removed(self):
    """Verifies login lives in the single Account tab and Section 2 profile bar is hidden."""
    tab_names = [b.get("data-tab") for b in self.dom.tab_buttons]
    self.assertIn("tab-account", tab_names)
    self.assertNotIn("tab-login", tab_names)
    self.assertIn("accountTabBtn", self.dom.by_id)
    self.assertIn("navAccountBadge", self.dom.by_id)
    self.assertIn("tab-account", self.dom.by_id)
    self.assertNotIn("tab-login", self.dom.by_id)
    self.assertNotIn("loginTabBtn", self.dom.by_id)

    for login_el_id in (
        "loginTabUsernameInput",
        "loginTabEmailInput",
        "loginTabSubmitBtn",
        "loginTabLogoutBtn",
        "loginTabSavedProfilesList",
        "loginTabSaveCurrentBtn",
        "loginTabRefreshArchivesBtn",
        "loginTabArchivesList",
    ):
      self.assertIn(
          login_el_id,
          self.dom.by_id,
          f"Missing #{login_el_id} in #tab-account",
      )

    # Section 2 profile bar inside #tab-delineation must be hidden from visible sidebar
    for s2_id in ("section2ProfileName", "section2LoginBtn"):
      self.assertIn(s2_id, self.dom.by_id)
      self.assertTrue(
          self.dom.by_id[s2_id]["hidden_or_ancestor_hidden"],
          f"#{s2_id} must be hidden inside #tab-delineation",
      )

  def test_task3_mega_simple_left_panel_and_1to1_dem_river_dropdown(self):
    """Verifies the single 1:1 DEM <-> River Network dropdown and hidden legacy selectors."""
    self.assertIn("datasetSelect", self.dom.by_id)
    self.assertFalse(
        self.dom.by_id["datasetSelect"]["hidden_or_ancestor_hidden"]
    )
    options = self.dom.options_by_select_id.get("datasetSelect", [])
    self.assertEqual(len(options), 2)

    values = {opt["attrs"].get("value"): opt for opt in options}
    self.assertIn("hydroatlas", values)
    self.assertIn("merit-hydro", values)
    self.assertEqual(
        values["hydroatlas"]["attrs"].get("data-dem-id"), "hydrosheds_90m"
    )
    self.assertEqual(
        values["merit-hydro"]["attrs"].get("data-dem-id"), "merit_hydro_90m"
    )
    self.assertEqual(values["hydroatlas"]["text"], "HydroRIVERS")
    self.assertEqual(values["merit-hydro"]["text"], "MERIT-Basins")
    for opt in options:
      self.assertNotIn("(", opt["text"])
      self.assertNotIn("—", opt["text"])
      self.assertNotIn("DEM", opt["text"])
    for opt in self.dom.options_by_select_id.get("basemapSelect", []):
      self.assertNotIn("(", opt["text"])

    # Stream order filter and delineation mode selector must be hidden in DOM
    self.assertIn("orderFilterSelect", self.dom.by_id)
    self.assertTrue(
        self.dom.by_id["orderFilterSelect"]["hidden_or_ancestor_hidden"]
    )
    self.assertIn("delinModeSelect", self.dom.by_id)
    self.assertTrue(
        self.dom.by_id["delinModeSelect"]["hidden_or_ancestor_hidden"]
    )

  def test_task4_dem_delineator_integration_and_profile_artifact_sync(self):
    """Verifies catchment_delineation.DemDelineator integration via /api/delineate and 13-folder sync."""
    self.assertTrue(ensure_flood_forecasting_on_sys_path())
    self.assertIsNotNone(DemDelineator)
    self.assertIsInstance(get_dem_delineator("hydrosheds_90m"), DemDelineator)

    # Verify config 1:1 DEM <-> River Network mapping and alias resolution
    self.assertEqual(resolve_hydro_dataset_id("hydrosheds_90m"), "hydroatlas")
    self.assertEqual(resolve_hydro_dataset_id("merit_hydro_90m"), "merit-hydro")
    self.assertEqual(
        HYDRO_DATASETS["hydroatlas"]["dem_id"], "hydrosheds_90m"
    )
    self.assertEqual(
        HYDRO_DATASETS["merit-hydro"]["dem_id"], "merit_hydro_90m"
    )

    # Verify /api/hydro/rivers returns distinct HydroRIVERS vs MERIT-Basins networks across zoom levels
    for z, bbox in [
        (2, "-180.0,-85.0,180.0,85.0"),
        (4, "-130.0,20.0,-60.0,55.0"),
        (7, "-89.0,39.0,-88.0,40.0"),
    ]:
      st_ha, ha_data = self._get(
          f"/api/hydro/rivers?dataset=hydroatlas&bbox={bbox}&zoom={z}"
      )
      st_mh, mh_data = self._get(
          f"/api/hydro/rivers?dataset=merit-hydro&bbox={bbox}&zoom={z}"
      )
      self.assertEqual(st_ha, 200)
      self.assertEqual(st_mh, 200)
      self.assertEqual(ha_data["properties"]["dataset"], "hydroatlas")
      self.assertEqual(mh_data["properties"]["dataset"], "merit-hydro")
      self.assertGreater(len(ha_data["features"]), 0)
      self.assertGreater(len(mh_data["features"]), 0)
      self.assertTrue(
          str(ha_data["features"][0]["properties"]["reach_id"]).startswith(
              "HYRIV_"
          )
      )
      self.assertTrue(
          str(mh_data["features"][0]["properties"]["reach_id"]).startswith(
              "MERIT_"
          )
      )

    # Log into a test profile and call POST /api/delineate (defaults to dem_flow_direction)
    test_user = "test_delineator_tab_user"
    login_status, _ = self._post(
        "/api/profile/login",
        {
            "username": test_user,
            "email": "test_delineator_tab_user@google.com",
            "create_if_missing": True,
        },
    )
    self.assertEqual(login_status, 200)

    try:
      delin_status, feat = self._post(
          "/api/delineate",
          {
              "dataset": "hydroatlas",
              "dem_id": "hydrosheds_90m",
              "latitude": 39.6828,
              "longitude": -88.7729,
              "username": test_user,
          },
      )
      self.assertEqual(delin_status, 200)
      self.assertEqual(feat["type"], "Feature")
      props = feat["properties"]
      self.assertEqual(props["delineation_mode"], "dem_flow_direction")
      self.assertEqual(props["dataset"], "hydroatlas")
      self.assertEqual(props["dem_id"], "hydrosheds_90m")
      self.assertIn("HydroRIVERS", props["river_network"])
      self.assertTrue(props["catchment_id"].startswith("catchment_hydroatlas_"))
      self.assertGreater(props["area_km2"], 10.0)
      self.assertGreater(props["upstream_cells_count"], 1000)
      self.assertIn(feat["geometry"]["type"], ("Polygon", "MultiPolygon"))

      # Also verify MERIT-Hydro 90m D8 flow-direction delineation uses MERIT-Hydro tiles (not HydroSHEDS)
      merit_dir = resolve_dem_tiles_dir("merit_hydro_90m")
      hydro_dir = resolve_dem_tiles_dir("hydrosheds_90m")
      self.assertNotEqual(merit_dir.resolve(), hydro_dir.resolve())
      self.assertIn("merit", str(merit_dir.resolve()).lower())

      m_status, m_feat = self._post(
          "/api/delineate",
          {
              "dataset": "merit-hydro",
              "dem_id": "merit_hydro_90m",
              "latitude": 39.6828,
              "longitude": -88.7729,
              "username": test_user,
          },
      )
      self.assertEqual(m_status, 200)
      m_props = m_feat["properties"]
      self.assertEqual(m_props["delineation_mode"], "dem_flow_direction")
      self.assertEqual(m_props["dataset"], "merit-hydro")
      self.assertEqual(m_props["dem_id"], "merit_hydro_90m")
      self.assertIn("MERIT-Hydro", m_props["delineation_method"])
      self.assertTrue(m_props["catchment_id"].startswith("catchment_merit_hydro_"))
      self.assertGreater(m_props["area_km2"], 10.0)
      self.assertNotEqual(
          m_props["upstream_cells_count"], props["upstream_cells_count"]
      )

      # Verify unknown dataset returns 400 rather than falling back to hydroatlas
      import urllib.error

      with self.assertRaises(urllib.error.HTTPError) as ctx:
        self._post(
            "/api/delineate",
            {
                "dataset": "nonexistent_dataset",
                "latitude": 39.6828,
                "longitude": -88.7729,
                "username": test_user,
            },
        )
      self.assertEqual(ctx.exception.code, 400)

      # Verify 13-folder googlehydrology workspace artifacts were written
      pm = get_profile_manager()
      cid = props["catchment_id"]
      watersheds_path = pm.get_catchments_dir(test_user) / "watersheds.geojson"
      shape_path = pm.get_catchment_shapes_dir(test_user) / f"{cid}.geojson"
      basin_list_path = (
          pm.get_basin_lists_dir(test_user) / "all_basins.txt"
      )
      self.assertTrue(watersheds_path.exists())
      self.assertTrue(shape_path.exists())
      self.assertTrue(basin_list_path.exists())
      basins_txt = basin_list_path.read_text(encoding="utf-8").splitlines()
      self.assertIn(cid, basins_txt)
      self.assertIn(m_props["catchment_id"], basins_txt)
    finally:
      self._post("/api/profile/logout", {})

  def test_issue86_unified_left_sidebar_across_all_tabs(self):
    """Verifies Issue #86 unified left-hand vertical sidebar across all 7 tabs with collapse toggle and drag-to-resize."""
    expected_sidebars = (
        "maasSidebar",
        "delineationSidebar",
        "weatherSidebar",
        "geoFeaturesSidebar",
        "trainingSidebar",
        "forecastingSidebar",
        "accountSidebar",
    )
    for sb_id in expected_sidebars:
      self.assertIn(sb_id, self.dom.by_id, f"Missing #{sb_id} in DOM")
      sb_info = self.dom.by_id[sb_id]
      self.assertEqual(sb_info["tag"], "aside")
      self.assertIn("unified-left-sidebar", sb_info["classes"])

    # Global toggle button in top navbar and floating expand tab on left edge
    self.assertIn("globalSidebarToggleBtn", self.dom.by_id)
    self.assertIn("sidebarExpandFloatingBtn", self.dom.by_id)
    self.assertIn("accountMainContent", self.dom.by_id)

    # Verify CSS custom properties, classes, and JS persistence keys in index.html
    for token in (
        "--sidebar-width: 380px",
        "--sidebar-min-width: 260px",
        "--sidebar-max-width: 640px",
        "body.sidebar-collapsed .unified-left-sidebar",
        ".sidebar-resizer",
        '<div class="sidebar-resizer"',
        'data-action="collapse-sidebar"',
        "openhydronet.sidebarWidth",
        "openhydronet.sidebarCollapsed",
        "initUnifiedSidebar",
        "notifySidebarLayoutChange",
    ):
      self.assertIn(token, self.html_text, f"Missing expected token: {token}")
    self.assertEqual(
        self.html_text.count('<div class="sidebar-resizer"'),
        len(expected_sidebars),
        "Every unified left sidebar must include a drag-to-resize handle",
    )
    self.assertEqual(
        self.html_text.count(
            '<button type="button" class="sidebar-collapse-btn" data-action="collapse-sidebar"'
        ),
        len(expected_sidebars),
        "Every unified left sidebar header must include a collapse button",
    )




if __name__ == "__main__":
  unittest.main()

