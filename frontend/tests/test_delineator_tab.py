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

"""Hermetic tests for Catchment Delineator Tab, Account Tab, and DEM routing."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, ClassVar

import pytest

import frontend.config as _frontend_config
import frontend.server as _server
from frontend import tests as _isolated_profiles  # noqa: F401
from frontend.config import (
    HYDRO_DATASETS,
    STATIC_DIR,
    ensure_flood_forecasting_on_sys_path,
    resolve_dem_tiles_dir,
    resolve_hydro_dataset_id,
)
from frontend.delineator import get_dem_delineator, reset_delineator_caches
from frontend.profile_manager import get_profile_manager
from frontend.river_indexer import reset_river_indexer_caches
from frontend.server import (
    EarthkitHydroHandler,
    ThreadingHTTPServer,
)
from frontend.tests.test_delineator import _build_hermetic_hydro_fixtures
from multimet.catchment_delineation import DemDelineator

_HTTP_OK: int = 200
_HTTP_BAD_REQUEST: int = 400
_EXPECTED_DATASET_OPTIONS: int = 2
_MIN_DEM_AREA_KM2: float = 10.0
_MIN_UPSTREAM_CELLS: int = 1000


def _write_fast_river_geojson_caches(river_dir: Path) -> None:
    """Write synthetic z2 and z4 fast GeoJSON caches in `river_dir`."""
    ha_fc = {
        'type': 'FeatureCollection',
        'properties': {'dataset': 'hydroatlas'},
        'features': [
            {
                'type': 'Feature',
                'geometry': {
                    'type': 'LineString',
                    'coordinates': [[-88.7729, 39.72], [-88.7729, 39.64]],
                },
                'properties': {
                    'reach_id': 'HYRIV_70000001',
                    'dataset': 'hydroatlas',
                    'stream_order': 6,
                    'upstream_area_km2': 12.1,
                    'length_km': 8.9,
                    'next_down': 0,
                },
            }
        ],
    }
    mh_fc = {
        'type': 'FeatureCollection',
        'properties': {'dataset': 'merit-hydro'},
        'features': [
            {
                'type': 'Feature',
                'geometry': {
                    'type': 'LineString',
                    'coordinates': [[-88.7729, 39.72], [-88.7729, 39.64]],
                },
                'properties': {
                    'reach_id': 'MERIT_74000002',
                    'dataset': 'merit-hydro',
                    'stream_order': 6,
                    'upstream_area_km2': 13.5,
                    'length_km': 8.9,
                    'next_down': 0,
                },
            }
        ],
    }
    for name in (
        'global_rivers_fast_z2.geojson',
        'global_rivers_fast_z4.geojson',
    ):
        (river_dir / name).write_text(json.dumps(ha_fc), encoding='utf-8')
    for name in (
        'merit_rivers_fast_z2.geojson',
        'merit_rivers_fast_z4.geojson',
    ):
        (river_dir / name).write_text(json.dumps(mh_fc), encoding='utf-8')


class _DOMInspector(HTMLParser):
    """Lightweight HTML parser that records elements by id and tag hierarchy."""

    _VOID_TAGS: ClassVar[set[str]] = {
        'area',
        'base',
        'br',
        'col',
        'embed',
        'hr',
        'img',
        'input',
        'link',
        'meta',
        'param',
        'source',
        'track',
        'wbr',
    }

    def __init__(self) -> None:
        """Initialize empty DOM inspection tables."""
        super().__init__()
        self.by_id: dict[str, dict[str, Any]] = {}
        self.options_by_select_id: dict[str, list[dict[str, Any]]] = {}
        self.tab_buttons: list[dict[str, str | None]] = []
        self._current_select_id: str | None = None
        self._current_option_attrs: dict[str, str | None] | None = None
        self._current_option_text: list[str] = []
        self._stack: list[dict[str, Any]] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        """Record element attributes and visibility state."""
        attr_dict = dict(attrs)
        el_id = attr_dict.get('id')
        classes = set((attr_dict.get('class') or '').split())
        ancestor_hidden = any(
            'hidden' in item['classes'] for item in self._stack
        )
        info: dict[str, Any] = {
            'tag': tag,
            'attrs': attr_dict,
            'classes': classes,
            'hidden_or_ancestor_hidden': (
                ('hidden' in classes) or ancestor_hidden
            ),
        }
        if tag not in self._VOID_TAGS:
            self._stack.append(info)
        if el_id:
            self.by_id[el_id] = info
        if tag == 'button' and 'data-tab' in attr_dict:
            self.tab_buttons.append(attr_dict)
        if tag == 'select' and el_id:
            self._current_select_id = el_id
            self.options_by_select_id.setdefault(el_id, [])
        elif tag == 'option' and self._current_select_id:
            self._current_option_attrs = attr_dict
            self._current_option_text = []

    def handle_data(self, data: str) -> None:
        """Accumulate text inside `<option>` tags."""
        if self._current_option_attrs is not None:
            self._current_option_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        """Close open `<option>`, `<select>`, and stacked tags."""
        if (
            tag == 'option'
            and self._current_option_attrs is not None
            and self._current_select_id is not None
        ):
            text = ''.join(self._current_option_text).strip()
            self.options_by_select_id[self._current_select_id].append(
                {
                    'attrs': self._current_option_attrs,
                    'text': text,
                }
            )
            self._current_option_attrs = None
            self._current_option_text = []
        elif tag == 'select':
            self._current_select_id = None
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i]['tag'] == tag:
                del self._stack[i:]
                break


class CatchmentDelineatorWorkplanTest(unittest.TestCase):
    """Tests all 4 tasks of the Catchment Delineator Workplan."""

    _tmp_dir: tempfile.TemporaryDirectory[str]
    _orig_dem_locs: dict[str, tuple[Path, ...]]
    _orig_river_dir: Path
    _orig_hb_dir: Path
    _orig_cache_dir: Path
    html_text: str
    dom: _DOMInspector
    server: ThreadingHTTPServer
    port: int
    base_url: str
    server_thread: threading.Thread

    @classmethod
    def setUpClass(cls) -> None:
        """Provision hermetic hydrography fixtures and start HTTP server."""
        super().setUpClass()
        cls._tmp_dir = tempfile.TemporaryDirectory()
        root = Path(cls._tmp_dir.name)
        _build_hermetic_hydro_fixtures(root)
        _write_fast_river_geojson_caches(root / 'river_networks')

        cls._orig_dem_locs = dict(_frontend_config.DEM_TILE_LOCATIONS)
        cls._orig_river_dir = _frontend_config.RIVER_NETWORKS_DIR
        cls._orig_hb_dir = _frontend_config.HYDRO_BASINS_DIR
        cls._orig_cache_dir = _frontend_config.CACHE_DIR

        _frontend_config.DEM_TILE_LOCATIONS['hydrosheds_90m'] = (
            root / 'dem' / 'hydrosheds_90m' / 'tiles_5deg',
        )
        _frontend_config.DEM_TILE_LOCATIONS['merit_hydro_90m'] = (
            root / 'dem' / 'merit_hydro_90m' / 'tiles_5deg',
        )
        _frontend_config.RIVER_NETWORKS_DIR = root / 'river_networks'
        _frontend_config.HYDRO_BASINS_DIR = root / 'hydro_basins'
        _frontend_config.CACHE_DIR = root / 'cache'
        reset_delineator_caches()
        reset_river_indexer_caches()
        _server._river_indexers.clear()  # noqa: SLF001
        _server._delineators.clear()  # noqa: SLF001

        index_path = STATIC_DIR / 'index.html'
        cls.html_text = index_path.read_text(encoding='utf-8')
        cls.dom = _DOMInspector()
        cls.dom.feed(cls.html_text)

        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), EarthkitHydroHandler)
        cls.port = cls.server.server_port
        cls.base_url = f'http://127.0.0.1:{cls.port}'
        cls.server_thread = threading.Thread(
            target=cls.server.serve_forever, daemon=True
        )
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        """Shut down test HTTP server and restore `frontend.config` paths."""
        cls.server.shutdown()
        cls.server.server_close()
        _frontend_config.DEM_TILE_LOCATIONS.clear()
        _frontend_config.DEM_TILE_LOCATIONS.update(cls._orig_dem_locs)
        _frontend_config.RIVER_NETWORKS_DIR = cls._orig_river_dir
        _frontend_config.HYDRO_BASINS_DIR = cls._orig_hb_dir
        _frontend_config.CACHE_DIR = cls._orig_cache_dir
        reset_delineator_caches()
        reset_river_indexer_caches()
        _server._river_indexers.clear()  # noqa: SLF001
        _server._delineators.clear()  # noqa: SLF001
        cls._tmp_dir.cleanup()
        super().tearDownClass()

    def _get(self, path: str) -> tuple[int, Any]:
        """Issue a GET request to the test server and decode JSON."""
        req = urllib.request.Request(  # noqa: S310
            f'{self.base_url}{path}', method='GET'
        )
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            return resp.status, json.loads(resp.read().decode('utf-8'))

    def _post(self, path: str, payload: dict[str, Any]) -> tuple[int, Any]:
        """Issue a POST request to the test server and decode JSON."""
        body = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(  # noqa: S310
            f'{self.base_url}{path}',
            data=body,
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
            return resp.status, json.loads(resp.read().decode('utf-8'))

    def test_task1_top_of_page_clutter_removed_and_legacy_ids_preserved(
        self,
    ) -> None:
        """Verify `#mainTabBar` is top visible element and header is hidden."""
        assert 'mainTabBar' in self.dom.by_id
        assert not self.dom.by_id['mainTabBar']['hidden_or_ancestor_hidden'], (
            '#mainTabBar must be visible at the top of the viewport'
        )

        for legacy_id in (
            'activeBasinHeaderContainer',
            'riverStatusCount',
            'backgroundJobStatusPill',
            'openArchivesBtn',
            'profileDropdownBtn',
        ):
            assert legacy_id in self.dom.by_id, f'Missing #{legacy_id} in DOM'
            assert self.dom.by_id[legacy_id]['hidden_or_ancestor_hidden'], (
                f'#{legacy_id} should be hidden inside the collapsed header'
            )

        assert 'legendBox' in self.dom.by_id
        assert self.dom.by_id['legendBox']['hidden_or_ancestor_hidden']

    def test_task2_login_merged_into_account_tab_and_section2_profile_removed(
        self,
    ) -> None:
        """Verify login lives in Account tab and Section 2 bar is hidden."""
        tab_names = [b.get('data-tab') for b in self.dom.tab_buttons]
        assert 'tab-account' in tab_names
        assert 'tab-login' not in tab_names
        assert 'accountTabBtn' in self.dom.by_id
        assert 'navAccountBadge' in self.dom.by_id
        assert 'tab-account' in self.dom.by_id
        assert 'tab-login' not in self.dom.by_id
        assert 'loginTabBtn' not in self.dom.by_id

        for login_el_id in (
            'loginTabUsernameInput',
            'loginTabEmailInput',
            'loginTabSubmitBtn',
            'loginTabLogoutBtn',
            'loginTabSavedProfilesList',
            'loginTabSaveCurrentBtn',
            'loginTabRefreshArchivesBtn',
            'loginTabArchivesList',
        ):
            assert login_el_id in self.dom.by_id, (
                f'Missing #{login_el_id} in #tab-account'
            )

        for s2_id in ('section2ProfileName', 'section2LoginBtn'):
            assert s2_id in self.dom.by_id
            assert self.dom.by_id[s2_id]['hidden_or_ancestor_hidden'], (
                f'#{s2_id} must be hidden inside #tab-delineation'
            )

    def test_task3_mega_simple_left_panel_and_1to1_dem_river_dropdown(
        self,
    ) -> None:
        """Verify 1:1 DEM <-> River Network dropdown and hidden selectors."""
        assert 'datasetSelect' in self.dom.by_id
        assert not self.dom.by_id['datasetSelect']['hidden_or_ancestor_hidden']
        options = self.dom.options_by_select_id.get('datasetSelect', [])
        assert len(options) == _EXPECTED_DATASET_OPTIONS

        values = {opt['attrs'].get('value'): opt for opt in options}
        assert 'hydroatlas' in values
        assert 'merit-hydro' in values
        assert (
            values['hydroatlas']['attrs'].get('data-dem-id') == 'hydrosheds_90m'
        )
        assert (
            values['merit-hydro']['attrs'].get('data-dem-id')
            == 'merit_hydro_90m'
        )
        assert values['hydroatlas']['text'] == 'HydroRIVERS'
        assert values['merit-hydro']['text'] == 'MERIT-Basins'
        for opt in options:
            assert '(' not in opt['text']
            assert '—' not in opt['text']
            assert 'DEM' not in opt['text']
        for opt in self.dom.options_by_select_id.get('basemapSelect', []):
            assert '(' not in opt['text']

        assert 'orderFilterSelect' in self.dom.by_id
        assert self.dom.by_id['orderFilterSelect']['hidden_or_ancestor_hidden']
        assert 'delinModeSelect' in self.dom.by_id
        assert self.dom.by_id['delinModeSelect']['hidden_or_ancestor_hidden']

    def _assert_river_endpoints(self) -> None:
        """Verify `/api/hydro/rivers` across zoom levels 2, 4, and 7."""
        for z, bbox in [
            (2, '-180.0,-85.0,180.0,85.0'),
            (4, '-130.0,20.0,-60.0,55.0'),
            (7, '-89.0,39.0,-88.0,40.0'),
        ]:
            st_ha, ha_data = self._get(
                f'/api/hydro/rivers?dataset=hydroatlas&bbox={bbox}&zoom={z}'
            )
            st_mh, mh_data = self._get(
                f'/api/hydro/rivers?dataset=merit-hydro&bbox={bbox}&zoom={z}'
            )
            assert st_ha == _HTTP_OK
            assert st_mh == _HTTP_OK
            assert ha_data['properties']['dataset'] == 'hydroatlas'
            assert mh_data['properties']['dataset'] == 'merit-hydro'
            assert len(ha_data['features']) > 0
            assert len(mh_data['features']) > 0
            assert str(
                ha_data['features'][0]['properties']['reach_id']
            ).startswith('HYRIV_')
            assert str(
                mh_data['features'][0]['properties']['reach_id']
            ).startswith('MERIT_')

    def test_task4_dem_delineator_integration_and_profile_artifact_sync(  # noqa: PLR0915
        self,
    ) -> None:
        """Verify `DemDelineator` integration via `/api/delineate` and sync."""
        assert ensure_flood_forecasting_on_sys_path()
        assert DemDelineator is not None
        assert isinstance(get_dem_delineator('hydrosheds_90m'), DemDelineator)

        assert resolve_hydro_dataset_id('hydrosheds_90m') == 'hydroatlas'
        assert resolve_hydro_dataset_id('merit_hydro_90m') == 'merit-hydro'
        assert HYDRO_DATASETS['hydroatlas']['dem_id'] == 'hydrosheds_90m'
        assert HYDRO_DATASETS['merit-hydro']['dem_id'] == 'merit_hydro_90m'
        self._assert_river_endpoints()

        test_user = 'test_delineator_tab_user'
        login_status, _ = self._post(
            '/api/profile/login',
            {
                'username': test_user,
                'email': 'user@example.com',
                'create_if_missing': True,
            },
        )
        assert login_status == _HTTP_OK

        try:
            delin_status, feat = self._post(
                '/api/delineate',
                {
                    'dataset': 'hydroatlas',
                    'dem_id': 'hydrosheds_90m',
                    'latitude': 39.6828,
                    'longitude': -88.7729,
                    'username': test_user,
                },
            )
            assert delin_status == _HTTP_OK
            assert feat['type'] == 'Feature'
            props = feat['properties']
            assert props['delineation_mode'] == 'dem_flow_direction'
            assert props['dataset'] == 'hydroatlas'
            assert props['dem_id'] == 'hydrosheds_90m'
            assert 'HydroRIVERS' in props['river_network']
            assert props['catchment_id'].startswith('catchment_hydroatlas_')
            assert props['area_km2'] > _MIN_DEM_AREA_KM2
            assert props['upstream_cells_count'] > _MIN_UPSTREAM_CELLS
            assert feat['geometry']['type'] in ('Polygon', 'MultiPolygon')

            merit_dir = resolve_dem_tiles_dir('merit_hydro_90m')
            hydro_dir = resolve_dem_tiles_dir('hydrosheds_90m')
            assert merit_dir.resolve() != hydro_dir.resolve()
            assert 'merit' in str(merit_dir.resolve()).lower()

            m_status, m_feat = self._post(
                '/api/delineate',
                {
                    'dataset': 'merit-hydro',
                    'dem_id': 'merit_hydro_90m',
                    'latitude': 39.6828,
                    'longitude': -88.7729,
                    'username': test_user,
                },
            )
            assert m_status == _HTTP_OK
            m_props = m_feat['properties']
            assert m_props['delineation_mode'] == 'dem_flow_direction'
            assert m_props['dataset'] == 'merit-hydro'
            assert m_props['dem_id'] == 'merit_hydro_90m'
            assert 'MERIT-Hydro' in m_props['delineation_method']
            assert m_props['catchment_id'].startswith('catchment_merit_hydro_')
            assert m_props['area_km2'] > _MIN_DEM_AREA_KM2
            assert (
                m_props['upstream_cells_count'] != props['upstream_cells_count']
            )

            with pytest.raises(urllib.error.HTTPError) as ctx:
                self._post(
                    '/api/delineate',
                    {
                        'dataset': 'nonexistent_dataset',
                        'latitude': 39.6828,
                        'longitude': -88.7729,
                        'username': test_user,
                    },
                )
            assert ctx.value.code == _HTTP_BAD_REQUEST

            pm = get_profile_manager()
            cid = props['catchment_id']
            watersheds_path = (
                pm.get_catchments_dir(test_user) / 'watersheds.geojson'
            )
            shape_path = (
                pm.get_catchment_shapes_dir(test_user) / f'{cid}.geojson'
            )
            basin_list_path = (
                pm.get_basin_lists_dir(test_user) / 'all_basins.txt'
            )
            assert watersheds_path.exists()
            assert shape_path.exists()
            assert basin_list_path.exists()
            basins_txt = basin_list_path.read_text(
                encoding='utf-8'
            ).splitlines()
            assert cid in basins_txt
            assert m_props['catchment_id'] in basins_txt
        finally:
            self._post('/api/profile/logout', {})

    def test_issue86_unified_left_sidebar_across_all_tabs(self) -> None:
        """Verify Issue #86 unified left sidebar across all 7 tabs."""
        expected_sidebars = (
            'maasSidebar',
            'delineationSidebar',
            'weatherSidebar',
            'geoFeaturesSidebar',
            'trainingSidebar',
            'forecastingSidebar',
            'accountSidebar',
        )
        for sb_id in expected_sidebars:
            assert sb_id in self.dom.by_id, f'Missing #{sb_id} in DOM'
            sb_info = self.dom.by_id[sb_id]
            assert sb_info['tag'] == 'aside'
            assert 'unified-left-sidebar' in sb_info['classes']

        assert 'globalSidebarToggleBtn' in self.dom.by_id
        assert 'sidebarExpandFloatingBtn' in self.dom.by_id
        assert 'accountMainContent' in self.dom.by_id
        assert 'weatherCatchmentsToggleBtn' in self.dom.by_id
        assert 'weatherCatchmentsToggleLabel' in self.dom.by_id
        assert 'weatherCatchmentsToggleState' in self.dom.by_id

        for token in (
            '--sidebar-width: 380px',
            '--sidebar-min-width: 260px',
            '--sidebar-max-width: 640px',
            'body.sidebar-collapsed .unified-left-sidebar',
            '.sidebar-resizer',
            '<div class="sidebar-resizer"',
            'data-action="collapse-sidebar"',
            'openhydronet.sidebarWidth',
            'openhydronet.sidebarCollapsed',
            'openhydronet.weatherShowCatchments',
            'initUnifiedSidebar',
            'notifySidebarLayoutChange',
        ):
            assert token in self.html_text, f'Missing expected token: {token}'
        assert self.html_text.count('<div class="sidebar-resizer"') == len(
            expected_sidebars
        ), 'Every unified left sidebar must include a drag-to-resize handle'
        collapse_btn_html = (
            '<button type="button" class="sidebar-collapse-btn" '
            'data-action="collapse-sidebar"'
        )
        assert self.html_text.count(collapse_btn_html) == len(
            expected_sidebars
        ), 'Every unified left sidebar header must include a collapse button'

    def test_dataset_switch_preserves_watersheds_and_mutations_bind_profile(
        self,
    ) -> None:
        """Verify dataset switch keeps basins and mutations bind profile."""
        ds_listener_marker = (
            "document.getElementById('datasetSelect').addEventListener('change'"
        )
        assert ds_listener_marker in self.html_text
        ds_block = self.html_text.split(ds_listener_marker, 1)[1].split(
            '});', 1
        )[0]
        assert 'clearAllWatersheds' not in ds_block, (
            'Switching datasetSelect must not wipe active watersheds'
        )
        assert 'fetchAndRenderRivers(true)' in ds_block

        for endpoint in (
            '/api/profile/save_watersheds',
            '/api/watersheds/delete',
            '/api/watersheds/clear',
            '/api/delineate',
        ):
            call_block = self.html_text.split(f"fetch('{endpoint}'", 1)[
                1
            ].split('});', 1)[0]
            assert "'X-User-Profile': targetUser" in call_block, (
                f'{endpoint} fetch must include X-User-Profile header'
            )
            assert 'username: targetUser' in call_block, (
                f'{endpoint} fetch must include username in request body'
            )

        user_a = 'profile_isolation_a'
        user_b = 'profile_isolation_b'
        self.addCleanup(lambda: self._post('/api/profile/logout', {}))
        self._post('/api/profile/login', {'username': user_a})
        st_a, delin_a = self._post(
            '/api/delineate',
            {
                'username': user_a,
                'dataset': 'hydroatlas',
                'dem_id': 'hydrosheds_90m',
                'latitude': 39.66,
                'longitude': -88.7729,
                'mode': 'dem_flow_direction',
                'snap_radius_km': 2.0,
            },
        )
        assert st_a == _HTTP_OK
        cid_a = delin_a['properties']['catchment_id']

        # Switch active singleton profile to user_b and delineate for user_b
        self._post('/api/profile/login', {'username': user_b})
        st_b, delin_b = self._post(
            '/api/delineate',
            {
                'username': user_b,
                'dataset': 'merit-hydro',
                'dem_id': 'merit_hydro_90m',
                'latitude': 39.66,
                'longitude': -88.7729,
                'mode': 'dem_flow_direction',
                'snap_radius_km': 2.0,
            },
        )
        assert st_b == _HTTP_OK
        cid_b = delin_b['properties']['catchment_id']

        # Clearing user_a while user_b is active must not wipe user_b
        st_clr, clr_res = self._post(
            '/api/watersheds/clear',
            {'username': user_a},
        )
        assert st_clr == _HTTP_OK
        assert clr_res['status'] == 'cleared'

        pm = get_profile_manager()
        ws_a = json.loads(
            (pm.get_catchments_dir(user_a) / 'watersheds.geojson').read_text(
                encoding='utf-8'
            )
        )
        ws_b = json.loads(
            (pm.get_catchments_dir(user_b) / 'watersheds.geojson').read_text(
                encoding='utf-8'
            )
        )
        assert ws_a['features'] == []
        assert [f['properties']['catchment_id'] for f in ws_b['features']] == [
            cid_b
        ]
        assert not (
            pm.get_catchment_shapes_dir(user_a) / f'{cid_a}.geojson'
        ).exists()
        assert (
            pm.get_catchment_shapes_dir(user_b) / f'{cid_b}.geojson'
        ).exists()


if __name__ == '__main__':
    unittest.main()
