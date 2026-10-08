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

"""Command-line entry point (`fetch-maas-forecast`) for the `maas` package."""

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from maas.config import MaaSConfig, normalize_requested_models
from maas.fetcher import MaaSDataFetcher


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for `fetch-maas-forecast`."""
    parser = argparse.ArgumentParser(
        prog='fetch-maas-forecast',
        description=(
            'Fetch and compare operational streamflow forecasts across '
            'Google FloodHub, Copernicus GloFAS, and GEOGLOWS ECMWF.'
        ),
    )
    parser.add_argument(
        '--lat',
        type=float,
        required=True,
        help='Probe latitude in decimal degrees (EPSG:4326).',
    )
    parser.add_argument(
        '--lon',
        type=float,
        required=True,
        help='Probe longitude in decimal degrees (EPSG:4326).',
    )
    parser.add_argument(
        '--cache-dir',
        type=Path,
        required=True,
        help='Explicit local directory for SQLite return-period caches.',
    )
    parser.add_argument(
        '--river-networks-dir',
        type=Path,
        default=None,
        help=(
            'Directory containing static river network datasets '
            '(defaults to --cache-dir when omitted).'
        ),
    )
    parser.add_argument(
        '--models',
        type=str,
        default='floodhub,glofas,geoglows',
        help=(
            'Comma-separated list of providers to query '
            '(floodhub, glofas, geoglows).'
        ),
    )
    parser.add_argument(
        '--gauge-id',
        type=str,
        default=None,
        help='Optional Google FloodHub gauge ID (e.g. hybas_7120012340).',
    )
    parser.add_argument(
        '--river-id',
        type=int,
        default=None,
        help='Optional GEOGLOWS v2 9-digit reach ID (LINKNO).',
    )
    parser.add_argument(
        '--reach-id',
        type=str,
        default=None,
        help='Optional HydroRIVERS reach ID (e.g. HYRIV_41000001).',
    )
    parser.add_argument(
        '--floodhub-api-key',
        type=str,
        default=None,
        help='Google FloodHub API key (defaults to $FLOODHUB_API_KEY).',
    )
    parser.add_argument(
        '--timeout',
        type=float,
        default=12.0,
        help='HTTP request timeout in seconds (default: 12.0).',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Optional output JSON file path (prints to stdout if omitted).',
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Execute `fetch-maas-forecast` CLI command."""
    args = parse_args(argv)
    if not (-90.0 <= args.lat <= 90.0):
        raise ValueError(f'--lat must be in [-90, 90], got {args.lat}.')
    if not (-180.0 <= args.lon <= 180.0):
        raise ValueError(f'--lon must be in [-180, 180], got {args.lon}.')

    cache_dir: Path = args.cache_dir
    river_networks_dir: Path = (
        args.river_networks_dir
        if args.river_networks_dir is not None
        else cache_dir
    )
    api_key = (
        args.floodhub_api_key
        if args.floodhub_api_key is not None
        else os.environ.get('FLOODHUB_API_KEY', '')
    )

    config = MaaSConfig(
        cache_dir=cache_dir,
        river_networks_dir=river_networks_dir,
        floodhub_api_key=api_key,
        http_timeout_s=args.timeout,
    )
    fetcher = MaaSDataFetcher(config)
    requested_models = normalize_requested_models(
        [m.strip() for m in args.models.split(',') if m.strip()]
    )
    result = fetcher.fetch_forecasts(
        lat=args.lat,
        lon=args.lon,
        gauge_id=args.gauge_id,
        river_id=args.river_id,
        reach_id=args.reach_id,
        requested_models=requested_models,
    )
    formatted = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(formatted + '\n', encoding='utf-8')
    else:
        sys.stdout.write(formatted + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
