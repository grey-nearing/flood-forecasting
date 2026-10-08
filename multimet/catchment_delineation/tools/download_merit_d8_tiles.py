# Copyright 2025 Google LLC
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

"""CLI tool to download 5x5 degree MERIT-Hydro 90m D8 flow-direction tiles."""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from multimet.catchment_delineation.datasets import MERIT_HYDRO_90M
from multimet.catchment_delineation.merit import download_merit_d8_tile
from multimet.catchment_delineation.tiles import (
    filename_to_tile_key,
    get_required_tiles_for_bbox,
    is_tile_in_coverage,
    tile_key_to_filename,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Download 5x5 degree MERIT-Hydro 90m D8 flow-direction .npy tiles '
            'from Google Earth Engine. All paths and project IDs must be '
            'supplied explicitly.'
        )
    )
    parser.add_argument(
        '--target-dir',
        type=Path,
        required=True,
        help='Destination directory for .npy tiles.',
    )
    parser.add_argument(
        '--ee-project',
        type=str,
        required=True,
        help='Google Cloud project ID for Earth Engine requests.',
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        '--tiles',
        nargs='+',
        help="Explicit 5x5 tile stems or filenames (e.g. 'n40w090' 'n45w090').",
    )
    source_group.add_argument(
        '--reference-tiles-dir',
        type=Path,
        help='Directory of existing .npy tile filenames to mirror.',
    )
    source_group.add_argument(
        '--bbox',
        type=float,
        nargs=4,
        metavar=('MIN_LON', 'MIN_LAT', 'MAX_LON', 'MAX_LAT'),
        help='Bounding box (min_lon min_lat max_lon max_lat) to cover.',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=12,
        help='Number of concurrent download threads (default: 12).',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the MERIT-Hydro D8 tile downloader."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    target_dir = args.target_dir.expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    if args.tiles:
        tile_keys = sorted(
            {filename_to_tile_key(tile_str) for tile_str in args.tiles}
        )
    elif args.reference_tiles_dir:
        ref_dir = args.reference_tiles_dir.expanduser().resolve()
        if not ref_dir.is_dir():
            raise FileNotFoundError(
                f'Reference tiles directory does not exist: {ref_dir}'
            )
        ref_files = sorted(ref_dir.glob('*.npy'))
        tile_keys = [filename_to_tile_key(p.name) for p in ref_files]
    else:
        min_lon, min_lat, max_lon, max_lat = args.bbox
        tile_keys = sorted(
            tk
            for tk in get_required_tiles_for_bbox(
                min_lat, min_lon, max_lat, max_lon
            )
            if is_tile_in_coverage(tk[0], tk[1], dataset=MERIT_HYDRO_90M)
        )

    missing = [
        (lat, lon)
        for (lat, lon) in tile_keys
        if not (target_dir / tile_key_to_filename(lat, lon)).is_file()
    ]
    sys.stdout.write(
        f'Total 5x5 tiles: {len(tile_keys)} | '
        f'Already downloaded: {len(tile_keys) - len(missing)} | '
        f'Remaining: {len(missing)}\n'
    )
    if not missing:
        return 0

    t0 = time.time()
    completed = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        fut_map = {
            pool.submit(
                download_merit_d8_tile,
                lat,
                lon,
                target_dir,
                ee_project=args.ee_project,
            ): (lat, lon)
            for (lat, lon) in missing
        }
        for fut in as_completed(fut_map):
            lat, lon = fut_map[fut]
            try:
                fut.result()
                completed += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                sys.stderr.write(
                    f'FAILED {tile_key_to_filename(lat, lon)}: {exc}\n'
                )
            if (completed + failed) % 25 == 0 or (completed + failed) == len(
                missing
            ):
                elapsed = time.time() - t0
                rate = (completed + failed) / max(elapsed, 1e-3)
                sys.stdout.write(
                    f'[{completed + failed}/{len(missing)}] '
                    f'ok={completed} failed={failed} '
                    f'({rate:.2f} tiles/s, elapsed={elapsed:.1f}s)\n'
                )
    return 1 if failed > 0 else 0


if __name__ == '__main__':
    sys.exit(main())
