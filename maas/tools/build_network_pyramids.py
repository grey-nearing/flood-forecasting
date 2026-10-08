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

"""Offline builder for zoom-stratified river network `.npz` pyramids."""

import argparse
import logging
import time
from collections.abc import Sequence
from pathlib import Path

from maas.networks import (
    CACHE_VERSION,
    GEOGLOWS_LOD,
    GLOFAS_LOD,
    build_geoglows_pyramid,
    build_glofas_pyramid,
    pyramid_signature,
    save_network_pyramid,
)


def build_all_pyramids(
    river_networks_dir: Path,
    output_dir: Path | None = None,
    models: Sequence[str] = ('glofas', 'geoglows'),
) -> dict[str, list[int]]:
    """Build and persist zoom-stratified river network pyramids for `models`.

    Args:
        river_networks_dir: Directory containing `glofas_v4/` and `geoglows_v2/`.
        output_dir: Optional output directory for `.npz` pyramids (defaults to
            each model's subdirectory under `river_networks_dir`).
        models: Sequence of model identifiers to build (`'glofas'`,
            `'geoglows'`).

    Returns:
        Mapping from model name to list of polyline counts per zoom level.
    """
    if not river_networks_dir.exists():
        raise FileNotFoundError(
            f'river_networks_dir does not exist: {river_networks_dir}'
        )
    glofas_dir = river_networks_dir / 'glofas_v4'
    geoglows_dir = river_networks_dir / 'geoglows_v2'

    out_counts: dict[str, list[int]] = {}
    if 'glofas' in models:
        glofas_net = build_glofas_pyramid(glofas_dir)
        glofas_out = (
            output_dir if output_dir is not None else glofas_dir
        ) / f'glofas_network_v{CACHE_VERSION}.npz'
        save_network_pyramid(
            glofas_out,
            glofas_net['levels'],
            pyramid_signature(GLOFAS_LOD),
            cell_lin=glofas_net['cell_lin'],
            cell_area=glofas_net['cell_area'],
        )
        out_counts['glofas'] = [
            len(lvl['offsets']) - 1 for lvl in glofas_net['levels']
        ]

    if 'geoglows' in models:
        gg_net = build_geoglows_pyramid(geoglows_dir)
        gg_out = (
            output_dir if output_dir is not None else geoglows_dir
        ) / f'geoglows_network_v{CACHE_VERSION}.npz'
        save_network_pyramid(
            gg_out,
            gg_net['levels'],
            pyramid_signature(GEOGLOWS_LOD),
        )
        out_counts['geoglows'] = [
            len(lvl['offsets']) - 1 for lvl in gg_net['levels']
        ]
    return out_counts


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for `build_network_pyramids`."""
    parser = argparse.ArgumentParser(
        description='Build zoom-stratified river network pyramids for MaaS.',
    )
    parser.add_argument(
        '--river-networks-dir',
        type=Path,
        required=True,
        help='Directory containing glofas_v4/ and geoglows_v2/ input datasets.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Optional output directory for derived *_network_v1.npz files.',
    )
    parser.add_argument(
        '--models',
        type=str,
        default='glofas,geoglows',
        help='Comma-separated list of models to build.',
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run offline network pyramid builder."""
    args = parse_args(argv)
    models = [m.strip() for m in args.models.split(',') if m.strip()]
    t0 = time.time()
    counts = build_all_pyramids(
        river_networks_dir=args.river_networks_dir,
        output_dir=args.output_dir,
        models=models,
    )
    for model, sizes in counts.items():
        logging.info(
            f'{model}: lines per level {sizes} ({time.time() - t0:.1f} s)'
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
