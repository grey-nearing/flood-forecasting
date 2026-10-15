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
"""Scaler utilities."""

from collections.abc import Hashable, Iterator
from pathlib import Path

import dask
import dask.array
import numpy as np
import pandas as pd
import xarray as xr
import yaml

from utils.file_paths import SCALER_ZARR_NAME

SCALER_FILE_NAME = SCALER_ZARR_NAME


def resolve_scaler_dir(cfg: object, *, is_train: bool = False) -> Path | None:
    """Resolve the directory used to load ``scaler.zarr``.

    During fine-tuning, training starts from ``cfg.base_run_dir`` (and later
    saves the extended scaler to ``cfg.run_dir``), whereas validation/test
    loads the extended scaler from ``cfg.run_dir`` if it exists and falls back
    to ``cfg.base_run_dir`` for legacy fine-tuned runs.
    """
    run_dir: Path | None = getattr(cfg, 'run_dir', None)
    if getattr(cfg, 'is_finetuning', False):
        if is_train:
            return cfg.base_run_dir
        if run_dir is not None and (run_dir / SCALER_ZARR_NAME).exists():
            return run_dir
        return cfg.base_run_dir
    return run_dir


def _calc_stats(
    dataset: xr.Dataset, needed: set[str]
) -> dict[str, xr.DataArray]:
    stats = {
        'mean': dataset.mean(skipna=True),
        'median': (
            dataset.quantile(q=0.5, skipna=True) if 'median' in needed else None
        ),
        'min': dataset.min(skipna=True) if {'min', 'minmax'} & needed else None,
        'max': dataset.max(skipna=True) if {'max', 'minmax'} & needed else None,
    }

    stats['minmax'] = (
        stats['max'] - stats['min'] if 'minmax' in needed else None
    )

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance (Naive):  # noqa: E501
    # Var(X) = E[X**2] - E**2[X] instead of xr.std that subtracts mean from all
    # elements, to force dask work element wise. Non negative var failsafes if
    # there is propagating cancellation. For benchmark ds abs error was < 0.0001  # noqa: E501
    var = (dataset**2).mean(skipna=True) - stats['mean'] ** 2
    stats['std'] = var.clip(min=0) ** 0.5

    return stats


def _calc_types(
    dataset: xr.Dataset,
    types: dict[Hashable, str],
    none_value: float,
    stats: dict[str, xr.DataArray],
) -> Iterator[xr.DataArray]:
    """Yield pre-calculated statistics for each feature.

    Helper for building final 'center' and 'scale' params for the scaler.
    Determines needed statistic type ('mean', 'std', 'none', etc) for `types`
    for each feature.
    Looks up those in `stats` contains already-computed values for the feature.
    """
    for feature in dataset.data_vars:
        a_type = types[feature].lower()  # Get stat type like 'mean' for feature  # noqa: E501
        if a_type == 'none':
            yield xr.DataArray(none_value, name=feature)
        else:
            if a_type not in stats:
                raise ValueError(f'Unknown method {a_type}')
            yield stats[a_type][feature]


class Scaler:
    """Scaler for a dataset that contains multiple features.

    Statistics are computed over every value in the dataset passed to
    `calculate`. The `Multimet` dataset passes its loaded training slice, which
    extends `seq_length` days before `train_start_date` and `lead_time` days
    after `train_end_date`. The same window is used when fine-tuning adds new
    variables to a pre-trained scaler.

    Parameters
    ----------
    scaler_dir : pathlib.Path
        Directory for loading a pre-calculated scaler or saving this scaler
        if it is calculated.
    calculate_scaler : bool
        Flag to indicate if the scaler should be computed
        (the alternative is to load an existing scaler file).
    custom_normalization : dict[str, dict[str, float]]
        Feature-specific scaling instructions as a mapping from feature name
        to centering and/or scaling type.
        See docs for a list of accepted types and their meaning.
    dataset : xr.Dataset | None
        Dataset to use for calculating a new scaler. Cannot be supplied if
        `calculate_scaler` is False.

    Raises
    ------
    ValueError for incompatible loading/calculating instructions.
    """

    def __init__(
        self,
        scaler_dir: Path,
        calculate_scaler: bool,  # noqa: FBT001
        custom_normalization: dict[str, dict[str, float]] | None = None,
        dataset: xr.Dataset | None = None,
    ) -> None:
        """Initialize the scaler."""
        if custom_normalization is None:
            custom_normalization = {}
        # Consistency check.
        if not calculate_scaler and dataset is not None:
            raise ValueError(
                'Do not pass a dataset if you are loading a pre-calculated scaler.'  # noqa: E501
            )

        # Load or calculate scaling parameters.
        self.scaler = None
        self.scaler_dir = scaler_dir
        self._custom_normalization = custom_normalization
        if not calculate_scaler:
            self.load()
            self.check_zero_scale()
        elif dataset is not None:
            self.calculate(dataset)

    def load(self) -> None:
        """Load a pre-computed scaler dataset from zarr or legacy dictionary."""
        scaler_zarr = self.scaler_dir / SCALER_FILE_NAME
        if scaler_zarr.exists():
            self.scaler = xr.open_zarr(scaler_zarr).load()
            return

        if self._load_legacy():
            return

        raise FileNotFoundError(f'Scaler file not found at {scaler_zarr}')

    def _load_legacy(self) -> bool:
        """Load legacy scaler dictionary into self.scaler."""
        legacy_p = self.scaler_dir / 'train_data' / 'train_data_scaler.p'
        legacy_yml = self.scaler_dir / 'train_data' / 'train_data_scaler.yml'

        legacy_dict = None
        if legacy_p.exists():
            legacy_dict = np.load(legacy_p, allow_pickle=True).item()
        elif legacy_yml.exists():
            with legacy_yml.open() as f:
                legacy_dict = yaml.safe_load(f)

        if legacy_dict is None:
            return False

        scaler_dict = {'center': [], 'scale': []}
        for key, value in legacy_dict.items():
            if key in ['attribute_means', 'camels_attr_means']:
                scaler_dict['center'].append(
                    xr.Dataset(value).astype(np.float32)
                )
            elif key in ['attribute_stds', 'camels_attr_stds']:
                scaler_dict['scale'].append(
                    xr.Dataset(value).astype(np.float32)
                )
            elif key == 'xarray_feature_center':
                scaler_dict['center'].append(
                    xr.Dataset.from_dict(value).astype(np.float32)
                )
            elif key == 'xarray_feature_scale':
                scaler_dict['scale'].append(
                    xr.Dataset.from_dict(value).astype(np.float32)
                )

        if scaler_dict['center'] and scaler_dict['scale']:
            center_ds = xr.merge(scaler_dict['center'])
            scale_ds = xr.merge(scaler_dict['scale'])
            self.scaler = xr.concat(
                [center_ds, scale_ds],
                dim=pd.Index(['center', 'scale'], name='parameter'),
            )
            obs_scaler = self.scaler.copy().rename(
                {f: f + '_obs' for f in self.scaler.data_vars}
            )
            sim_scaler = self.scaler.copy().rename(
                {f: f + '_sim' for f in self.scaler.data_vars}
            )
            self.scaler = xr.merge([self.scaler, obs_scaler, sim_scaler])
            return True
        return False

    def calculate(
        self,
        dataset: xr.Dataset,
    ) -> None:
        """Calculate scaler statistics."""
        # Option for custom scaling for each feature.
        centering_types = dict.fromkeys(dataset.data_vars, 'mean')
        scaling_types = dict.fromkeys(dataset.data_vars, 'std')
        for feature, norm in self._custom_normalization.items():
            if 'centering' in norm:
                centering_types[feature] = norm['centering']
            if 'scaling' in norm:
                scaling_types[feature] = norm['scaling']

        needed = set(centering_types.values()) | set(scaling_types.values())
        stats = _calc_stats(dataset, needed)

        # Select the appropriate center and scale statistic for each feature.
        center = xr.merge(_calc_types(dataset, centering_types, 0.0, stats))
        scale = xr.merge(_calc_types(dataset, scaling_types, 1.0, stats))

        # Combine parameters into a single xarray.Dataset with a 'parameter' coordinate.  # noqa: E501
        param_names = ['center', 'scale', 'mean', 'std']
        param_index = pd.Index(param_names, name='parameter', dtype=object)
        scaler = xr.concat(
            [center, scale, stats['mean'], stats['std']], dim=param_index
        )

        # Expand the scaler dataset to include 'obs' and 'sim' versions of all variables.  # noqa: E501
        obs_scaler = scaler.rename(
            {var: f'{var}_obs' for var in scaler.data_vars}
        )
        sim_scaler = scaler.rename(
            {var: f'{var}_sim' for var in scaler.data_vars}
        )
        scaler = xr.merge([scaler, obs_scaler, sim_scaler])

        # Handle cases where part of the scaler is already calculated. Simply add new features.  # noqa: E501
        if self.scaler is not None:
            self.scaler = xr.merge([self.scaler, scaler])
        else:
            self.scaler = scaler

        if not is_any_lazy(
            self.scaler
        ):  # ensure allowing side-effects on compute
            self.scaler = self.scaler.chunk('auto')

    def save(self) -> None:
        """Save the scaler."""
        if self.scaler is None:
            raise ValueError(
                'You are trying to save a scaler that has not been computed.'
            )
        _assert_computed(self.scaler)

        self.scaler_dir.mkdir(parents=True, exist_ok=True)
        scaler_file = self.scaler_dir / SCALER_FILE_NAME
        self.scaler.to_zarr(scaler_file, mode='w')

    def check_zero_scale(self) -> None:
        """Check scaler parameters are finite and scales are non-zero.

        Raises:
            ValueError: If any feature has a zero `scale` or `std` parameter,
                or
                if any feature has a `NaN` or `Inf` value in any of its
                `center`, `scale`, `mean`, or `std` parameters.
        """
        _assert_computed(self.scaler)

        scales_to_check = self.scaler.sel(parameter=['scale', 'std'])
        is_zero = (scales_to_check == 0).any('parameter').to_dataarray()
        zero_features = is_zero['variable'][is_zero]
        if zero_features.size > 0:
            raise ValueError(
                'Zero scale values found for features: '
                f'{zero_features.to_numpy().tolist()}.'
            )

        # `NaN == 0` is False, so NaN and Inf parameters need a separate check.
        is_invalid = (~np.isfinite(self.scaler.to_dataarray())).any('parameter')
        invalid_features = is_invalid['variable'][is_invalid]
        if invalid_features.size > 0:
            raise ValueError(
                'NaN or Inf values found in scaler parameters for features: '
                f'{invalid_features.to_numpy().tolist()}.'
            )

    def scale(
        self,
        dataset: xr.Dataset,
    ) -> xr.Dataset:
        """Scale a data set with a precaculated_scaler.

        $$ unscaled_dataset = (dataset - center) / scale $$

        Applies a linear transformation to the features (data_vars)
        in an xr.Dataset.
        This transformation is the inverse of the one applied
        by self.unscale().
        Agnostic to the dimensions and coordinates of the dataset.

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset to be scaled.

        Returns
        -------
        xr.Dataset
            The new dataset where all scalable features are scaled.

        Raises
        ------
        ValueError if the dataset contains features that are not in the
        scaler parameters.
        """
        missing_features = [
            feature
            for feature in dataset
            if feature not in self.scaler.data_vars
        ]
        if any(missing_features):
            raise ValueError(
                f'Requesting to scale variables that are not part of the scaler: {missing_features}'  # noqa: E501
            )
        return (
            dataset - self.scaler.sel(parameter='center')
        ) / self.scaler.sel(parameter='scale')

    def unscale(self, dataset: xr.Dataset) -> xr.Dataset:
        """Un-scale a data set with a precalculated scaler.

        $$ scaled_dataset = dataset * scale + center $$

        Applies a linear transformation to the features (data_vars)
        in an xr.Dataset.
        This transformation is the inverse of the one applied
        by self.scale().
        Agnostic to the dimensions and coordinates of the dataset.

        Parameters
        ----------
        dataset : xr.Dataset
            Dataset to be un-scaled.

        Returns
        -------
        xr.Dataset
            The new dataset where all scalable features are un-scaled.

        Raises
        ------
        ValueError if the dataset contains features that are not in the
        scaler parameters.
        """
        missing_features = [
            feature
            for feature in dataset
            if feature not in self.scaler.data_vars
        ]
        if any(missing_features):
            raise ValueError(
                f'Requesting to unscale variables that are not part of the scaler: {missing_features}'  # noqa: E501
            )
        return dataset * self.scaler.sel(parameter='scale') + self.scaler.sel(
            parameter='center'
        )


def is_any_lazy(dataset: xr.Dataset) -> bool:
    """Check if any variable in the dataset is a lazy dask array."""
    return any(
        isinstance(var.data, dask.array.Array)
        for var in dataset.data_vars.values()
    )


def _assert_computed(da: xr.DataArray | xr.Dataset | None) -> None:
    assert da is not None
    chunks = da.chunks
    assert not chunks, f'`scaler` needs to be computed yet has {chunks=}'
