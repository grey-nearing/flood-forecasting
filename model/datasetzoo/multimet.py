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

"""Caravan-MultiMet forecast dataset, data loader, and sample indexer."""

import functools
import itertools
import logging
from collections.abc import Hashable, Iterable, Iterator, Sequence
from pathlib import Path

import dask
import dask.array
import numpy as np
import pandas as pd
import torch
import xarray as xr
from dask.sizeof import sizeof
from torch.utils.data import Dataset

from model.datasetzoo.caravan import (
    load_caravan_attributes,
    load_caravan_timeseries,
)
from model.datautils.scaler import (
    Scaler,
)
from model.datautils.scaler import (
    resolve_scaler_dir as _resolve_scaler_dir,
)
from model.datautils.union_features import union_features
from model.datautils.utils import load_basin_file
from model.datautils.validate_samples import validate_samples
from model.utils import memory
from model.utils.config import Config
from model.utils.configutils import (
    canonical_product_name as _canonical_product_name,
)
from model.utils.configutils import (
    flatten_feature_list,
    group_features_list,
)
from model.utils.configutils import (
    product_name_from_feature as _product_name_from_feature,
)
from model.utils.errors import NoEvaluationDataError, NoTrainDataError
from model.utils.tqdm import AutoRefreshTqdm
from utils.file_paths import SCALER_ZARR_NAME, TIMESERIES_ZARR_NAME

LOGGER = logging.getLogger(__name__)

# Data types for all keys in the sample dictionary.
NUMPY_VARS = ['date']
TENSOR_VARS = [
    'x_s',
    'x_s_local',
    'x_d',
    'x_d_hindcast',
    'x_d_forecast',
    'y',
    'per_basin_target_stds',
    'basin_index',
]
MULTIMET_MINIMUM_LEAD_TIME = 1


def _needs_per_basin_target_stds(cfg: Config) -> bool:
    """Whether the training loss or the data assimilation loss is NSE."""
    if cfg.loss.lower() == 'nse':
        return True
    assimilation_config = cfg.assimilation_config
    return (
        assimilation_config is not None
        and assimilation_config.loss.lower() == 'nse'
    )


class MultimetDataLoader(torch.utils.data.DataLoader):
    """Custom DataLoader that handles lazy data loading.

    Ignores num_workers to avoid issues with dask/xarray in subprocesses.
    Triggers compute() every batch using dask IFF `lazy_load` is True.

    Args:
        *args: Positional arguments passed to the parent class.
        lazy_load: Iff True, the data is computed using dask before collating.
        logging_level: The value of the logging level e.g. DEBUG INFO etc.
        **kwargs: Keyword arguments passed to the parent class.
    """

    def __init__(
        self,
        *args: object,
        lazy_load: bool,
        logging_level: int,
        **kwargs: object,
    ):
        """Initialize the parent DataLoader with `num_workers` forced to 0."""
        kwargs['num_workers'] = 0
        super().__init__(*args, **kwargs)
        self._lazy_load = lazy_load
        self._debug = logging_level <= logging.DEBUG

    def __iter__(self):
        """Yield collated batches, one per index list of the batch sampler."""
        if not self._lazy_load and isinstance(self.dataset, Multimet):
            for indices in self.batch_sampler:
                yield self.dataset.get_batch(indices)
            return

        for indices in self.batch_sampler:
            # Known limitations: samples are fetched one at a time (there is
            # no batched `__getitems__`), which costs memory and runtime due to
            # many independent dask graphs, especially in lazy mode. Streaming
            # results with dask.Bag, or in non-lazy mode streaming the batch
            # samples straight into the tensor conversion below, would save
            # memory.
            batch = AutoRefreshTqdm(
                (self.dataset[i] for i in indices),
                desc='Prepare batch',
                unit='sample',
                disable=not self._lazy_load or not self._debug,
                total=len(indices),
            )

            batch = dask.compute(*batch) if self._lazy_load else tuple(batch)

            # Samples are converted to tensors before collating; collating
            # first might save memory.
            batch = [
                {k: _convert_to_tensor(k, v) for k, v in sample.items()}
                for sample in batch
            ]
            batch = self.collate_fn(batch)

            del indices
            yield batch

    def __len__(self):
        """Return the number of batches produced by the batch sampler."""
        return len(self.batch_sampler)


class Multimet(Dataset):
    """Base data set class for forecast models.

    Use subclasses of this class for training/evaluating a model with forecast
    capabilities. Currently, the only supported forecast dataset is
    Caravan-Multimet.

    Args:
        cfg: The run configuration.
        is_train: Defines if the dataset is used for training or evaluating.
            If True (training), means/stds for each feature are computed and
            stored to the run directory. If one-hot encoding is used, the
            mapping for the one-hot encoding is created and also stored to
            disk. If False, the scaler must be calculated (`compute_scaler`
            must be True).
        period: One of {'train', 'validation', 'test'}. Defines the period for
            which the data will be loaded.
        basins: Optional. If passed, the data for only these basins will be
            loaded. Otherwise, the basin(s) is(are) read from the appropriate
            basin file, corresponding to the `period`.
        compute_scaler: Forces the dataset to calculate a new scaler instead of
            loading a precalculated scaler. Used during training, but not
            finetuning.
    """

    def __init__(  # noqa: C901, PLR0915
        self,
        cfg: Config,
        is_train: bool,  # noqa: FBT001
        period: str,
        basins: list[str] | None = None,
        compute_scaler: bool = True,  # noqa: FBT001, FBT002
    ):
        """Load, validate, normalize, and index the dataset for `period`."""
        self._cfg = cfg

        # Sequence length parameters.
        # These are inherited from the old forecast functionality of
        # basedataset.
        self.lead_time = cfg.lead_time
        self._seq_length = cfg.seq_length
        self._predict_last_n = cfg.predict_last_n
        self._forecast_overlap = cfg.forecast_overlap
        self._allzero_samples_are_invalid = cfg.allzero_samples_are_invalid

        # Feature lists by type.
        self._static_features = cfg.static_attributes
        self._local_static_inputs = cfg.local_static_attributes
        self._local_static_features = (
            flatten_feature_list(cfg.local_static_attributes)
            if cfg.local_static_attributes
            else []
        )
        self._all_static_features = list(
            dict.fromkeys(
                [
                    *(self._static_features or []),
                    *self._local_static_features,
                ]
            )
        )
        self._target_features = cfg.target_variables
        if not cfg.hindcast_inputs:
            raise ValueError('hindcast_inputs must be supplied.')
        self._forecast_features = flatten_feature_list(cfg.forecast_inputs)
        self._hindcast_features = flatten_feature_list(cfg.hindcast_inputs)
        self._hindcast_inputs = cfg.hindcast_inputs
        self._forecast_inputs = cfg.forecast_inputs
        self._union_mapping = cfg.union_mapping

        # Feature data paths by type. This allows the option to load some data
        # from cloud and some locally.
        self._statics_data_path = cfg.statics_data_dir
        self._dynamics_data_path = cfg.dynamics_data_dir
        self._targets_data_path = cfg.targets_data_dir

        # NaN-handling options are required to apply the correct sample
        # validation algorithms.
        self._configure_feature_groups(cfg)

        # Validating samples depends on whether we are training or testing.
        self.is_train = is_train

        self._period = period
        if period not in ['train', 'validation', 'test']:
            raise ValueError(
                "'period' must be one of 'train', 'validation' or 'test' "
            )

        if (
            period in ['validation', 'test'] or cfg.is_finetuning
        ) and compute_scaler:
            raise ValueError(
                'Scaler must be loaded (not computed) for validation, test, '
                'and finetuning.'
            )

        # Basin list loading is not consolidated: it also happens in the
        # trainer and tester.
        self._basins = basins or load_basin_file(
            getattr(cfg, f'{period}_basin_file')
        )

        # Load & preprocess the data.
        LOGGER.debug('load data')
        self._dataset = self._load_data()
        memory.release()
        LOGGER.debug('validate all floats are float32')
        _assert_floats_are_float32(self._dataset)

        # Extract date ranges.
        # Assumes continuous date ranges and daily data.
        self._min_lead_time = 0
        self._lead_times = []
        if self._forecast_features:
            self._min_lead_time = int(
                (self._dataset.lead_time.min() / np.timedelta64(1, 'D')).item()
            )
            # The date arithmetic in `_calc_date_range` and `_extract_hindcasts`
            # relies on the shortest loaded lead time being the Caravan-MultiMet
            # first lead (1 day, valid on the issue date itself).
            if self._min_lead_time != MULTIMET_MINIMUM_LEAD_TIME:
                raise ValueError(
                    'Expected the minimum forecast lead time to be '
                    f'{MULTIMET_MINIMUM_LEAD_TIME} day(s), got '
                    f'{self._min_lead_time}.'
                )
            self._lead_times = list(
                range(self._min_lead_time, self.lead_time + 1)
            )

        # Split hindcast features to groups with/without lead_time in the
        # dataset. These lists will be used for efficient data selection during
        # sampling.
        self._hindcast_features_with_lead_time = [
            feature
            for feature in self._hindcast_features
            if 'lead_time' in self._dataset[feature].dims
        ]
        self._hindcast_features_without_lead_time = [
            feature
            for feature in self._hindcast_features
            if feature not in self._hindcast_features_with_lead_time
        ]

        start_dates, end_dates = self._get_period_dates(cfg)
        self._sample_dates = self._union_ranges(start_dates, end_dates)
        # The convention in NH is that the period dates define the SAMPLE dates.
        # All hindcast (and forecast) seqences are extra. Therefore, when
        # cropping the dataset for sampling, we keep all the hindcast and
        # forecast sequence data on both sides of the period dates. This would
        # be more memory efficient in `_load_data()` but that approach adds
        # complexity to the child classes.
        extended_start_dates = [
            start_date - pd.Timedelta(days=self._seq_length)
            for start_date in start_dates
        ]
        extended_end_dates = [
            end_date + pd.Timedelta(days=self.lead_time)
            for end_date in end_dates
        ]
        extended_dates = self._union_ranges(
            extended_start_dates, extended_end_dates
        )
        LOGGER.debug('reindex data')
        self._dataset = self._dataset.reindex(date=extended_dates).sel(
            date=extended_dates
        )

        # Timestep counters indicate the lead time of each forecast timestep.
        self._hindcast_counter = None
        self._forecast_counter = None
        if cfg.timestep_counter:
            self._hindcast_counter = np.full((self._seq_length,), 0)
            self._forecast_counter = np.asarray(
                self._lead_times, dtype=np.int64
            )
            if self._forecast_overlap:
                overlap_counter = np.full(
                    (self._forecast_overlap,), self._min_lead_time
                )
                self._forecast_counter = np.concatenate(
                    [overlap_counter, self._forecast_counter], 0
                )

        # Union features to extend certain data records.
        # Martin suggests doing this step prior to training models and then
        # saving the unioned dataset locally. If you do that, then remove this
        # line.
        if self._union_mapping:
            LOGGER.debug('union features')
            self._dataset = union_features(self._dataset, self._union_mapping)

        # Scale the dataset AFTER cropping dates.
        self._setup_normalization(cfg, compute_scaler=compute_scaler)

        if not cfg.lazy_load:
            LOGGER.debug('[eager load] compute dataset')
            (self._dataset,) = dask.compute(self._dataset)
            memory.release()
        else:
            LOGGER.debug('[lazy load] not computing dataset')

        LOGGER.debug('create valid sample mask and indices plan')
        valid_sample_mask, indices = self._create_valid_sample_mask()

        # Compute stats for NSE-based loss functions alongside sample indices
        # so lazy_load batches do not re-evaluate target .std(dim='date') per
        # batch. Whether to calculate these is decided by
        # `_needs_per_basin_target_stds`. Alternatives are keeping a list of
        # the losses that require them somewhere like `training/__init__.py`,
        # or simply always calculating them.
        self._per_basin_target_stds = None
        if _needs_per_basin_target_stds(cfg):
            LOGGER.debug('create per_basin_target_stds')
            self._per_basin_target_stds = self._dataset[
                self._target_features
            ].std(
                dim=[
                    d
                    for d in self._dataset[self._target_features].dims
                    if d != 'basin'
                ],
                skipna=True,
            )

        LOGGER.debug('compute indices')
        indices, self._per_basin_target_stds = dask.compute(
            indices, self._per_basin_target_stds
        )
        memory.release()

        LOGGER.debug('Dataset size: %s MB', sizeof(self._dataset) / 1024**2)
        LOGGER.debug('Dataset on disk: %s MB', self._dataset.nbytes / 1024**2)
        LOGGER.debug('Sample index size: %s MB', sizeof(indices) / 1024**2)

        # Create sample index lookup table for `__getitem__`.
        LOGGER.debug('create sample index')
        self._create_sample_index(valid_sample_mask, indices)

        self._data_cache: dict[str, xr.DataArray] = {}
        self._fast_ready = False
        if not cfg.lazy_load:
            self._init_fast_batch_extractor()

        LOGGER.debug('forecast dataset init complete (%s)', self._period)

    def _configure_feature_groups(self, cfg: Config) -> None:
        """Validate NaN-handling configuration and build per-product groups."""
        self._nan_handling_method = cfg.nan_handling_method
        hindcast_is_flat = isinstance(cfg.hindcast_inputs, list) and (
            not cfg.hindcast_inputs or isinstance(cfg.hindcast_inputs[0], str)
        )
        forecast_is_flat = bool(cfg.forecast_inputs) and (
            isinstance(cfg.forecast_inputs, list)
            and isinstance(cfg.forecast_inputs[0], str)
        )
        if (
            hindcast_is_flat or forecast_is_flat
        ) and self._nan_handling_method in [
            'masked_mean',
            'attention',
            'unioning',
        ]:
            raise ValueError(
                'Feature groups are required for '
                f'{self._nan_handling_method} NaN-handling.'
            )
        self._hindcast_groups = list(
            group_features_list(cfg.hindcast_inputs).values()
        )
        self._forecast_groups = (
            list(group_features_list(cfg.forecast_inputs).values())
            if cfg.forecast_inputs
            else []
        )
        self._feature_groups = [
            self._hindcast_groups,
            self._forecast_groups,
        ]

    def _extend_scaler_with_new_features(self, new_vars: list[str]) -> None:
        """Calculate and validate scaler stats for new fine-tune features."""
        LOGGER.info(
            'Extending base scaler with new fine-tuning features: %s',
            new_vars,
        )
        static_only_vars = [
            v
            for v in new_vars
            if 'date' not in self._dataset[v].dims
            and self._dataset.sizes.get('basin', 0) < 2  # noqa: PLR2004
        ]
        dyn_or_multi_vars = [v for v in new_vars if v not in static_only_vars]
        if dyn_or_multi_vars:
            self.scaler.calculate(self._dataset[dyn_or_multi_vars])
        if static_only_vars:
            full_static_ds = load_caravan_attributes(
                data_dir=self._statics_data_path,
                basins=None,
                features=static_only_vars,
            )
            self.scaler.calculate(full_static_ds[static_only_vars])
        (self.scaler.scaler,) = dask.compute(self.scaler.scaler)
        invalid_vars: list[str] = []
        for var in new_vars:
            stats_da = self.scaler.scaler[var]
            vals = np.asarray(
                stats_da.sel(
                    parameter=['center', 'scale', 'mean', 'std']
                ).values,
                dtype=np.float64,
            )
            scale_vals = np.asarray(
                stats_da.sel(parameter=['scale', 'std']).values,
                dtype=np.float64,
            )
            if not np.all(np.isfinite(vals)) or np.any(scale_vals == 0):
                invalid_vars.append(var)
        if invalid_vars:
            raise ValueError(
                'Non-finite or zero normalization statistics '
                '(center/scale/mean/std) computed for new fine-tuning '
                f'features: {invalid_vars}. Ensure each new product has '
                'valid (non-NaN) observations with non-zero variance.'
            )

    def _setup_normalization(
        self, cfg: Config, *, compute_scaler: bool
    ) -> None:
        """Initialize, extend, validate, save, and apply the dataset scaler."""
        LOGGER.debug('init scaler')
        scaler_dir = _resolve_scaler_dir(cfg, is_train=self.is_train)
        self.scaler = Scaler(
            scaler_dir=scaler_dir,
            calculate_scaler=compute_scaler,
            custom_normalization=cfg.custom_normalization,
            dataset=(self._dataset if compute_scaler else None),
        )
        if cfg.is_finetuning and isinstance(self.scaler.scaler, xr.Dataset):
            missing_vars = [
                v
                for v in self._dataset.data_vars
                if v not in self.scaler.scaler.data_vars
            ]
            if self.is_train and missing_vars:
                self._extend_scaler_with_new_features(missing_vars)
            elif not self.is_train and missing_vars:
                expected_scaler = (
                    cfg.run_dir / SCALER_ZARR_NAME
                    if cfg.run_dir is not None
                    else Path(SCALER_ZARR_NAME)
                )
                raise ValueError(
                    'Fine-tuning evaluation dataset contains features '
                    f'{missing_vars} missing from scaler loaded at '
                    f'{self.scaler.scaler_dir}. Ensure fine-tuning training '
                    f'has written the extended scaler to {expected_scaler}.'
                )

        LOGGER.debug('compute scaler')
        (self.scaler.scaler,) = dask.compute(self.scaler.scaler)
        memory.release()

        LOGGER.debug('scaler check zero scale')
        self.scaler.check_zero_scale()
        if compute_scaler or (
            cfg.is_finetuning
            and self.is_train
            and cfg.run_dir is not None
            and isinstance(self.scaler.scaler, xr.Dataset)
        ):
            LOGGER.debug('scaler save')
            self.scaler.scaler_dir = cfg.run_dir
            self.scaler.save()

        LOGGER.debug('scale data')
        self._dataset = self.scaler.scale(self._dataset)

    def _init_fast_batch_extractor(self) -> None:
        """Precompute contiguous arrays and sliding window views for batching.

        Builds zero-copy `sliding_window_view` arrays over the eager in-memory
        `(basin, date)` and `(basin, date, lead_time)` dataset arrays so that
        `get_batch` can slice a full batch of samples in a single vectorized
        NumPy indexing step.
        """
        swv = np.lib.stride_tricks.sliding_window_view
        self._fast_basin_col = self._sample_index.get_column('basin').astype(
            np.intp
        )
        self._fast_date_col = self._sample_index.get_column('date').astype(
            np.intp
        )
        self._fast_date_win = swv(
            self._dataset['date'].data, self._seq_length, axis=0
        )
        self._fast_statics = np.ascontiguousarray(
            np.stack(
                [
                    self._dataset[feat].transpose('basin').data
                    for feat in self._static_features
                ],
                axis=-1,
            )
        )
        self._fast_local_statics: dict[str, np.ndarray] = {
            feat: np.ascontiguousarray(
                self._dataset[feat].transpose('basin').data
            )
            for feat in self._local_static_features
        }
        target_arr = np.ascontiguousarray(
            np.stack(
                [
                    self._dataset[feat].transpose('basin', 'date').data
                    for feat in self._target_features
                ],
                axis=-1,
            )
        )
        self._fast_target_win = np.moveaxis(
            swv(target_arr, self._seq_length, axis=1), -1, 2
        )
        if self._per_basin_target_stds is not None:
            self._fast_stds = np.ascontiguousarray(
                np.expand_dims(
                    np.stack(
                        [
                            self._per_basin_target_stds[feat]
                            .transpose('basin')
                            .data
                            for feat in self._target_features
                        ],
                        axis=-1,
                    ),
                    axis=1,
                )
            )
        else:
            self._fast_stds = None

        self._fast_hindcast_wins: dict[str, np.ndarray] = {}
        for feature in self._hindcast_features_without_lead_time:
            arr = np.ascontiguousarray(
                self._dataset[feature].transpose('basin', 'date').data
            )
            self._fast_hindcast_wins[feature] = swv(
                arr, self._seq_length, axis=1
            )
        for feature in self._hindcast_features_with_lead_time:
            arr = np.ascontiguousarray(
                self._dataset[feature]
                .isel(lead_time=0)
                .transpose('basin', 'date')
                .data
            )
            self._fast_hindcast_wins[feature] = swv(
                arr, self._seq_length, axis=1
            )

        self._fast_forecast_3d: dict[str, np.ndarray] = {}
        self._fast_forecast_overlap_wins: dict[str, np.ndarray] = {}
        for feature in self._forecast_features:
            arr_3d = np.ascontiguousarray(
                self._dataset[feature]
                .transpose('basin', 'date', 'lead_time')
                .data
            )
            self._fast_forecast_3d[feature] = arr_3d
            if (
                self._forecast_overlap is not None
                and self._forecast_overlap > 0
            ):
                arr_l0 = np.ascontiguousarray(arr_3d[:, :, 0])
                self._fast_forecast_overlap_wins[feature] = swv(
                    arr_l0, self._forecast_overlap, axis=1
                )
        self._fast_ready = True

    def _build_batch_forecast(
        self,
        b_idx: np.ndarray,
        d_idx: np.ndarray,
        h_end: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        """Build the `x_d_forecast` tensor dictionary for a batch of indices."""
        if not self._forecast_features:
            return {}
        if self._forecast_overlap is not None and self._forecast_overlap > 0:
            ov_start = h_end - self._forecast_overlap
            batch_forecast: dict[str, torch.Tensor] = {
                feature: torch.from_numpy(
                    np.concatenate(
                        [
                            self._fast_forecast_overlap_wins[feature][
                                b_idx, ov_start
                            ],
                            self._fast_forecast_3d[feature][b_idx, d_idx, :],
                        ],
                        axis=1,
                    )
                ).unsqueeze(-1)
                for feature in self._forecast_features
            }
        else:
            batch_forecast = {
                feature: torch.from_numpy(
                    self._fast_forecast_3d[feature][b_idx, d_idx, :]
                ).unsqueeze(-1)
                for feature in self._forecast_features
            }
        if self._forecast_counter is not None:
            forecast_cnt = np.broadcast_to(
                self._forecast_counter[None, :, None],
                (len(b_idx), len(self._forecast_counter), 1),
            ).copy()
            batch_forecast['forecast_counter'] = torch.from_numpy(forecast_cnt)
        return batch_forecast

    def get_batch(
        self, indices: Sequence[int] | np.ndarray
    ) -> dict[str, torch.Tensor | np.ndarray | dict[str, torch.Tensor]]:
        """Extract a collated batch of samples via vectorized indexing.

        Args:
            indices: 1D sequence or NumPy array of integer sample indices.

        Returns:
            Dictionary of collated batch tensors and date arrays matching
            `self.collate_fn([_convert_to_tensor(self[i]) for i in indices])`.

        Raises:
            RuntimeError: If the dataset was initialized with `lazy_load=True`
                and fast batch views are not available.
        """
        if not self._fast_ready:
            raise RuntimeError(
                'Fast batch extractor is not initialized because the dataset '
                'was constructed with lazy_load=True.'
            )
        if len(indices) == 0:
            return {}
        idx = np.asarray(indices, dtype=np.intp)
        b_idx = self._fast_basin_col[idx]
        d_idx = self._fast_date_col[idx]
        bsz = len(idx)

        target_end = d_idx + self.lead_time - self._min_lead_time
        target_start = target_end - (self._seq_length - 1)
        date_end = d_idx if not self._lead_times else target_end
        date_start = date_end - (self._seq_length - 1)

        h_end = d_idx + 1 - self._min_lead_time
        h_start = h_end - self._seq_length

        batch_hindcast: dict[str, torch.Tensor] = {
            feature: torch.from_numpy(win[b_idx, h_start]).unsqueeze(-1)
            for feature, win in self._fast_hindcast_wins.items()
        }
        if self._hindcast_counter is not None:
            hindcast_cnt = np.broadcast_to(
                self._hindcast_counter[None, :, None],
                (bsz, len(self._hindcast_counter), 1),
            ).copy()
            batch_hindcast['hindcast_counter'] = torch.from_numpy(hindcast_cnt)

        batch_forecast = self._build_batch_forecast(b_idx, d_idx, h_end)

        batch: dict[
            str, torch.Tensor | np.ndarray | dict[str, torch.Tensor]
        ] = {
            'date': self._fast_date_win[date_start],
            'x_s': torch.from_numpy(self._fast_statics[b_idx]),
            'x_d_hindcast': batch_hindcast,
            'x_d_forecast': batch_forecast,
            'y': torch.from_numpy(self._fast_target_win[b_idx, target_start]),
        }
        if self._fast_local_statics:
            batch['x_s_local'] = {
                feat: torch.from_numpy(arr[b_idx]).unsqueeze(-1)
                for feat, arr in self._fast_local_statics.items()
            }
        if self._fast_stds is not None:
            batch['per_basin_target_stds'] = torch.from_numpy(
                self._fast_stds[b_idx]
            )
        if not self._forecast_features:
            batch['x_d'] = batch.pop('x_d_hindcast')
            _ = batch.pop('x_d_forecast')
        batch['basin_index'] = torch.from_numpy(b_idx.astype(np.int64))
        return batch

    def __len__(self) -> int:
        """Return the number of valid samples."""
        return self._num_samples

    @property
    def min_lead_time(self) -> int:
        """Shortest forecast lead time in days, or 0 without forecast inputs."""
        return self._min_lead_time

    def __getitem__(
        self, item: int
    ) -> dict[str, torch.Tensor | np.ndarray | dict[str, torch.Tensor]]:
        """Retrieve a sample by integer index."""
        # Stop iteration.
        if item >= self._num_samples:
            raise IndexError(
                f'Requested index {item} > the total number of samples '
                f'{self._num_samples}.'
            )

        # Negative and non-integer indexes raise an error instead of stop
        # iterating.
        if item < 0:
            raise ValueError(f'Requested index {item} < 0.')
        if item % 1 != 0:
            raise ValueError(f'Requested index {item} is not an integer.')

        # The sample keeps the outer keys ('x_s', 'x_d_hindcast', ...) instead
        # of using only feature names; removing them would be a major change.
        sample_index = self._sample_index[item]
        sample = {
            'date': self._extract_dates(sample_index),
            'x_s': self._extract_statics(sample_index),
            'x_d_hindcast': self._extract_hindcasts(sample_index),
            'x_d_forecast': self._extract_forecasts(sample_index),
            'y': self._extract_targets(sample_index),
        }
        if self._local_static_features:
            sample['x_s_local'] = self._extract_local_statics(sample_index)
        if self._per_basin_target_stds is not None:
            sample['per_basin_target_stds'] = self._extract_per_basin_stds(
                sample_index
            )
        if self._hindcast_counter is not None:
            sample['x_d_hindcast']['hindcast_counter'] = np.expand_dims(
                self._hindcast_counter, -1
            )
        if self._forecast_counter is not None:
            sample['x_d_forecast']['forecast_counter'] = np.expand_dims(
                self._forecast_counter, -1
            )

        # Rename the hindcast data key if we are not doing forecasting.
        if not self._forecast_features:
            sample['x_d'] = sample.pop('x_d_hindcast')
            _ = sample.pop('x_d_forecast')

        # Can't use strings. Torch does not support it in tensors.
        basin_index = sample_index['basin']
        sample['basin_index'] = np.array(basin_index, dtype=np.int64)

        return sample

    def _calc_date_range(
        self, sample_index: dict[str, int], *, lead: bool = False
    ) -> range:
        date = sample_index['date']
        duration = self._seq_length - 1
        if not lead and not self._lead_times:
            return range(date - duration, date + 1)
        end = date + self.lead_time - self._min_lead_time
        return range(end - duration, end + 1)

    def _extract_dates(self, sample_index: dict[str, int]) -> np.ndarray:
        date = self._calc_date_range(sample_index)
        features = self._extract_dataset(
            self._dataset, ['date'], {'date': date}
        )
        return features['date']

    def _extract_statics(self, sample_index: dict[str, int]) -> np.ndarray:
        basin = sample_index['basin']
        features = self._extract_dataset(
            self._dataset, self._static_features, {'basin': basin}
        )
        return np.stack([features[e] for e in self._static_features], axis=-1)

    def _extract_local_statics(
        self, sample_index: dict[str, int]
    ) -> dict[str, np.ndarray]:
        basin = sample_index['basin']
        features = self._extract_dataset(
            self._dataset, self._local_static_features, {'basin': basin}
        )
        return {
            name: np.expand_dims(np.asarray(val, dtype=np.float32), -1)
            for name, val in features.items()
        }

    def _extract_hindcasts(
        self, sample_index: dict[str, int]
    ) -> dict[str, np.ndarray]:
        # In Caravan-MultiMet the first lead time (1 day) of a forecast issued
        # on date D covers [D 00:00, D+1 00:00]. The completed hindcast days
        # before D 00:00 are therefore [D - seq_length, ..., D - 1] when
        # forecasting (_min_lead_time = 1), and [D - seq_length + 1, ..., D]
        # when hindcast-only (_min_lead_time = 0).
        hindcast_end = sample_index['date'] + 1 - self._min_lead_time
        hindcast_date_range = range(
            hindcast_end - self._seq_length,
            hindcast_end,
        )

        # Extract hindcast features without lead_time.
        dim_indexes_without_lead_time = sample_index.copy()
        dim_indexes_without_lead_time['date'] = hindcast_date_range
        features = self._extract_dataset(
            self._dataset,
            self._hindcast_features_without_lead_time,
            dim_indexes_without_lead_time,
        )

        # Forecast features with lead_time may be used as hindcast features.
        # The first lead time (index 0) on date t covers [t 00:00, t+1 00:00],
        # so it shares the valid date range of the 2D hindcast features.
        dim_indexes_with_lead_time = sample_index.copy()
        dim_indexes_with_lead_time['lead_time'] = 0
        dim_indexes_with_lead_time['date'] = hindcast_date_range
        features |= self._extract_dataset(
            self._dataset,
            self._hindcast_features_with_lead_time,
            dim_indexes_with_lead_time,
        )

        # This adds a dimension to many features, as required by some models.
        # There is no need for this except that it is how basedataset works,
        # and everything else expects the trailing dim.
        return {
            name: np.expand_dims(feature, -1)
            for name, feature in features.items()
        }

    def _extract_forecasts(
        self, sample_index: dict[str, int]
    ) -> dict[str, np.ndarray]:
        features = self._extract_dataset(
            self._dataset, self._forecast_features, sample_index
        )
        if self._forecast_overlap is not None and self._forecast_overlap > 0:
            dim_indexes = sample_index.copy()
            dim_indexes['date'] = range(
                dim_indexes['date']
                + 1
                - self._min_lead_time
                - self._forecast_overlap,
                dim_indexes['date'] + 1 - self._min_lead_time,
            )
            dim_indexes['lead_time'] = 0
            overlaps = self._extract_dataset(
                self._dataset, self._forecast_features, dim_indexes
            )
            features = {
                name: np.concatenate([overlaps[name], feature])
                for name, feature in features.items()
            }
        # This adds a dimension to many features, as required by some models.
        # There is no need for this except that it is how basedataset works,
        # and everything else expects the trailing dim.
        return {
            name: np.expand_dims(feature, -1)
            for name, feature in features.items()
        }

    def _extract_targets(self, sample_index: dict[str, int]) -> np.ndarray:
        dim_indexes = sample_index.copy()
        dim_indexes['date'] = self._calc_date_range(sample_index, lead=True)
        features = self._extract_dataset(
            self._dataset, self._target_features, dim_indexes
        )
        return np.stack([features[e] for e in self._target_features], axis=-1)

    def _extract_per_basin_stds(
        self, sample_index: dict[str, int]
    ) -> np.ndarray:
        assert self._per_basin_target_stds is not None
        features = self._extract_dataset(
            self._per_basin_target_stds,
            self._target_features,
            {'basin': sample_index['basin']},
        )
        # This adds a dimension to many features, as required by some models.
        # There is no need for this except that it is how basedataset works,
        # and everything else expects the trailing dim.
        return np.expand_dims(
            np.stack([features[e] for e in self._target_features], axis=-1),
            axis=0,
        )

    def _get_period_dates(
        self, cfg: Config
    ) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
        if self._period == 'train':
            start_dates, end_dates = cfg.train_start_date, cfg.train_end_date
        elif self._period == 'test':
            start_dates, end_dates = cfg.test_start_date, cfg.test_end_date
        elif self._period == 'validation':
            start_dates, end_dates = (
                cfg.validation_start_date,
                cfg.validation_end_date,
            )
        else:
            raise ValueError(f'Unknown period {self._period}')
        if len(start_dates) != len(end_dates):
            raise ValueError(
                f'Start and end date lists for period {self._period} must '
                'have the same length.'
            )
        if any(
            start >= end
            for start, end in zip(start_dates, end_dates, strict=True)
        ):
            raise ValueError(
                f'Start dates {start_dates} are before matched end dates '
                f'{end_dates}.'
            )
        return start_dates, end_dates

    def _union_ranges(
        self, start_dates: list[pd.Timestamp], end_dates: list[pd.Timestamp]
    ) -> pd.DatetimeIndex:
        ranges = [
            pd.date_range(start, end)
            for start, end in zip(start_dates, end_dates, strict=True)
        ]
        return functools.reduce(pd.Index.union, ranges)

    def _create_valid_sample_mask(
        self,
    ) -> tuple[xr.DataArray, tuple[dask.array.Array, ...]]:
        """Map int sample indexes to the int positions into the xr.Dataset.

        Allows index-based sample retrieval, faster than coordinate-based sample
        retrieval.
        """
        # Create a boolean mask for the original dataset noting valid (True)
        # vs. invalid (False) samples.
        valid_sample_mask = validate_samples(
            is_train=self.is_train,
            dataset=self._dataset,
            nan_handling_method=self._nan_handling_method,
            sample_dates=self._sample_dates,
            lead_time=self.lead_time,
            seq_length=self._seq_length,
            predict_last_n=self._predict_last_n,
            forecast_overlap=self._forecast_overlap,
            min_lead_time=self._min_lead_time,
            static_features=(
                self._static_features
                if self._nan_handling_method == 'masked_mean'
                else self._all_static_features
            ),
            forecast_features=self._forecast_features,
            hindcast_features=self._hindcast_features,
            target_features=self._target_features,
            feature_groups=self._feature_groups,
            allzero_samples_are_invalid=self._allzero_samples_are_invalid,
        )[0]

        # Convert boolean valid sample mask into indexes of all samples. This
        # retains only the portion of the valid sample mask with True values.
        # Each element is a list of valid integer positions (indexers) for which
        # values are True for a dimension.
        indices = dask.array.nonzero(valid_sample_mask.data)
        # Compact memory widths. Values are indexes within each mask's shape.
        min_dtypes = map(np.min_scalar_type, valid_sample_mask.shape)
        indices = tuple(
            idx.astype(dt) for idx, dt in zip(indices, min_dtypes, strict=True)
        )

        return valid_sample_mask, indices

    def _create_sample_index(
        self, valid_sample_mask: xr.DataArray, indices: np.ndarray
    ) -> None:
        """Create the sample index structure to access the mapping."""
        # Count the number of valid samples.
        num_samples = len(indices[0]) if indices else 0
        if num_samples == 0:
            if self._period == 'train':
                raise NoTrainDataError
            raise NoEvaluationDataError

        # Align dim name with its respective list of int indices (index arrays),
        # i.e. columns of all basins, all dates, etc.
        aligned_indices = tuple(
            (dim, indices[i])
            for i, dim in enumerate(valid_sample_mask.dims)
            if dim != 'sample'
        )

        self._sample_index = SampleIndexer(aligned_indices)
        self._num_samples = num_samples

    def _extract_dataset(
        self,
        data: xr.Dataset,
        features: list[str],
        indexers: dict[Hashable, int | range | slice],
    ) -> dict[str, np.ndarray | np.float32]:
        def extract(feature_name: str) -> np.ndarray | np.float32:
            key = f'{id(data)}{feature_name}'
            feature = self._data_cache.get(key)
            if feature is None:
                feature = self._data_cache[key] = data[feature_name]
            return _extract_dataarray(feature, indexers)

        return {
            feature_name: extract(feature_name) for feature_name in features
        }

    def _load_data(self) -> xr.Dataset:
        """Load and merge all Caravan-Multimet features into one dataset.

        Returns an xr dataset of features with the following dimensions:
        (basin, date, lead_time). This loading function aggregates hindcast,
        forecast, statics, and target data.

        Returns:
            xr.Dataset: Dataset containing the loaded features with various
                dimensions.
        """
        datasets = []
        if self._static_features is not None:
            LOGGER.debug('load attributes')
            datasets.append(self._load_static_features())
        if self._hindcast_features is not None:
            LOGGER.debug('load hindcast features')
            datasets.extend(self._load_hindcast_features())
        if self._forecast_features is not None:
            LOGGER.debug('load forecast features')
            datasets.extend(self._load_forecast_features())
        if self._target_features is not None:
            LOGGER.debug('load target features')
            datasets.append(self._load_target_features())
        if not datasets:
            raise ValueError('At least one type of data must be loaded.')

        LOGGER.debug('merge')
        ds = xr.merge(datasets, join='outer', compat='no_conflicts')

        LOGGER.debug('rechunk')
        return rechunk(ds)

    def _load_hindcast_features(self) -> list[xr.Dataset]:
        """Load Caravan-Multimet data for hindcast features.

        Returns:
            list[xr.Dataset]: Datasets containing the loaded features with
                dimensions (date, basin).
        """
        return self._load_hindcast_as_zarr()

    def _load_hindcast_as_zarr(self) -> list[xr.Dataset]:  # noqa: C901
        """Load Caravan-Multimet data for hindcast features.

        Returns:
            list[xr.Dataset]: Datasets containing the loaded hindcast features.
        """
        # Prepare hindcast features to load, including the masks of
        # union_mapping. dict.fromkeys (rather than a set) dedups while
        # preserving insertion order, so the resulting `data_vars` order is
        # stable across processes.
        features = list(
            dict.fromkeys(
                [
                    *self._hindcast_features,
                    *(self._union_mapping or {}).values(),
                ]
            )
        )

        # Check if single unified dynamics zarr store contains the features
        single_store_path = _find_single_dynamics_zarr_path(
            self._dynamics_data_path
        )
        if single_store_path is not None:
            ds = _open_zarr(single_store_path)
            missing = sorted(set(features) - set(ds.data_vars))
            if missing:
                raise ValueError(
                    f'Requested hindcast features {missing} not found in '
                    f'{single_store_path}.'
                )
            if 'lead_time' in ds:
                ds = ds.sel(
                    basin=self._basins, lead_time=self._lead_time_slice()
                )
            else:
                ds = ds.sel(basin=self._basins)
            return [ds[features]]

        # Separate products and bands for each product from the configured
        # hindcast inputs.
        product_bands = _get_products_and_bands_from_features(
            self._hindcast_inputs
        )

        # Also load fallback variables used by union_mapping.
        if self._union_mapping:
            union_product_bands = _get_products_and_bands_from_feature_strings(
                self._union_mapping.values()
            )
            for product, bands in union_product_bands.items():
                existing_bands = product_bands.setdefault(product, [])
                for band in bands:
                    if band not in existing_bands:
                        existing_bands.append(band)

        # Initialize storage for product/band datasets to concatenate.
        product_dss = []

        # Load data for the selected products, bands, and basins.
        for product, bands in product_bands.items():
            product_path = _find_product_zarr_path(
                self._dynamics_data_path, product
            )
            LOGGER.info(
                "Loading hindcast product '%s' with bands %s", product, bands
            )
            product_ds = _open_zarr(product_path)

            missing = sorted(set(bands) - set(product_ds.data_vars))
            if missing:
                raise ValueError(
                    f'Requested features {missing} not found in product '
                    f"'{product}'. Available variables: "
                    f'{list(product_ds.data_vars)}'
                )

            if 'lead_time' in product_ds:
                # The same product may be used both for forecast and hindcast
                # features. For hindcast, we load it with the full lead_time
                # similar to forecast, and filter minimal lead_time in sampling.
                product_ds = product_ds.sel(
                    basin=self._basins, lead_time=self._lead_time_slice()
                )
            else:
                product_ds = product_ds.sel(basin=self._basins)

            product_ds = product_ds[bands]
            product_dss.append(product_ds)

        return product_dss

    def _load_forecast_features(self) -> list[xr.Dataset]:
        """Load Caravan-Multimet data for forecast features.

        Returns:
            list[xr.Dataset]: Datasets containing the loaded features with
                dimensions (date, lead_time, basin).
        """
        return self._load_forecast_as_zarr()

    def _load_forecast_as_zarr(self) -> list[xr.Dataset]:
        """Load Caravan-Multimet data for forecast features.

        Returns:
            list[xr.Dataset]: Datasets containing loaded features with
                dimensions (date, lead_time, basin).
        """
        # Check if single unified dynamics zarr store contains the forecast
        # features.
        single_store_path = _find_single_dynamics_zarr_path(
            self._dynamics_data_path
        )
        if single_store_path is not None:
            ds = _open_zarr(single_store_path)
            missing = sorted(set(self._forecast_features) - set(ds.data_vars))
            if missing:
                raise ValueError(
                    f'Requested forecast features {missing} not found in '
                    f'{single_store_path}.'
                )
            if 'lead_time' not in ds:
                raise ValueError(
                    f'Lead times do not exist in forecast dataset at '
                    f'{single_store_path}.'
                )
            ds = ds.sel(basin=self._basins, lead_time=self._lead_time_slice())
            return [ds[self._forecast_features]]

        # Separate products and bands for each product from configured inputs.
        product_bands = _get_products_and_bands_from_features(
            self._forecast_inputs
        )

        # Initialize storage for product/band datasets to concatenate.
        product_dss = []

        # Load data for the selected products, bands, and basins.
        for product, bands in product_bands.items():
            product_path = _find_product_zarr_path(
                self._dynamics_data_path, product
            )
            LOGGER.info(
                "Loading forecast product '%s' with bands %s", product, bands
            )
            product_ds = _open_zarr(product_path)

            missing = sorted(set(bands) - set(product_ds.data_vars))
            if missing:
                raise ValueError(
                    f'Requested features {missing} not found in product '
                    f"'{product}'. Available variables: "
                    f'{list(product_ds.data_vars)}'
                )

            # If this is a forecast product, extract only leadtime 0.
            if 'lead_time' not in product_ds:
                raise ValueError(
                    f'Lead times do not exist for forecast product ({product}).'
                )

            product_ds = product_ds.sel(
                basin=self._basins, lead_time=self._lead_time_slice()
            )[bands]
            product_dss.append(product_ds)

        return product_dss

    def _load_target_features(self) -> xr.Dataset:
        """Load Caravan streamflow data.

        Returns:
            xr.Dataset: Dataset containing the loaded features with dimensions
                (date, basin).
        """
        return load_caravan_timeseries(
            data_dir=self._targets_data_path,
            basins=self._basins,
            target_features=self._target_features,
            csv=self._cfg.load_as_csv,
        )

    def _load_static_features(self) -> xr.Dataset:
        """Load Caravan static attributes.

        Returns:
            xr.Dataset: Dataset containing the loaded features with dimensions
                (basin).
        """
        return load_caravan_attributes(
            data_dir=self._statics_data_path,
            basins=self._basins,
            features=self._all_static_features,
        )

    def _lead_time_slice(self) -> slice:
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#endpoints-are-inclusive
        max_lead_time = max(self.lead_time, MULTIMET_MINIMUM_LEAD_TIME)
        return slice(
            pd.Timedelta(days=MULTIMET_MINIMUM_LEAD_TIME),
            pd.Timedelta(days=max_lead_time),
        )

    @staticmethod
    def collate_fn(
        samples: list[
            dict[str, torch.Tensor | np.ndarray, dict[str, torch.Tensor]]
        ],
    ) -> dict[str, torch.Tensor | np.ndarray, dict[str, torch.Tensor]]:
        """Stack a list of tensor samples into a single batch dictionary."""
        batch = {}
        if not samples:
            return batch
        features = list(samples[0].keys())
        for feature in features:
            if feature.startswith('date'):
                # Dates are stored as a numpy array of datetime64, which we
                # maintain as numpy array.
                batch[feature] = np.stack(
                    [sample[feature] for sample in samples], axis=0
                )
            elif feature.startswith('x_d') or feature == 'x_s_local':
                # Dynamics and local static attribute groups are stored as
                # dictionaries with feature names as keys.
                batch[feature] = {
                    k: torch.stack(
                        [sample[feature][k] for sample in samples], dim=0
                    )
                    for k in samples[0][feature]
                }
            else:
                # Everything else is a torch.Tensor.
                batch[feature] = torch.stack(
                    [sample[feature] for sample in samples], dim=0
                )
        return batch


def _extract_dataarray(
    data: xr.DataArray, indexers: dict[Hashable, int | range | slice]
) -> np.ndarray | np.float32:
    """Return the values in array according to dims given by indexers.

    This function replaces uses of `isel` with data and indexers.
    """
    locs = (
        indexers[dim] if dim in indexers else slice(None) for dim in data.dims
    )
    # Convert range(0, n) to [0, 1, ..., n-1] as arrays don't support range.
    locs = (list(loc) if isinstance(loc, range) else loc for loc in locs)
    return data.data[tuple(locs)]


def _assert_floats_are_float32(dataset: xr.Dataset) -> None:
    items = itertools.chain(dataset.data_vars.items(), dataset.coords.items())
    for name, data_array_or_coord in items:
        if np.issubdtype(data_array_or_coord.dtype, np.floating):
            assert data_array_or_coord.dtype == np.float32, (
                f"Data variable or coord '{name}' is a float but not float32. "
                f'Actual dtype: {data_array_or_coord.dtype}'
            )


def _convert_to_tensor(
    key: str, value: np.ndarray
) -> torch.Tensor | np.ndarray:
    if key in NUMPY_VARS:
        return value
    if key not in TENSOR_VARS:
        raise ValueError(f'Unrecognized data key: {key}')
    if isinstance(value, dict):
        return {k: torch.from_numpy(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    raise ValueError(f'Unrecognized data type: {type(value)}')


def _find_single_dynamics_zarr_path(
    dynamics_path: Path | str | list[Path | str],
) -> Path | None:
    if isinstance(dynamics_path, list):
        return None
    path_str = (
        dynamics_path.as_posix()
        if isinstance(dynamics_path, Path)
        else str(dynamics_path)
    )
    if path_str.startswith(('gs://', 'gs:/')):
        if path_str.rstrip('/').endswith('.zarr'):
            return Path(path_str.rstrip('/'))
        return None

    p = Path(dynamics_path)
    if p.suffix == '.zarr':
        if not p.exists():
            raise FileNotFoundError(f'Dynamics Zarr store not found: {p}')
        return p
    if not p.exists():
        raise FileNotFoundError(f'Dynamics data path not found: {p}')
    if (p / TIMESERIES_ZARR_NAME).exists():
        return p / TIMESERIES_ZARR_NAME
    return None


def _find_product_zarr_path(
    dynamics_path: Path | str | list[Path | str], product: str
) -> Path:
    paths = (
        dynamics_path if isinstance(dynamics_path, list) else [dynamics_path]
    )
    checked: list[Path] = []
    local_matches: list[Path] = []
    cloud_candidates: list[Path] = []
    for dp in paths:
        path_str = dp.as_posix() if isinstance(dp, Path) else str(dp)
        if path_str.startswith(('gs://', 'gs:/')):
            cloud_candidates.append(
                Path(f'{path_str.rstrip("/")}/{product}/{TIMESERIES_ZARR_NAME}')
            )
            continue
        p = Path(dp)
        if not p.exists():
            raise FileNotFoundError(f'Dynamics data path not found: {p}')
        candidate = p / product / TIMESERIES_ZARR_NAME
        checked.append(candidate)
        if candidate.exists():
            local_matches.append(candidate)
    if local_matches:
        resolved = local_matches[0]
        if len(local_matches) > 1:
            LOGGER.warning(
                "Product '%s' found in multiple local directories %s; using %s",
                product,
                [str(m) for m in local_matches],
                resolved,
            )
        LOGGER.info("Resolved product '%s' Zarr store to %s", product, resolved)
        return resolved
    if cloud_candidates:
        resolved = cloud_candidates[0]
        LOGGER.info("Resolved product '%s' Zarr store to %s", product, resolved)
        return resolved
    if len(checked) == 1:
        raise FileNotFoundError(
            f"Zarr store for product '{product}' not found at {checked[0]}"
        )
    raise FileNotFoundError(
        f"Zarr store for product '{product}' not found in {checked}"
    )


@functools.cache
def _open_zarr(path: Path) -> xr.Dataset:
    str_path = str(path)
    if str_path.startswith(('gs:', 'gs/')):
        store = path.as_posix().replace('gs:/', 'gs://')
    else:
        store = str_path
    is_cloud = str_path.startswith(('gs:', 'gs/'))
    # `chunks={}` keeps the on-disk Zarr chunks (e.g. `basin=128` in
    # Caravan-MultiMet) so that lazily loaded samples only read the chunk of
    # their own basin.
    return xr.open_zarr(
        store=store,
        chunks={},
        decode_timedelta=True,
        consolidated=is_cloud,
    )


def _get_products_and_bands_from_feature_strings(
    features: Iterable[str],
) -> dict[str, list[str]]:
    """Process feature strings to create a dictionary of product to band(s).

    Args:
        features: Feature names in the format '<product>_<band>'.

    Returns:
        dict[str, list[str]]: Keys are canonical product names and values are
            lists of features. Feature names are preserved.
    """
    product_bands: dict[str, list[str]] = {}
    for feature in features:
        product = _product_name_from_feature(feature)
        existing = product_bands.setdefault(product, [])
        if feature not in existing:
            existing.append(feature)
    return product_bands


def _get_products_and_bands_from_features(
    features: dict[str, list[str]] | list[list[str]] | Iterable[str],
) -> dict[str, list[str]]:
    """Create a mapping of canonical product names to feature bands.

    Args:
        features: Either:
            - A dictionary where keys are product names from the config and
              values are lists of features belonging to that product, or
            - A flat or nested iterable of feature names in the format
              '<product>_<band>'.

    Returns:
        dict[str, list[str]]: Dictionary mapping canonical product names to
            their associated feature bands.
    """
    if isinstance(features, dict):
        product_bands: dict[str, list[str]] = {}
        for product, bands in features.items():
            canonical = _canonical_product_name(product)
            existing = product_bands.setdefault(canonical, [])
            for band in bands:
                if band not in existing:
                    existing.append(band)
        return product_bands

    flat_features = flatten_feature_list(features)
    return _get_products_and_bands_from_feature_strings(flat_features)


class SampleIndexer:
    """Reorg columns to rows.

    Map sample index i [0, num_samples) to a dict that maps an int
    position for that sample in each dim.
    E.g. {1: {'basin': 2, 'date': 3}}

    This allows integer indexing into each coordinate dimension of
    the original dataset, while ONLY selecting valid samples. The full
    original dataset is retained (including not-valid samples) for
    sequence construction.
    """

    def __init__(self, aligned_indices: tuple[tuple[str, np.ndarray]]) -> None:
        """Store the `(dim, index array)` pairs that make up the columns."""
        self._aligned_indices = aligned_indices

    def __getitem__(self, item: int) -> dict[str, int]:
        """Return the per-dimension int positions of sample `item`."""
        return {dim: indexes[item] for dim, indexes in self._aligned_indices}

    def keys(self) -> Iterator[int]:
        """Return the sample indexes `range(len(self))`."""
        return range(len(self))

    def values(self) -> Iterator[dict[str, int]]:
        """Iterate over the per-dimension positions of every sample."""
        return (self[i] for i in self.keys())

    def items(self) -> Iterator[tuple[int, dict[str, int]]]:
        """Iterate over `(sample index, per-dimension positions)` pairs."""
        return zip(self.keys(), self.values(), strict=True)

    def __len__(self) -> int:
        """Return the number of samples."""
        _, indexes = self._aligned_indices[0]
        return len(indexes)

    def get_column(self, dim: str) -> np.ndarray:
        """Return the index array of all samples for dimension `dim`."""
        return next(v for (k, v) in self._aligned_indices if k == dim)


def _finest_chunks(ds: xr.Dataset, dim: str) -> tuple[int, ...] | None:
    """Return the finest multi-element chunking of `dim` across variables.

    The result is the union of the chunk boundaries of every dask-backed
    variable that has `dim` with multi-element chunks (ignoring trivial
    1-element-per-chunk arrays from legacy per-file NetCDF concatenation).
    Returns None when no variable has multi-element chunks along `dim`.
    """
    boundaries: set[int] = set()
    dim_size = ds.sizes.get(dim, 0)
    for variable in ds.variables.values():
        if variable.chunks is None or dim not in variable.dims:
            continue
        dim_chunks = variable.chunksizes[dim]
        if not dim_chunks or (dim_size > 1 and max(dim_chunks) == 1):
            continue
        boundaries.update(itertools.accumulate(dim_chunks))
    if not boundaries:
        return None
    edges = np.array([0, *sorted(boundaries)])
    return tuple(int(size) for size in np.diff(edges))


def rechunk(ds: xr.Dataset) -> xr.Dataset:
    """Return `ds` with dask-backed variables and consistent chunks.

    Every variable is coerced to a dask array. The `basin` dimension keeps the
    finest chunking already present in `ds` (the on-disk Zarr chunks, e.g.
    `basin=128` in Caravan-MultiMet), so that a lazily loaded sample only reads
    the chunk of its own basin. All other dimensions are chunked with dask's
    `'auto'` heuristic, which would otherwise also merge all basins of a small
    dataset into a single chunk.
    """
    chunks: dict[Hashable, str | tuple[int, ...]] = dict.fromkeys(
        ds.dims, 'auto'
    )
    basin_chunks = _finest_chunks(ds, 'basin')
    if basin_chunks is not None:
        chunks['basin'] = basin_chunks
    return ds.chunk(chunks).unify_chunks()
