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

"""Abstract base class for flood return period distribution fitters.

Implements data validation, base-10 logarithmic transformations, and the
interface for forward and inverse exceedance probability calculations following
USGS Bulletin 17C (England et al., 2019):
https://pubs.usgs.gov/tm/04/b05/tm4b5.pdf
"""

import abc
from collections.abc import Sequence

import numpy as np

from return_periods import exceptions

# Minimum number of positive observations required to fit a 3-parameter
# distribution reliably.
_MINIMUM_FIT_DATA_POINTS = 5

# Small value used to keep exceedance probabilities strictly within (0, 1).
_EPSILON = 1e-6


class BaseFitter(abc.ABC):
    """Abstract base class for flood frequency distribution fitters."""

    def __init__(
        self,
        data: Sequence[float] | np.ndarray,
        log_transform: bool = True,  # noqa: FBT001, FBT002
    ):
        """Initialize the distribution fitter and validate peak flow data.

        Args:
            data: Sequence or 1D array of annual peak streamflow values.
            log_transform: Whether to apply a base-10 logarithmic transform to
                the flow data prior to fitting.

        Raises:
            InvalidFlowValueError: If any flow value is negative or NaN.
            NotEnoughDataError: If fewer than 5 positive flow values are
                provided.
        """
        raw = np.asarray(data, dtype=float).ravel()

        if np.any(~np.isfinite(raw)) or np.any(raw < 0):
            raise exceptions.InvalidFlowValueError(
                'Flow values cannot be negative, NaN, or infinite.'
            )

        self.raw_sample = raw
        self.num_zero_flows = int(np.sum(raw == 0))

        positive_data = raw[raw > 0]
        if len(positive_data) < _MINIMUM_FIT_DATA_POINTS:
            raise exceptions.NotEnoughDataError(
                'Need at least '
                f'{_MINIMUM_FIT_DATA_POINTS} positive data points to fit a '
                f'distribution; got {len(positive_data)}.'
            )

        self._log_transform = log_transform
        self.sample = positive_data
        self.transformed_sample = self._transform_data(positive_data)

        # Subclasses that perform low-outlier screening (e.g., GEMAFitter) set
        # _pilf_threshold in transformed space.
        self._pilf_threshold: float | None = None

    @property
    def record_length(self) -> int:
        """Return the number of positive peak flow observations."""
        return len(self.sample)

    @property
    def total_record_length(self) -> int:
        """Return the total number of annual peaks including zero flows."""
        return len(self.raw_sample)

    @property
    def pilf_threshold(self) -> float:
        """Return the PILF threshold in untransformed discharge units."""
        if self._pilf_threshold is not None and np.isfinite(
            self._pilf_threshold
        ):
            untransformed = self._untransform_data(
                np.asarray([self._pilf_threshold], dtype=float)
            )
            return float(untransformed[0])
        return 0.0

    def _transform_data(self, data: Sequence[float] | np.ndarray) -> np.ndarray:
        """Transform positive flow values into fitting space."""
        arr = np.asarray(data, dtype=float)
        if np.any(~np.isfinite(arr)) or np.any(arr <= 0):
            raise exceptions.InvalidFlowValueError(
                f'Flow values must be finite and strictly positive, got: {data}'
            )
        if self._log_transform:
            return np.log10(arr)
        return arr.copy()

    def _untransform_data(
        self, data: Sequence[float] | np.ndarray
    ) -> np.ndarray:
        """Inverse-transform values from fitting space back to flow space."""
        arr = np.asarray(data, dtype=float)
        if self._log_transform:
            return np.power(10.0, arr)
        return arr.copy()

    def _validate_flow_array(
        self,
        flows: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Validate that flow values are finite and non-negative."""
        arr = np.asarray(flows, dtype=float)
        if np.any(~np.isfinite(arr)) or np.any(arr < 0):
            raise exceptions.InvalidFlowValueError(
                f'Flow values must be finite and non-negative, got: {flows}'
            )
        return arr

    def _check_exceedance_probabilities(
        self,
        exceedance_probabilities: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Validate that exceedance probabilities lie strictly in (0, 1)."""
        probs = np.asarray(exceedance_probabilities, dtype=float)
        if np.any(np.isnan(probs)) or np.any(probs <= 0) or np.any(probs >= 1):
            raise ValueError(
                'All exceedance probabilities must be strictly in (0, 1).'
            )
        return probs

    @abc.abstractmethod
    def exceedance_probabilities_from_flow_values(
        self,
        flows: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Calculate annual exceedance probabilities for given flow values."""

    @abc.abstractmethod
    def flow_values_from_exceedance_probabilities(
        self,
        exceedance_probabilities: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Calculate flow quantiles for given exceedance probabilities."""
