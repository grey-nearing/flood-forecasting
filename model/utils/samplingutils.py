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

from typing import Callable

import torch

from model.datautils.scaler import Scaler
from model.utils import cmal_deterministic
from model.utils.config import Config


def sample_pointpredictions(
    model: 'BaseModel',
    data: dict[str, torch.Tensor],
    n_samples: int,
    scaler: Scaler,
    *,
    outputs: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Point prediction samplers for the different uncertainty estimation approaches.

    This function provides different point sampling functions for the supported uncertainty estimation approaches
    (e.g. Countable Mixtures of Asymmetric Laplacians (CMAL)).

    There are also options to handle negative point prediction samples that arise while sampling from the uncertainty
    estimates. This functionality currently supports (a) 'clip' for directly clipping values at zero and
    (b) 'truncate' for resampling values that are below zero.

    Parameters
    ----------
    model : BaseModel
        The OpenHydroNet model from which to sample from.
    data : dict[str, torch.Tensor]
        Dictionary, containing input features as key-value pairs.
    n_samples : int
        The number of point prediction samples that should be created.
    scaler : Scaler
        Scaler of the run.
    outputs, optional
        Model forward result

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary, containing the sampled model outputs for the `predict_last_n` (config argument) time steps.
    """

    if model.cfg.head.lower() == 'cmal':
        samples = sample_cmal(model, data, n_samples, scaler, outputs=outputs)
    elif model.cfg.head.lower() == 'cmal_deterministic':
        samples = sample_cmal_deterministic(
            model, data, scaler, outputs=outputs
        )
    else:
        raise NotImplementedError(
            f'Sampling mode not supported for head {model.cfg.head.lower()}!'
        )

    return samples


def _subset_target(
    parameter: dict[str, torch.Tensor], n_target: int, steps: int
) -> dict[str, torch.Tensor]:
    # determine which output neurons correspond to the n_target target variable
    start = n_target * steps
    end = start + steps
    parameter_sub = parameter[:, :, start:end]
    return parameter_sub


def _calc_normalized_zero_thresholds(
    scaler: Scaler,
    targets: list[str],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Calculate the normalized zero value threshold for target vars.

    Extract 'center' and 'scale' params for each target var from scaler
    and return -center/scale via tensors (gpu).
    """
    parameter = ['center', 'scale']
    params = [scaler.scaler[e].sel(parameter=parameter).data for e in targets]
    centers, scales = zip(*params)
    centers = torch.tensor(centers, device=device, dtype=dtype)
    scales = torch.tensor(scales, device=device, dtype=dtype)
    return -(centers / scales)


def _handle_negative_values(
    cfg: Config,
    values: torch.Tensor,
    sample_values: Callable,
    normalized_zero: torch.Tensor,
) -> torch.Tensor:
    """Handle negative samples that arise while sampling from the uncertainty estimates.

    Currently supports (a) 'clip' for directly clipping values at physical zero
    (``normalized_zero``), (b) 'truncate' for resampling values that are below zero,
    and (c) 'none' (or ``None`` / omitted) to leave CMAL draws and summary statistics
    completely unclipped.

    When ``negative_sample_handling: 'clip'`` is configured, clamping at physical zero
    (``normalized_zero``) is applied inside ``model.sample()`` (``sample_cmal`` and
    ``sample_cmal_deterministic``). Therefore, during evaluation
    (``BaseTester.evaluate``), evaluation metrics (``NSE``, ``KGE``, etc.) are
    computed after clipping and sample reduction (``tester_sample_reduction``),
    whereas losses (``cmalloss``) are computed on the raw distribution parameters
    before sampling or clipping.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    values : torch.Tensor
        Tensor with the sampled values.
    sample_values : Callable
        Sampling function to allow for repeated sampling in the case of truncation-handling.
    normalized_zero : torch.Tensor
        1D tensor of shape [cfg.target_variables] with normalized zero threshold
        foreach target variable.

    Returns
    -------
    torch.Tensor
        Bound values according to user specifications.
    """
    match (cfg.negative_sample_handling or '').lower():
        case 'clip':
            return torch.clamp(values, min=normalized_zero)
        case 'truncate':
            values_smaller_zero = values < normalized_zero
            try_count = 0
            while torch.any(values_smaller_zero.flatten()):
                values[values_smaller_zero] = sample_values(values_smaller_zero)
                values_smaller_zero = values < normalized_zero
                try_count += 1
                if try_count >= cfg.negative_sample_max_retries:
                    break
            return values
        case '' | 'none':
            return values
        case _:
            raise NotImplementedError(
                f'The option {cfg.negative_sample_handling} is not supported for handling negative samples!'
            )


def _sample_asymmetric_laplacians(
    ids: list[int],
    m_sub: torch.Tensor,
    b_sub: torch.Tensor,
    t_sub: torch.Tensor,
) -> torch.Tensor:
    # The ids are used for location-specific resampling for 'truncation' in '_handle_negative_values'
    m_sub_ids = m_sub[ids]
    prob = torch.rand_like(m_sub_ids)  # sample uniformly in [0,1)
    t_sub_ids = torch.clamp(t_sub[ids], 1e-6, 1.0 - 1e-6)
    t_sub_ids_c = 1 - t_sub_ids
    b_sub_ids = b_sub[ids]
    values = torch.where(
        prob < t_sub_ids,  # needs to be in accordance with the loss
        m_sub_ids + ((b_sub_ids * torch.log(prob / t_sub_ids)) / t_sub_ids_c),
        m_sub_ids
        - ((b_sub_ids * torch.log((1 - prob) / t_sub_ids_c)) / t_sub_ids),
    )
    return values.flatten()


class _SamplingSetup:
    def __init__(
        self, model: 'BaseModel', data: dict[str, torch.Tensor], head: str
    ):
        # make model checks:
        cfg = model.cfg
        if not cfg.head.lower() == head.lower():
            raise NotImplementedError(
                f'{head} sampling not supported for the {cfg.head} head!'
            )

        # assign setup properties:
        self.cfg = cfg
        self.device = next(model.parameters()).device
        self.number_of_targets = len(cfg.target_variables)
        self.predict_last_n = cfg.predict_last_n
        self.batch_size_data = data['y'].shape[0]


def sample_cmal_deterministic(
    model: 'BaseModel',
    data: dict[str, torch.Tensor],
    scaler: Scaler,
    *,
    outputs: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Sample 10 point predictions with the Countable Mixture of Asymmetric Laplacians (CMAL) head.

    Setting ``negative_sample_handling: 'none'`` (or omitting it / ``None``) leaves
    CMAL summary statistics completely unclipped. When ``negative_sample_handling: 'clip'``
    is configured, clamping at physical zero (``normalized_zero``) is applied inside
    ``model.sample()`` (``sample_cmal`` and ``sample_cmal_deterministic``). Therefore,
    during evaluation (``BaseTester.evaluate``), evaluation metrics (``NSE``, ``KGE``,
    etc.) are computed after clipping and sample reduction (``tester_sample_reduction``),
    whereas losses (``cmalloss``) are computed on the raw distribution parameters
    before sampling or clipping.

    Parameters
    ----------
    model : BaseModel
        A model with a CMAL head.
    data : dict[str, torch.Tensor]
        Dictionary, containing input features as key-value pairs.
    scaler : Scaler
        Scaler of the run.
    outputs, optional
        Model forward result

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary, containing the sampled model outputs for the `predict_last_n` (config argument) time steps.
        The shape of the output tensor is ``[batch size, predict_last_n, target, n_samples]``.
    """
    setup = _SamplingSetup(model, data, 'cmal_deterministic')

    # Make predictions (forward pass). For CMAL head those are dist params and
    # not point predictions.
    pred = outputs or model(data)

    # This head returns summary statistics (the mixture mean plus nine
    # quantiles), not random draws, so 'truncate' has nothing to resample and
    # is a no-op here. Everything else is delegated to the shared negative
    # value handler, so an unsupported value still raises.
    negative_handling = (setup.cfg.negative_sample_handling or '').lower()
    normalized_zeros = None
    if negative_handling not in ('', 'none', 'truncate'):
        normalized_zeros = _calc_normalized_zero_thresholds(
            scaler=scaler,
            targets=setup.cfg.target_variables,
            device=next(model.parameters()).device,
            dtype=next(model.parameters()).dtype,
        )

    mu = pred['mu']  # means
    b = pred['b']  # scales
    tau = pred['tau']  # asymmetries
    pi = pred['pi']  # weights

    values = cmal_deterministic.generate_predictions(mu, b, tau, pi)
    # Element 0 is the mixture mean. Clipping it applies a physical floor
    # to E[X]; it is not the distributional correction E[max(X, 0)].
    if normalized_zeros is not None:
        values = _handle_negative_values(
            setup.cfg,
            values,
            # Unused: 'clip' never resamples, and a summary statistic
            # cannot be redrawn.
            sample_values=lambda _: values,
            # generate_predictions collapses every target into a single
            # mixture, so values only ever holds target 0.
            normalized_zero=normalized_zeros[0],
        )
    return {'y_hat': torch.stack([values], 2)}


def sample_cmal(
    model: 'BaseModel',
    data: dict[str, torch.Tensor],
    n_samples: int,
    scaler: Scaler,
    *,
    outputs: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Sample point predictions with the Countable Mixture of Asymmetric Laplacians (CMAL) head.

    This function generates `n_samples` CMAL sample points for each entry in the batch. Concretely, the model is
    executed once (forward pass) and then the sample points are generated by sampling from the resulting mixtures.
    General information about CMAL can be found in [#]_.

    The negative sample handling currently supports (a) 'clip' for directly clipping sample_points at zero and (b)
    'truncate' for resampling sample_points that are below zero. The mode can be defined by the config argument
    'negative_sample_handling'.

    Parameters
    ----------
    model : BaseModel
        A model with a CMAL head.
    data : dict[str, torch.Tensor]
        Dictionary, containing input features as key-value pairs.
    n_samples : int
        Number of samples to generate for each input sample.
    scaler : Scaler
        Scaler of the run.
    outputs, optional
        Model forward result

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary, containing the sampled model outputs for the `predict_last_n` (config argument) time steps.
        The shape of the output tensor is ``[batch size, predict_last_n, target, n_samples]``.

    References
    ----------
    .. [#] D.Klotz, F. Kratzert, M. Gauch, A. K. Sampson, G. Klambauer, S. Hochreiter, and G. Nearing:
        Uncertainty Estimation with Deep Learning for Rainfall-Runoff Modelling. arXiv preprint arXiv:2012.14295,
        2020.
    """
    setup = _SamplingSetup(model, data, 'cmal')

    # Make predictions (forward pass). For CMAL head those are dist params and
    # not point predictions.
    pred = outputs or model(data)

    normalized_zeros = _calc_normalized_zero_thresholds(
        scaler=scaler,
        targets=setup.cfg.target_variables,
        device=next(model.parameters()).device,
        dtype=next(model.parameters()).dtype,
    )

    predict_last_n = setup.cfg.predict_last_n

    # Extract the four parameters of the CMAL distributions.
    m = pred['mu']  # location means
    b = pred['b']  # scales
    t = pred['tau']  # asymmetries
    p = pred['pi']  # mixture weights

    sample_points = []  # (for each target parameter)
    for nth_target in range(
        setup.number_of_targets
    ):  # e.g. streamflow, temp
        # Slice each full param tensor from the model's concat'd params to get
        # only the portion relevant to the current target.
        m_target = _subset_target(
            m[:, -predict_last_n:, :],
            nth_target,
            setup.cfg.n_distributions,
        )
        b_target = _subset_target(
            b[:, -predict_last_n:, :],
            nth_target,
            setup.cfg.n_distributions,
        )
        t_target = _subset_target(
            t[:, -predict_last_n:, :],
            nth_target,
            setup.cfg.n_distributions,
        )
        p_target = _subset_target(
            p[:, -predict_last_n:, :],
            nth_target,
            setup.cfg.n_distributions,
        )

        assert (
            m_target.shape
            == b_target.shape
            == t_target.shape
            == p_target.shape
        )
        batch_size, time_steps, n_dist = m_target.shape  # WLOG

        # Make [batch, sample, time, dist] (expanded) tensor views of the targets.
        # Unsqueeze to add a dim for samples: [batch, rime, dist] -> [batch, 1, time, dist].
        # Expand to repeat the new dim without allocating new memory for it. So:
        #     [batch, 1, time, dist] -> [batch, sample, time, dist].
        m_exp = m_target.unsqueeze(1).expand(-1, n_samples, -1, -1)
        b_exp = b_target.unsqueeze(1).expand(-1, n_samples, -1, -1)
        t_exp = t_target.unsqueeze(1).expand(-1, n_samples, -1, -1)
        p_exp = p_target.unsqueeze(1).expand(-1, n_samples, -1, -1)

        # Distribute:

        # Prepare data for the Categorical dist.
        # Categorical dist descrbies a random event with a fixed number of results
        # where each one has a probability. Number of outcomes here is from 0 to
        # n_dist-1, and probabilities are given by pi.
        # Replace nan with uniform probability. Later those nans will be restored.
        p_invalid = torch.isnan(p_exp)
        p_safe = torch.where(
            p_invalid, torch.ones_like(p_exp) / n_dist, p_exp
        )

        dist = torch.distributions.Categorical(probs=p_safe)

        # Sample:

        # Draw dist index for each sample and time step from the dist dim, to get
        # [batch, sample, time].
        # And add a dim to match the shape of the expanded params for gathering,
        # for selecting which dist it is gathered from.
        choices = dist.sample().unsqueeze(-1)
        # For each param point, select from [batch, sample, time, dist] using
        # choices index which is [batch, sample, time, 1] to find which dist
        # to use, gathering that param using the index for it.
        # Then squeeze out the dist dim.
        m_sub = torch.gather(m_exp, dim=3, index=choices).squeeze(-1)
        b_sub = torch.gather(b_exp, dim=3, index=choices).squeeze(-1)
        t_sub = torch.gather(t_exp, dim=3, index=choices).squeeze(-1)

        def sample_values(ids: torch.Tensor) -> torch.Tensor:
            return _sample_asymmetric_laplacians(ids, m_sub, b_sub, t_sub)

        # Generate an initial value for every single pos via a mask of all `True`s,
        # with the _sample_asymmetric_laplacians helper.
        values_unbound = sample_values(
            ids=torch.ones_like(m_sub, dtype=torch.bool)
        )
        # Reshape it back since the helper has flattened dims.
        values_unbound = values_unbound.reshape(
            batch_size, n_samples, time_steps
        )
        # Restore nans (squeezing out the dist dim from which it was gathered)
        was_nan_mask = torch.gather(
            p_invalid, dim=3, index=choices
        ).squeeze(-1)
        values_unbound[was_nan_mask] = torch.nan

        values = _handle_negative_values(  # Resample as needed
            setup.cfg,
            values_unbound,
            sample_values=sample_values,
            normalized_zero=normalized_zeros[nth_target],
        )
        # Swap [batch, sample, time] to [batch, time, sample]
        values = values.permute(0, 2, 1)
        sample_points.append(values)

    # torch.stack results for all targets into a single tensor.
    # It stacks into a new dim for the targets at dim 2, so shape should be
    # [batch, time, sample] -> [batch, time, target, sample].
    return {'y_hat': torch.stack(sample_points, dim=2)}
