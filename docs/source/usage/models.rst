Modelzoo
========

This section describes the deep learning flood forecasting models and output heads included in OpenHydroNet.

Model Heads
-----------

A **model head** sits on top of the recurrent neural network (LSTM) and maps the hidden state to the predicted river discharge (`m³/s`, normalized by watershed area to `mm/day` during training).

Regression
^^^^^^^^^^

:py:class:`model.modelzoo.head.Regression` is a single linear layer that predicts one deterministic value (point estimate) for each target variable and time step. It supports ``linear``, ``relu``, and ``softplus`` output activations.

CMAL
^^^^

:py:class:`model.modelzoo.head.CMAL` implements a **Countable Mixture of Asymmetric Laplacians (CMAL)** head. Instead of predicting a single number, CMAL predicts the parameters of a probability distribution (location, scale, and asymmetry for ``n_distributions`` mixture components). Sampling from this distribution produces an ensemble of streamflow trajectories that captures forecast uncertainty and skewed flood peaks.

Model Classes
-------------

BaseModel
^^^^^^^^^

:py:class:`model.modelzoo.basemodel.BaseModel` is the abstract base class that all forecasting models inherit from. Do not select ``BaseModel`` directly in a configuration file.

.. _handoff-forecast-lstm:

Handoff-Forecast-LSTM
^^^^^^^^^^^^^^^^^^^^^

:py:class:`model.modelzoo.handoff_forecast_lstm.HandoffForecastLSTM` (config key ``handoff_forecast_lstm``) splits each prediction into two time windows:

1. **Hindcast window (past weather):** A hindcast LSTM reads historical weather observations (such as satellite and gauge precipitation, temperature, and solar radiation) up to the forecast issue date.
2. **State handoff:** At the forecast issue date, a feed-forward neural network (`state_handoff_network`) transforms the final hidden state and cell state of the hindcast LSTM into the initial states of a second LSTM.
3. **Forecast window (future weather):** The forecast LSTM rolls forward over the forecast lead time (for example, 7 days ahead) using numerical or AI weather forecasts as inputs.

``HandoffForecastLSTM`` is a **former operational model** used in an earlier (2024) release of `Google FloodHub <https://sites.research.google/floods/>`__. The architecture and global ungauged evaluation are described in [Nearing2024]_.

.. _mean-embedding-forecast-lstm:

Mean-Embedding-Forecast-LSTM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:py:class:`model.modelzoo.mean_embedding_forecast_lstm.MeanEmbeddingForecastLSTM` (config key ``mean_embedding_forecast_lstm``) extends the encoder-decoder LSTM design to handle multiple weather products that may have missing dates or different start years:

1. **Per-product embeddings:** Each weather product (such as ``imerg``, ``cpc``, ``era5land``, ``hres``, ``graphcast``, or ``aifs``) passes through its own small feed-forward embedding network.
2. **Masked-mean aggregation:** At each daily time step, the model averages the embeddings of all weather products that have valid (non-``NaN``) data on that day. If one satellite or forecast product is missing on a given day, the model continues smoothly using the remaining products.
3. **Hindcast and forecast LSTMs:** The combined hindcast embedding feeds into the hindcast LSTM, whose state transitions into the forecast LSTM over the future lead-time window.

``MeanEmbeddingForecastLSTM`` is a **former operational model** used in a previous (2025) release of `Google FloodHub <https://sites.research.google/floods/>`__. The multi-product embedding design is described in [Gauch2025]_.

Implementing a New Model
^^^^^^^^^^^^^^^^^^^^^^^^

The code block below shows the template for adding a custom model architecture to OpenHydroNet.

**Required steps:**

1. **Inherit from BaseModel:** Subclass :py:class:`model.modelzoo.basemodel.BaseModel`.
2. **Define ``module_parts``:** List the attribute names of all trainable sub-modules in ``module_parts``. Fine-tuning (`run finetune`) uses this list to know which layers to freeze or update.
3. **Register the model:** Add your model name and class to :py:func:`model.modelzoo.get_model` in ``model/modelzoo/__init__.py``.

.. code-block:: python

    import torch

    from model.modelzoo.basemodel import BaseModel


    class TemplateModel(BaseModel):

        # List all trainable sub-module attribute names so fine-tuning can
        # selectively freeze or unfreeze individual model parts.
        module_parts = [...]

        def __init__(self, cfg: dict):
            """Initialize the model from a parsed Config object."""
            super(TemplateModel, self).__init__(cfg=cfg)

            ###########################
            # Create model parts here #
            ###########################

        def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            """Run a forward pass through the model.

            Parameters
            ----------
            data : dict[str, torch.Tensor]
                Input batch dictionary containing:
                - 'x_d_hindcast': Historical dynamic inputs, shape [batch, seq_length, features]
                - 'x_d_forecast': Forecast dynamic inputs, shape [batch, lead_time, features]
                - 'x_s': Static watershed attributes, shape [batch, features]

            Returns
            -------
            dict[str, torch.Tensor]
                Dictionary containing ``'y_hat'`` with shape
                ``[batch, seq_length, self.output_size]``.
            """
            hindcast = data['x_d_hindcast']
            forecast = data['x_d_forecast']
            statics = data['x_s']

            # Replace with your custom forecasting logic:
            batch_size, seq_len, _ = hindcast.shape
            output = torch.zeros(batch_size, seq_len, self.output_size)

            return {'y_hat': output}

References
^^^^^^^^^^

.. [Nearing2024] Nearing, Grey, et al. "`Global prediction of extreme floods in ungauged watersheds <https://www.nature.com/articles/s41586-024-07145-1>`__." *Nature* 627, 559–563 (2024).

.. [Gauch2025] Gauch, Martin, et al. "`How to deal with missing input data <https://hess.copernicus.org/articles/29/6221/2025/>`__." *Hydrology and Earth System Sciences* 29, 6221–6238 (2025).
