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

"""Training loop, fine-tuning freeze logic, and checkpoint handling."""

import itertools
import logging
import math
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim.lr_scheduler
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset

from model.datasetzoo import get_dataset
from model.datasetzoo.multimet import MultimetDataLoader
from model.datautils.utils import load_basin_file
from model.evaluation import get_tester
from model.evaluation.tester import BaseTester
from model.modelzoo import get_model, load_model_weights
from model.training import (
    get_loss_obj,
    get_optimizer,
    get_regularization_obj,
    loss,
)
from model.training.logger import Logger
from model.utils.config import Config
from model.utils.logging_utils import setup_logging
from model.utils.tqdm import AutoRefreshTqdm

LOGGER = logging.getLogger(__name__)


class BaseTrainer:
    """Default class to train a model.

    Args:
        cfg: The run configuration.
    """

    def __init__(self, cfg: Config):
        """Create the run folder, set up logging, and seed the RNGs."""
        super().__init__()
        self.cfg = cfg
        self.model = None
        self.optimizer = None
        self.loss_obj = None
        self.experiment_logger = None
        self.loader = None
        self.validator = None
        self.noise_sampler_y = None
        self._target_mean = None
        self._target_std = None
        self._allow_subsequent_nan_losses = cfg.allow_subsequent_nan_losses
        self._disable_pbar = cfg.verbose == 0
        self._max_updates_per_epoch = cfg.max_updates_per_epoch

        # load train basin list and add number of basins to the config
        self.basins = load_basin_file(cfg.train_basin_file)
        self.cfg.number_of_basins = len(self.basins)

        # check at which epoch the training starts
        self._epoch = self._get_start_epoch_number()

        self._create_folder_structure()
        setup_logging(
            str(self.cfg.run_dir / 'output.log'),
            cfg.logging_level,
            cfg.print_warnings_once,
        )
        LOGGER.info('### Folder structure created at %s', self.cfg.run_dir)

        if self.cfg.is_continue_training:
            LOGGER.info(
                '### Continue training of run stored in %s',
                self.cfg.base_run_dir,
            )

        if self.cfg.is_finetuning:
            LOGGER.info(
                '### Start finetuning with pretrained model stored in %s',
                self.cfg.base_run_dir,
            )

        LOGGER.info('### Run configurations for %s', self.cfg.experiment_name)
        for key, val in self.cfg.as_dict().items():
            LOGGER.info('%s: %s', key, val)

        self._set_random_seeds()
        self._set_device()

    def _get_dataset(self, *, compute_scaler: bool) -> Dataset:
        return get_dataset(
            cfg=self.cfg,
            period='train',
            is_train=True,
            compute_scaler=compute_scaler,
        )

    def _get_model(self) -> torch.nn.Module:
        return get_model(cfg=self.cfg)

    def _get_optimizer(self) -> torch.optim.Optimizer:
        return get_optimizer(
            model_or_params=self.model,
            cfg=self.cfg,
            is_gpu=self.device.type == 'cuda',
        )

    def _get_loss_obj(self) -> loss.BaseLoss:
        return get_loss_obj(cfg=self.cfg)

    def _set_regularization(self) -> None:
        self.loss_obj.set_regularization_terms(
            get_regularization_obj(cfg=self.cfg)
        )

    def _get_tester(self) -> BaseTester:
        return get_tester(
            cfg=self.cfg,
            run_dir=self.cfg.run_dir,
            period='validation',
            init_model=False,
        )

    def _get_data_loader(self, ds: Dataset) -> MultimetDataLoader:
        return MultimetDataLoader(
            ds,
            lazy_load=self.cfg.lazy_load,
            logging_level=self.cfg.logging_level,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            collate_fn=ds.collate_fn,
        )

    @staticmethod
    def _unfreeze_module_dict(
        group_obj: torch.nn.ModuleDict,
        module_group: str,
        module_parts: object,
    ) -> None:
        """Validate and unfreeze submodules inside a ModuleDict."""
        if isinstance(module_parts, bool):
            if module_parts:
                for param in group_obj.parameters():
                    param.requires_grad = True
            return
        if isinstance(module_parts, str):
            parts_list: list[str] = [module_parts]
        elif isinstance(module_parts, list) and all(
            isinstance(p, str) for p in module_parts
        ):
            parts_list = module_parts
        else:
            raise ValueError(
                f'Invalid value {module_parts!r} for ModuleDict '
                f"'{module_group}' in 'finetune_modules'; expected bool, "
                'str, or list[str].'
            )
        for module_part in parts_list:
            if module_part not in group_obj:
                raise KeyError(
                    f"Submodule '{module_part}' not found in "
                    f"'{module_group}'. Available keys: "
                    f'{list(group_obj.keys())}'
                )
            for param in group_obj[module_part].parameters():
                param.requires_grad = True

    def _unfreeze_module_group(
        self,
        group_obj: torch.nn.Module,
        module_group: str,
        module_parts: object,
    ) -> None:
        """Validate and unfreeze a module or ModuleDict entry."""
        if isinstance(group_obj, torch.nn.ModuleDict):
            self._unfreeze_module_dict(group_obj, module_group, module_parts)
            return
        if not isinstance(module_parts, bool):
            raise ValueError(  # noqa: TRY004
                f'Invalid value {module_parts!r} for non-ModuleDict module '
                f"'{module_group}' in 'finetune_modules'; expected a bool "
                '(True or False).'
            )
        if module_parts:
            for param in group_obj.parameters():
                param.requires_grad = True

    def _validate_newly_initialized_embeddings_unfrozen(self) -> None:
        """Raise ValueError if a newly initialized embedding is frozen."""
        target_model = getattr(self.model, '_orig_mod', self.model)
        newly_init = getattr(
            target_model, '_newly_initialized_embeddings', set()
        )
        for module_group, product_key in sorted(newly_init):
            group_obj = getattr(target_model, module_group, None)
            if (
                isinstance(group_obj, torch.nn.ModuleDict)
                and product_key in group_obj
            ):
                submodule = group_obj[product_key]
                if not any(p.requires_grad for p in submodule.parameters()):
                    raise ValueError(
                        'Newly added product embedding '
                        f"'{module_group}.{product_key}' is randomly "
                        'initialized (not present in checkpoint) and must '
                        "be unfrozen in 'finetune_modules'."
                    )

    def _freeze_model_parts(self) -> None:
        """Freeze every weight except the parts listed in `finetune_modules`.

        The model is unwrapped from a `torch.compile` `OptimizedModule`
        (`_orig_mod`) before its module parts are inspected, so the names in
        `finetune_modules` always match the underlying model attributes.

        Raises:
            ValueError: If a name in `finetune_modules` is not one of the
                model's `module_parts`.
        """
        target_model = getattr(self.model, '_orig_mod', self.model)

        # freeze all model weights
        for param in target_model.parameters():
            param.requires_grad = False

        available_modules = set(target_model.module_parts)
        finetune_modules = self.cfg.finetune_modules
        if isinstance(finetune_modules, list):
            # a plain list unfreezes every parameter of each listed part
            module_specs = dict.fromkeys(finetune_modules, True)
        else:
            # if it was no list, it has to be a dictionary
            module_specs = finetune_modules

        unresolved_modules = [
            module_group
            for module_group in module_specs
            if module_group not in available_modules
        ]
        if unresolved_modules:
            raise ValueError(
                "Unresolved module name(s) in 'finetune_modules': "
                f'{unresolved_modules}. Available modules: '
                f'{sorted(available_modules)}'
            )

        # unfreeze parameters specified in config as tuneable parameters
        for module_group, module_parts in module_specs.items():
            group_obj = getattr(target_model, module_group)
            self._unfreeze_module_group(group_obj, module_group, module_parts)
        self._validate_newly_initialized_embeddings_unfrozen()

    def _load_initial_weights(self) -> None:
        """Load the checkpoint a new (non-resumed) training run starts from."""
        if self.cfg.checkpoint_path is not None:
            LOGGER.info(
                'Starting training from Checkpoint %s',
                self.cfg.checkpoint_path,
            )
            self._load_model_weights(
                self.cfg.checkpoint_path,
                allow_new_embeddings=self.cfg.is_finetuning,
            )
        elif self.cfg.is_finetuning:
            # the default for finetuning is the last model state
            checkpoint_path = max(self.cfg.base_run_dir.glob('model_epoch*.pt'))
            LOGGER.info('Starting training from checkpoint %s', checkpoint_path)
            self._load_model_weights(
                checkpoint_path,
                allow_new_embeddings=True,
            )

    def initialize_training(self) -> None:
        """Initialize the training class.

        This method will load the model, initialize loss, regularization,
        optimizer, dataset and dataloader, tensorboard logging, and Tester
        class. If called in a ``continue_training`` context, this model will
        also restore the model and optimizer state.

        Raises:
            ValueError: If the training dataset contains no samples.
        """
        # Initialize dataset before the model is loaded.
        ds = self._get_dataset(compute_scaler=(not self.cfg.is_finetuning))
        if len(ds) == 0:
            raise ValueError('Dataset contains no samples.')
        self.loader = self._get_data_loader(ds=ds)

        LOGGER.debug('init model')
        self.model = self._get_model().to(self.device)
        self._load_initial_weights()

        # Freeze model parts from pre-trained model.
        if self.cfg.is_finetuning:
            self._freeze_model_parts()

        self.optimizer = self._get_optimizer()
        self.scaler = GradScaler(enabled=self.device.type == 'cuda')
        self.loss_obj = self._get_loss_obj().to(self.device)

        # Add possible regularization terms to the loss function.
        self._set_regularization()

        # restore optimizer and model state if training is continued
        if self.cfg.is_continue_training:
            self._restore_training_state()

        self.experiment_logger = Logger(cfg=self.cfg)
        if self.cfg.log_tensorboard:
            self.experiment_logger.start_tb()

        if self.cfg.is_continue_training:
            # set epoch and iteration step counter to continue from the
            # selected checkpoint
            self.experiment_logger.epoch = self._epoch
            self.experiment_logger.update = len(self.loader) * self._epoch

        if self.cfg.validate_every is not None:
            if self.cfg.validate_n_random_basins < 1:
                LOGGER.warning(
                    'Validation set to validate every %s epoch(s), but '
                    "'validate_n_random_basins' not set or set to zero. Will "
                    'validate on the entire validation set.',
                    self.cfg.validate_every,
                )
                self.cfg.validate_n_random_basins = self.cfg.number_of_basins
            self.validator = self._get_tester()

        if self.cfg.target_noise_std is not None:
            self.noise_sampler_y = torch.distributions.Normal(
                loc=0, scale=self.cfg.target_noise_std
            )
            target_means = [
                ds.scaler.scaler.sel(parameter='mean')[feature].item()
                for feature in self.cfg.target_variables
            ]
            self._target_mean = torch.tensor(target_means).to(self.device)
            target_stds = [
                ds.scaler.scaler.sel(parameter='std')[feature].item()
                for feature in self.cfg.target_variables
            ]
            self._target_std = torch.tensor(target_stds).to(self.device)

    def _create_lr_scheduler(self) -> tuple[object, object]:
        match self.cfg.learning_rate_strategy:
            case 'ConstantLR':
                # Keep learning rate constant.
                lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                    self.optimizer,
                    factor=1.0,
                    total_iters=1,
                )

                def lr_step(loss: float) -> None:
                    pass

            case 'StepLR':
                # Step down by a factor every step size epocs, regardless of
                # loss.
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=self.cfg.learning_rate_epochs_drop,
                    gamma=self.cfg.learning_rate_drop_factor,
                )

                def lr_step(loss: float) -> None:  # noqa: ARG001
                    lr_scheduler.step()
            case 'ReduceLROnPlateau':
                # Step down by a factor every epoc w.r.t. to change in loss
                # between patience epocs.
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    factor=self.cfg.learning_rate_drop_factor,
                    patience=self.cfg.learning_rate_epochs_drop,
                )

                def lr_step(loss: float) -> None:
                    lr_scheduler.step(loss)
            case _:
                raise ValueError(
                    'learning_rate_strategy unsupported: '
                    f'{self.cfg.learning_rate_strategy}'
                )
        return lr_scheduler, lr_step

    def train_and_validate(self) -> None:
        """Train and validate the model.

        Train the model for the number of epochs specified in the run
        configuration, and perform validation after every ``validate_every``
        epochs. Model and optimizer state are saved after every
        ``save_weights_every`` epochs.
        """
        lr_scheduler, lr_step = self._create_lr_scheduler()

        for epoch in range(self._epoch + 1, self._epoch + self.cfg.epochs + 1):
            LOGGER.info('learning rate is %s', lr_scheduler.get_last_lr())

            self._train_epoch(epoch=epoch)
            avg_losses = self.experiment_logger.summarise()
            lr_step(avg_losses['avg_loss'])

            loss_str = ', '.join(f'{k}: {v:.5f}' for k, v in avg_losses.items())
            LOGGER.info('Epoch %d average loss: %s', epoch, loss_str)

            if epoch % self.cfg.save_weights_every == 0:
                self._save_weights_and_optimizer(epoch)

            if (self.validator is not None) and (
                epoch % self.cfg.validate_every == 0
            ):
                self.validator.evaluate(
                    epoch=epoch,
                    save_results=self.cfg.save_validation_results,
                    metrics=self.cfg.metrics,
                    model=self.model,
                    experiment_logger=self.experiment_logger.valid(),
                    # Validation during training never assimilates, even if
                    # the run config sets `assimilate: true` for evaluation.
                    data_assimilation=False,
                )

                valid_metrics = {
                    'avg_total_loss': math.nan
                } | self.experiment_logger.summarise()
                print_msg = (
                    f'Epoch {epoch} average validation loss: '
                    f'{valid_metrics["avg_total_loss"]:.5f}'
                )
                if self.cfg.metrics:
                    print_msg += ' -- Median validation metrics: '
                    print_msg += ', '.join(
                        f'{k}: {v:.5f}'
                        for k, v in valid_metrics.items()
                        if k != 'avg_total_loss'
                    )
                    LOGGER.info(print_msg)

            self.experiment_logger.log_step(
                learning_rate=lr_scheduler.get_last_lr()[-1]
            )

        # make sure to close tensorboard to avoid losing the last epoch
        if self.cfg.log_tensorboard:
            self.experiment_logger.stop_tb()

    def _get_start_epoch_number(self) -> int:
        if self.cfg.is_continue_training:
            if self.cfg.continue_from_epoch is not None:
                epoch = self.cfg.continue_from_epoch
            else:
                weight_path = max(self.cfg.run_dir.glob('model_epoch*.pt'))
                epoch = weight_path.name[-6:-3]
        else:
            epoch = 0
        return int(epoch)

    def _restore_training_state(self) -> None:
        if self.cfg.continue_from_epoch is not None:
            epoch = f'{self.cfg.continue_from_epoch:03d}'
            weight_path = self.cfg.base_run_dir / f'model_epoch{epoch}.pt'
        else:
            weight_path = max(self.cfg.base_run_dir.glob('model_epoch*.pt'))
            epoch = weight_path.name[-6:-3]

        optimizer_path = (
            self.cfg.base_run_dir / f'optimizer_state_epoch{epoch}.pt'
        )

        LOGGER.info('Continue training from epoch %d', int(epoch))
        # A resumed fine-tuning run may carry product embeddings that are
        # not in the checkpoint; only fine-tuning may add them.
        self._load_model_weights(
            weight_path, allow_new_embeddings=self.cfg.is_finetuning
        )
        self.optimizer.load_state_dict(
            torch.load(
                str(optimizer_path), map_location=self.device, weights_only=True
            )
        )

    def _load_model_weights(
        self,
        checkpoint_path: Path | str,
        *,
        allow_new_embeddings: bool = False,
    ) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
        """Load the model state_dict while handling torch.compile prefixes."""
        return load_model_weights(
            self.model,
            checkpoint_path,
            self.device,
            allow_new_embeddings=allow_new_embeddings,
        )

    def _save_weights_and_optimizer(self, epoch: int) -> None:
        weight_path = self.cfg.run_dir / f'model_epoch{epoch:03d}.pt'
        torch.save(self.model.state_dict(), str(weight_path))

        optimizer_path = (
            self.cfg.run_dir / f'optimizer_state_epoch{epoch:03d}.pt'
        )
        torch.save(self.optimizer.state_dict(), str(optimizer_path))

    def _move_batch_to_device(self, batch: dict) -> None:
        """Move every tensor of `batch` to `self.device` in place."""
        for key, value in batch.items():
            if isinstance(value, dict):
                batch[key] = {k: v.to(self.device) for k, v in value.items()}
            elif not key.startswith('date'):
                batch[key] = value.to(self.device)

    def _train_epoch(self, epoch: int) -> None:
        self.model.train()
        self.experiment_logger.train()

        # process bar handle
        n_limit = self._max_updates_per_epoch
        n_length = len(self.loader)
        n_iter = n_limit if 0 < n_limit <= n_length else n_length
        pbar = AutoRefreshTqdm(
            itertools.islice(self.loader, n_iter),
            file=sys.stdout,
            disable=self._disable_pbar,
            total=n_iter,
        )
        pbar.set_description(f'# Epoch {epoch}')

        # Iterate in batches over training set
        nan_count = 0
        gradient_norms = []
        for i, batch in enumerate(pbar):
            self._move_batch_to_device(batch)

            with autocast(
                self.device.type, enabled=(self.device.type == 'cuda')
            ):
                # apply possible pre-processing to the batch before the
                # forward pass
                data = self.model.pre_model_hook(batch, is_train=True)

                # get predictions
                predictions = self.model(data)

                if self.noise_sampler_y is not None:
                    for key in filter(lambda k: 'y' in k, data):
                        noise = self.noise_sampler_y.sample(data[key].shape)
                        # make sure we add near-zero noise to originally
                        # near-zero targets
                        data[key] += (
                            data[key] + self._target_mean / self._target_std
                        ) * noise.to(self.device)

                loss, all_losses = self.loss_obj(predictions, data)

            # early stop training if loss is NaN
            if torch.isnan(loss):
                nan_count += 1
                if nan_count > self._allow_subsequent_nan_losses:
                    raise RuntimeError(
                        f'Loss was NaN for {nan_count} times in a row. '
                        'Stopped training.'
                    )
                LOGGER.warning(
                    'Loss is Nan; ignoring step. (#%d/%d)',
                    nan_count,
                    self._allow_subsequent_nan_losses,
                )
            else:
                nan_count = 0

                # delete old gradients
                self.optimizer.zero_grad()

                # get gradients
                self.scaler.scale(loss).backward()

                if self.cfg.clip_gradient_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.clip_gradient_norm
                    )
                    # Keep detached scalars on device until epoch end to avoid
                    # an extra GPU synchronization on every training step.
                    gradient_norms.append(gradient_norm.detach())

                # update weights
                self.scaler.step(self.optimizer)

                self.scaler.update()  # Update scale for the next iteration

            if i % self.cfg.log_loss_every_nth_update == 0 or i + 1 == n_iter:
                # Report loss every nth update or finally
                pbar.set_postfix_str(f'Loss: {loss.item():.4f}')
                self.experiment_logger.log_step(
                    **{k: v.item() for k, v in all_losses.items()}
                )

        if self.cfg.clip_gradient_norm is not None:
            norms = (
                torch.stack(gradient_norms).cpu().double().numpy()
                if gradient_norms
                else np.empty(0)
            )
            self.experiment_logger.log_gradient_norms(
                norms, self.cfg.clip_gradient_norm, epoch
            )

    def _set_random_seeds(self) -> None:
        # The global legacy NumPy RNG is what downstream sampling code uses,
        # so it is seeded (and drawn from) directly.
        if self.cfg.seed is None:
            self.cfg.seed = int(np.random.uniform(low=0, high=1e6))  # noqa: NPY002

        # fix random seeds for various packages
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)  # noqa: NPY002
        torch.cuda.manual_seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)

    def _set_device(self) -> None:
        if self.cfg.device is not None:
            if self.cfg.device.startswith('cuda'):
                gpu_id = int(self.cfg.device.split(':')[-1])
                if gpu_id >= torch.cuda.device_count():
                    raise RuntimeError(
                        f'This machine does not have GPU #{gpu_id} '
                    )
                self.device = torch.device(self.cfg.device)
            elif self.cfg.device == 'mps':
                if torch.backends.mps.is_available():
                    self.device = torch.device('mps')
                else:
                    raise RuntimeError('MPS device is not available.')
            else:
                self.device = torch.device('cpu')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        LOGGER.info('### Device %s will be used for training', self.device)

    def _create_folder_structure(self) -> None:
        # create as subdirectory within run directory of base run
        if self.cfg.is_continue_training:
            folder_name = f'continue_training_from_epoch{self._epoch:03d}'

            # store dir of base run for easier access in weight loading
            self.cfg.base_run_dir = self.cfg.run_dir
            self.cfg.run_dir = self.cfg.run_dir / folder_name

        # create as new folder structure
        else:
            # Run folders are stamped with the local wall-clock time.
            now = datetime.now()  # noqa: DTZ005
            day = f'{now.day}'.zfill(2)
            month = f'{now.month}'.zfill(2)
            hour = f'{now.hour}'.zfill(2)
            minute = f'{now.minute}'.zfill(2)
            second = f'{now.second}'.zfill(2)
            timestamp = f'{day}{month}_{hour}{minute}{second}'
            run_name = f'{self.cfg.experiment_name}_{timestamp}'

            # if no directory for the runs is specified, a 'runs' folder will
            # be created in the current working dir
            if self.cfg.run_dir is None:
                self.cfg.run_dir = Path().cwd() / 'runs' / run_name
            else:
                self.cfg.run_dir = self.cfg.run_dir / run_name

        # create folder + necessary subfolder
        if not self.cfg.run_dir.is_dir():
            self.cfg.train_dir = self.cfg.run_dir / 'train_data'
            self.cfg.train_dir.mkdir(parents=True)
        else:
            raise RuntimeError(
                f'There is already a folder at {self.cfg.run_dir}'
            )
        if self.cfg.log_n_figures is not None:
            self.cfg.img_log_dir = self.cfg.run_dir / 'img_log'
            self.cfg.img_log_dir.mkdir(parents=True)
