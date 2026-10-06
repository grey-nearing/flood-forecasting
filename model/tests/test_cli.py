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

"""Unit tests for model.run and model.run_scheduler CLI."""

import sys
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest

from model import run, run_scheduler
from model.utils.config import Config


@pytest.mark.unit
def test_run_get_args_valid_modes(monkeypatch):
    # Train mode
    monkeypatch.setattr(
        sys, 'argv', ['run.py', 'train', '--config-file', 'test_config.yml']
    )
    args = run._get_args()
    assert args['mode'] == 'train'
    assert args['config_file'] == 'test_config.yml'

    # Continue training mode
    monkeypatch.setattr(
        sys, 'argv', ['run.py', 'continue_training', '--run-dir', '/tmp/run']
    )
    args = run._get_args()
    assert args['mode'] == 'continue_training'
    assert args['run_dir'] == '/tmp/run'

    # Evaluate mode
    monkeypatch.setattr(
        sys,
        'argv',
        ['run.py', 'evaluate', '--run-dir', '/tmp/run', '--period', 'test'],
    )
    args = run._get_args()
    assert args['mode'] == 'evaluate'
    assert args['period'] == 'test'

    # Infer mode
    monkeypatch.setattr(
        sys, 'argv', ['run.py', 'infer', '--run-dir', '/tmp/run']
    )
    args = run._get_args()
    assert args['mode'] == 'infer'


@pytest.mark.unit
def test_run_get_args_missing_required_args(monkeypatch):
    # Train missing config file
    monkeypatch.setattr(sys, 'argv', ['run.py', 'train'])
    with pytest.raises(ValueError, match='Missing path to config file'):
        run._get_args()

    # Continue training missing run dir
    monkeypatch.setattr(sys, 'argv', ['run.py', 'continue_training'])
    with pytest.raises(
        ValueError, match='Missing path to run directory file'
    ):
        run._get_args()

    # Evaluate missing run dir
    monkeypatch.setattr(sys, 'argv', ['run.py', 'evaluate'])
    with pytest.raises(ValueError, match='Missing path to run directory'):
        run._get_args()


@pytest.mark.unit
def test_run_dispatch_start_run(monkeypatch, minimal_config):
    calls = []
    monkeypatch.setattr(run, 'start_training', lambda cfg: calls.append(cfg))

    run.start_run(config=minimal_config, gpu=0)
    assert minimal_config.device == 'cuda:0'
    assert calls == [minimal_config]

    run.start_run(config=minimal_config, gpu=-1)
    assert minimal_config.device == 'cpu'
    assert len(calls) == 2


@pytest.mark.unit
def test_run_dispatch_eval_run(monkeypatch, minimal_config, tmp_path):
    calls = []
    monkeypatch.setattr(
        run,
        'start_evaluation',
        lambda **kwargs: calls.append(kwargs),
    )

    run.eval_run(
        config=minimal_config,
        run_dir=tmp_path,
        period='test',
        epoch=1,
        gpu=-1,
    )
    assert minimal_config.device == 'cpu'
    assert len(calls) == 1
    assert calls[0]['cfg'] is minimal_config
    assert calls[0]['run_dir'] == tmp_path
    assert calls[0]['period'] == 'test'
    assert calls[0]['epoch'] == 1


@pytest.mark.unit
def test_run_scheduler_get_args(monkeypatch, tmp_path):
    # Valid arguments
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'schedule-runs',
            'train',
            '--directory',
            str(tmp_path),
            '--gpu-ids',
            '0',
            '1',
            '--runs-per-gpu',
            '2',
        ],
    )
    args = run_scheduler._get_args()
    assert args['mode'] == 'train'
    assert args['directory'] == tmp_path
    assert args['gpu_ids'] == [0, 1]
    assert args['runs_per_gpu'] == 2

    # Non-existent directory
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'schedule-runs',
            'train',
            '--directory',
            str(tmp_path / 'nonexistent'),
            '--gpu-ids',
            '0',
            '--runs-per-gpu',
            '1',
        ],
    )
    with pytest.raises(ValueError, match='No folder at'):
        run_scheduler._get_args()


def _da_config(get_config: Callable[[str], Config]) -> Config:
    """Return a forecast config with an assimilation_config block."""
    cfg = get_config('forecast')
    cfg.update_config(
        {
            'assimilation_config': {
                'assimilation_components': ['hindcast_embedding'],
                'assimilation_window': 10,
                'initial_learning_rate': 0.01,
            }
        }
    )
    return cfg


@pytest.mark.unit
@pytest.mark.parametrize(
    ('mode', 'flag', 'expected'),
    [
        ('evaluate', False, False),
        ('evaluate', True, True),
        ('infer', False, False),
        ('infer', True, True),
    ],
)
def test_run_get_args_assimilate_flag(
    *, mode: str, flag: bool, expected: bool
) -> None:
    """--assimilate is parsed (default False) for evaluate and infer."""
    argv = ['run.py', mode, '--run-dir', 'runs/run']
    if flag:
        argv.append('--assimilate')
    with patch.object(sys, 'argv', argv):
        assert run._get_args()['assimilate'] is expected  # noqa: SLF001


@pytest.mark.unit
@pytest.mark.parametrize(
    'argv',
    [
        ['train', '--config-file', 'c.yml'],
        ['continue_training', '--run-dir', 'runs/run'],
        ['finetune', '--config-file', 'c.yml'],
    ],
)
def test_run_get_args_assimilate_rejected_outside_evaluation(
    argv: list[str],
) -> None:
    """--assimilate is rejected for modes that do not evaluate a model."""
    with (
        patch.object(sys, 'argv', ['run.py', *argv, '--assimilate']),
        pytest.raises(ValueError, match='--assimilate is only supported'),
    ):
        run._get_args()  # noqa: SLF001


@pytest.mark.unit
@pytest.mark.parametrize('mode', ['evaluate', 'infer'])
@pytest.mark.parametrize('flag', [False, True])
def test_run_main_forwards_assimilate(
    *, get_config: Callable[[str], Config], mode: str, flag: bool
) -> None:
    """_main forwards the flag as True and its absence as None (config)."""
    cfg = _da_config(get_config)
    argv = ['run.py', mode, '--run-dir', str(cfg.run_dir), '--gpu', '-1']
    if flag:
        argv.append('--assimilate')
    with (
        patch.object(sys, 'argv', argv),
        patch('model.run.Config', return_value=cfg),
        patch('model.run.eval_run') as mock_eval,
    ):
        run._main()  # noqa: SLF001
    mock_eval.assert_called_once()
    assert mock_eval.call_args.kwargs['assimilate'] is (True if flag else None)
    assert cfg.inference_mode is (mode == 'infer')


@pytest.mark.unit
@pytest.mark.parametrize(
    ('config_value', 'override', 'expected'),
    [
        (False, None, False),
        (True, None, True),
        (False, True, True),
        (True, False, False),
    ],
)
def test_run_eval_run_assimilate_override(
    *,
    get_config: Callable[[str], Config],
    config_value: bool,
    override: bool | None,
    expected: bool,
) -> None:
    """eval_run overrides cfg.assimilate on a copy; None keeps the config."""
    cfg = _da_config(get_config)
    cfg.assimilate = config_value
    with patch('model.run.start_evaluation') as mock_eval:
        run.eval_run(
            config=cfg,
            run_dir=cfg.run_dir,
            period='test',
            epoch=1,
            gpu=-1,
            assimilate=override,
        )
    mock_eval.assert_called_once()
    used_cfg = mock_eval.call_args.kwargs['cfg']
    assert used_cfg.assimilate is expected
    assert used_cfg.device == 'cpu'
    # The caller's config is left untouched by an override.
    assert cfg.assimilate is config_value
    if override is not None:
        assert used_cfg is not cfg


@pytest.mark.unit
def test_run_eval_run_assimilate_requires_config(
    get_config: Callable[[str], Config],
) -> None:
    """Requesting DA on a config without assimilation_config fails early."""
    cfg = get_config('forecast')
    assert cfg.assimilation_config is None
    with (
        patch('model.run.start_evaluation') as mock_eval,
        pytest.raises(ValueError, match='no assimilation_config'),
    ):
        run.eval_run(
            config=cfg, run_dir=cfg.run_dir, period='test', assimilate=True
        )
    mock_eval.assert_not_called()
    assert cfg.assimilate is False
