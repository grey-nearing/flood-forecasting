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
from pathlib import Path

import pytest

from model import run, run_scheduler


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
