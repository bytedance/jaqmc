# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import shutil
import tempfile

import chex
import numpy as np
import pytest
from jax import numpy as jnp

from jaqmc_legacy.dmc.ckpt_handler import CkptHandler
from jaqmc_legacy.dmc.data_path import DataPath, _resolve_path, _setup_path
from jaqmc_legacy.dmc.metric_manager import MetricManager
from jaqmc_legacy.dmc.state import State
from jaqmc_legacy.dmc.storage_handler import (
    dummy_storage_handler,
    local_storage_handler,
)


def test_state_default_state() -> None:
    init_position = jnp.arange(12).reshape((6, 2))
    test_energy = -88.88

    def calc_energy_func(_self, *_args, **_kwargs):
        return test_energy

    state = State.default(
        init_position=init_position,
        calc_energy_func=calc_energy_func,
        mixed_estimator_num_steps=88,
        energy_window_size=22,
        time_step=1e-3,
    )

    chex.assert_trees_all_close(state.position, init_position)
    chex.assert_trees_all_close(state.walker_age, jnp.ones(len(init_position)))
    chex.assert_trees_all_close(state.weight, jnp.ones(len(init_position)))
    assert state.local_energy is None
    assert state.energy_offset == test_energy
    assert state.target_num_walkers == len(init_position)
    assert state.mixed_estimator == test_energy
    assert state.mixed_estimator_calculator.mixed_estimator_num_steps == 88
    assert state.mixed_estimator_calculator.all_energy.maxlen == 22
    assert state.effective_time_step_calculator.time_step == pytest.approx(1e-3)


@pytest.mark.parametrize("return_fullpath", [True, False])
def test_local_storage_handler_ls_path(
    tmp_path: pathlib.Path,
    return_fullpath: bool,
) -> None:
    all_file_paths: list[str] = []
    for i in range(6):
        tmp_file = tmp_path / f"test_{i}.txt"
        tmp_file.touch()
        all_file_paths.append(str(tmp_file))

    expected_set = (
        set(all_file_paths)
        if return_fullpath
        else {os.path.basename(f) for f in all_file_paths}
    )

    ls_results = local_storage_handler.ls(
        str(tmp_path),
        return_fullpath=return_fullpath,
    )
    assert set(ls_results) == expected_set


def test_local_storage_handler_rm_file(tmp_path: pathlib.Path) -> None:
    tmp_file = tmp_path / "test.txt"
    tmp_file.touch()
    assert tmp_file.exists()

    local_storage_handler.rm(str(tmp_file))
    assert not tmp_file.exists()


def test_local_storage_handler_rm_dir(tmp_path: pathlib.Path) -> None:
    tmp_dir = tmp_path / "test_dir"
    tmp_dir.mkdir()
    assert tmp_dir.exists() and tmp_dir.is_dir()

    local_storage_handler.rm(str(tmp_dir))
    assert not tmp_dir.exists()


def test_local_storage_handler_exists(tmp_path: pathlib.Path) -> None:
    fake_path = str(tmp_path / "fake.txt")
    assert local_storage_handler.exists(fake_path) is False

    test_file = tmp_path / "test.txt"
    test_file.touch()
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()

    assert local_storage_handler.exists(str(test_file)) is True
    assert local_storage_handler.exists(str(test_dir)) is True


def test_local_storage_handler_exists_dir(tmp_path: pathlib.Path) -> None:
    test_file = tmp_path / "test.txt"
    test_file.touch()
    fake_path = str(tmp_path / "fake")
    assert local_storage_handler.exists_dir(str(test_file)) is False
    assert local_storage_handler.exists_dir(fake_path) is False

    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    assert local_storage_handler.exists_dir(str(test_dir)) is True


def test_ckpt_handler_saving_ckpt(tmp_path: pathlib.Path) -> None:
    init_position = jnp.arange(12).reshape((6, 2))
    test_energy = -88.88

    def calc_energy_func(_self, *_args, **_kwargs):
        return test_energy

    ckpt_handler = CkptHandler(
        ckpt_file_prefix="test_none",
        local_save_path=str(tmp_path),
    )
    default_state = State.default(
        init_position=init_position,
        calc_energy_func=calc_energy_func,
        mixed_estimator_num_steps=88,
        energy_window_size=22,
        time_step=1e-3,
    )
    ckpt_handler.save(5, default_state)

    target_ckpt_path = os.path.join(
        ckpt_handler.local_save_path,
        ckpt_handler.ckpt_file_pattern.format(step=5),
    )
    step, state = ckpt_handler._load_ckpt(target_ckpt_path)

    assert step == 5
    chex.assert_trees_all_close(state.position, init_position)
    chex.assert_trees_all_close(state.walker_age, jnp.ones(len(init_position)))
    chex.assert_trees_all_close(state.weight, jnp.ones(len(init_position)))
    assert state.local_energy is None
    assert state.energy_offset == test_energy
    assert state.target_num_walkers == len(init_position)
    assert state.mixed_estimator == test_energy
    assert state.mixed_estimator_calculator.mixed_estimator_num_steps == 88
    assert state.mixed_estimator_calculator.all_energy.maxlen == 22
    assert state.effective_time_step_calculator.time_step == pytest.approx(1e-3)


def test_data_path_resolve_string_path() -> None:
    path = _resolve_path("test/local/path")
    assert isinstance(path, DataPath)
    assert path.local_path == "test/local/path"
    assert path.remote_path is None


def test_data_path_setup_local_path(tmp_path: pathlib.Path) -> None:
    save_path = _resolve_path(str(tmp_path / "test/local/path"))
    restore_path = DataPath()
    updated_save_path, updated_restore_path = _setup_path(
        save_path=save_path,
        restore_path=restore_path,
        remote_storage_handler=dummy_storage_handler,
    )

    assert updated_save_path.local_path == save_path.local_path
    assert updated_save_path.remote_path == save_path.remote_path
    assert updated_restore_path.remote_path == restore_path.remote_path
    assert updated_restore_path.local_path != restore_path.local_path
    assert local_storage_handler.exists_dir(updated_save_path.local_path)
    assert local_storage_handler.exists_dir(updated_restore_path.local_path)

    shutil.rmtree(updated_restore_path.local_path)


def test_data_path_setup_remote_path(tmp_path: pathlib.Path) -> None:
    with tempfile.TemporaryDirectory() as tmp_remote_dir:
        save_path = DataPath(
            local_path=str(tmp_path / "test/local/path"),
            remote_path=os.path.join(tmp_remote_dir, "test/remote/path"),
        )
        restore_path = DataPath(remote_path=str(tmp_path / "test/remote/path2"))
        updated_save_path, updated_restore_path = _setup_path(
            save_path=save_path,
            restore_path=restore_path,
            remote_storage_handler=local_storage_handler,
        )

        assert updated_save_path.local_path == save_path.local_path
        assert updated_save_path.remote_path == save_path.remote_path
        assert updated_restore_path.remote_path == restore_path.remote_path
        assert updated_restore_path.local_path != restore_path.local_path
        assert local_storage_handler.exists_dir(updated_save_path.local_path)
        assert local_storage_handler.exists_dir(updated_save_path.remote_path)
        assert local_storage_handler.exists_dir(updated_restore_path.local_path)

    shutil.rmtree(updated_restore_path.local_path)


def test_metric_manager_write(tmp_path: pathlib.Path) -> None:
    metric_manager = MetricManager("test.csv", ["a", "b"], str(tmp_path))

    test_data = np.arange(8).reshape(4, 2)
    with metric_manager:
        for i, data in enumerate(test_data):
            metric_manager.write(i, data)
        data = metric_manager.get_metric_data()

    chex.assert_trees_all_close(data.a.to_numpy(), test_data[:, 0])
    chex.assert_trees_all_close(data.b.to_numpy(), test_data[:, 1])
