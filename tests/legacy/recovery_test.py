# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import pathlib

import chex
import jax
import pytest
from jax import numpy as jnp

from jaqmc_legacy.dmc.ckpt_metric_manager import CkptMetricManager, NoSafeDataAvailable
from jaqmc_legacy.dmc.dmc import IterationOutput, recovery_wrapper
from jaqmc_legacy.dmc.state import State


def test_recovery_wrapper_no_failure() -> None:
    test_iteration_output = IterationOutput(
        succeeded=True,
        state=State(position=jnp.arange(20).reshape((10, 2))),
        key=None,
    )

    def test_dmc_iteration(*, should_raise: bool):
        if should_raise:
            raise RuntimeError("boom")
        return 666, test_iteration_output

    wrapped_func = recovery_wrapper(
        1,
        test_dmc_iteration,
        0,
        ckpt_metric_manager=None,
        key=None,
    )
    step, output = wrapped_func(should_raise=False)
    assert step == 666
    chex.assert_trees_all_close(
        output.state.position,
        test_iteration_output.state.position,
    )


def test_recovery_wrapper_no_safe_data_available() -> None:
    ckpt_metric_manager = CkptMetricManager(
        metric_schema=[],
        block_size=3,
        lazy_setup=False,
    )

    def test_dmc_iteration(*, should_raise: bool):
        raise RuntimeError("boom")

    wrapped_func = recovery_wrapper(
        1,
        test_dmc_iteration,
        max_restore_nums=1,
        ckpt_metric_manager=ckpt_metric_manager,
        key=None,
    )
    with pytest.raises(NoSafeDataAvailable):
        wrapped_func(should_raise=True)


def _get_manager_with_safe_data(tmp_path: pathlib.Path, block_size: int = 10):
    state = State(position=jnp.arange(120).reshape((10, 12)))
    ckpt_metric_manager = CkptMetricManager(
        metric_schema=[],
        block_size=block_size,
        save_path=str(tmp_path),
        lazy_setup=False,
    )
    with ckpt_metric_manager:
        for i in range(block_size + 1):
            ckpt_metric_manager.run(i, state, [])
    return ckpt_metric_manager, state


def test_recovery_wrapper_has_safe_data_available(tmp_path: pathlib.Path) -> None:
    counter = {"n": 0}

    def test_func():
        counter["n"] += 1
        if counter["n"] == 1:
            raise RuntimeError("boom")
        return counter["n"], None

    ckpt_metric_manager, state = _get_manager_with_safe_data(tmp_path, 10)
    data = ckpt_metric_manager.alive_metric_manager.get_metric_data()
    assert int(data.step.iloc[-1]) == 10

    wrapped_func = recovery_wrapper(
        1,
        test_func,
        max_restore_nums=3,
        ckpt_metric_manager=ckpt_metric_manager,
        key=jax.random.PRNGKey(666),
    )
    step, output = wrapped_func()
    assert step == 1
    assert output.succeeded is False
    chex.assert_trees_all_close(output.state.position, state.position)

    rolled_back_data = ckpt_metric_manager.alive_metric_manager.get_metric_data()
    assert int(rolled_back_data.step.iloc[-1]) == 0


def test_recovery_wrapper_exceeds_max_restore_num(tmp_path: pathlib.Path) -> None:
    def test_func():
        raise RuntimeError("boom")

    ckpt_metric_manager, _state = _get_manager_with_safe_data(tmp_path)
    wrapped_func = recovery_wrapper(
        1,
        test_func,
        max_restore_nums=3,
        ckpt_metric_manager=ckpt_metric_manager,
        key=jax.random.PRNGKey(666),
    )

    for _ in range(3):
        _step, output = wrapped_func()
        assert output.succeeded is False

    with pytest.raises(Exception):
        wrapped_func()
