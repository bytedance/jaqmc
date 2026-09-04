# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from typing import NamedTuple

import jax
import numpy as np
import pytest
from jaqmc_contrib_lit.config import LITConfig
from jaqmc_contrib_lit.continuation_policy import (
    _continuation_min_step,
    _continuation_probe_is_acceptable,
    _physics_continuation_step,
)
from jaqmc_contrib_lit.optimization import _is_better_lit_checkpoint
from jaqmc_contrib_lit.state import _FidelityPlateauTracker
from jax import numpy as jnp


class _Stats(NamedTuple):
    loss: jax.Array
    fidelity: jax.Array
    reverse_kl: jax.Array
    invalid_sample_fraction: jax.Array
    reweight_ess_fraction: jax.Array
    signed_lit: jax.Array
    source_norm: jax.Array


def _stats(
    fidelity: float = 1.0,
    *,
    ess: float = 1.0,
    signed_lit: float = 1.0,
    source_norm: float = 1.0,
) -> _Stats:
    return _Stats(
        loss=jnp.asarray(1.0 - fidelity),
        fidelity=jnp.asarray(fidelity),
        reverse_kl=jnp.asarray(0.0),
        invalid_sample_fraction=jnp.asarray(0.0),
        reweight_ess_fraction=jnp.asarray(ess),
        signed_lit=jnp.asarray(signed_lit),
        source_norm=jnp.asarray(source_norm),
    )


def test_default_continuation_step_tracks_finer_spectrum_spacing():
    assert _continuation_min_step(
        LITConfig(), np.asarray([0.772, 0.77225])
    ) == pytest.approx(0.00025)


def test_physics_continuation_step_uses_lit_residual_scale():
    step = _physics_continuation_step(
        _stats(signed_lit=4.0),
        gap=1.0,
        fraction=0.2,
        min_step=0.01,
    )
    assert step == pytest.approx(0.1)


def test_continuation_probe_requires_relative_fidelity_and_absolute_ess():
    current = _stats(fidelity=0.9)
    assert _continuation_probe_is_acceptable(
        current,
        _stats(fidelity=0.86, ess=0.5),
        retention=0.95,
        min_reweight_ess_fraction=0.2,
    )
    assert not _continuation_probe_is_acceptable(
        current,
        _stats(fidelity=0.84, ess=0.5),
        retention=0.95,
        min_reweight_ess_fraction=0.2,
    )
    assert not _continuation_probe_is_acceptable(
        current,
        _stats(fidelity=0.88, ess=0.1),
        retention=0.95,
        min_reweight_ess_fraction=0.2,
    )


def test_plateau_tracker_resets_only_after_cumulative_significant_gain():
    tracker = _FidelityPlateauTracker(
        start_iteration=2,
        patience_iterations=2,
        min_delta=0.01,
    )

    assert not tracker.observe(2, 0.80)
    assert not tracker.observe(3, 0.805)
    assert not tracker.observe(4, 0.811)
    assert tracker.last_significant_iteration == 4
    assert not tracker.observe(5, 0.811)
    assert tracker.observe(6, 0.811)


def test_checkpoint_selection_prefers_healthy_fidelity_over_regularized_loss():
    incumbent = _stats(fidelity=0.8)._replace(
        loss=jnp.asarray(0.2),
        reverse_kl=jnp.asarray(0.0),
    )
    candidate = _stats(fidelity=0.995)._replace(
        loss=jnp.asarray(0.505),
        reverse_kl=jnp.asarray(0.5),
    )
    unhealthy = _stats(fidelity=0.999, ess=0.01)

    assert _is_better_lit_checkpoint(
        candidate,
        incumbent,
        min_reweight_ess_fraction=0.05,
    )
    assert not _is_better_lit_checkpoint(
        unhealthy,
        candidate,
        min_reweight_ess_fraction=0.05,
    )
