# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from jaqmc.data import BatchedData, Data
from jaqmc.estimator import StreamingLossAndGrad
from jaqmc.utils import parallel_jax
from jaqmc.utils.config import ConfigManager
from jaqmc.wavefunction.determinant_state import SubspaceSpec, take_replica
from jaqmc.workflow.base import init_batched_data
from jaqmc.workflow.subspace_vmc import (
    SubspaceVMCWorkflow,
    SubspaceVMCWorkStage,
    make_subspace_data_init,
    replicate_walker_replicas,
)


class ToyData(Data):
    electrons: jax.Array
    atoms: jax.Array


def test_replica_axis_is_inside_walker_axis_and_physical_data_is_recoverable():
    physical = BatchedData(
        ToyData(electrons=jnp.zeros((5, 2, 3)), atoms=jnp.ones((1, 3))),
        ["electrons"],
    )
    spec = SubspaceSpec(4)

    replica_data = replicate_walker_replicas(physical, spec)
    one_walker = replica_data.unbatched_example()
    recovered = take_replica(one_walker, 2, spec)

    assert replica_data.data.electrons.shape == (5, 4, 2, 3)
    assert replica_data.fields_with_batch == ["electrons"]
    assert recovered.electrons.shape == (2, 3)
    np.testing.assert_array_equal(recovered.atoms, physical.data.atoms)


def test_subspace_data_init_groups_independent_native_samples():
    requested_sizes = []

    def physical_data_init(size, rngs):
        del rngs
        requested_sizes.append(size)
        return BatchedData(
            ToyData(
                electrons=jnp.arange(size, dtype=float)[:, None, None],
                atoms=jnp.ones((1, 3)),
            ),
            ["electrons"],
        )

    data_init = make_subspace_data_init(physical_data_init, SubspaceSpec(3))
    result = data_init(4, jax.random.key(0))

    assert requested_sizes == [12]
    assert result.data.electrons.shape == (4, 3, 1, 1)
    np.testing.assert_array_equal(
        result.data.electrons[:, :, 0, 0], jnp.arange(12).reshape(4, 3)
    )


class ToyWavefunction:
    def init_params(self, data, rngs):
        del data
        return {"slope": jax.random.uniform(rngs, (), minval=0.5, maxval=1.5)}

    def logpsi(self, params, data):
        return params["slope"] * jnp.sum(data.electrons)

    def phase_logpsi(self, params, data):
        return jnp.array(1.0), self.logpsi(params, data)


def test_workflow_accepts_documented_nested_subspace_config():
    cfg = ConfigManager(
        {
            "workflow": {"batch_size": 4, "save_path": ".tmp/subspace-test"},
            "subspace": {
                "n_states": 2,
                "sampling": {"steps": 3, "initial_width": 0.04},
                "evaluation": {
                    "pair_chunk_size": 2,
                    "matrix_dtype": "complex64",
                },
                "diagnostics": {
                    "condition_warning": 1e8,
                    "solve_residual_warning": 1e-5,
                    "max_imag_eigenvalue_warning": 1e-4,
                },
            },
            "train": {
                "grads": {
                    "vmap_chunk_size": 3,
                    "clip_method": "none",
                    "clip_scale": 7.0,
                    "loss_key": "must_be_overridden",
                }
            },
        }
    )

    def physical_data_init(size, rngs):
        del rngs
        return BatchedData(
            ToyData(
                electrons=jnp.arange(size, dtype=float)[:, None, None],
                atoms=jnp.ones((1, 3)),
            ),
            ["electrons"],
        )

    def energy(params, data, prev_stats, state, rngs):
        del params, data, prev_stats, rngs
        return {"total_energy": jnp.array(1.0)}, state

    workflow = SubspaceVMCWorkflow(cfg)
    workflow.configure_subspace(
        base_wavefunction=ToyWavefunction(),
        physical_data_init=physical_data_init,
        physical_energy_estimators={"total": energy},
    )
    workflow.prepare(dry_run=True)

    sampler = workflow.train_stage.sample_plan.samplers[("electrons",)]
    optimizer = workflow.train_stage.optimizer
    grads = workflow.train_stage.estimators.estimators["grads"]
    assert sampler.n_states == 2
    assert sampler.steps == 3
    assert sampler.initial_width == 0.04
    assert type(optimizer).__name__ == "adam"
    assert isinstance(grads, StreamingLossAndGrad)
    assert grads.vmap_chunk_size == 3
    assert grads.clip_method == "none"
    assert grads.clip_scale == 7.0
    assert grads.loss_key == "subspace_local_energy"

    console_fields = workflow.default_preset()["train"]["writers"]["console"][
        "fields"
    ]
    assert "grad_norm=grad_norm" in console_fields
    assert "update_norm=update_norm" in console_fields

    data = init_batched_data(workflow.data_init, 4, jax.random.key(2))
    state = workflow.train_stage.create_state(jax.random.key(3), batched_data=data)
    partition = state.partition()
    compute = parallel_jax.jit_sharded(
        workflow.train_stage.compute_step,
        in_specs=(partition, parallel_jax.DATA_PARTITION),
        out_specs=(partition, parallel_jax.SHARE_PARTITION),
        check_vma=False,
    )
    rngs = jax.device_put(
        jax.random.split(jax.random.PRNGKey(4), jax.device_count()).flatten(),
        parallel_jax.make_sharding(parallel_jax.DATA_PARTITION),
    )
    updated, stats = compute(state, rngs)

    assert jnp.isfinite(stats["subspace_energy"])
    assert jnp.isfinite(stats["subspace_energy_var"])
    assert all(
        np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(updated.params)
    )


def test_invalid_subspace_step_skips_optimizer_update():
    @dataclass
    class ToyState:
        params: dict
        batched_data: object
        sampler_state: object
        estimator_state: object
        opt_state: dict

    original = ToyState(
        params={"w": jnp.array(1.0)},
        batched_data=object(),
        sampler_state=None,
        estimator_state=None,
        opt_state={"count": jnp.array(2)},
    )

    class SamplePlan:
        def step(self, params, data, sampler_state, rngs):
            del params, rngs
            return data, {}, sampler_state

    class Estimators:
        def evaluate(self, params, data, estimator_state, rngs):
            del params, data, rngs
            return {}, estimator_state

        def finalize_stats(self, stats, estimator_state):
            del stats, estimator_state
            return {
                "grads": {"w": jnp.array(jnp.nan)},
                "training_step_valid": jnp.array(False),
            }

    class Optimizer:
        def update(self, grads, opt_state, **kwargs):
            del grads, kwargs
            return {"w": jnp.array(jnp.nan)}, {"count": opt_state["count"] + 1}

    stage = object.__new__(SubspaceVMCWorkStage)
    stage.sample_plan = SamplePlan()
    stage.estimators = Estimators()
    stage.optimizer = Optimizer()

    updated, stats = stage.compute_step(original, jax.random.key(0))

    np.testing.assert_array_equal(updated.params["w"], original.params["w"])
    np.testing.assert_array_equal(
        updated.opt_state["count"], original.opt_state["count"]
    )
    np.testing.assert_array_equal(stats["update_norm"], 0)
    assert stage._has_nan(stats)
