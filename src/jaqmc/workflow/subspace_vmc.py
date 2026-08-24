# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""JaQMC-native assembly of determinant-state variational training."""

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

import jax
from jax import numpy as jnp

from jaqmc.array_types import PRNGKey
from jaqmc.data import BatchedData
from jaqmc.estimator import CrossLocalEnergyEvaluator, EstimatorLike
from jaqmc.estimator.loss_grad import LossAndGrad
from jaqmc.estimator.rayleigh import RayleighMatrixEstimator
from jaqmc.optimizer.kfac import KFACOptimizer
from jaqmc.sampler.determinant import DeterminantMCMCSampler
from jaqmc.utils.config import ConfigManager, configurable_dataclass
from jaqmc.utils.wiring import wire
from jaqmc.wavefunction.determinant_state import (
    DeterminantStateWavefunction,
    SubspaceSpec,
)
from jaqmc.workflow.stage.vmc import VMCWorkStage
from jaqmc.workflow.vmc import VMCWorkflow


@configurable_dataclass
class SubspaceConfig:
    """Configuration shared by molecule and solid subspace workflows."""

    n_states: int = 2
    condition_warning: float = 1e10
    solve_residual_warning: float = 1e-6
    max_imag_eigenvalue_warning: float = 1e-6

    def __post_init__(self):
        if self.n_states < 1:
            raise ValueError("subspace.n_states must be positive")


def replicate_walker_replicas(
    batched_data: BatchedData, spec: SubspaceSpec
) -> BatchedData:
    """Repeat batched fields along a new replica axis.

    This shape helper is useful in tests and adapters but must not initialize a
    determinant chain: identical replica rows produce a singular amplitude
    matrix.  Use :func:`make_subspace_data_init` for workflow initialization.
    """
    missing = set(spec.replica_fields) - set(batched_data.fields_with_batch)
    if missing:
        raise ValueError(
            "Replica fields must already carry the walker batch axis: "
            f"{sorted(missing)}"
        )
    data = replace(
        batched_data.data,
        **{
            name: jax.tree.map(
                lambda x: jnp.repeat(x[:, None], spec.n_states, axis=1),
                batched_data.data[name],
            )
            for name in spec.replica_fields
        },
    )
    return replace(batched_data, data=data)


def make_subspace_data_init(
    physical_data_init: Callable[[int, PRNGKey], BatchedData], spec: SubspaceSpec
):
    """Initialize independent physical configurations and group them by walker.

    Repeating one configuration across replicas makes the initial amplitude
    matrix singular.  Requesting ``B*M`` native samples keeps initialization
    owned by the app while introducing only a reshape in this adapter.
    """

    def data_init(size: int, rngs: PRNGKey) -> BatchedData:
        physical = physical_data_init(size * spec.n_states, rngs)
        physical.check()
        if extra := set(physical.fields_with_batch) - set(spec.replica_fields):
            raise ValueError(
                "All native batched fields must be listed in replica_fields; "
                f"missing {sorted(extra)}"
            )
        data = replace(
            physical.data,
            **{
                name: jax.tree.map(
                    lambda x: x.reshape(size, spec.n_states, *x.shape[1:]),
                    physical.data[name],
                )
                for name in physical.fields_with_batch
                if name in spec.replica_fields
            },
        )
        result = replace(physical, data=data)
        result.check()
        return result

    return data_init


class SubspaceVMCWorkflow(VMCWorkflow):
    """Base workflow that reuses JaQMC's VMC stage, gradients, and optimizers."""

    config_namespace = "subspace_train"

    @classmethod
    def default_preset(cls) -> dict[str, Any]:
        fields = (
            "pmove:.2f,energy=subspace_energy:.4f,"
            "variance=subspace_energy_var:.4f,max_imag=max_ritz_imag:.2e"
        )
        return {
            "train": {
                "run": {"iterations": 200_000},
                "writers": {"console": {"fields": fields}},
            }
        }

    def __init__(self, cfg: ConfigManager) -> None:
        super().__init__(cfg)
        # Read leaves separately so the ``subspace`` namespace can also hold
        # nested sampler/evaluation configuration without dataclass decoding
        # treating those sections as unknown SubspaceConfig fields.
        self.subspace = SubspaceConfig(
            n_states=cfg.get("subspace.n_states", 2),
            condition_warning=cfg.get(
                "subspace.diagnostics.condition_warning", 1e10
            ),
            solve_residual_warning=cfg.get(
                "subspace.diagnostics.solve_residual_warning", 1e-6
            ),
            max_imag_eigenvalue_warning=cfg.get(
                "subspace.diagnostics.max_imag_eigenvalue_warning", 1e-6
            ),
        )
        self.spec = SubspaceSpec(self.subspace.n_states)

    def configure_subspace(
        self,
        *,
        base_wavefunction,
        physical_data_init: Callable[[int, PRNGKey], BatchedData],
        physical_energy_estimators: Mapping[str, EstimatorLike],
        physical_proposal=None,
    ) -> None:
        """Assemble the native VMC stage around app-provided physical pieces."""
        self.base_wavefunction = base_wavefunction
        self.wf = DeterminantStateWavefunction(base_wavefunction, self.spec)
        self.data_init = make_subspace_data_init(physical_data_init, self.spec)

        sampler_default = DeterminantMCMCSampler(
            n_states=self.spec.n_states,
            initial_width=0.02,
            **(
                {"sampling_proposal": physical_proposal}
                if physical_proposal is not None
                else {}
            ),
        )
        sampler = self.cfg.get("subspace.sampling", sampler_default)
        rayleigh = self.cfg.get_module(
            "subspace.evaluation",
            RayleighMatrixEstimator,
        )
        rayleigh.condition_warning = self.subspace.condition_warning
        rayleigh.solve_residual_warning = self.subspace.solve_residual_warning
        rayleigh.max_imag_eigenvalue_warning = (
            self.subspace.max_imag_eigenvalue_warning
        )
        cross_energy = CrossLocalEnergyEvaluator(
            physical_energy_estimators,
            self.spec,
            pair_chunk_size=rayleigh.pair_chunk_size,
        )
        wire(
            rayleigh,
            f_component_logpsi_matrix=self.wf.component_logpsi_matrix,
            f_cross_local_energy=cross_energy,
        )

        train = VMCWorkStage.builder(self.cfg.scoped("train"), self.wf)
        train.configure_sample_plan(self.wf.logpsi, {"electrons": sampler})
        train.configure_optimizer(default=KFACOptimizer, f_log_psi=self.wf.logpsi)
        train.configure_estimators(rayleigh=rayleigh)
        train.configure_loss_grads(
            LossAndGrad(loss_key="subspace_energy"), f_log_psi=self.wf.logpsi
        )
        self.train_stage = train.build()


def energy_estimators_only(
    estimators: Mapping[str, EstimatorLike],
) -> dict[str, EstimatorLike]:
    """Keep the ordered physical energy pipeline and discard unrelated observables."""
    names = ("potential", "kinetic", "ecp", "ph", "total")
    return {name: estimators[name] for name in names if name in estimators}
