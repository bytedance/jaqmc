# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from dataclasses import field
from typing import Annotated

import jax
from flax import linen as nn
from jax import numpy as jnp

from jaqmc.data import Data
from jaqmc.estimator import EstimatorCollection, PerWalkerEstimator, WithToggles
from jaqmc.estimator.density.cartesian import CartesianAxis, CartesianDensity
from jaqmc.estimator.kinetic import EuclideanKinetic
from jaqmc.estimator.total_energy import TotalEnergy
from jaqmc.optimizer.optax import adam
from jaqmc.sampler import SamplerLike
from jaqmc.sampler.mcmc import MCMCSampler
from jaqmc.utils.config import ModuleConfig, configurable_dataclass
from jaqmc.utils.wiring import wire
from jaqmc.wavefunction import Wavefunction, WavefunctionLike
from jaqmc.workflow.evaluation import EvaluationWorkflow
from jaqmc.workflow.stage import EvaluationWorkStage, VMCWorkStage
from jaqmc.workflow.vmc import VMCTrainConfig, VMCWorkflow, stage_writers

__all__ = ["HydrogenAtomEvalWorkflow", "HydrogenAtomTrainWorkflow"]


class HydrogenAtomData(Data):
    electrons: jnp.ndarray


class HydrogenAtom(Wavefunction[HydrogenAtomData, jnp.ndarray]):
    initial_alpha: float = -0.8

    @nn.compact
    def __call__(self, data: HydrogenAtomData) -> jnp.ndarray:
        alpha = self.param("alpha", lambda *_: jnp.float32(self.initial_alpha), ())
        return jnp.linalg.norm(data.electrons) * alpha


@configurable_dataclass
class PotentialEnergy(PerWalkerEstimator):
    """Hydrogen electron-nucleus potential energy."""

    def evaluate_single_walker(self, params, data, prev_walker_stats, state, rngs):
        del params, prev_walker_stats, rngs
        return {"energy:potential": -1 / jnp.linalg.norm(data.electrons)}, state


def data_init(size, rngs):
    shape = (size, 1)
    kr, kt, kp = jax.random.split(rngs, 3)
    r = jax.random.exponential(kr, shape)
    theta = jax.random.uniform(kt, shape, maxval=jnp.pi)
    phi = jax.random.uniform(kp, shape, maxval=2 * jnp.pi)
    return HydrogenAtomData(
        jnp.concatenate(
            [
                r * jnp.sin(theta) * jnp.cos(phi),
                r * jnp.sin(theta) * jnp.sin(phi),
                r * jnp.cos(theta),
            ],
            axis=-1,
        )
    )


@configurable_dataclass
class HydrogenEnergyEstimators(EstimatorCollection):
    kinetic: EuclideanKinetic = field(default_factory=EuclideanKinetic)
    potential: PotentialEnergy = field(default_factory=PotentialEnergy)
    total: TotalEnergy = field(default_factory=TotalEnergy)


@configurable_dataclass
class HydrogenEstimators(EstimatorCollection):
    energy: HydrogenEnergyEstimators = field(default_factory=HydrogenEnergyEstimators)
    density: CartesianDensity = field(
        default_factory=lambda: CartesianDensity(
            axes={"z": CartesianAxis(direction=(0, 0, 1), bins=100, range=(-8, 8))},
        )
    )


@configurable_dataclass
class HydrogenAtomWorkflowBase:
    wf: Annotated[WavefunctionLike, ModuleConfig()] = field(
        default_factory=HydrogenAtom
    )
    sampler: Annotated[SamplerLike, ModuleConfig(package="jaqmc.sampler")] = field(
        default_factory=MCMCSampler
    )
    estimators: Annotated[HydrogenEstimators, WithToggles(density=False)] = field(
        default_factory=HydrogenEstimators
    )


@configurable_dataclass
class HydrogenAtomTrainWorkflow(HydrogenAtomWorkflowBase, VMCWorkflow):
    train: VMCTrainConfig = field(
        default_factory=lambda: VMCTrainConfig(
            optim=adam(),
            writers=stage_writers(
                "pmove:.2f,energy=total_energy:.4f,variance=total_energy_var:.4f"
            ),
        )
    )

    def build_runtime(self) -> None:
        train = VMCWorkStage.builder(self.train.run, self.wf, name="train")
        train.configure_sample_plan(self.wf.evaluate, {"electrons": self.sampler})
        train.configure_optimizer(self.train.optim, f_log_psi=self.wf.evaluate)
        wire(self.estimators.energy.kinetic, f_log_psi=self.wf.evaluate)
        train.configure_estimators(self.estimators)
        train.configure_loss_grads(self.train.grads, f_log_psi=self.wf.evaluate)
        train.configure_writers(self.train.writers)
        self.train_stage = train.build()
        self.data_init = data_init


@configurable_dataclass
class HydrogenAtomEvalWorkflow(HydrogenAtomWorkflowBase, EvaluationWorkflow):
    estimators: Annotated[
        HydrogenEstimators, WithToggles(energy=True, density=False)
    ] = field(default_factory=HydrogenEstimators)

    def build_runtime(self) -> None:
        evaluation = EvaluationWorkStage.builder(
            self.run_config, self.wf, name="evaluation"
        )
        evaluation.configure_sample_plan(self.wf.evaluate, {"electrons": self.sampler})
        wire(self.estimators.energy.kinetic, f_log_psi=self.wf.evaluate)
        evaluation.configure_estimators(self.estimators)
        evaluation.configure_writers(self.writers)
        self.evaluation_stage = evaluation.build()
        self.data_init = data_init
