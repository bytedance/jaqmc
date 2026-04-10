# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import jax
from flax import linen as nn
from jax import numpy as jnp

from jaqmc.data import Data
from jaqmc.estimator import EstimatorLike
from jaqmc.estimator.density.cartesian import CartesianAxis, CartesianDensity
from jaqmc.estimator.kinetic import EuclideanKinetic
from jaqmc.estimator.total_energy import TotalEnergy
from jaqmc.optimizer.optax import adam
from jaqmc.sampler.mcmc import MCMCSampler
from jaqmc.utils.config import ConfigManager
from jaqmc.wavefunction import Wavefunction
from jaqmc.workflow.evaluation import EvaluationWorkflow
from jaqmc.workflow.stage import EvaluationWorkStage, VMCWorkStage
from jaqmc.workflow.vmc import VMCWorkflow

__all__ = ["hydrogen_atom_eval_workflow", "hydrogen_atom_train_workflow"]


class HydrogenAtomData(Data):
    electrons: jnp.ndarray


class HydrogenAtom(Wavefunction):
    initial_alpha: float = -0.8

    @nn.compact
    def __call__(self, data: HydrogenAtomData):
        alpha = self.param("alpha", lambda *_: jnp.float32(self.initial_alpha), ())
        r_ae = jnp.linalg.norm(data.electrons)
        return r_ae * alpha


def potential_energy(params, data, stats, state, rngs):
    del params, rngs, stats
    r_ae = jnp.linalg.norm(data.electrons)
    return {"energy:potential": -1 / r_ae}, state


def data_init(size, rngs):
    shape = (size, 1)
    kr, kt, kp = jax.random.split(rngs, 3)
    r = jax.random.exponential(kr, shape)
    theta = jax.random.uniform(kt, shape, maxval=jnp.pi)
    phi = jax.random.uniform(kp, shape, maxval=2 * jnp.pi)
    electrons = jnp.concatenate(
        [
            r * jnp.sin(theta) * jnp.cos(phi),
            r * jnp.sin(theta) * jnp.sin(phi),
            r * jnp.cos(theta),
        ],
        axis=-1,
    )
    return HydrogenAtomData(electrons)


def hydrogen_atom_train_workflow(cfg: ConfigManager):
    train_workflow = VMCWorkflow(cfg)
    console_fields = "pmove:.2f,energy=total_energy:.4f,variance=total_energy_var:.4f"
    cfg.use_preset({"train": {"writers": {"console": {"fields": console_fields}}}})

    wf = HydrogenAtom()
    train_workflow.data_init = data_init

    energy_estimators: dict[str, EstimatorLike] = {
        "kinetic": cfg.get("energy.kinetic", EuclideanKinetic(f_log_psi=wf.evaluate)),
        "potential": potential_energy,
        "total": TotalEnergy(),
    }
    train = VMCWorkStage.builder(cfg.scoped("train"), wf)
    train.configure_sample_plan(
        wf.evaluate, {"electrons": cfg.get("sampler", MCMCSampler)}
    )
    train.configure_optimizer(default=adam, f_log_psi=wf.evaluate)
    train.configure_estimators(**energy_estimators)
    train.configure_loss_grads(f_log_psi=wf.evaluate)
    train_workflow.train_stage = train.build()
    return train_workflow


def hydrogen_atom_eval_workflow(cfg: ConfigManager):
    eval_workflow = EvaluationWorkflow(cfg)

    wf = HydrogenAtom()
    eval_workflow.data_init = data_init

    evaluation = EvaluationWorkStage.builder(cfg, wf, name="evaluation")
    evaluation.configure_sample_plan(
        wf.evaluate, {"electrons": cfg.get("sampler", MCMCSampler)}
    )
    estimators: dict[str, EstimatorLike] = {
        "kinetic": cfg.get("energy.kinetic", EuclideanKinetic(f_log_psi=wf.evaluate)),
        "potential": potential_energy,
        "total": TotalEnergy(),
    }
    if cfg.get("estimators.enabled.density", False):
        estimators["density"] = cfg.get(
            "estimators.density",
            CartesianDensity(
                axes={"z": CartesianAxis(direction=(0, 0, 1), bins=100, range=(-8, 8))}
            ),
        )
    evaluation.configure_estimators(**estimators)
    eval_workflow.evaluation_stage = evaluation.build()
    return eval_workflow
