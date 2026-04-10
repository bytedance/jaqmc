# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Workflows for the fractional quantum Hall effect on a Haldane sphere."""

import logging
from functools import partial
from typing import Any

from jax import numpy as jnp

from jaqmc.estimator import EstimatorLike
from jaqmc.estimator.density import SphericalDensity
from jaqmc.estimator.kinetic import SphericalKinetic
from jaqmc.estimator.loss_grad import LossAndGrad
from jaqmc.estimator.total_energy import TotalEnergy
from jaqmc.geometry.sphere import sphere_proposal
from jaqmc.optimizer.kfac import KFACOptimizer
from jaqmc.sampler.mcmc import MCMCSampler
from jaqmc.utils.config import ConfigManager, ConfigManagerLike
from jaqmc.workflow.evaluation import EvaluationWorkflow
from jaqmc.workflow.stage.evaluation import EvaluationWorkStage
from jaqmc.workflow.stage.vmc import VMCWorkStage
from jaqmc.workflow.vmc import VMCWorkflow

from .config import HallConfig
from .data import data_init
from .estimator import OneRDM, PairCorrelation, PenalizedLoss
from .hamiltonian import SpherePotential

logger = logging.getLogger(__name__)


class HallTrainWorkflow(VMCWorkflow):
    """VMC training workflow for quantum Hall effect simulations."""

    @classmethod
    def default_preset(cls) -> dict[str, Any]:
        console_fields = (
            "pmove:.2f,"
            "energy=total_energy_real:.4f,"
            "variance=total_energy_var:.4f,"
            "Lz=angular_momentum_z:+.4f,"
            "L_square=angular_momentum_square:.4f"
        )
        return {
            "train": {
                "run": {"iterations": 200_000},
                "writers": {"console": {"fields": console_fields}},
            }
        }

    def __init__(self, cfg: ConfigManager) -> None:
        super().__init__(cfg)
        system_config, wf = configure_system(cfg)

        self.data_init = partial(data_init, system_config)

        has_penalties = system_config.lz_penalty or system_config.l2_penalty
        loss_key = "penalized_loss" if has_penalties else "total_energy"

        estimators = make_estimators(cfg, wf, system_config, always_enable_energy=True)

        train = VMCWorkStage.builder(cfg.scoped("train"), wf)
        sampler = cfg.get("sampler", MCMCSampler(sampling_proposal=sphere_proposal))
        train.configure_sample_plan(wf.logpsi, {"electrons": sampler})
        train.configure_optimizer(default=KFACOptimizer, f_log_psi=wf.logpsi)
        train.configure_estimators(**estimators)
        train.configure_loss_grads(LossAndGrad(loss_key=loss_key), f_log_psi=wf.logpsi)
        self.train_stage = train.build()


class HallEvalWorkflow(EvaluationWorkflow):
    """Evaluation workflow for quantum Hall effect simulations."""

    def __init__(self, cfg: ConfigManager) -> None:
        super().__init__(cfg)
        system_config, wf = configure_system(cfg)

        self.data_init = partial(data_init, system_config)

        evaluation = EvaluationWorkStage.builder(cfg, wf, name="evaluation")
        sampler = cfg.get("sampler", MCMCSampler(sampling_proposal=sphere_proposal))
        evaluation.configure_sample_plan(wf.logpsi, {"electrons": sampler})
        eval_estimators: dict[str, EstimatorLike] = make_estimators(
            cfg, wf, system_config
        )
        evaluation.configure_estimators(**eval_estimators)
        self.evaluation_stage = evaluation.build()


def configure_system(
    cfg: ConfigManagerLike,
) -> tuple[HallConfig, Any]:
    """Build the shared system objects for quantum Hall workflows.

    Returns:
        Tuple of (system_config, wavefunction).
    """
    system_config: HallConfig = cfg.get_module(
        "system", "jaqmc.app.hall.config:HallConfig"
    )

    wf = cfg.get_module("wf", "jaqmc.app.hall.wavefunction.mhpo")
    wf.nspins = system_config.nspins
    wf.monopole_strength = system_config.flux / 2
    wf.flux = system_config.flux

    return system_config, wf


def make_estimators(
    cfg: ConfigManagerLike,
    wf: Any,
    system_config: HallConfig,
    always_enable_energy: bool = False,
) -> dict[str, EstimatorLike]:
    estimators: dict[str, EstimatorLike] = {}
    if always_enable_energy or cfg.get("estimators.enabled.energy", True):
        Q = system_config.flux / 2
        radius = (
            system_config.radius
            if system_config.radius is not None
            else float(jnp.sqrt(Q))
        )

        estimators["kinetic"] = cfg.get(
            "estimators.energy.kinetic",
            SphericalKinetic(
                monopole_strength=Q,
                radius=radius,
                f_log_psi=wf.logpsi,
            ),
        )
        estimators["potential"] = cfg.get(
            "estimators.energy.potential",
            SpherePotential(
                interaction_type=system_config.interaction_type,
                monopole_strength=Q,
                radius=radius,
                interaction_strength=system_config.interaction_strength,
            ),
        )
        estimators["total"] = TotalEnergy()

        if system_config.lz_penalty or system_config.l2_penalty:
            estimators["penalty"] = PenalizedLoss(
                lz_center=system_config.lz_center,
                lz_penalty=system_config.lz_penalty,
                l2_penalty=system_config.l2_penalty,
            )

    if cfg.get("estimators.enabled.density", False):
        estimators["density"] = cfg.get(
            "estimators.density",
            SphericalDensity(),
        )

    if cfg.get("estimators.enabled.pair_correlation", False):
        estimators["pair_correlation"] = cfg.get(
            "estimators.pair_correlation",
            PairCorrelation(),
        )

    if cfg.get("estimators.enabled.one_rdm", False):
        estimators["one_rdm"] = cfg.get(
            "estimators.one_rdm",
            OneRDM(flux=system_config.flux, f_log_psi=wf.logpsi),
        )

    return estimators
