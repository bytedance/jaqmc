# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Fractional quantum Hall workflows."""

from dataclasses import field
from functools import partial
from typing import Annotated

from jax import numpy as jnp

from jaqmc.estimator import EstimatorCollection, WithToggles
from jaqmc.estimator.angular_momentum import (
    SphericalAngularMomentum,
    SphericalAngularMomentumProjection,
)
from jaqmc.estimator.density import SphericalDensity
from jaqmc.estimator.kinetic import SphericalKinetic
from jaqmc.estimator.loss_grad import LossAndGrad
from jaqmc.estimator.total_energy import TotalEnergy
from jaqmc.geometry.sphere import sphere_proposal
from jaqmc.optimizer.kfac import KFACOptimizer
from jaqmc.optimizer.schedule import Standard
from jaqmc.sampler import SamplerLike
from jaqmc.sampler.mcmc import MCMCSampler
from jaqmc.utils.config import ModuleConfig, configurable_dataclass
from jaqmc.utils.wiring import wire
from jaqmc.workflow.evaluation import EvaluationWorkflow
from jaqmc.workflow.stage.evaluation import EvaluationWorkStage
from jaqmc.workflow.stage.vmc import VMCWorkStage, VMCWorkStageConfig
from jaqmc.workflow.vmc import VMCTrainConfig, VMCWorkflow, stage_writers

from .config import HallConfig
from .data import data_init
from .estimator import OneRDM, PairCorrelation, PenalizedLoss
from .hamiltonian import SpherePotential
from .wavefunction import HallWavefunction
from .wavefunction.mhpo import MHPO


@configurable_dataclass
class HallEnergyEstimators(EstimatorCollection):
    """Energy component estimators on the Haldane sphere.

    Args:
        kinetic: Covariant spherical kinetic energy.
        potential: Pairwise chord-distance Coulomb repulsion on the sphere.
        total: Sums enabled ``energy:``-prefixed components into
            ``total_energy``.
    """

    kinetic: SphericalKinetic = field(default_factory=SphericalKinetic)
    potential: SpherePotential = field(default_factory=SpherePotential)
    total: TotalEnergy = field(default_factory=TotalEnergy)


@configurable_dataclass
class HallEstimators(EstimatorCollection):
    r"""Observables evaluated by quantum Hall workflows.

    Args:
        energy: Energy component estimators.
        angular_momentum: Total angular momentum (:math:`L_z`, :math:`L_z^2`,
            and :math:`L^2`) on the sphere.
        angular_projection: Total-angular-momentum sector projection
            (:math:`p_L` and :math:`E_L`) from Haar-averaged rotational
            overlaps. Requires ``estimators.energy.enabled=true``.
        penalty: Angular-momentum penalty loss (``penalized_loss``). Enabled
            when ``system.lz_penalty`` or ``system.l2_penalty`` is nonzero;
            requires both ``energy`` and ``angular_momentum`` to be enabled.
        density: Polar (and optional azimuthal) electron density histogram on
            the sphere.
        pair_correlation: Geodesic pair-angle histogram weighted by
            :math:`1/\sin\theta_{ij}`.
        one_rdm: One-body reduced density matrix in the lowest-Landau-level
            monopole harmonic basis of dimension ``system.flux + 1``.
    """

    energy: HallEnergyEstimators = field(default_factory=HallEnergyEstimators)
    angular_momentum: SphericalAngularMomentum = field(
        default_factory=SphericalAngularMomentum
    )
    angular_projection: SphericalAngularMomentumProjection = field(
        default_factory=SphericalAngularMomentumProjection
    )
    penalty: PenalizedLoss = field(default_factory=PenalizedLoss)
    density: SphericalDensity = field(default_factory=SphericalDensity)
    pair_correlation: PairCorrelation = field(default_factory=PairCorrelation)
    one_rdm: OneRDM = field(default_factory=OneRDM)


_CONSOLE_FIELDS = (
    "pmove:.2f,"
    "energy=total_energy_real:.4f,"
    "variance=total_energy_real_var:.4f,"
    "Lz=angular_momentum_z:+.4f,"
    "L_square=angular_momentum_square:.4f"
)


def _configure(wf: HallWavefunction, system: HallConfig, sampler: SamplerLike) -> None:
    q = system.flux / 2
    radius = system.radius if system.radius is not None else float(jnp.sqrt(q))
    wf.nspins = system.nspins
    wf.flux = system.flux
    wf.interaction_strength = system.interaction_strength
    wf.radius = radius
    wire(sampler, sampling_proposal=sphere_proposal)


def _wire_estimators(
    settings: HallEstimators, wf: HallWavefunction, system: HallConfig
) -> None:
    q = system.flux / 2
    radius = system.radius if system.radius is not None else float(jnp.sqrt(q))
    settings.energy.kinetic.monopole_strength = q
    settings.energy.kinetic.radius = radius
    settings.energy.kinetic.f_log_psi_from_spinor = wf.logpsi_from_spinor
    settings.energy.potential.monopole_strength = q
    settings.energy.potential.radius = radius
    settings.energy.potential.interaction_strength = system.interaction_strength
    settings.angular_momentum.f_log_psi_from_spinor = wf.logpsi_from_spinor
    settings.angular_projection.f_log_psi_from_spinor = wf.logpsi_from_spinor
    settings.penalty.lz_center = system.lz_center
    settings.penalty.lz_penalty = system.lz_penalty
    settings.penalty.l2_penalty = system.l2_penalty
    settings.one_rdm.flux = system.flux
    settings.one_rdm.f_log_psi = wf.logpsi


@configurable_dataclass
class HallWorkflowBase:
    system: HallConfig = field(default_factory=HallConfig)
    wf: Annotated[
        HallWavefunction,
        ModuleConfig(package="jaqmc.app.hall.wavefunction"),
    ] = field(default_factory=MHPO)
    sampler: Annotated[SamplerLike, ModuleConfig(package="jaqmc.sampler")] = field(
        default_factory=MCMCSampler
    )
    estimators: Annotated[
        HallEstimators,
        WithToggles(
            angular_momentum=True,
            angular_projection=False,
            density=False,
            pair_correlation=False,
            one_rdm=False,
        ),
    ] = field(default_factory=HallEstimators)

    def __post_init__(self) -> None:
        self.estimators.set_enabled(penalty=self._penalized)
        if self.estimators.is_enabled("penalty"):
            for key, label in (
                ("total", "total energy"),
                ("angular_momentum", "angular-momentum"),
            ):
                if not self.estimators.is_enabled(key):
                    raise ValueError(
                        f"Angular-momentum penalties require the {label} estimator."
                    )
        if self.estimators.is_enabled(
            "angular_projection"
        ) and not self.estimators.is_enabled("total"):
            raise ValueError(
                "Angular-momentum projection requires the total energy estimator."
            )

    @property
    def _penalized(self) -> bool:
        return bool(self.system.lz_penalty or self.system.l2_penalty)


@configurable_dataclass
class HallTrainWorkflow(HallWorkflowBase, VMCWorkflow):
    train: VMCTrainConfig = field(
        default_factory=lambda: VMCTrainConfig(
            run=VMCWorkStageConfig(iterations=200_000),
            optim=KFACOptimizer(learning_rate=Standard(delay=10_000, warmup=1_000)),
            grads=LossAndGrad(clip_scale=100),
            writers=stage_writers(_CONSOLE_FIELDS),
        )
    )

    def build_runtime(self) -> None:
        _configure(self.wf, self.system, self.sampler)
        train = VMCWorkStage.builder(self.train.run, self.wf, name="train")
        train.configure_sample_plan(self.wf.logpsi, {"electrons": self.sampler})
        train.configure_optimizer(self.train.optim, f_log_psi=self.wf.logpsi)
        _wire_estimators(self.estimators, self.wf, self.system)
        train.configure_estimators(self.estimators)
        train.configure_loss_grads(
            self.train.grads,
            f_log_psi=self.wf.logpsi,
            loss_key="penalized_loss" if self._penalized else None,
        )
        train.configure_writers(self.train.writers)
        self.train_stage = train.build()
        self.data_init = partial(data_init, self.system)


@configurable_dataclass
class HallEvalWorkflow(HallWorkflowBase, EvaluationWorkflow):
    estimators: Annotated[
        HallEstimators,
        WithToggles(
            energy=True,
            angular_momentum=True,
            angular_projection=False,
            density=False,
            pair_correlation=False,
            one_rdm=False,
        ),
    ] = field(default_factory=HallEstimators)

    def build_runtime(self) -> None:
        _configure(self.wf, self.system, self.sampler)
        evaluation = EvaluationWorkStage.builder(
            self.run_config, self.wf, name="evaluation"
        )
        evaluation.configure_sample_plan(self.wf.logpsi, {"electrons": self.sampler})
        _wire_estimators(self.estimators, self.wf, self.system)
        evaluation.configure_estimators(self.estimators)
        evaluation.configure_writers(self.writers)
        self.evaluation_stage = evaluation.build()
        self.data_init = partial(data_init, self.system)
