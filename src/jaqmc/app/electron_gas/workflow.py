# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Homogeneous electron-gas workflows."""

from dataclasses import field
from functools import partial
from typing import Annotated

from jax import numpy as jnp

from jaqmc.app.solid.hamiltonian import PotentialEnergy
from jaqmc.estimator import EstimatorCollection, HideLoss, WithToggles
from jaqmc.estimator.kinetic import EuclideanKinetic
from jaqmc.estimator.overlap import Overlap, ReferenceState
from jaqmc.estimator.spin import SpinSquared
from jaqmc.estimator.total_energy import TotalEnergy
from jaqmc.geometry.pbc import make_pbc_gaussian_proposal
from jaqmc.sampler import SamplerLike
from jaqmc.sampler.mcmc import MCMCSampler
from jaqmc.utils.atomic.pretrain import make_pretrain_log_amplitude, make_pretrain_loss
from jaqmc.utils.config import ModuleConfig, configurable_dataclass
from jaqmc.utils.wiring import wire
from jaqmc.wavefunction.output.envelope import EnvelopeType
from jaqmc.workflow.evaluation import EvaluationWorkflow
from jaqmc.workflow.stage.evaluation import EvaluationWorkStage
from jaqmc.workflow.stage.vmc import VMCWorkStage, VMCWorkStageConfig
from jaqmc.workflow.vmc import (
    VMCPretrainConfig,
    VMCTrainConfig,
    VMCWorkflow,
    stage_writers,
)

from .config import ElectronGasConfig
from .data import data_init
from .reference import FreeElectronReference
from .wavefunction import ElectronGasWavefunction, ElectronGasWavefunctionLike


@configurable_dataclass
class ElectronGasEnergyEstimators(EstimatorCollection):
    """Energy component estimators for homogeneous electron-gas systems.

    Args:
        potential: Periodic Coulomb potential energy via Ewald summation.
        kinetic: Euclidean kinetic energy.
        total: Sums enabled ``energy:``-prefixed components into
            ``total_energy``.
    """

    potential: PotentialEnergy = field(default_factory=PotentialEnergy)
    kinetic: EuclideanKinetic = field(default_factory=EuclideanKinetic)
    total: TotalEnergy = field(default_factory=TotalEnergy)


@configurable_dataclass
class ElectronGasReferenceState(ReferenceState):
    """Reference state for electron-gas overlap estimation."""

    wf: Annotated[
        ElectronGasWavefunctionLike | None,
        ModuleConfig(package="jaqmc.app.electron_gas.wavefunction"),
    ] = None


@configurable_dataclass
class ElectronGasEstimators(EstimatorCollection):
    """Observables evaluated by electron-gas workflows.

    Args:
        energy: Energy component estimators.
        spin: Total spin-squared (:math:`S^2`) estimator.
        overlap: Wavefunction overlap estimator.
    """

    energy: ElectronGasEnergyEstimators = field(
        default_factory=ElectronGasEnergyEstimators
    )
    spin: SpinSquared = field(default_factory=SpinSquared)
    overlap: Overlap[ElectronGasReferenceState] = field(default_factory=Overlap)


def _validate_wavefunction(wf: ElectronGasWavefunctionLike) -> None:
    if wf.envelope_type != EnvelopeType.null:
        raise ValueError("Electron-gas wavefunctions require envelope_type='null'.")
    if not wf.full_det:
        raise ValueError("Electron-gas wavefunctions require full_det=True.")


@configurable_dataclass
class ElectronGasWorkflowBase:
    system: ElectronGasConfig = field(default_factory=ElectronGasConfig)
    wf: Annotated[
        ElectronGasWavefunctionLike,
        ModuleConfig(package="jaqmc.app.electron_gas.wavefunction"),
    ] = field(default_factory=ElectronGasWavefunction)
    sampler: Annotated[SamplerLike, ModuleConfig(package="jaqmc.sampler")] = field(
        default_factory=MCMCSampler
    )
    estimators: Annotated[
        ElectronGasEstimators, WithToggles(spin=False, overlap=False)
    ] = field(default_factory=ElectronGasEstimators)

    def __post_init__(self) -> None:
        _validate_wavefunction(self.wf)


@configurable_dataclass
class ElectronGasTrainWorkflow(ElectronGasWorkflowBase, VMCWorkflow):
    """VMC training with analytic free-electron pretraining."""

    pretrain: VMCPretrainConfig = field(
        default_factory=lambda: VMCPretrainConfig(
            run=VMCWorkStageConfig(iterations=1_000)
        )
    )
    train: VMCTrainConfig = field(
        default_factory=lambda: VMCTrainConfig(
            run=VMCWorkStageConfig(iterations=200_000),
            writers=stage_writers(
                "pmove:.2f,energy=total_energy:.4f,variance=total_energy_real_var:.4f"
            ),
        )
    )

    def build_runtime(self) -> None:
        reference = _configure_system(self.system, self.wf)
        lattice = jnp.asarray(self.system.lattice)
        wire(self.sampler, sampling_proposal=make_pbc_gaussian_proposal(lattice))
        loss = make_pretrain_loss(
            orbitals_fn=self.wf.orbitals,
            orbital_ref=reference,
            nspins=self.system.nspins,
            full_det=self.wf.full_det,
        )
        log_amplitude = make_pretrain_log_amplitude(
            self.wf.logpsi,
            lambda data: reference.eval_slater(data.electrons, self.system.nspins).real,
            ref_fraction=1.0,
        )
        pretrain = VMCWorkStage.builder(self.pretrain.run, self.wf, name="pretrain")
        pretrain.configure_sample_plan(log_amplitude, {"electrons": self.sampler})
        pretrain.configure_optimizer(self.pretrain.optim, f_log_psi=self.wf.logpsi)
        pretrain.configure_estimators({"grads": loss})
        pretrain.configure_writers(self.pretrain.writers)
        train = VMCWorkStage.builder(self.train.run, self.wf, name="train")
        train.configure_sample_plan(self.wf.logpsi, {"electrons": self.sampler})
        train.configure_optimizer(self.train.optim, f_log_psi=self.wf.logpsi)
        _wire_estimators(self.estimators, self.wf, self.system, self.sampler)
        train.configure_estimators(self.estimators)
        train.configure_loss_grads(self.train.grads)
        train.configure_writers(self.train.writers)
        self.pretrain_stage = pretrain.build()
        self.train_stage = train.build()
        self.data_init = partial(data_init, self.system)


@configurable_dataclass
class ElectronGasEvalWorkflow(ElectronGasWorkflowBase, EvaluationWorkflow):
    """Evaluate a trained homogeneous electron gas wavefunction."""

    estimators: Annotated[
        ElectronGasEstimators,
        WithToggles(energy=True, spin=False, overlap=False),
        HideLoss(),
    ] = field(default_factory=ElectronGasEstimators)

    def build_runtime(self) -> None:
        _configure_system(self.system, self.wf)
        wire(
            self.sampler,
            sampling_proposal=make_pbc_gaussian_proposal(
                jnp.asarray(self.system.lattice)
            ),
        )
        evaluation = EvaluationWorkStage.builder(
            self.run_config, self.wf, name="evaluation"
        )
        evaluation.configure_sample_plan(self.wf.logpsi, {"electrons": self.sampler})
        _wire_estimators(self.estimators, self.wf, self.system, self.sampler)
        evaluation.configure_estimators(self.estimators)
        evaluation.configure_writers(self.writers)
        self.evaluation_stage = evaluation.build()
        self.data_init = partial(data_init, self.system)


def _configure_system(
    system: ElectronGasConfig, wf: ElectronGasWavefunctionLike
) -> FreeElectronReference:
    _validate_wavefunction(wf)
    reference = FreeElectronReference(system.nspins, system.lattice, system.twist)
    lattice = jnp.asarray(system.lattice)
    wf.nspins = system.nspins
    wf.klist = reference.get_orbital_kpoints()
    wf.primitive_lattice = lattice
    wf.simulation_lattice = lattice
    return reference


def _wire_estimators(
    settings: ElectronGasEstimators,
    wf: ElectronGasWavefunctionLike,
    system: ElectronGasConfig,
    sampler: SamplerLike,
) -> None:
    settings.energy.potential.supercell_lattice = jnp.asarray(system.lattice)
    settings.energy.kinetic.f_log_psi = wf.logpsi
    settings.spin.n_up, settings.spin.n_down = system.nspins
    settings.spin.phase_logpsi = wf.phase_logpsi
    for ref in filter(None, settings.overlap.states.values()):
        if ref.wf is not None:
            _configure_system(system, ref.wf)
    settings.overlap.restore(
        wf=wf,
        samplers={"electrons": sampler},
        data_init=partial(data_init, system),
        enabled=settings.is_enabled("overlap"),
    )
