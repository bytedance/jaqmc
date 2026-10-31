# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Moire workflows."""

from dataclasses import field
from functools import partial
from typing import Annotated, Protocol, runtime_checkable

import jax
from jax import numpy as jnp

from jaqmc.estimator import EstimatorCollection, HideLoss, WithToggles
from jaqmc.estimator.kinetic import EuclideanKinetic
from jaqmc.estimator.loss_grad import LossGrad
from jaqmc.estimator.overlap import Overlap, ReferenceState
from jaqmc.estimator.total_energy import TotalEnergy
from jaqmc.geometry.pbc import wrap_positions
from jaqmc.optimizer.kfac import KFACOptimizer
from jaqmc.optimizer.schedule import Standard
from jaqmc.sampler import SamplerLike
from jaqmc.sampler.mcmc import MCMCSampler, SamplingProposal
from jaqmc.utils.config import ModuleConfig, configurable_dataclass
from jaqmc.utils.wiring import runtime_dep
from jaqmc.workflow.evaluation import EvaluationWorkflow
from jaqmc.workflow.stage.evaluation import EvaluationWorkStage
from jaqmc.workflow.stage.vmc import VMCWorkStage, VMCWorkStageConfig
from jaqmc.workflow.vmc import VMCTrainConfig, VMCWorkflow, stage_writers

from .config import MoireConfig
from .data import data_init
from .hamiltonian import CoulombInteractionEnergy, MoirePotential, MoireSOC
from .wavefunction import MoireWavefunction, MoireWavefunctionLike


@runtime_checkable
class MoireSamplerLike(SamplerLike, Protocol):
    """Runtime interface required by moire workflows."""

    def configure(self, lattice: jnp.ndarray) -> None: ...


@configurable_dataclass
class MoireJointMCMCSampler(MCMCSampler):
    """Joint Metropolis-Hastings sampler for spatial and layer-pseudospin coordinates.

    Args:
        spin_mass: Effective mass scaling the proposal width of the continuous
            spin angle relative to the spatial proposal width.
    """

    steps: int = 20
    initial_width: float = 0.02
    adapt_frequency: int = 50
    spin_mass: float = 1.0
    sampling_proposal: SamplingProposal = runtime_dep()

    def configure(self, lattice: jnp.ndarray) -> None:
        spin_mass = self.spin_mass

        def proposal(rngs, state, stddev):
            positions = state["positions"]
            spin_coords = state["spin_coords"]
            spatial_key, spin_key = jax.random.split(rngs)
            positions = wrap_positions(
                positions
                + stddev * jax.random.normal(spatial_key, positions.shape) @ lattice,
                lattice,
            )
            spin_coords = (
                spin_coords
                + 2
                * jnp.pi
                * stddev
                / spin_mass
                * jax.random.normal(spin_key, spin_coords.shape)
            ) % (2 * jnp.pi)
            return {"positions": positions, "spin_coords": spin_coords}

        self.sampling_proposal = proposal


@configurable_dataclass
class MoireEnergyEstimators(EstimatorCollection):
    """Energy component estimators for moire superlattice systems.

    Args:
        kinetic: Common Euclidean kinetic term.
        coulomb: 2-D Ewald electron-electron Coulomb interaction.
        moire_potential: Bilayer layer-pseudospin moire potential.
        soc: Valley-momentum kinetic-shift correction.
        total: Sums enabled ``energy:``-prefixed components into
            ``total_energy``.
    """

    kinetic: EuclideanKinetic = field(default_factory=EuclideanKinetic)
    coulomb: CoulombInteractionEnergy = field(default_factory=CoulombInteractionEnergy)
    moire_potential: MoirePotential = field(default_factory=MoirePotential)
    soc: MoireSOC = field(default_factory=MoireSOC)
    total: TotalEnergy = field(default_factory=TotalEnergy)


@configurable_dataclass
class MoireReferenceState(ReferenceState):
    """Reference state for moire overlap estimation."""

    wf: Annotated[
        MoireWavefunctionLike | None,
        ModuleConfig(package="jaqmc.app.moire.wavefunction"),
    ] = None


@configurable_dataclass
class MoireEstimators(EstimatorCollection):
    """Observables evaluated by moire workflows.

    Args:
        energy: Energy component estimators.
        overlap: Wavefunction overlap estimator.
    """

    energy: Annotated[
        MoireEnergyEstimators,
        WithToggles(kinetic=True, coulomb=True, moire_potential=True, soc=True),
    ] = field(default_factory=MoireEnergyEstimators)
    overlap: Overlap[MoireReferenceState] = field(default_factory=Overlap)


def _wire_wf(wf: MoireWavefunctionLike, system: MoireConfig) -> None:
    wf.nspins = system.electron_spins
    wf.primitive_lattice = jnp.asarray(system.lattice_vectors)
    wf.simulation_lattice = jnp.asarray(system.supercell_lattice)
    wf.twist = jnp.asarray(system.twist)
    wf.configure_phase_inputs(system)


def _configure(
    wf: MoireWavefunctionLike, sampler: MoireSamplerLike, system: MoireConfig
) -> None:
    _wire_wf(wf, system)
    sampler.configure(jnp.asarray(system.supercell_lattice))


def _wire_estimators(
    settings: MoireEstimators,
    wf: MoireWavefunctionLike,
    system: MoireConfig,
    sampler: MoireSamplerLike,
) -> None:
    energy = settings.energy
    primitive = jnp.asarray(system.moire_lattice_vectors)
    prefactor = system.kinetic_prefactor_mev / system.effective_mass
    energy.kinetic.f_log_psi = wf.logpsi
    energy.kinetic.data_field = "positions"
    energy.kinetic.prefactor = prefactor
    energy.coulomb.supercell_lattice = jnp.asarray(system.supercell_lattice)
    energy.coulomb.coulomb_prefactor_mev = system.coulomb_prefactor_mev
    energy.moire_potential.f_layer_components = wf.layer_components
    energy.moire_potential.primitive_lattice = primitive
    energy.moire_potential.nspins = system.electron_spins
    energy.moire_potential.v1_mev = system.v1_mev
    energy.moire_potential.phi1_rad = system.phi1_rad
    energy.moire_potential.omega1_mev = system.omega1_mev
    energy.soc.f_layer_components = wf.layer_components
    energy.soc.primitive_lattice = primitive
    energy.soc.nspins = system.electron_spins
    energy.soc.prefactor = prefactor
    for ref in filter(None, settings.overlap.states.values()):
        if ref.wf is not None:
            _wire_wf(ref.wf, system)
    settings.overlap.restore(
        wf=wf,
        samplers={("positions", "spin_coords"): sampler},
        data_init=partial(data_init, system),
        enabled=settings.is_enabled("overlap"),
    )


@configurable_dataclass
class MoireWorkflowBase:
    system: MoireConfig = field(default_factory=MoireConfig)
    wf: Annotated[
        MoireWavefunctionLike,
        ModuleConfig(package="jaqmc.app.moire.wavefunction"),
    ] = field(default_factory=MoireWavefunction)
    sampler: Annotated[MoireSamplerLike, ModuleConfig(package="jaqmc.app.moire")] = (
        field(default_factory=MoireJointMCMCSampler)
    )
    estimators: Annotated[MoireEstimators, WithToggles(overlap=False)] = field(
        default_factory=MoireEstimators
    )

    def __post_init__(self) -> None:
        if self.system.is_integer_filling and self.wf.k_com is not None:
            raise ValueError("k_com must be None when phase_mode='singlephase'")
        if self.wf.nlayers != 2:
            if self.estimators.is_enabled("moire_potential"):
                raise ValueError("MoirePotential supports exactly two layers.")
            if self.estimators.is_enabled("soc"):
                raise ValueError("MoireSOC supports exactly two layers.")


@configurable_dataclass
class MoireTrainWorkflow(MoireWorkflowBase, VMCWorkflow):
    train: VMCTrainConfig = field(
        default_factory=lambda: VMCTrainConfig(
            run=VMCWorkStageConfig(iterations=200_000),
            optim=KFACOptimizer(
                damping=3e-4,
                learning_rate=Standard(rate=0.003, delay=10000),
            ),
            grads=LossGrad(clip_scale=20.0),
            writers=stage_writers(
                "pmove:.2f,energy=total_energy_real:.4f,"
                "variance=total_energy_real_var:.4f"
            ),
        )
    )

    def build_runtime(self) -> None:
        _configure(self.wf, self.sampler, self.system)
        train = VMCWorkStage.builder(self.train.run, self.wf, name="train")
        train.configure_sample_plan(
            self.wf.logpsi, {("positions", "spin_coords"): self.sampler}
        )
        train.configure_optimizer(self.train.optim, f_log_psi=self.wf.logpsi)
        _wire_estimators(self.estimators, self.wf, self.system, self.sampler)
        train.configure_estimators(self.estimators)
        train.configure_loss_grads(self.train.grads)
        train.configure_writers(self.train.writers)
        self.train_stage = train.build()
        self.data_init = partial(data_init, self.system)


@configurable_dataclass
class MoireEvalWorkflow(MoireWorkflowBase, EvaluationWorkflow):
    estimators: Annotated[
        MoireEstimators, WithToggles(energy=True, overlap=False), HideLoss()
    ] = field(default_factory=MoireEstimators)

    def build_runtime(self) -> None:
        _configure(self.wf, self.sampler, self.system)
        evaluation = EvaluationWorkStage.builder(
            self.run_config, self.wf, name="evaluation"
        )
        evaluation.configure_sample_plan(
            self.wf.logpsi, {("positions", "spin_coords"): self.sampler}
        )
        _wire_estimators(self.estimators, self.wf, self.system, self.sampler)
        evaluation.configure_estimators(self.estimators)
        evaluation.configure_writers(self.writers)
        self.evaluation_stage = evaluation.build()
        self.data_init = partial(data_init, self.system)
