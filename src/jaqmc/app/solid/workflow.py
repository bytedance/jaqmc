# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Periodic solid workflows."""

import logging
from collections.abc import Callable, Sequence
from dataclasses import field
from functools import partial
from typing import Annotated

import numpy as np
from jax import numpy as jnp
from upath import UPath

from jaqmc.estimator import EstimatorCollection, HideLoss, WithToggles
from jaqmc.estimator.density import FractionalAxis, FractionalDensity
from jaqmc.estimator.ecp import ECPEnergy
from jaqmc.estimator.kinetic import EuclideanKinetic
from jaqmc.estimator.overlap import Overlap, ReferenceState
from jaqmc.estimator.ph import PHEnergy
from jaqmc.estimator.spin import SpinSquared
from jaqmc.estimator.total_energy import TotalEnergy
from jaqmc.geometry.pbc import build_distance_fn, make_pbc_gaussian_proposal
from jaqmc.sampler import SamplerLike
from jaqmc.sampler.mcmc import MCMCSampler, SamplingProposal
from jaqmc.utils.atomic.pretrain import make_pretrain_log_amplitude, make_pretrain_loss
from jaqmc.utils.config import ModuleConfig, configurable_dataclass
from jaqmc.utils.reference import auto_generate_reference
from jaqmc.utils.wiring import wire
from jaqmc.workflow.evaluation import EvaluationWorkflow
from jaqmc.workflow.stage.evaluation import EvaluationWorkStage
from jaqmc.workflow.stage.vmc import VMCWorkStage, VMCWorkStageConfig
from jaqmc.workflow.vmc import (
    VMCPretrainConfig,
    VMCTrainConfig,
    VMCWorkflow,
    stage_writers,
)

from .config import SolidConfig, SolidPySCFSolverConfig
from .data import data_init
from .hamiltonian import PotentialEnergy
from .reference import SolidReference, load_reference
from .reference import pyscf as reference_pyscf
from .wavefunction import SolidWavefunction, SolidWavefunctionLike

logger = logging.getLogger(__name__)


@configurable_dataclass
class SolidEnergyEstimators(EstimatorCollection):
    """Energy component estimators for periodic solid systems.

    Args:
        potential: Periodic Coulomb potential energy via Ewald summation.
        kinetic: Euclidean kinetic energy. Disabled when ``system.pp`` selects
            PH for any atom because :class:`~jaqmc.estimator.ph.PHEnergy`
            computes the kinetic term, and enabled otherwise.
        ecp: Semi-local effective core potential energy. Enabled when
            ``system.pp`` selects an ECP for at least one atom.
        ph: Local Pseudo-Hamiltonian energy correction. Enabled when
            ``system.pp`` selects ``"ph"`` for at least one atom.
        total: Sums enabled ``energy:``-prefixed components into
            ``total_energy``.
    """

    potential: PotentialEnergy = field(default_factory=PotentialEnergy)
    kinetic: EuclideanKinetic = field(default_factory=EuclideanKinetic)
    ecp: ECPEnergy = field(default_factory=ECPEnergy)
    ph: PHEnergy = field(default_factory=PHEnergy)
    total: TotalEnergy = field(default_factory=TotalEnergy)


@configurable_dataclass
class SolidReferenceState(ReferenceState):
    """Reference state for periodic solid overlap estimation."""

    wf: Annotated[
        SolidWavefunctionLike | None,
        ModuleConfig(package="jaqmc.app.solid.wavefunction"),
    ] = None


@configurable_dataclass
class SolidEstimators(EstimatorCollection):
    """Observables evaluated by periodic solid workflows.

    Args:
        energy: Energy component estimators.
        spin: Total spin-squared (:math:`S^2`) estimator.
        overlap: Wavefunction overlap estimator.
        density: Electron density histogram in fractional supercell
            coordinates over :math:`[0, 1)`.
    """

    energy: SolidEnergyEstimators = field(default_factory=SolidEnergyEstimators)
    spin: SpinSquared = field(default_factory=SpinSquared)
    overlap: Overlap[SolidReferenceState] = field(default_factory=Overlap)
    density: FractionalDensity = field(
        default_factory=lambda: FractionalDensity(
            axes={
                name: FractionalAxis(lattice_index=index, bins=50)
                for name, index in [("a", 0), ("b", 1), ("c", 2)]
            },
        )
    )


@configurable_dataclass
class SolidPretrainConfig(VMCPretrainConfig):
    """Configuration for the solid orbital pretraining stage.

    Args:
        sample_fraction: Weight of the reference in the log amplitude sampled
            during pretraining (0.0 = pure neural ansatz, 1.0 = pure reference).
    """

    sample_fraction: float = 1.0


def _supercell_spins(system: SolidConfig) -> tuple[int, int]:
    n_up, n_down = system.electron_spins
    return n_up * system.scale, n_down * system.scale


def _configure_system(
    system: SolidConfig, wf: SolidWavefunctionLike
) -> SamplingProposal:
    wf.nspins = _supercell_spins(system)
    wf.primitive_lattice = jnp.asarray(system.lattice_vectors)
    wf.simulation_lattice = jnp.asarray(system.supercell_lattice)
    return make_pbc_gaussian_proposal(jnp.asarray(system.supercell_lattice))


def _load_reference(
    reference: str | None,
    system: SolidConfig,
    save_path: UPath,
    search_dirs: Sequence[UPath | None],
) -> SolidReference:
    """Load ``reference``, or find or generate ``reference.npz`` for the run.

    Without an explicit path, the first ``search_dirs`` entry holding
    ``reference.npz`` wins, then ``save_path``; if none exists, a default PySCF
    reference is generated under ``save_path``.

    Returns:
        The reference validated against ``system``.
    """
    if reference:
        return load_reference(UPath(reference), system)
    filename = "reference.npz"
    reference_path = save_path / filename
    for directory in search_dirs:
        if directory is not None and (candidate := directory / filename).exists():
            reference_path = candidate
            break
    if not reference_path.exists():
        auto_generate_reference(
            path=reference_path,
            job=reference_pyscf.SolidPySCFReferenceJob(
                system, SolidPySCFSolverConfig()
            ),
        )
    return load_reference(reference_path, system)


def _assign_klist(wf: SolidWavefunctionLike, reference: SolidReference) -> None:
    wf.klist = reference.get_orbital_kpoints()
    logger.info(
        "Using following k-points:\n%s",
        "\n".join(
            f"{alpha=}, {beta=}: {np.asarray(kpt).tolist()}"
            for kpt, alpha, beta in reference.get_kpoint_occupancies()
        ),
    )


def _wire_estimators(
    settings: SolidEstimators,
    wf: SolidWavefunctionLike,
    system: SolidConfig,
    sampler: SamplerLike,
) -> None:
    lattice = jnp.asarray(system.supercell_lattice)
    symbols = [atom.symbol for atom in system.atoms] * system.scale
    energy = settings.energy
    energy.potential.supercell_lattice = lattice
    energy.kinetic.f_log_psi = wf.logpsi
    energy.ecp.ecp_coefficients = system.ecp_coefficients
    energy.ecp.atom_symbols = symbols
    energy.ecp.phase_logpsi = wf.phase_logpsi
    energy.ecp.lattice = lattice
    energy.ecp.twist = jnp.asarray(system.twist)
    energy.ph.f_log_psi = wf.logpsi
    energy.ph.atom_symbols = symbols
    energy.ph.ph = list(system.ph_elements)
    energy.ph.distance_fn = build_distance_fn(lattice)
    settings.spin.n_up, settings.spin.n_down = _supercell_spins(system)
    settings.spin.phase_logpsi = wf.phase_logpsi
    for ref in filter(None, settings.overlap.states.values()):
        if ref.wf is not None:
            _configure_system(system, ref.wf)
            ref.wf.klist = wf.klist
    settings.overlap.restore(
        wf=wf,
        samplers={"electrons": sampler},
        data_init=partial(data_init, system),
        enabled=settings.is_enabled("overlap"),
    )
    settings.density.inv_lattice = jnp.linalg.inv(lattice)


@configurable_dataclass
class SolidWorkflowBase:
    system: Annotated[
        SolidConfig | Callable[[], SolidConfig],
        ModuleConfig(package="jaqmc.app.solid.config"),
    ] = field(default_factory=SolidConfig)
    wf: Annotated[
        SolidWavefunctionLike,
        ModuleConfig(package="jaqmc.app.solid.wavefunction"),
    ] = field(default_factory=SolidWavefunction)
    sampler: Annotated[SamplerLike, ModuleConfig(package="jaqmc.sampler")] = field(
        default_factory=MCMCSampler
    )
    estimators: Annotated[
        SolidEstimators, WithToggles(spin=False, overlap=False, density=False)
    ] = field(default_factory=SolidEstimators)
    reference: str | None = None

    def __post_init__(self) -> None:
        system = self.system() if callable(self.system) else self.system
        self.estimators.energy.set_enabled(
            ph=bool(system.ph_elements),
            kinetic=not system.ph_elements,
            ecp=bool(system.ecp_coefficients),
        )


@configurable_dataclass
class SolidTrainWorkflow(SolidWorkflowBase, VMCWorkflow):
    pretrain: SolidPretrainConfig = field(
        default_factory=lambda: SolidPretrainConfig(
            run=VMCWorkStageConfig(iterations=2_000)
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
        system = self.system() if callable(self.system) else self.system
        proposal = _configure_system(system, self.wf)
        wire(self.sampler, sampling_proposal=proposal)
        reference = _load_reference(
            self.reference, system, self.save_path, [self.restore_dir]
        )
        _assign_klist(self.wf, reference)
        if self.pretrain.run.iterations > 0:
            self.pretrain_stage = self._build_pretrain_stage(reference, system)
        train = VMCWorkStage.builder(self.train.run, self.wf, name="train")
        train.configure_sample_plan(self.wf.logpsi, {"electrons": self.sampler})
        train.configure_optimizer(self.train.optim, f_log_psi=self.wf.logpsi)
        _wire_estimators(self.estimators, self.wf, system, self.sampler)
        train.configure_estimators(self.estimators)
        train.configure_loss_grads(self.train.grads)
        train.configure_writers(self.train.writers)
        self.train_stage = train.build()
        self.data_init = partial(data_init, system)

    def _build_pretrain_stage(
        self, reference: SolidReference, system: SolidConfig
    ) -> VMCWorkStage:
        nspins = _supercell_spins(system)
        loss = make_pretrain_loss(
            orbitals_fn=self.wf.orbitals,
            orbital_ref=reference,
            nspins=nspins,
            full_det=self.wf.full_det,
        )
        log_amplitude = make_pretrain_log_amplitude(
            self.wf.logpsi,
            lambda data: reference.eval_slater(data.electrons, nspins).real,
            ref_fraction=self.pretrain.sample_fraction,
        )
        pretrain = VMCWorkStage.builder(self.pretrain.run, self.wf, name="pretrain")
        pretrain.configure_sample_plan(log_amplitude, {"electrons": self.sampler})
        pretrain.configure_optimizer(self.pretrain.optim, f_log_psi=self.wf.logpsi)
        pretrain.configure_estimators({"grads": loss})
        pretrain.configure_writers(self.pretrain.writers)
        return pretrain.build()


@configurable_dataclass
class SolidEvalWorkflow(SolidWorkflowBase, EvaluationWorkflow):
    estimators: Annotated[
        SolidEstimators,
        WithToggles(energy=True, spin=False, overlap=False, density=False),
        HideLoss(),
    ] = field(default_factory=SolidEstimators)

    def build_runtime(self) -> None:
        system = self.system() if callable(self.system) else self.system
        proposal = _configure_system(system, self.wf)
        wire(self.sampler, sampling_proposal=proposal)
        reference = _load_reference(
            self.reference,
            system,
            self.save_path,
            [self.restore_dir, self.source_dir],
        )
        _assign_klist(self.wf, reference)
        evaluation = EvaluationWorkStage.builder(
            self.run_config, self.wf, name="evaluation"
        )
        evaluation.configure_sample_plan(self.wf.logpsi, {"electrons": self.sampler})
        _wire_estimators(self.estimators, self.wf, system, self.sampler)
        evaluation.configure_estimators(self.estimators)
        evaluation.configure_writers(self.writers)
        self.evaluation_stage = evaluation.build()
        self.data_init = partial(data_init, system)
