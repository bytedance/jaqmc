# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Molecular workflows."""

from collections.abc import Callable
from dataclasses import field
from functools import partial
from typing import Annotated

import numpy as np
from upath import UPath

from jaqmc.estimator import EstimatorCollection, HideLoss, WithToggles
from jaqmc.estimator.density import CartesianAxis, CartesianDensity
from jaqmc.estimator.ecp import ECPEnergy
from jaqmc.estimator.kinetic import EuclideanKinetic
from jaqmc.estimator.overlap import Overlap, ReferenceState
from jaqmc.estimator.ph import PHEnergy
from jaqmc.estimator.spin import SpinSquared
from jaqmc.estimator.total_energy import TotalEnergy
from jaqmc.sampler import SamplerLike
from jaqmc.sampler.mcmc import MCMCSampler
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

from .config import MoleculeConfig, MoleculeSolverConfig
from .data import data_init
from .hamiltonian import PotentialEnergy
from .reference import MoleculeReference, load_reference
from .reference import pyscf as reference_pyscf
from .wavefunction.base import MoleculeWavefunction
from .wavefunction.ferminet import FermiNetWavefunction


@configurable_dataclass
class MoleculeEnergyEstimators(EstimatorCollection):
    """Energy component estimators for molecular systems.

    Args:
        potential: Electron-electron, electron-nucleus, and nucleus-nucleus
            Coulomb potential energy.
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
class MoleculeReferenceState(ReferenceState):
    """Reference state for molecular overlap estimation."""

    wf: Annotated[
        MoleculeWavefunction | None,
        ModuleConfig(package="jaqmc.app.molecule.wavefunction"),
    ] = None


@configurable_dataclass
class MoleculeEstimators(EstimatorCollection):
    """Observables evaluated by molecular workflows.

    Args:
        energy: Energy component estimators.
        spin: Total spin-squared (:math:`S^2`) estimator.
        overlap: Wavefunction overlap estimator.
        density: 3-D Cartesian electron density histogram. Axes with
            ``range: null`` are auto-padded by 5 bohr around the atomic
            coordinates at runtime.
    """

    energy: MoleculeEnergyEstimators = field(default_factory=MoleculeEnergyEstimators)
    spin: SpinSquared = field(default_factory=SpinSquared)
    overlap: Overlap[MoleculeReferenceState] = field(default_factory=Overlap)
    density: CartesianDensity = field(
        default_factory=lambda: CartesianDensity(
            axes={
                name: CartesianAxis(
                    direction=tuple(
                        1.0 if index == axis else 0.0 for index in range(3)
                    ),
                    bins=50,
                    range=None,
                )
                for name, axis in [("x", 0), ("y", 1), ("z", 2)]
            },
        )
    )


@configurable_dataclass
class MoleculePretrainConfig(VMCPretrainConfig):
    """Configuration for the molecular orbital pretraining stage.

    Args:
        sample_fraction: Weight of the reference in the log amplitude sampled
            during pretraining (0.0 = pure neural ansatz, 1.0 = pure reference).
    """

    sample_fraction: float = 1.0


def _wire_estimators(
    settings: MoleculeEstimators,
    wf: MoleculeWavefunction,
    system: MoleculeConfig,
    sampler: SamplerLike,
) -> None:
    energy = settings.energy
    symbols = [atom.symbol for atom in system.atoms]
    energy.kinetic.f_log_psi = wf.logpsi
    energy.ecp.ecp_coefficients = system.ecp_coefficients
    energy.ecp.atom_symbols = symbols
    energy.ecp.phase_logpsi = wf.phase_logpsi
    energy.ph.f_log_psi = wf.logpsi
    energy.ph.atom_symbols = symbols
    energy.ph.ph = list(system.ph_elements)
    settings.spin.n_up, settings.spin.n_down = system.electron_spins
    settings.spin.phase_logpsi = wf.phase_logpsi
    for ref in filter(None, settings.overlap.states.values()):
        if ref.wf is not None:
            wire(ref.wf, nspins=system.electron_spins)
    settings.overlap.restore(
        wf=wf,
        samplers={"electrons": sampler},
        data_init=partial(data_init, system),
        enabled=settings.is_enabled("overlap"),
    )
    positions = np.array([atom.coords for atom in system.atoms])
    for name, index in [("x", 0), ("y", 1), ("z", 2)]:
        axis = settings.density.axes.get(name)
        if axis is not None and axis.range is None:
            axis.range = (
                float(positions[:, index].min()) - 5.0,
                float(positions[:, index].max()) + 5.0,
            )


@configurable_dataclass
class MoleculeWorkflowBase:
    system: Annotated[
        MoleculeConfig | Callable[[], MoleculeConfig],
        ModuleConfig(package="jaqmc.app.molecule.config"),
    ] = field(default_factory=MoleculeConfig)
    wf: Annotated[
        MoleculeWavefunction,
        ModuleConfig(package="jaqmc.app.molecule.wavefunction"),
    ] = field(default_factory=FermiNetWavefunction)
    sampler: Annotated[SamplerLike, ModuleConfig(package="jaqmc.sampler")] = field(
        default_factory=MCMCSampler
    )
    estimators: Annotated[
        MoleculeEstimators, WithToggles(spin=False, overlap=False, density=False)
    ] = field(default_factory=MoleculeEstimators)

    def __post_init__(self) -> None:
        system = self.system() if callable(self.system) else self.system
        self.estimators.energy.set_enabled(
            ph=bool(system.ph_elements),
            kinetic=not system.ph_elements,
            ecp=bool(system.ecp_coefficients),
        )


@configurable_dataclass
class MoleculeTrainWorkflow(MoleculeWorkflowBase, VMCWorkflow):
    reference: str | None = None
    pretrain: MoleculePretrainConfig = field(
        default_factory=lambda: MoleculePretrainConfig(
            run=VMCWorkStageConfig(iterations=2_000)
        )
    )
    train: VMCTrainConfig = field(
        default_factory=lambda: VMCTrainConfig(
            run=VMCWorkStageConfig(iterations=200_000),
            writers=stage_writers(
                "pmove:.2f,energy=total_energy:.4f,variance=total_energy_var:.4f"
            ),
        )
    )

    def build_runtime(self) -> None:
        system = self.system() if callable(self.system) else self.system
        self.wf.nspins = system.electron_spins
        wire(self.sampler)
        if self.pretrain.run.iterations > 0:
            self.pretrain_stage = self._build_pretrain_stage(
                self._load_reference(system), system
            )
        train = VMCWorkStage.builder(self.train.run, self.wf, name="train")
        train.configure_sample_plan(self.wf.logpsi, {"electrons": self.sampler})
        train.configure_optimizer(self.train.optim, f_log_psi=self.wf.logpsi)
        _wire_estimators(self.estimators, self.wf, system, self.sampler)
        train.configure_estimators(self.estimators)
        train.configure_loss_grads(self.train.grads)
        train.configure_writers(self.train.writers)
        self.train_stage = train.build()
        self.data_init = partial(data_init, system)

    def _load_reference(self, system: MoleculeConfig) -> MoleculeReference:
        """Load ``reference``, or find or generate ``reference.npz`` for the run.

        Returns:
            The reference validated against ``system``.
        """
        if self.reference:
            return load_reference(UPath(self.reference), system)
        reference_path = self.save_path / "reference.npz"
        if (restore_reference_path := self.restore_dir / "reference.npz").exists():
            reference_path = restore_reference_path
        if not reference_path.exists():
            auto_generate_reference(
                path=reference_path,
                job=reference_pyscf.MoleculePySCFReferenceJob(
                    system, MoleculeSolverConfig()
                ),
            )
        return load_reference(reference_path, system)

    def _build_pretrain_stage(
        self, reference: MoleculeReference, system: MoleculeConfig
    ) -> VMCWorkStage:
        nspins = system.electron_spins
        loss = make_pretrain_loss(
            orbitals_fn=self.wf.orbitals,
            orbital_ref=reference,
            nspins=nspins,
            full_det=self.wf.full_det,
        )
        log_amplitude = make_pretrain_log_amplitude(
            self.wf.logpsi,
            lambda data: reference.eval_slater(data.electrons, nspins)[1],
            ref_fraction=self.pretrain.sample_fraction,
        )
        pretrain = VMCWorkStage.builder(self.pretrain.run, self.wf, name="pretrain")
        pretrain.configure_sample_plan(log_amplitude, {"electrons": self.sampler})
        pretrain.configure_optimizer(self.pretrain.optim, f_log_psi=self.wf.logpsi)
        pretrain.configure_estimators({"grads": loss})
        pretrain.configure_writers(self.pretrain.writers)
        return pretrain.build()


@configurable_dataclass
class MoleculeEvalWorkflow(MoleculeWorkflowBase, EvaluationWorkflow):
    estimators: Annotated[
        MoleculeEstimators,
        WithToggles(energy=True, spin=False, overlap=False, density=False),
        HideLoss(),
    ] = field(default_factory=MoleculeEstimators)

    def build_runtime(self) -> None:
        system = self.system() if callable(self.system) else self.system
        self.wf.nspins = system.electron_spins
        wire(self.sampler)
        evaluation = EvaluationWorkStage.builder(
            self.run_config, self.wf, name="evaluation"
        )
        evaluation.configure_sample_plan(self.wf.logpsi, {"electrons": self.sampler})
        _wire_estimators(self.estimators, self.wf, system, self.sampler)
        evaluation.configure_estimators(self.estimators)
        evaluation.configure_writers(self.writers)
        self.evaluation_stage = evaluation.build()
        self.data_init = partial(data_init, system)
