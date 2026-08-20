# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
from jax import numpy as jnp
from upath import UPath

from jaqmc.estimator import EstimatorLike
from jaqmc.estimator.density import FractionalAxis, FractionalDensity
from jaqmc.estimator.ecp import ECPEnergy
from jaqmc.estimator.kinetic import EuclideanKinetic
from jaqmc.estimator.spin import SpinSquared
from jaqmc.estimator.total_energy import TotalEnergy
from jaqmc.geometry.pbc import make_pbc_gaussian_proposal
from jaqmc.optimizer.kfac import KFACOptimizer
from jaqmc.optimizer.optax import adam
from jaqmc.sampler.mcmc import MCMCSampler
from jaqmc.utils.atomic import make_pretrain_loss
from jaqmc.utils.atomic.pretrain import make_pretrain_log_amplitude
from jaqmc.utils.config import ConfigManager, ConfigManagerLike
from jaqmc.utils.reference import auto_generate_reference
from jaqmc.utils.wiring import wire
from jaqmc.wavefunction import Wavefunction
from jaqmc.workflow import EvaluationWorkflow, VMCWorkflow
from jaqmc.workflow.stage import EvaluationWorkStage, VMCWorkStage
from jaqmc.workflow.stage.vmc import VMCStageBuilder

from .config import SolidConfig, SolidPySCFSolverConfig
from .data import data_init
from .hamiltonian import PotentialEnergy
from .reference import SolidReference, load_reference
from .reference import pyscf as reference_pyscf
from .wavefunction import SolidWavefunction

logger = logging.getLogger(__name__)


class SolidTrainWorkflow(VMCWorkflow):
    """VMC training workflow for periodic solid-state systems."""

    @classmethod
    def default_preset(cls) -> dict[str, Any]:
        console_fields = (
            "pmove:.2f,energy=total_energy:.4f,variance=total_energy_real_var:.4f"
        )
        return {
            "pretrain": {
                "run": {"iterations": 2_000},
                "optim": {
                    "learning_rate": {
                        "module": "jaqmc.optimizer.schedule:Constant",
                        "rate": 3e-4,
                    }
                },
            },
            "train": {
                "run": {"iterations": 200_000},
                "writers": {"console": {"fields": console_fields}},
            },
        }

    def __init__(self, cfg: ConfigManager) -> None:
        super().__init__(cfg)
        system_config, wf, sampling_proposal = configure_system(cfg)

        self.wf = wf
        self.reference: SolidReference | None = None
        self._system_config = system_config
        if reference_path := cfg.get("reference", ""):
            self.reference = load_reference(UPath(reference_path), self._system_config)
        self.data_init = partial(data_init, system_config)

        sampler = cfg.get("sampler", MCMCSampler(sampling_proposal=sampling_proposal))

        self._pretrain_builder: VMCStageBuilder | None = None
        if cfg.get("pretrain.run.iterations", 2_000) > 0:
            self._pretrain_sample_fraction = cfg.get("pretrain.sample_fraction", 1.0)
            pretrain = VMCWorkStage.builder(cfg.scoped("pretrain"), wf, name="pretrain")
            pretrain.configure_optimizer(default=adam, f_log_psi=wf.logpsi)
            pretrain.configure_writers()
            self._pretrain_builder = pretrain

        train = VMCWorkStage.builder(cfg.scoped("train"), wf)
        train.configure_sample_plan(wf.logpsi, {"electrons": sampler})
        train.configure_optimizer(default=KFACOptimizer, f_log_psi=wf.logpsi)
        estimators = make_estimators(cfg, wf, system_config, always_enable_energy=True)
        train.configure_estimators(**estimators)
        train.configure_loss_grads(f_log_psi=wf.logpsi)
        self.train_stage = train.build()

    def prepare(self, dry_run: bool = False) -> None:
        super().prepare(dry_run)
        if dry_run and self.reference is None:
            return

        if self.reference is None:
            reference_path = self.save_path / "reference.npz"
            if (restore_reference_path := self.restore_dir / "reference.npz").exists():
                reference_path = restore_reference_path
            if not reference_path.exists():
                auto_generate_reference(
                    path=reference_path,
                    job=reference_pyscf.SolidPySCFReferenceJob(
                        self._system_config, SolidPySCFSolverConfig()
                    ),
                )
            self.reference = load_reference(reference_path, self._system_config)
        reference = self.reference
        _assign_klist(self.wf, reference)

        if self._pretrain_builder is None:
            return
        nspins = (
            self._system_config.electron_spins[0] * self._system_config.scale,
            self._system_config.electron_spins[1] * self._system_config.scale,
        )
        loss_estimator = make_pretrain_loss(
            orbitals_fn=self.wf.orbitals,
            orbital_ref=reference,
            nspins=nspins,
            full_det=self.wf.full_det,
        )
        f_log_amplitude = make_pretrain_log_amplitude(
            self.wf.logpsi,
            lambda data: reference.eval_slater(data.electrons, nspins).real,
            ref_fraction=self._pretrain_sample_fraction,
        )
        self._pretrain_builder.configure_sample_plan(
            f_log_amplitude,
            {"electrons": self.train_stage.sample_plan.samplers["electrons",]},
        )
        self._pretrain_builder.configure_estimators(grads=loss_estimator)
        self.pretrain_stage = self._pretrain_builder.build()


class SolidEvalWorkflow(EvaluationWorkflow):
    """Evaluation workflow for periodic solid-state systems."""

    def __init__(self, cfg: ConfigManager) -> None:
        super().__init__(cfg)
        system_config, wf, sampling_proposal = configure_system(cfg)

        self.wf = wf
        self._system_config = system_config
        self.reference: SolidReference | None = None
        if reference_path := cfg.get("reference", ""):
            self.reference = load_reference(UPath(reference_path), self._system_config)
        self.data_init = partial(data_init, system_config)

        evaluation = EvaluationWorkStage.builder(cfg, wf, name="evaluation")
        sampler = cfg.get("sampler", MCMCSampler(sampling_proposal=sampling_proposal))
        evaluation.configure_sample_plan(wf.logpsi, {"electrons": sampler})

        eval_estimators = make_estimators(cfg, wf, system_config)
        evaluation.configure_estimators(**eval_estimators)
        self.evaluation_stage = evaluation.build()

    def prepare(self, dry_run: bool = False) -> None:
        super().prepare(dry_run)
        if dry_run and self.reference is None:
            return
        if self.reference is None:
            filename = "reference.npz"
            reference_path = self.save_path / filename
            if (restore_reference_path := self.restore_dir / filename).exists():
                reference_path = restore_reference_path
            elif (
                self.source_dir
                and (source_reference_path := self.source_dir / filename).exists()
            ):
                reference_path = source_reference_path
            if not reference_path.exists():
                auto_generate_reference(
                    path=reference_path,
                    job=reference_pyscf.SolidPySCFReferenceJob(
                        self._system_config, SolidPySCFSolverConfig()
                    ),
                )
            self.reference = load_reference(reference_path, self._system_config)
        _assign_klist(self.wf, self.reference)


def configure_system(
    cfg: ConfigManagerLike,
) -> tuple[SolidConfig, SolidWavefunction, Callable]:
    """Build the shared system objects for solid workflows.

    Returns:
        Tuple of (system_config, wavefunction, sampling_proposal).

    Raises:
        TypeError: If the wavefunction does not implement SolidWavefunction.
        ValueError: PH is used
    """
    system_config: SolidConfig | Callable[[], SolidConfig] = cfg.get_module(
        "system", SolidConfig
    )
    if callable(system_config):
        system_config = system_config()
    if system_config.ph_elements:
        raise ValueError(
            "solid workflows do not support PH pseudopotentials; "
            "system.pp may only select ECP or all-electron treatment."
        )

    nspins = (
        system_config.electron_spins[0] * system_config.scale,
        system_config.electron_spins[1] * system_config.scale,
    )

    supercell_lattice = jnp.asarray(system_config.supercell_lattice)
    lattice_vectors = system_config.lattice_vectors

    wf = cfg.get_module("wf", "jaqmc.app.solid.wavefunction")
    wf.nspins = nspins
    wf.primitive_lattice = lattice_vectors
    wf.simulation_lattice = supercell_lattice

    sampling_proposal = make_pbc_gaussian_proposal(supercell_lattice)

    if not isinstance(wf, Wavefunction) or not isinstance(wf, SolidWavefunction):
        raise TypeError(
            f"Wavefunction must implement SolidWavefunction, got {type(wf).__name__}"
        )
    return system_config, wf, sampling_proposal


def _assign_klist(wf: SolidWavefunction, reference: SolidReference) -> None:
    wf.klist = reference.get_orbital_kpoints()
    logger.info(
        "Using following k-points:\n%s",
        "\n".join(
            f"{alpha=}, {beta=}: {np.asarray(kpt).tolist()}"
            for kpt, alpha, beta in reference.get_kpoint_occupancies()
        ),
    )


def make_estimators(
    cfg: ConfigManagerLike,
    wf: SolidWavefunction,
    system_config: SolidConfig,
    always_enable_energy: bool = False,
) -> dict[str, EstimatorLike]:
    estimators: dict[str, EstimatorLike] = {}
    if always_enable_energy or cfg.get("estimators.enabled.energy", True):
        supercell_lattice = jnp.asarray(system_config.supercell_lattice)
        twist = jnp.array(system_config.twist)

        estimators["potential"] = PotentialEnergy(supercell_lattice=supercell_lattice)
        estimators["kinetic"] = cfg.get(
            "estimators.energy.kinetic", EuclideanKinetic(f_log_psi=wf.logpsi)
        )
        if ecp_coefficients := system_config.ecp_coefficients:
            logger.info("ECP enabled for elements: %s", list(ecp_coefficients.keys()))
            estimators["ecp"] = cfg.get(
                "estimators.energy.ecp",
                ECPEnergy(
                    ecp_coefficients=ecp_coefficients,
                    atom_symbols=[atom.symbol for atom in system_config.atoms]
                    * system_config.scale,
                    phase_logpsi=wf.phase_logpsi,
                    lattice=supercell_lattice,
                    twist=twist,
                ),
            )
        estimators["total"] = TotalEnergy()
    if cfg.get("estimators.enabled.spin", False):
        estimators["spin"] = cfg.get(
            "estimators.spin",
            SpinSquared(
                n_up=system_config.electron_spins[0] * system_config.scale,
                n_down=system_config.electron_spins[1] * system_config.scale,
                phase_logpsi=wf.phase_logpsi,
            ),
        )
    if cfg.get("estimators.enabled.density", False):
        supercell_lattice = jnp.asarray(system_config.supercell_lattice)
        inv_lattice = jnp.linalg.inv(supercell_lattice)
        density = cfg.get(
            "estimators.density",
            FractionalDensity(
                axes={
                    "a": FractionalAxis(lattice_index=0, bins=50),
                    "b": FractionalAxis(lattice_index=1, bins=50),
                    "c": FractionalAxis(lattice_index=2, bins=50),
                }
            ),
        )
        wire(density, inv_lattice=inv_lattice)
        estimators["density"] = density
    return estimators
