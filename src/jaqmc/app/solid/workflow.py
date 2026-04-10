# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
from jax import numpy as jnp

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
from jaqmc.utils.atomic import PeriodicSCF, get_core_electrons
from jaqmc.utils.atomic.pretrain import make_pretrain_log_amplitude, make_pretrain_loss
from jaqmc.utils.config import ConfigManager, ConfigManagerLike
from jaqmc.utils.supercell import get_reciprocal_vectors, get_supercell_kpts
from jaqmc.utils.wiring import wire
from jaqmc.wavefunction import Wavefunction
from jaqmc.workflow.evaluation import EvaluationWorkflow
from jaqmc.workflow.stage.evaluation import EvaluationWorkStage
from jaqmc.workflow.stage.vmc import VMCWorkStage
from jaqmc.workflow.vmc import VMCWorkflow

from .config import SolidConfig
from .data import data_init
from .hamiltonian import PotentialEnergy
from .wavefunction import SolidWavefunction

logger = logging.getLogger(__name__)


class SolidTrainWorkflow(VMCWorkflow):
    """VMC training workflow for periodic solid-state systems."""

    @classmethod
    def default_preset(cls) -> dict[str, Any]:
        console_fields = (
            "pmove:.2f,energy=total_energy:.4f,variance=total_energy_var:.4f"
        )
        return {
            "pretrain": {
                "run": {"iterations": 2_000},
                "optim": {"learning_rate": {"rate": 3e-4}},
            },
            "train": {
                "run": {"iterations": 200_000},
                "writers": {"console": {"fields": console_fields}},
            },
        }

    def __init__(self, cfg: ConfigManager) -> None:
        super().__init__(cfg)
        system_config, wf, scf, sampling_proposal = configure_system(cfg)

        nspins = (
            system_config.electron_spins[0] * system_config.scale,
            system_config.electron_spins[1] * system_config.scale,
        )
        self.scf = scf
        self.wf = wf
        self.data_init = partial(data_init, system_config)

        loss_estimator = make_pretrain_loss(
            orbitals_fn=wf.orbitals, scf=scf, nspins=nspins, full_det=wf.full_det
        )
        f_log_amplitude = make_pretrain_log_amplitude(
            wf.logpsi, lambda data: scf.eval_slater(data.electrons, nspins).real
        )
        sampler = cfg.get("sampler", MCMCSampler(sampling_proposal=sampling_proposal))

        pretrain = VMCWorkStage.builder(cfg.scoped("pretrain"), wf, name="pretrain")
        pretrain.configure_sample_plan(f_log_amplitude, {"electrons": sampler})
        pretrain.configure_optimizer(default=adam, f_log_psi=wf.logpsi)
        pretrain.configure_estimators(grads=loss_estimator)
        self.pretrain_stage = pretrain.build()

        train = VMCWorkStage.builder(cfg.scoped("train"), wf)
        train.configure_sample_plan(wf.logpsi, {"electrons": sampler})
        train.configure_optimizer(default=KFACOptimizer, f_log_psi=wf.logpsi)
        estimators = make_estimators(
            cfg, wf, scf, system_config, always_enable_energy=True
        )
        train.configure_estimators(**estimators)
        train.configure_loss_grads(f_log_psi=wf.logpsi)
        self.train_stage = train.build()

    def run(self) -> None:
        self.scf.run()
        self.wf.klist = self.scf.get_orbital_kpoints()
        super().run()


class SolidEvalWorkflow(EvaluationWorkflow):
    """Evaluation workflow for periodic solid-state systems."""

    def __init__(self, cfg: ConfigManager) -> None:
        super().__init__(cfg)
        system_config, wf, scf, sampling_proposal = configure_system(cfg)

        self.scf = scf
        self.wf = wf
        self.data_init = partial(data_init, system_config)

        evaluation = EvaluationWorkStage.builder(cfg, wf, name="evaluation")
        sampler = cfg.get("sampler", MCMCSampler(sampling_proposal=sampling_proposal))
        evaluation.configure_sample_plan(wf.logpsi, {"electrons": sampler})
        eval_estimators: dict[str, EstimatorLike] = make_estimators(
            cfg, wf, scf, system_config
        )
        evaluation.configure_estimators(**eval_estimators)
        self.evaluation_stage = evaluation.build()

    def run(self) -> None:
        self.scf.run()
        self.wf.klist = self.scf.get_orbital_kpoints()
        super().run()


def configure_system(
    cfg: ConfigManagerLike,
) -> tuple[SolidConfig, SolidWavefunction, PeriodicSCF, Callable]:
    """Build the shared system objects for solid workflows.

    Returns:
        Tuple of (system_config, wavefunction, scf, sampling_proposal).

    Raises:
        TypeError: If the wavefunction does not implement SolidWavefunction.
    """
    system_config: SolidConfig | Callable[[], SolidConfig] = cfg.get_module(
        "system", "jaqmc.app.solid.config.base"
    )
    if callable(system_config):
        system_config = system_config()

    nspins = (
        system_config.electron_spins[0] * system_config.scale,
        system_config.electron_spins[1] * system_config.scale,
    )

    supercell_lattice = jnp.asarray(system_config.supercell_lattice)
    lattice_vectors = jnp.asarray(system_config.lattice_vectors)

    wf = cfg.get_module("wf", "jaqmc.app.solid.wavefunction")
    wf.nspins = nspins
    wf.primitive_lattice = lattice_vectors
    wf.simulation_lattice = supercell_lattice

    # Compute k-points for PeriodicSCF
    S = jnp.array(system_config.supercell_matrix)
    prim_rec_vecs = get_reciprocal_vectors(lattice_vectors)
    kpts_folding = get_supercell_kpts(S, prim_rec_vecs)
    twist = jnp.array(system_config.twist)
    sim_rec_vecs = get_reciprocal_vectors(supercell_lattice)
    k_twist = jnp.dot(twist, sim_rec_vecs)
    kpts = kpts_folding + k_twist[None, :]

    core_electrons = get_core_electrons(system_config.atoms, system_config.ecp)
    scf = PeriodicSCF(
        atoms=system_config.atoms,
        nelectrons=system_config.electron_spins,
        lattice_vectors=np.asarray(lattice_vectors),
        kpts=np.asarray(kpts),
        basis=system_config.basis,
        ecp=system_config.ecp,
        core_electrons=core_electrons,
    )

    sampling_proposal = make_pbc_gaussian_proposal(supercell_lattice)

    if not isinstance(wf, Wavefunction) or not isinstance(wf, SolidWavefunction):
        raise TypeError(
            f"Wavefunction must implement SolidWavefunction, got {type(wf).__name__}"
        )
    return system_config, wf, scf, sampling_proposal


def make_estimators(
    cfg: ConfigManagerLike,
    wf: SolidWavefunction,
    scf: PeriodicSCF,
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
        if system_config.ecp is not None:
            logger.info("ECP enabled for elements: %s", list(scf._cell._ecp.keys()))
            estimators["ecp"] = cfg.get(
                "estimators.energy.ecp",
                ECPEnergy(
                    ecp_coefficients=scf._cell._ecp,
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
                    "a": FractionalAxis(lattice_index=0),
                    "b": FractionalAxis(lattice_index=1),
                    "c": FractionalAxis(lattice_index=2),
                }
            ),
        )
        wire(density, inv_lattice=inv_lattice)
        estimators["density"] = density
    return estimators
