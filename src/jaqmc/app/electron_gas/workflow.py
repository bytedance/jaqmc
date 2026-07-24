# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Training and evaluation workflows for the homogeneous electron gas."""

from collections.abc import Callable
from functools import partial
from typing import Any

from jax import numpy as jnp

from jaqmc.app.solid.hamiltonian import PotentialEnergy
from jaqmc.app.solid.wavefunction import SolidWavefunction
from jaqmc.estimator import EstimatorLike
from jaqmc.estimator.kinetic import EuclideanKinetic
from jaqmc.estimator.spin import SpinSquared
from jaqmc.estimator.total_energy import TotalEnergy
from jaqmc.geometry.pbc import make_pbc_gaussian_proposal
from jaqmc.optimizer.kfac import KFACOptimizer
from jaqmc.optimizer.optax import adam
from jaqmc.sampler.mcmc import MCMCSampler
from jaqmc.utils.atomic.pretrain import make_pretrain_log_amplitude, make_pretrain_loss
from jaqmc.utils.config import ConfigManager, ConfigManagerLike
from jaqmc.wavefunction.output.envelope import EnvelopeType
from jaqmc.workflow.evaluation import EvaluationWorkflow
from jaqmc.workflow.stage.evaluation import EvaluationWorkStage
from jaqmc.workflow.stage.vmc import VMCWorkStage
from jaqmc.workflow.vmc import VMCWorkflow

from .config import ElectronGasConfig
from .data import data_init
from .reference import FreeElectronReference


class ElectronGasTrainWorkflow(VMCWorkflow):
    """VMC training with analytic free-electron pretraining."""

    @classmethod
    def default_preset(cls) -> dict[str, Any]:
        console_fields = (
            "pmove:.2f,energy=total_energy:.4f,variance=total_energy_real_var:.4f"
        )
        return {
            "pretrain": {
                "run": {"iterations": 1_000},
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
        system, wf, reference, proposal = configure_system(cfg)

        self.wf = wf
        self.data_init = partial(data_init, system)
        sampler = cfg.get("sampler", MCMCSampler(sampling_proposal=proposal))

        loss_estimator = make_pretrain_loss(
            orbitals_fn=wf.orbitals,
            scf=reference,
            nspins=system.nspins,
            full_det=wf.full_det,
        )
        f_log_amplitude = make_pretrain_log_amplitude(
            wf.logpsi,
            lambda data: reference.eval_slater(data.electrons, system.nspins).real,
            scf_fraction=1.0,
        )

        pretrain = VMCWorkStage.builder(cfg.scoped("pretrain"), wf, name="pretrain")
        pretrain.configure_sample_plan(f_log_amplitude, {"electrons": sampler})
        pretrain.configure_optimizer(default=adam, f_log_psi=wf.logpsi)
        pretrain.configure_estimators(grads=loss_estimator)
        self.pretrain_stage = pretrain.build()

        train = VMCWorkStage.builder(cfg.scoped("train"), wf)
        train.configure_sample_plan(wf.logpsi, {"electrons": sampler})
        train.configure_optimizer(default=KFACOptimizer, f_log_psi=wf.logpsi)
        train.configure_estimators(
            **make_estimators(cfg, wf, system, always_enable_energy=True)
        )
        train.configure_loss_grads(f_log_psi=wf.logpsi)
        self.train_stage = train.build()


class ElectronGasEvalWorkflow(EvaluationWorkflow):
    """Evaluate a trained homogeneous electron gas wavefunction."""

    def __init__(self, cfg: ConfigManager) -> None:
        super().__init__(cfg)
        system, wf, _, proposal = configure_system(cfg)

        self.wf = wf
        self.data_init = partial(data_init, system)
        sampler = cfg.get("sampler", MCMCSampler(sampling_proposal=proposal))

        evaluation = EvaluationWorkStage.builder(cfg, wf, name="evaluation")
        evaluation.configure_sample_plan(wf.logpsi, {"electrons": sampler})
        evaluation.configure_estimators(**make_estimators(cfg, wf, system))
        self.evaluation_stage = evaluation.build()


def configure_system(
    cfg: ConfigManagerLike,
) -> tuple[ElectronGasConfig, SolidWavefunction, FreeElectronReference, Callable]:
    """Build the shared HEG system, ansatz, reference, and sampler proposal.

    Returns:
        System config, neural ansatz, analytic reference, and PBC proposal.

    Raises:
        TypeError: If the selected wavefunction is not solid-compatible.
        ValueError: If a nuclear envelope or block determinant is requested.
    """
    system: ElectronGasConfig = cfg.get_module(
        "system", "jaqmc.app.electron_gas.config"
    )
    lattice = jnp.asarray(system.lattice)
    reference = FreeElectronReference(system.nspins, system.lattice, system.twist)

    wf = cfg.get_module("wf", "jaqmc.app.electron_gas.wavefunction")
    if not isinstance(wf, SolidWavefunction):
        raise TypeError(
            "Electron-gas wavefunction must implement SolidWavefunction. "
            f"Got {type(wf).__name__}."
        )
    if wf.envelope_type != EnvelopeType.null:
        raise ValueError("Electron-gas wavefunctions require envelope_type='null'.")
    if not wf.full_det:
        raise ValueError("Electron-gas wavefunctions require full_det=True.")

    wf.nspins = system.nspins
    wf.klist = reference.get_orbital_kpoints()
    wf.primitive_lattice = lattice
    wf.simulation_lattice = lattice

    proposal = make_pbc_gaussian_proposal(lattice)
    return system, wf, reference, proposal


def make_estimators(
    cfg: ConfigManagerLike,
    wf: SolidWavefunction,
    system: ElectronGasConfig,
    always_enable_energy: bool = False,
) -> dict[str, EstimatorLike]:
    """Create the small estimator set required by the HEG workflow.

    Returns:
        Named estimators enabled for the requested workflow.
    """
    estimators: dict[str, EstimatorLike] = {}
    if always_enable_energy or cfg.get("estimators.enabled.energy", True):
        lattice = jnp.asarray(system.lattice)
        estimators["potential"] = PotentialEnergy(supercell_lattice=lattice)
        estimators["kinetic"] = cfg.get(
            "estimators.energy.kinetic", EuclideanKinetic(f_log_psi=wf.logpsi)
        )
        estimators["total"] = TotalEnergy()
    if cfg.get("estimators.enabled.spin", False):
        estimators["spin"] = cfg.get(
            "estimators.spin",
            SpinSquared(
                n_up=system.nspins[0],
                n_down=system.nspins[1],
                phase_logpsi=wf.phase_logpsi,
            ),
        )
    return estimators
