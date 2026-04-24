# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np

from jaqmc.estimator import EstimatorLike
from jaqmc.estimator.density import CartesianAxis, CartesianDensity
from jaqmc.estimator.ecp import ECPEnergy
from jaqmc.estimator.kinetic import EuclideanKinetic
from jaqmc.estimator.spin import SpinSquared
from jaqmc.estimator.total_energy import TotalEnergy
from jaqmc.optimizer.kfac import KFACOptimizer
from jaqmc.optimizer.optax import adam
from jaqmc.sampler.mcmc import MCMCSampler
from jaqmc.utils.atomic import MolecularSCF, get_core_electrons
from jaqmc.utils.atomic.pretrain import make_pretrain_log_amplitude, make_pretrain_loss
from jaqmc.utils.config import ConfigManager, ConfigManagerLike
from jaqmc.wavefunction import Wavefunction
from jaqmc.workflow.evaluation import EvaluationWorkflow
from jaqmc.workflow.stage.evaluation import EvaluationWorkStage
from jaqmc.workflow.stage.vmc import VMCWorkStage
from jaqmc.workflow.vmc import VMCWorkflow

from .config import MoleculeConfig, MoleculePretrainReferenceConfig
from .data import data_init
from .hamiltonian import potential_energy
from .wavefunction import MoleculeWavefunction

logger = logging.getLogger(__name__)


class MoleculeTrainWorkflow(VMCWorkflow):
    """VMC training workflow for molecular systems."""

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
        system_config, wf = configure_system(cfg)

        nspins = system_config.electron_spins
        pretrain_config = cfg.get("pretrain.reference", MoleculePretrainReferenceConfig)
        self.scf = make_scf(pretrain_config, system_config)
        self.data_init = partial(data_init, system_config)
        sampler = cfg.get("sampler", MCMCSampler)

        pretrain_loss = make_pretrain_loss(
            orbitals_fn=wf.orbitals, scf=self.scf, nspins=nspins, full_det=wf.full_det
        )
        pretrain_f_log_amplitude = make_pretrain_log_amplitude(
            wf.logpsi,
            lambda data: self.scf.eval_slater(data.electrons, nspins)[1],
            scf_fraction=pretrain_config.sample_fraction,
        )

        pretrain = VMCWorkStage.builder(cfg.scoped("pretrain"), wf)
        pretrain.configure_sample_plan(pretrain_f_log_amplitude, {"electrons": sampler})
        pretrain.configure_optimizer(default=adam, f_log_psi=wf.logpsi)
        pretrain.configure_estimators(grads=pretrain_loss)
        self.pretrain_stage = pretrain.build()

        train = VMCWorkStage.builder(cfg.scoped("train"), wf)
        train.configure_sample_plan(wf.logpsi, {"electrons": sampler})
        train.configure_optimizer(default=KFACOptimizer, f_log_psi=wf.logpsi)
        estimators = make_estimators(
            cfg, wf, system_config, self.scf._mol._ecp, always_enable_energy=True
        )
        train.configure_estimators(**estimators)
        train.configure_loss_grads(f_log_psi=wf.logpsi)
        self.train_stage = train.build()

    def run(self) -> None:
        self.scf.run()
        super().run()


class MoleculeEvalWorkflow(EvaluationWorkflow):
    """Evaluation workflow for molecular systems."""

    def __init__(self, cfg: ConfigManager) -> None:
        super().__init__(cfg)
        system_config, wf = configure_system(cfg)

        self.data_init = partial(data_init, system_config)

        evaluation = EvaluationWorkStage.builder(cfg, wf, name="evaluation")
        sampler = cfg.get("sampler", MCMCSampler)
        evaluation.configure_sample_plan(wf.logpsi, {"electrons": sampler})

        # Just a simple SCF object to get ecp coefficients
        scf = make_scf(MoleculePretrainReferenceConfig(), system_config)
        eval_estimators: dict[str, EstimatorLike] = make_estimators(
            cfg, wf, system_config, scf._mol._ecp
        )
        evaluation.configure_estimators(**eval_estimators)

        self.evaluation_stage = evaluation.build()


def configure_system(
    cfg: ConfigManagerLike,
) -> tuple[MoleculeConfig, MoleculeWavefunction]:
    system_config: MoleculeConfig | Callable[[], MoleculeConfig] = cfg.get_module(
        "system", "jaqmc.app.molecule.config.base"
    )
    if callable(system_config):
        system_config = system_config()

    wf = cfg.get_module("wf", "jaqmc.app.molecule.wavefunction.ferminet")
    wf.nspins = system_config.electron_spins

    if not isinstance(wf, Wavefunction) or not isinstance(wf, MoleculeWavefunction):
        raise TypeError(
            f"Wavefunction must implement MoleculeWavefunction protocol, "
            f"got {type(wf).__name__}"
        )
    return system_config, wf


def make_scf(
    pretrain_config: MoleculePretrainReferenceConfig, system_config: MoleculeConfig
) -> MolecularSCF:
    restricted = pretrain_config.method == "RHF"
    return MolecularSCF(
        system_config.atoms,
        system_config.electron_spins,
        basis=pretrain_config.basis,
        restricted=restricted,
        ecp=system_config.ecp,
        core_electrons=get_core_electrons(system_config.atoms, system_config.ecp),
        verbose=pretrain_config.verbose,
        pyscf_options=pretrain_config.extra,
    )


def make_estimators(
    cfg: ConfigManagerLike,
    wf: MoleculeWavefunction,
    system_config: MoleculeConfig,
    ecp_coefficients: dict[str, Any] | None = None,
    always_enable_energy: bool = False,
) -> dict[str, EstimatorLike]:
    estimators: dict[str, EstimatorLike] = {}
    if always_enable_energy or cfg.get("estimators.enabled.energy", True):
        estimators["potential"] = potential_energy
        estimators["kinetic"] = cfg.get(
            "estimators.energy.kinetic", EuclideanKinetic(f_log_psi=wf.logpsi)
        )
        if ecp_coefficients:
            logger.info("ECP enabled for elements: %s", list(ecp_coefficients.keys()))
            estimators["ecp"] = cfg.get(
                "estimators.energy.ecp",
                ECPEnergy(
                    ecp_coefficients=ecp_coefficients,
                    atom_symbols=[atom.symbol for atom in system_config.atoms],
                    phase_logpsi=wf.phase_logpsi,
                ),
            )
        estimators["total"] = TotalEnergy()
    if cfg.get("estimators.enabled.spin", False):
        estimators["spin"] = cfg.get(
            "estimators.spin",
            SpinSquared(
                n_up=system_config.electron_spins[0],
                n_down=system_config.electron_spins[1],
                phase_logpsi=wf.phase_logpsi,
            ),
        )
    if cfg.get("estimators.enabled.density", False):
        positions = np.array([a.coords for a in system_config.atoms])
        padding = 5.0  # bohr
        axes: dict[str, CartesianAxis | None] = {}
        for name, idx in [("x", 0), ("y", 1), ("z", 2)]:
            lo = float(positions[:, idx].min()) - padding
            hi = float(positions[:, idx].max()) + padding
            axes[name] = CartesianAxis(
                direction=tuple(1.0 if i == idx else 0.0 for i in range(3)),
                bins=50,
                range=(lo, hi),
            )
        estimators["density"] = cfg.get(
            "estimators.density",
            CartesianDensity(axes=axes),
        )
    return estimators
