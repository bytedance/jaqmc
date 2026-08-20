# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
from upath import UPath

from jaqmc.estimator import EstimatorLike
from jaqmc.estimator.density import CartesianAxis, CartesianDensity
from jaqmc.estimator.ecp import ECPEnergy
from jaqmc.estimator.kinetic import EuclideanKinetic
from jaqmc.estimator.ph import PHEnergy
from jaqmc.estimator.spin import SpinSquared
from jaqmc.estimator.total_energy import TotalEnergy
from jaqmc.optimizer.kfac import KFACOptimizer
from jaqmc.optimizer.optax import adam
from jaqmc.sampler.mcmc import MCMCSampler
from jaqmc.utils.atomic import make_pretrain_loss
from jaqmc.utils.atomic.pretrain import make_pretrain_log_amplitude
from jaqmc.utils.config import ConfigManager, ConfigManagerLike
from jaqmc.utils.reference import auto_generate_reference
from jaqmc.wavefunction import Wavefunction
from jaqmc.workflow import EvaluationWorkflow, VMCWorkflow
from jaqmc.workflow.stage import EvaluationWorkStage, VMCWorkStage
from jaqmc.workflow.stage.vmc import VMCStageBuilder

from .config import MoleculeConfig, MoleculeSolverConfig
from .data import data_init
from .hamiltonian import potential_energy
from .reference import MoleculeReference, load_reference
from .reference import pyscf as reference_pyscf
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
        system_config, wf = configure_system(cfg)

        self.wf = wf
        self.data_init = partial(data_init, system_config)
        sampler = cfg.get("sampler", MCMCSampler)
        self.reference: MoleculeReference | None = None
        self._system_config = system_config

        self._pretrain_builder: VMCStageBuilder | None = None
        if cfg.get("pretrain.run.iterations", 2_000) > 0:
            if reference_path := cfg.get("reference", ""):
                self.reference = load_reference(
                    UPath(reference_path), self._system_config
                )
            self._pretrain_sample_fraction = cfg.get("pretrain.sample_fraction", 1.0)
            pretrain = VMCWorkStage.builder(cfg.scoped("pretrain"), wf)
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
        if self._pretrain_builder is None or (dry_run and self.reference is None):
            return

        if self.reference is None:
            reference_path = self.save_path / "reference.npz"
            if (restore_reference_path := self.restore_dir / "reference.npz").exists():
                reference_path = restore_reference_path
            if not reference_path.exists():
                auto_generate_reference(
                    path=reference_path,
                    job=reference_pyscf.MoleculePySCFReferenceJob(
                        self._system_config, MoleculeSolverConfig()
                    ),
                )
            self.reference = load_reference(reference_path, self._system_config)
        reference = self.reference

        pretrain_loss = make_pretrain_loss(
            orbitals_fn=self.wf.orbitals,
            orbital_ref=reference,
            nspins=self._system_config.electron_spins,
            full_det=self.wf.full_det,
        )
        pretrain_f_log_amplitude = make_pretrain_log_amplitude(
            self.wf.logpsi,
            lambda data: reference.eval_slater(
                data.electrons, self._system_config.electron_spins
            )[1],
            ref_fraction=self._pretrain_sample_fraction,
        )
        self._pretrain_builder.configure_sample_plan(
            pretrain_f_log_amplitude,
            {"electrons": self.train_stage.sample_plan.samplers["electrons",]},
        )
        self._pretrain_builder.configure_estimators(grads=pretrain_loss)
        self.pretrain_stage = self._pretrain_builder.build()


class MoleculeEvalWorkflow(EvaluationWorkflow):
    """Evaluation workflow for molecular systems."""

    def __init__(self, cfg: ConfigManager) -> None:
        super().__init__(cfg)
        system_config, wf = configure_system(cfg)

        self.data_init = partial(data_init, system_config)

        evaluation = EvaluationWorkStage.builder(cfg, wf, name="evaluation")
        sampler = cfg.get("sampler", MCMCSampler)
        evaluation.configure_sample_plan(wf.logpsi, {"electrons": sampler})

        eval_estimators = make_estimators(cfg, wf, system_config)
        evaluation.configure_estimators(**eval_estimators)

        self.evaluation_stage = evaluation.build()


def configure_system(
    cfg: ConfigManagerLike,
) -> tuple[MoleculeConfig, MoleculeWavefunction]:
    system_config: MoleculeConfig | Callable[[], MoleculeConfig] = cfg.get_module(
        "system", MoleculeConfig
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


def make_estimators(
    cfg: ConfigManagerLike,
    wf: MoleculeWavefunction,
    system_config: MoleculeConfig,
    always_enable_energy: bool = False,
) -> dict[str, EstimatorLike]:
    estimators: dict[str, EstimatorLike] = {}
    if always_enable_energy or cfg.get("estimators.enabled.energy", True):
        estimators["potential"] = potential_energy
        if not system_config.ph_elements:
            estimators["kinetic"] = cfg.get(
                "estimators.energy.kinetic", EuclideanKinetic(f_log_psi=wf.logpsi)
            )
        else:
            logger.warning(
                "PH is active: the regular EuclideanKinetic estimator is "
                "inactive, and `estimators.energy.kinetic.*` overrides are "
                "ignored. Use `estimators.energy.ph.kinetic_backend` to "
                "select the PH derivative backend."
            )
        if ecp_coefficients := system_config.ecp_coefficients:
            logger.info("ECP enabled for elements: %s", list(ecp_coefficients))
            estimators["ecp"] = cfg.get(
                "estimators.energy.ecp",
                ECPEnergy(
                    ecp_coefficients=ecp_coefficients,
                    atom_symbols=[atom.symbol for atom in system_config.atoms],
                    phase_logpsi=wf.phase_logpsi,
                ),
            )
        if system_config.ph_elements:
            logger.info("PH enabled for elements: %s", list(system_config.ph_elements))
            estimators["ph"] = cfg.get(
                "estimators.energy.ph",
                PHEnergy(
                    f_log_psi=wf.logpsi,
                    atom_symbols=[atom.symbol for atom in system_config.atoms],
                    ph=list(system_config.ph_elements),
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
