# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Callable, Mapping
from dataclasses import replace
from functools import partial
from typing import Any

import numpy as np

from jaqmc.app.molecule.config import MoleculeConfig
from jaqmc.app.molecule.data import data_init
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
from jaqmc.utils.atomic import (
    MolecularSCF,
    ResolvedPseudopotentialConfig,
    resolve_pseudopotential_config,
)
from jaqmc.utils.atomic.pp import get_ph_effective_charge
from jaqmc.utils.atomic.pretrain import make_pretrain_log_amplitude, make_pretrain_loss
from jaqmc.utils.config import ConfigManager, ConfigManagerLike
from jaqmc.utils.units import ONE_ANGSTROM_IN_BOHR, LengthUnit
from jaqmc.wavefunction import Wavefunction
from jaqmc.workflow.evaluation import EvaluationWorkflow
from jaqmc.workflow.stage.evaluation import EvaluationWorkStage
from jaqmc.workflow.stage.vmc import VMCWorkStage
from jaqmc.workflow.vmc import VMCWorkflow

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
        self.scf = make_scf(system_config)
        self.data_init = partial(data_init, system_config)
        sampler = cfg.get("sampler", MCMCSampler)

        pretrain_loss = make_pretrain_loss(
            orbitals_fn=wf.orbitals, scf=self.scf, nspins=nspins, full_det=wf.full_det
        )
        pretrain_f_log_amplitude = make_pretrain_log_amplitude(
            wf.logpsi, lambda data: self.scf.eval_slater(data.electrons, nspins)[1]
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
            cfg, wf, self.scf, system_config, always_enable_energy=True
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
        scf = make_scf(system_config)

        evaluation = EvaluationWorkStage.builder(cfg, wf, name="evaluation")
        sampler = cfg.get("sampler", MCMCSampler)
        evaluation.configure_sample_plan(wf.logpsi, {"electrons": sampler})
        eval_estimators: dict[str, EstimatorLike] = make_estimators(
            cfg, wf, scf, system_config
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
    system_config = _normalize_molecule_config_units(system_config)

    wf = cfg.get_module("wf", "jaqmc.app.molecule.wavefunction.ferminet")
    wf.nspins = system_config.electron_spins

    if not isinstance(wf, Wavefunction) or not isinstance(wf, MoleculeWavefunction):
        raise TypeError(
            f"Wavefunction must implement MoleculeWavefunction protocol, "
            f"got {type(wf).__name__}"
        )
    return system_config, wf


def _normalize_molecule_config_units(system_config: MoleculeConfig) -> MoleculeConfig:
    if system_config.unit == LengthUnit.bohr:
        return system_config
    if system_config.unit != LengthUnit.angstrom:
        raise ValueError(f"Unsupported molecule length unit: {system_config.unit!r}")

    atoms = [
        replace(
            atom,
            coords=[coord * ONE_ANGSTROM_IN_BOHR for coord in atom.coords],
        )
        for atom in system_config.atoms
    ]
    return replace(
        system_config,
        atoms=atoms,
        unit=LengthUnit.bohr,
    )


def make_scf(system_config: MoleculeConfig) -> MolecularSCF:
    pseudopotential = resolve_pseudopotential_config(
        system_config.atoms, system_config.pp
    )
    return MolecularSCF(
        system_config.atoms,
        system_config.electron_spins,
        basis=system_config.basis,
        pseudopotential=pseudopotential,
    )


def make_estimators(
    cfg: ConfigManagerLike,
    wf: MoleculeWavefunction,
    scf: MolecularSCF,
    system_config: MoleculeConfig,
    always_enable_energy: bool = False,
) -> dict[str, EstimatorLike]:
    pseudopotential = resolve_pseudopotential_config(
        system_config.atoms, system_config.pp
    )

    estimators: dict[str, EstimatorLike] = {}
    if always_enable_energy or cfg.get("estimators.enabled.energy", True):
        estimators["potential"] = potential_energy
        uses_runtime_ph = pseudopotential.uses_runtime_ph()
        if not uses_runtime_ph:
            estimators["kinetic"] = cfg.get(
                "estimators.energy.kinetic", EuclideanKinetic(f_log_psi=wf.logpsi)
            )
        else:
            # Skip reading `estimators.energy.kinetic.*` so that any user
            # override on a PH run is left unread and surfaces as a
            # `cfg.finalize(raise_on_unused=True)` error rather than a silent
            # no-op. The PH derivative backend is selected by
            # `estimators.energy.ph.kinetic_backend`.
            logger.warning(
                "PH is active: the regular EuclideanKinetic estimator is "
                "inactive, and `estimators.energy.kinetic.*` overrides are "
                "ignored. Use `estimators.energy.ph.kinetic_backend` to "
                "select the PH derivative backend."
            )
        if pseudopotential.uses_runtime_ecp():
            runtime_ecp_coefficients = _runtime_ecp_coefficients(
                pseudopotential,
                scf._mol._ecp,
            )
            logger.info(
                "ECP enabled for elements: %s",
                list(runtime_ecp_coefficients),
            )
            estimators["ecp"] = cfg.get(
                "estimators.energy.ecp",
                ECPEnergy(
                    ecp_coefficients=runtime_ecp_coefficients,
                    atom_symbols=[atom.symbol for atom in system_config.atoms],
                    phase_logpsi=wf.phase_logpsi,
                ),
            )
        if uses_runtime_ph:
            logger.info(
                "PH enabled for elements: %s", list(pseudopotential.runtime_ph_symbols)
            )
            _validate_ph_atom_charges(system_config, pseudopotential.runtime_ph_symbols)
            estimators["ph"] = cfg.get(
                "estimators.energy.ph",
                PHEnergy(
                    f_log_psi=wf.logpsi,
                    atom_symbols=[atom.symbol for atom in system_config.atoms],
                    ph=list(pseudopotential.runtime_ph_symbols),
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


def _runtime_ecp_coefficients(
    pseudopotential: ResolvedPseudopotentialConfig,
    scf_ecp_coefficients: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        symbol: scf_ecp_coefficients[symbol]
        for symbol in pseudopotential.runtime_ecp_symbols
    }


def _validate_ph_atom_charges(
    system_config: MoleculeConfig,
    ph_symbols: tuple[str, ...],
) -> None:
    # PH composes additively with ``potential_energy``: the bare ``-Z/r`` term
    # uses ``Atom.charge`` (which feeds ``data.charges``) while PHEnergy uses
    # the table-baked ``get_ph_effective_charge(symbol)``. The cancellation is
    # only correct when those two ``Z`` values agree, so flag the mismatch
    # loudly at workflow construction rather than silently emitting wrong
    # energies during training.
    ph_set = set(ph_symbols)
    for atom_index, atom in enumerate(system_config.atoms):
        if atom.symbol not in ph_set:
            continue
        expected = float(get_ph_effective_charge(atom.symbol))
        actual = float(atom.charge)
        if actual != expected:
            raise ValueError(
                f"system_config.atoms[{atom_index}].charge = {actual} for PH "
                f"atom {atom.symbol!r} does not match get_ph_effective_charge"
                f"({atom.symbol!r}) = {expected}; the PH residual cancellation "
                "with potential_energy requires these to match."
            )
