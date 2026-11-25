# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Explicit precision keywords in workflows, estimators, samplers, and optimizers."""

from pathlib import Path
from typing import Any, cast

import jax
import numpy as np
from jax import numpy as jnp

from jaqmc.app.hall.data import HallData
from jaqmc.app.hall.estimator.one_rdm import OneRDM
from jaqmc.app.molecule.data import MoleculeData
from jaqmc.app.molecule.wavefunction.lapnet import LapNetWavefunction
from jaqmc.app.molecule.workflow import MoleculeTrainWorkflow
from jaqmc.array_types import Params
from jaqmc.data import BatchedData
from jaqmc.estimator.angular_momentum import (
    SphericalAngularMomentum,
    SphericalAngularMomentumProjection,
)
from jaqmc.estimator.kinetic import EuclideanKinetic, SphericalKinetic
from jaqmc.estimator.loss_grad import LossGrad, loss_term
from jaqmc.estimator.spin import SpinSquared
from jaqmc.optimizer.kfac import KFACOptimizer
from jaqmc.optimizer.sr import SROptimizer
from jaqmc.sampler.base import SamplePlan
from jaqmc.sampler.mcmc import MCMCSampler
from jaqmc.utils.atomic.pretrain import make_pretrain_log_amplitude


def _mode_scale(precision: str | None) -> float:
    return {None: 1.0, "standard": 2.0, "fp16_mixed": 3.0}[precision]


def _make_molecule_sample() -> tuple[MoleculeData, BatchedData]:
    data = MoleculeData(
        jnp.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        jnp.zeros((1, 3)),
        jnp.array([2.0]),
    )
    batched = BatchedData(
        data.merge({"electrons": jnp.stack([data.electrons, 2.0 * data.electrons])}),
        ["electrons"],
    )
    return data, batched


def _scaled_log_psi(
    params: Any, data: MoleculeData, *, precision: str | None = None
) -> jnp.ndarray:
    scale = _mode_scale(precision) * params.get("w", jnp.float32(1.0))
    return -0.5 * scale * jnp.sum(data.electrons**2)


def _scaled_phase_logpsi(
    params: Any, data: MoleculeData, *, precision: str | None = None
) -> tuple[jnp.ndarray, jnp.ndarray]:
    del params
    scale = _mode_scale(precision)
    return jnp.float32(1.0), scale * (data.electrons[0, 0] - data.electrons[1, 0])


def test_workflow_configures_consumer_wavefunction_precisions(tmp_path: Path) -> None:
    workflow = MoleculeTrainWorkflow.from_config(
        {
            "workflow": {"save_path": str(tmp_path)},
            "system": {"atoms": [{"symbol": "Li", "coords": [0, 0, 0]}], "s_z": 0.5},
            "pretrain": {"run": {"iterations": 0}},
            "wf": {"module": "jaqmc.app.molecule.wavefunction.lapnet"},
        },
        [
            "wf.precision=fp16_jac",
            "sampler.wavefunction_precision=fp16_mixed",
            "estimators.spin.enabled=true",
            "estimators.energy.kinetic.wavefunction_precision=fp16_mixed",
            "estimators.spin.wavefunction_precision=fp16_mixed",
            "train.grads.wavefunction_precision=fp16_mixed",
            "train.optim.wavefunction_precision=standard",
        ],
    )
    workflow.build_runtime()
    stage = workflow.train_stage
    assert isinstance(stage.wavefunction, LapNetWavefunction)
    assert stage.wavefunction.precision == "fp16_jac"
    assert (
        cast(
            MCMCSampler, stage.sample_plan.samplers["electrons",]
        ).wavefunction_precision
        == "fp16_mixed"
    )
    assert (
        cast(
            EuclideanKinetic, stage.estimators.estimators["kinetic"]
        ).wavefunction_precision
        == "fp16_mixed"
    )
    assert (
        cast(SpinSquared, stage.estimators.estimators["spin"]).wavefunction_precision
        == "fp16_mixed"
    )
    assert (
        cast(LossGrad, stage.estimators.estimators["loss_grad"]).wavefunction_precision
        == "fp16_mixed"
    )
    assert cast(KFACOptimizer, stage.optimizer).wavefunction_precision == "standard"


def test_sampler_and_pretrain_forward_wavefunction_precision() -> None:
    _, batched = _make_molecule_sample()
    key = jax.random.key(0)
    pretrain_amp = make_pretrain_log_amplitude(
        lambda p, d, *, precision=None: (
            (1.0 if precision == "fp16_mixed" else -1.0) * jnp.sum(d.electrons**2)
        ),
        lambda d: jnp.float32(-0.5),
        ref_fraction=0.25,
    )
    for mode, expected_pmove in ((None, 0.0), ("fp16_mixed", 1.0)):
        sampler = MCMCSampler(
            steps=1,
            initial_width=10.0,
            wavefunction_precision=mode,
            sampling_proposal=lambda rngs, x, stddev: jax.tree.map(
                lambda a: a + stddev, x
            ),
        )
        plan = SamplePlan(pretrain_amp, {"electrons": sampler})
        _, stats, _ = jax.jit(plan.step)({}, batched, plan.init(batched, key), key)
        np.testing.assert_allclose(stats["pmove"], expected_pmove)


def test_molecule_estimators_evaluate_configured_precision() -> None:
    data, batched = _make_molecule_sample()
    key = jax.random.key(0)

    for mode, s in ((None, 1.0), ("fp16_mixed", 3.0)):
        # log_psi = -0.5 * s * r^2 (r^2 = 5) -> lap = -6s, |grad|^2 = 5s^2
        kinetic = EuclideanKinetic(
            f_log_psi=_scaled_log_psi,
            data_field="electrons",
            wavefunction_precision=mode,
        )
        kinetic_out, _ = kinetic.evaluate_single_walker({}, data, {}, None, key)
        np.testing.assert_allclose(
            kinetic_out["energy:kinetic"], -0.5 * (-6.0 * s + 5.0 * s**2), rtol=1e-5
        )

        # Swapping e0=(1,0,0) and e1=(0,2,0) flips (x0 - x1) from +1 to -1
        spin = SpinSquared(
            n_up=1,
            n_down=1,
            phase_logpsi=_scaled_phase_logpsi,
            wavefunction_precision=mode,
        )
        spin.init(data, key)
        spin_out, _ = spin.evaluate_single_walker({}, data, {}, None, key)
        np.testing.assert_allclose(
            spin_out["spin_squared"], 1.0 - np.exp(-2.0 * s), rtol=1e-5
        )

        loss_grads = LossGrad(
            clip_method="none",
            f_log_psi=_scaled_log_psi,
            wavefunction_precision=mode,
        )
        grad_out, _ = loss_grads.evaluate_batch_walkers(
            {"w": jnp.float32(2.0)},
            batched,
            loss_term("total_energy", jnp.array([1.0, 3.0], dtype=jnp.float32)),
            None,
            key,
        )
        np.testing.assert_allclose(grad_out["grads"]["w"], -7.5 * s, rtol=1e-6)


def test_sr_optimizer_evaluates_configured_precision() -> None:
    _, batched = _make_molecule_sample()
    params = {"w": jnp.float32(1.0)}
    grads = {"w": jnp.float32(0.5)}

    # Higher score scale increases Fisher curvature and shrinks preconditioned updates
    updates = []
    for mode in (None, "standard"):
        sr = SROptimizer(
            wavefunction_precision=mode,
            f_log_psi=_scaled_log_psi,
            damping=1.0,
            max_norm=None,
            score_chunk_size=None,
            gram_num_chunks=None,
        )
        upd, _ = sr.update(
            grads,
            sr.init(params, batched_data=batched),
            params,
            batched_data=batched,
        )
        updates.append(float(upd["w"]))

    assert abs(updates[1]) < abs(updates[0])


def test_hall_estimators_evaluate_configured_precision() -> None:
    def spinor_logpsi(
        params: Params,
        u: jnp.ndarray,
        v: jnp.ndarray,
        *,
        precision: str | None = None,
    ) -> jnp.ndarray:
        del params
        return _mode_scale(precision) * jnp.sum(u * v)

    def data_logpsi(
        params: Params,
        data: HallData,
        *,
        precision: str | None = None,
    ) -> jnp.ndarray:
        del params
        return _mode_scale(precision) * jnp.sum(data.electrons) + 0j

    hall_data = HallData(electrons=jnp.array([[0.7, 0.2], [1.4, -0.9]]))
    key = jax.random.key(0)

    for build, stat_key in (
        (
            lambda mode: SphericalKinetic(
                monopole_strength=1.0,
                radius=1.0,
                wavefunction_precision=mode,
                f_log_psi_from_spinor=spinor_logpsi,
            ),
            "energy:kinetic",
        ),
        (
            lambda mode: SphericalAngularMomentum(
                wavefunction_precision=mode,
                f_log_psi_from_spinor=spinor_logpsi,
            ),
            "angular_momentum_square",
        ),
        (
            lambda mode: SphericalAngularMomentumProjection(
                angular_momenta=(0.0,),
                n_rotations=2,
                wavefunction_precision=mode,
                f_log_psi_from_spinor=spinor_logpsi,
            ),
            "angular_projection:weight",
        ),
        (
            lambda mode: OneRDM(
                flux=2,
                wavefunction_precision=mode,
                f_log_psi=data_logpsi,
            ),
            "one_rdm",
        ),
    ):
        est_default = build(None)
        est_mixed = build("fp16_mixed")
        est_default.init(hall_data, key)
        est_mixed.init(hall_data, key)
        out_def, _ = est_default.evaluate_single_walker(
            {}, hall_data, {"total_energy": jnp.complex64(1.0)}, None, key
        )
        out_mix, _ = est_mixed.evaluate_single_walker(
            {}, hall_data, {"total_energy": jnp.complex64(1.0)}, None, key
        )
        assert not np.allclose(out_def[stat_key], out_mix[stat_key])
