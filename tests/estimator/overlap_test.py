# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from operator import itemgetter
from pathlib import Path

import jax
import numpy as np
from jax import numpy as jnp

from jaqmc.app.molecule.data import data_init
from jaqmc.app.molecule.workflow import MoleculeEvalWorkflow, MoleculeTrainWorkflow
from jaqmc.data import BatchedData, Data
from jaqmc.estimator.base import EstimatorPipeline
from jaqmc.estimator.loss_grad import LossGrad
from jaqmc.estimator.overlap import Overlap, ReferenceState
from jaqmc.sampler.base import SamplePlan
from jaqmc.sampler.mcmc import MCMCSampler
from jaqmc.utils.func_transform import transform_maybe_complex


@dataclass
class _OneDimData(Data):
    electrons: jnp.ndarray


class _OneDimWF:
    """1D wavefunction psi(x) = (x - w) exp(-x^2/2 + offset + i(k x + alpha))."""

    def init_params(self, data: _OneDimData, rngs):
        del data, rngs
        return {
            "w": jnp.array(0.0),
            "k": jnp.array(0.0),
            "offset": jnp.array(0.0),
            "alpha": jnp.array(0.0),
        }

    def logpsi(
        self, params, data: _OneDimData, *, precision: str | None = None
    ) -> jnp.ndarray:
        del precision
        x = data.electrons[0]
        val = x - params["w"]
        log_abs = jnp.log(jnp.abs(val) + 1e-12) - 0.5 * x**2 + params["offset"]
        angle = jnp.where(val >= 0, 0.0, jnp.pi) + params["k"] * x + params["alpha"]
        return log_abs + 1j * angle

    def phase_logpsi(
        self, params, data: _OneDimData, *, precision: str | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        lp = self.logpsi(params, data, precision=precision)
        return jnp.exp(1j * lp.imag), lp.real


def test_overlap_shift_invariance_and_analytic_gradient(monkeypatch):
    """Overlap value and gradient match analytic formula and are shift-invariant."""
    from jaqmc.utils import parallel_jax

    monkeypatch.setattr(parallel_jax, "all_gather", lambda x: x)

    # Current wavefunction has offset 0.0; reference has a huge log-offset (+500.0)
    # that would overflow float32 exp() without shift-invariant normalization, plus
    # distinct global phase alpha.
    wf = _OneDimWF()

    params_cur = {
        "w": jnp.array(0.3),
        "k": jnp.array(0.6),
        "offset": jnp.array(0.0),
        "alpha": jnp.array(1.1),
    }
    params_ref = {
        "w": jnp.array(-0.4),
        "k": jnp.array(-0.5),
        "offset": jnp.array(500.0),
        "alpha": jnp.array(-0.7),
    }

    # Fixed walkers (steps=0)
    x_walkers = jnp.linspace(-2.0, 2.0, 64)[:, None]
    y_walkers = jnp.linspace(-1.8, 2.2, 64)[:, None]
    batched_x = BatchedData(_OneDimData(electrons=x_walkers), ["electrons"])
    batched_y = BatchedData(_OneDimData(electrons=y_walkers), ["electrons"])

    ref_sampler = MCMCSampler(steps=0)
    ref_sample_plan = SamplePlan(
        lambda p, d, *, precision=None: wf.phase_logpsi(p, d, precision=precision)[1],
        {"electrons": ref_sampler},
    )
    ref_sampler_state = ref_sample_plan.init(batched_y, jax.random.key(0))

    weight = 2.5
    overlap_est = Overlap(
        states={
            "s0": ReferenceState(
                weight=weight,
                phase_logpsi=wf.phase_logpsi,
                sample_plan=ref_sample_plan,
                params=params_ref,
                batched_data=batched_y,
                sampler_state=ref_sampler_state,
            )
        },
        loss_weight=3.0,
        phase_logpsi=wf.phase_logpsi,
    )
    pipeline = EstimatorPipeline(
        {
            "overlap": overlap_est,
            "grads": LossGrad(clip_method="none", f_log_psi=wf.logpsi),
        }
    )

    state = pipeline.init(batched_x, jax.random.key(1))
    step_stats, state = pipeline.evaluate(
        params_cur, batched_x, state, jax.random.key(2)
    )
    final_stats = pipeline.finalize_stats(
        jax.tree.map(itemgetter(None), step_stats), state
    )
    grads = final_stats["grads"]

    # Analytic calculation with zero offset and zero global phase
    params_cur_zero = {
        **params_cur,
        "offset": jnp.array(0.0),
        "alpha": jnp.array(0.0),
    }
    params_ref_zero = {
        **params_ref,
        "offset": jnp.array(0.0),
        "alpha": jnp.array(0.0),
    }

    def eval_psi_ratio(p_num, p_den, walkers):
        ph_n, lg_n = jax.vmap(wf.phase_logpsi, in_axes=(None, 0))(
            p_num, _OneDimData(walkers)
        )
        ph_d, lg_d = jax.vmap(wf.phase_logpsi, in_axes=(None, 0))(
            p_den, _OneDimData(walkers)
        )
        return ph_n * jnp.conj(ph_d) * jnp.exp(lg_n - lg_d)

    r_x = eval_psi_ratio(params_ref_zero, params_cur_zero, x_walkers)
    q_y = eval_psi_ratio(params_cur_zero, params_ref_zero, y_walkers)
    p_local_x = jnp.mean(q_y) * r_x
    expected_overlap = jnp.real(jnp.mean(p_local_x))

    np.testing.assert_allclose(
        final_stats["overlap:s0"], expected_overlap, rtol=1e-4, atol=1e-5
    )
    np.testing.assert_allclose(
        final_stats["overlap"], expected_overlap, rtol=1e-4, atol=1e-5
    )
    np.testing.assert_allclose(
        final_stats["loss:overlap:s0"],
        3.0 * weight * expected_overlap,
        rtol=1e-4,
        atol=1e-5,
    )

    # Analytic projector VMC gradient for both amplitude (w) and phase (k):
    # 2 * 3.0 * weight * Re[Cov_x((d/dtheta log psi(x))^*, p_local(x))]
    grad_logpsi_fn = transform_maybe_complex(wf.logpsi, jax.grad)
    grad_logpsi_x = jax.vmap(grad_logpsi_fn, in_axes=(None, 0))(
        params_cur, _OneDimData(x_walkers)
    )
    for key in ("w", "k"):
        g_x = grad_logpsi_x[key]
        expected_grad = (
            3.0
            * weight
            * 2.0
            * jnp.mean(jnp.real(jnp.conj(g_x) * (p_local_x - jnp.mean(p_local_x))))
        )
        np.testing.assert_allclose(
            grads[key],
            expected_grad,
            rtol=1e-4,
            atol=1e-5,
        )


def test_molecule_workflow_overlap_checkpoint_restoration(
    tmp_path: Path,
):
    """Train and eval workflows restore reference checkpoints across architectures."""
    ref_dir = tmp_path / "ref_run"
    train_dir = tmp_path / "excited_run"
    eval_dir = tmp_path / "eval_run"

    # 1. Run a tiny reference job with LapNet and batch_size=6
    #    (different architecture and batch size from excited_wf).
    ref_wf = MoleculeTrainWorkflow.from_config(
        {
            "workflow": {
                "seed": 0,
                "batch_size": 6,
                "save_path": str(ref_dir),
            },
            "system": {
                "atoms": [{"symbol": "H", "coords": [0.0, 0.0, 0.0]}],
                "s_z": 0.5,
            },
            "wf": {
                "module": "jaqmc.app.molecule.wavefunction.lapnet:LapNetWavefunction",
                "ndets": 1,
                "num_layers": 1,
                "num_heads": 1,
                "heads_dim": 4,
            },
            "pretrain": {"run": {"iterations": 0}},
            "train": {
                "run": {
                    "iterations": 1,
                    "burn_in": 0,
                    "save_step_interval": 1,
                    "save_time_interval": 0,
                },
                "optim": {"module": "jaqmc.optimizer.optax:adam"},
                "writers": {"console": None, "csv": None, "hdf5": None},
            },
        }
    )
    ref_wf()
    assert list(ref_dir.glob("train_ckpt_*.npz"))

    ref_wf_cfg = {
        "module": "lapnet:LapNetWavefunction",
        "ndets": 1,
        "num_layers": 1,
        "num_heads": 1,
        "heads_dim": 4,
    }

    # 2. Run a training job with FermiNet and batch_size=4 that orthogonalizes
    #    against the LapNet reference in ref_dir.
    excited_wf = MoleculeTrainWorkflow.from_config(
        {
            "workflow": {
                "seed": 1,
                "batch_size": 4,
                "save_path": str(train_dir),
            },
            "system": {
                "atoms": [{"symbol": "H", "coords": [0.0, 0.0, 0.0]}],
                "s_z": 0.5,
            },
            "wf": {
                "ndets": 2,
                "hidden_dims_single": [8],
                "hidden_dims_double": [4],
            },
            "sampler": {"steps": 1},
            "estimators": {
                "overlap": {
                    "enabled": True,
                    "loss_weight": 5.0,
                    "states": {
                        "ground": {
                            "path": str(ref_dir),
                            "weight": 2.0,
                            "wf": ref_wf_cfg,
                        },
                    },
                },
            },
            "pretrain": {"run": {"iterations": 0}},
            "train": {
                "run": {
                    "iterations": 2,
                    "burn_in": 0,
                    "save_step_interval": 2,
                    "save_time_interval": 0,
                },
                "optim": {"module": "jaqmc.optimizer.optax:adam"},
                "writers": {"console": None, "csv": None, "hdf5": None},
            },
        }
    )
    excited_wf()
    ckpt_file = next(train_dir.glob("train_ckpt_*.npz"))

    # 3. Evaluate overlap against both the LapNet directory (explicit wf) and
    #    the FermiNet checkpoint file (default wf=None inheriting workflow.wf).
    eval_wf = MoleculeEvalWorkflow.from_config(
        {
            "workflow": {
                "seed": 2,
                "batch_size": 4,
                "save_path": str(eval_dir),
                "source_path": str(train_dir),
            },
            "system": {
                "atoms": [{"symbol": "H", "coords": [0.0, 0.0, 0.0]}],
                "s_z": 0.5,
            },
            "wf": {
                "ndets": 2,
                "hidden_dims_single": [8],
                "hidden_dims_double": [4],
            },
            "sampler": {"steps": 1},
            "estimators": {
                "overlap": {
                    "enabled": True,
                    "states": {
                        "ground": {
                            "path": str(ref_dir),
                            "wf": ref_wf_cfg,
                        },
                        "self_file": {"path": str(ckpt_file)},
                    },
                },
            },
            "run": {
                "iterations": 2,
                "burn_in": 0,
            },
            "writers": {"console": None, "csv": None, "hdf5": None},
        }
    )
    eval_wf()
    digest_path = eval_dir / "evaluation_digest.npz"
    assert digest_path.is_file()
    with np.load(digest_path) as digest:
        assert "overlap" in digest
        assert "overlap:ground" in digest
        assert "overlap:self_file" in digest
        assert np.isfinite(digest["overlap:ground"])
        assert np.isfinite(digest["overlap:self_file"])
        np.testing.assert_allclose(
            digest["overlap"],
            digest["overlap:ground"] + digest["overlap:self_file"],
            rtol=1e-6,
        )

    # 4. Verify Overlap.restore handles data_init returning plain Data with a
    #    string sampler key ("electrons").
    system = eval_wf.system() if callable(eval_wf.system) else eval_wf.system
    plain_overlap = Overlap(states={"s0": ReferenceState(path=str(ckpt_file))})
    plain_overlap.restore(
        wf=eval_wf.wf,
        samplers={"electrons": eval_wf.sampler},
        data_init=lambda n, r: data_init(system, n, r).data,
    )
    assert plain_overlap.states["s0"] is not None
    assert plain_overlap.states["s0"].batched_data.fields_with_batch == ["electrons"]
