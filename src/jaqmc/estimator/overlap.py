# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

r"""Overlap estimator and penalty for excited-state calculations.

Computes the squared wavefunction overlap with one or more reference states

.. math::

    \mathcal{O}_k = \frac{|\langle \psi_\theta | \psi_k \rangle|^2}
                         {\langle \psi_\theta | \psi_\theta \rangle
                          \langle \psi_k | \psi_k \rangle}
    = \left\langle \frac{\psi_\theta(y)}{\psi_k(y)}
      \right\rangle_{y \sim |\psi_k|^2}
      \left\langle \frac{\psi_k(x)}{\psi_\theta(x)}
      \right\rangle_{x \sim |\psi_\theta|^2}.

.. seealso:: :doc:`/guide/estimators/overlap` for the full derivation,
   excited-state penalty workflow, and configuration examples.
"""

from collections.abc import Callable, Mapping
from dataclasses import field
from operator import itemgetter
from typing import Annotated, Any, Protocol, Self, cast

import jax
from jax import numpy as jnp
from upath import UPath

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import BatchedData, Data
from jaqmc.estimator.base import Estimator, mean_reduce
from jaqmc.estimator.loss_grad import loss_term
from jaqmc.sampler.base import SamplePlan, SamplerLike
from jaqmc.utils import parallel_jax
from jaqmc.utils.checkpoint import NumPyCheckpointManager
from jaqmc.utils.chunked_vmap import chunked_vmap
from jaqmc.utils.config import ModuleConfig, configurable_dataclass
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction.base import WavefunctionEvaluate, WavefunctionLike

__all__ = ["Overlap", "ReferenceState"]

type PhaseLogPsi[DataT: Data] = WavefunctionEvaluate[
    DataT, tuple[jnp.ndarray, jnp.ndarray]
]


class WavefunctionWithPhaseLogPsi[DataT: Data, OutputT](
    WavefunctionLike[DataT, OutputT], Protocol
):
    def phase_logpsi(
        self, params: Params, data: DataT, *, precision: str | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray]: ...


@configurable_dataclass
class ReferenceState:
    """Configuration for a single reference state in :class:`Overlap`.

    Args:
        path: Checkpoint directory or ``.npz`` file for the reference state.
        weight: Multiplier applied to this reference state's overlap penalty
            relative to ``Overlap.loss_weight``.
        wf: Wavefunction ansatz for the reference state. When ``None``
            (default), uses the current workflow's wavefunction architecture.
        phase_logpsi: Reference wavefunction ``(phase, log|psi|)`` evaluate
            function (runtime dep).
        sample_plan: Sampling plan used to advance the reference walkers
            (runtime dep).
        params: Restored reference wavefunction parameters (runtime dep).
        batched_data: Restored process-local reference walkers (runtime dep).
        sampler_state: Restored reference sampler state (runtime dep).
    """

    path: str | None = None
    weight: float = 1.0
    wf: Annotated[WavefunctionWithPhaseLogPsi | None, ModuleConfig()] = None
    phase_logpsi: PhaseLogPsi[Any] = runtime_dep()
    sample_plan: SamplePlan = runtime_dep()
    params: Params = runtime_dep()
    batched_data: BatchedData[Any] = runtime_dep()
    sampler_state: Any = runtime_dep()


# Overlap is an architectural compromise: unlike standard estimators that
# evaluate observables on the stage's walkers, it manages its own reference
# checkpoints, sharded walker populations, and MCMC burn-in/stepping inside
# estimator state to keep multi-state reference sampling out of core workflow
# stages.


@configurable_dataclass
class Overlap[RefT: ReferenceState](Estimator):
    r"""Wavefunction overlap estimator and orthogonalization penalty.

    For each configured reference state :math:`\psi_k`, maintains an independent
    MCMC walker population :math:`y \sim |\psi_k|^2` inside ``estimator_state``
    and estimates the squared overlap

    .. math::

        \mathcal{O}_k = \operatorname{Re}\left(
            \langle Q_k(y) \rangle \, \langle R_k(x) \rangle
        \right),

    where :math:`R_k(x) = \psi_k(x) / \psi_\theta(x)` on current walkers
    :math:`x \sim |\psi_\theta|^2` and
    :math:`Q_k(y) = \psi_\theta(y) / \psi_k(y)` on reference walkers
    :math:`y \sim |\psi_k|^2`. Both ratios are shifted by their batch-wise
    maximum log difference so arbitrary normalization constants cancel
    without overflow or underflow.

    Args:
        states: Mapping from state label to :class:`ReferenceState` config
            (or ``None`` to disable an entry).
        loss_weight: Overall multiplier applied to the overlap penalty in the
            VMC loss and gradient. Set to ``0.0`` to disable.
        vmap_chunk_size: Number of walkers to evaluate per ``vmap`` chunk.
            ``None`` evaluates the full batch at once.
        phase_logpsi: Current wavefunction ``(phase, log|psi|)`` evaluate
            function (runtime dep).
    """

    states: dict[str, RefT | None] = field(default_factory=dict)
    loss_weight: float = 1.0
    vmap_chunk_size: int | None = None
    phase_logpsi: PhaseLogPsi[Any] = runtime_dep()

    def _active_states(self) -> list[tuple[str, RefT]]:
        return [(name, cfg) for name, cfg in self.states.items() if cfg is not None]

    def restore[KeyT: str | tuple[str, ...]](
        self,
        *,
        wf: WavefunctionWithPhaseLogPsi,
        samplers: Mapping[KeyT, SamplerLike],
        data_init: Callable[[int, PRNGKey], Data | BatchedData],
        enabled: bool = True,
    ) -> Self:
        """Restore active reference states when enabled.

        Args:
            wf: Current workflow wavefunction (also used as fallback when a
                reference state's ``wf`` is ``None``).
            samplers: Mapping from data field(s) to sampler instance(s). Each
                reference checkpoint must have a ``sampler_state`` compatible
                with these samplers.
            data_init: Batched data initializer for 1-walker template creation.
            enabled: Whether this overlap estimator is enabled in the workflow.

        Returns:
            ``self`` with runtime dependencies wired.

        Raises:
            ValueError: If an active reference state has ``path=None`` or a
                checkpoint batch size not divisible by ``jax.device_count()``.
        """
        self.phase_logpsi = wf.phase_logpsi
        if not enabled:
            return self
        rngs = jax.random.key(0)
        raw = data_init(1, rngs)
        fields = [
            f for key in samplers for f in ((key,) if isinstance(key, str) else key)
        ]
        tmpl_data = raw if isinstance(raw, BatchedData) else BatchedData(raw, fields)
        for _, ref in self._active_states():
            if ref.path is None:
                raise ValueError(
                    "ReferenceState.path is required to restore from a checkpoint."
                )
            ref_wf = wf if ref.wf is None else ref.wf
            ref_phase_logpsi = ref_wf.phase_logpsi
            plan = SamplePlan(
                lambda p, d, *, precision=None, _f=ref_phase_logpsi: _f(
                    p, d, precision=precision
                )[1],
                samplers,
            )
            path = UPath(ref.path).resolve()
            _, ckpt = NumPyCheckpointManager(
                path, path, prefix="" if path.is_file() else "train"
            ).restore(
                {
                    "params": ref_wf.init_params(tmpl_data.unbatched_example(), rngs),
                    "batched_data": tmpl_data,
                    "sampler_state": plan.init(tmpl_data, rngs),
                },
                strict=True,
            )
            bdata = cast(BatchedData, ckpt["batched_data"])
            if bdata.batch_size % jax.device_count() != 0:
                raise ValueError(
                    f"Checkpoint batch size {bdata.batch_size} must be "
                    f"divisible by number of devices {jax.device_count()}."
                )
            local_n = bdata.batch_size // jax.process_count()
            idx = jax.process_index()
            slc = itemgetter(slice(idx * local_n, (idx + 1) * local_n))
            local_fields = {
                k: jax.tree.map(slc, bdata.data[k]) for k in bdata.fields_with_batch
            }
            ref.phase_logpsi = ref_phase_logpsi
            ref.sample_plan = plan
            ref.params = cast(Params, ckpt["params"])
            ref.batched_data = BatchedData(
                bdata.data.merge(local_fields), bdata.fields_with_batch
            )
            ref.sampler_state = ckpt["sampler_state"]
        return self

    def _eval_batch(
        self, fn: PhaseLogPsi[Any], params: Params, batched: BatchedData[Any]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return chunked_vmap(
            fn, in_axes=(None, batched.vmap_axis), chunk_size=self.vmap_chunk_size
        )(params, batched.data)

    def init(self, data: Data, rngs: PRNGKey) -> dict[str, dict[str, Any]]:
        del data, rngs
        return {
            name: {
                "batched_data": ref_cfg.batched_data,
                "sampler_state": ref_cfg.sampler_state,
            }
            for name, ref_cfg in self._active_states()
        }

    def evaluate_batch_walkers(
        self,
        params: Params,
        batched_data: BatchedData[Any],
        prev_walker_stats: Mapping[str, Any],
        state: dict[str, dict[str, Any]],
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        del prev_walker_stats
        active = self._active_states()
        if not active:
            zeros = jnp.zeros(batched_data.batch_size)
            return {"overlap": zeros}, state

        phase_cur_x, log_cur_x = self._eval_batch(
            self.phase_logpsi, params, batched_data
        )

        new_state: dict[str, dict[str, Any]] = {}
        walker_stats: dict[str, Any] = {}
        total_overlap = jnp.zeros(batched_data.batch_size)

        for name, ref_cfg in active:
            rngs, sub_rng = jax.random.split(rngs)
            ref_params = ref_cfg.params
            ref_state = state[name]

            new_ref_data, _, new_sampler_state = ref_cfg.sample_plan.step(
                ref_params,
                ref_state["batched_data"],
                ref_state["sampler_state"],
                sub_rng,
            )
            new_state[name] = {
                "batched_data": new_ref_data,
                "sampler_state": new_sampler_state,
            }

            phase_ref_x, log_ref_x = self._eval_batch(
                ref_cfg.phase_logpsi, ref_params, batched_data
            )
            phase_cur_y, log_cur_y = self._eval_batch(
                self.phase_logpsi, params, new_ref_data
            )
            phase_ref_y, log_ref_y = self._eval_batch(
                ref_cfg.phase_logpsi, ref_params, new_ref_data
            )

            # Q(y) = psi_cur(y) / psi_ref(y) on reference walkers y
            log_diff_q = log_cur_y - log_ref_y
            phase_q = phase_cur_y * jnp.conj(phase_ref_y)
            m_q = jax.lax.stop_gradient(parallel_jax.pmax(jnp.nanmax(log_diff_q)))
            mean_q_shifted = parallel_jax.pnanmean(phase_q * jnp.exp(log_diff_q - m_q))

            # R(x) = psi_ref(x) / psi_cur(x) on current walkers x
            log_diff_r = log_ref_x - log_cur_x
            phase_r = phase_ref_x * jnp.conj(phase_cur_x)
            p_local = mean_q_shifted * phase_r * jnp.exp(log_diff_r + m_q)
            tag = f"overlap:{name}"
            walker_stats[tag] = p_local
            if self.loss_weight and ref_cfg.weight:
                walker_stats.update(
                    loss_term(tag, p_local, weight=self.loss_weight * ref_cfg.weight)
                )
            total_overlap = total_overlap + p_local

        walker_stats["overlap"] = total_overlap
        return walker_stats, new_state

    def reduce(self, walker_stats: Mapping[str, Any]) -> dict[str, Any]:
        return mean_reduce(jax.tree.map(jnp.real, walker_stats))
