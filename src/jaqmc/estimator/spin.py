# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

r"""Spin-squared (:math:`S^2`) estimator and penalty.

Computes the local value of :math:`S^2` using the identity
:math:`S^2 = S_z(S_z+1) + S_- S_+` with wavefunction coordinate swaps.

.. seealso:: :doc:`/guide/estimators/spin` for the full derivation
   and implementation notes.
"""

from collections.abc import Mapping
from functools import partial
from typing import Any, Literal

import jax
from jax import numpy as jnp

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import BatchedData, Data
from jaqmc.estimator.base import PerWalkerEstimator
from jaqmc.estimator.loss_grad import loss_term
from jaqmc.utils import parallel_jax
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.wiring import runtime_dep

__all__ = ["SpinSquared"]


@configurable_dataclass
class SpinSquared(PerWalkerEstimator):
    r"""Estimator for total spin :math:`S^2` (reported as ``spin_squared``).

    In ``"full"`` mode, computes the local value of :math:`S^2` for a single
    walker by summing over all minority-channel electrons:

    .. math::

        S^2_\text{local}
            = |S_z|(|S_z| + 1)
            + n_\text{min}
            - \sum_{i \in \text{min}} \sum_{j \in \text{maj}}
              \frac{\Psi(\mathbf{r}_{i \leftrightarrow j})}
                   {\Psi(\mathbf{r})}

    where :math:`|S_z| = |n_\uparrow - n_\downarrow| / 2` and ``min``
    is the spin channel with fewer (or equal) electrons.

    In ``"single_electron"`` mode, cycles through minority-channel electrons
    across steps using :math:`i = \text{step} \bmod n_\text{min}` and scales
    that electron's contribution by :math:`n_\text{min}`:

    .. math::

        S^{2,(i)}_\text{local}
            = |S_z|(|S_z| + 1)
            + n_\text{min} \left(
                1 - \sum_{j \in \text{maj}}
                    \frac{\Psi(\mathbf{r}_{i \leftrightarrow j})}
                         {\Psi(\mathbf{r})}
              \right).

    Args:
        mode: Evaluation strategy over minority-channel electrons.
            ``"full"`` sums over all :math:`n_\text{min}` minority electrons
            each step (:math:`n_\text{min} \times n_\text{maj}` swaps per
            walker). ``"single_electron"`` evaluates one minority electron
            per step (:math:`n_\text{maj}` swaps per walker) and cycles
            across steps.
        loss_weight: Penalty multiplier applied to :math:`\langle P_L \rangle^2`
            in the VMC loss and gradient. Set to ``0.0`` (default) to disable.
        wavefunction_precision: Wavefunction-defined precision mode passed to
            wavefunction evaluation. ``None`` inherits the wavefunction default.
        n_up: Number of spin-up electrons (runtime dep).
        n_down: Number of spin-down electrons (runtime dep).
        phase_logpsi: Wavefunction evaluate function returning
            ``(sign, log|psi|)`` (runtime dep).
        data_field: Name of the coordinate field (runtime dep, default
            ``"electrons"``).
    """

    mode: Literal["full", "single_electron"] = "full"
    loss_weight: float = 0.0
    n_up: int = runtime_dep()
    n_down: int = runtime_dep()
    wavefunction_precision: str | None = None
    phase_logpsi: Any = runtime_dep()
    data_field: str = runtime_dep(default="electrons")

    def init(self, data: Data, rngs: PRNGKey) -> jnp.ndarray | None:
        del data, rngs
        n_up, n_down = self.n_up, self.n_down
        n_elec = n_up + n_down
        self._n_min = min(n_up, n_down)

        # Ties: minority = up.
        if n_up > n_down:
            self._majority_idx = jnp.arange(n_up)
            self._minority_idx = jnp.arange(n_up, n_elec)
        else:
            self._majority_idx = jnp.arange(n_up, n_elec)
            self._minority_idx = jnp.arange(n_up)

        self._idx_all = jnp.arange(n_elec)
        sz = abs(n_up - n_down) * 0.5
        self._sz_term = sz * (sz + 1)
        self._vmapped_phase_logpsi = jax.vmap(
            partial(self.phase_logpsi, precision=self.wavefunction_precision),
            in_axes=(None, 0),
        )
        return jnp.array(0) if self.mode == "single_electron" else None

    def evaluate_batch_walkers(
        self,
        params: Params,
        batched_data: BatchedData[Any],
        prev_walker_stats: Mapping[str, Any],
        state: jnp.ndarray | None,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], jnp.ndarray | None]:
        walker_stats, state = super().evaluate_batch_walkers(
            params, batched_data, prev_walker_stats, state, rngs
        )
        walker_stats.update(
            loss_term(
                "spin_squared",
                walker_stats["spin_squared"] - self._sz_term,
                weight=self.loss_weight,
                direct_grad=walker_stats.pop("aux:grad_p_local", None),
                loss_fn=jnp.square,
            )
        )
        return walker_stats, state

    def evaluate_single_walker(
        self,
        params: Params,
        data: Data,
        prev_walker_stats: Mapping[str, Any],
        state: jnp.ndarray | None,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], jnp.ndarray | None]:
        del prev_walker_stats, rngs
        next_state = state + 1 if state is not None else None
        if self._n_min == 0:
            return {"spin_squared": jnp.asarray(self._sz_term)}, next_state
        if self.loss_weight:
            p_local, grad_p_local = jax.value_and_grad(self._compute_p_local)(
                parallel_jax.pvary(params), data, state
            )
            return {
                "spin_squared": self._sz_term + p_local,
                "aux:grad_p_local": grad_p_local,
            }, next_state

        p_local = self._compute_p_local(params, data, state)
        return {"spin_squared": self._sz_term + p_local}, next_state

    def _compute_p_local(
        self, params: Params, data: Data, state: jnp.ndarray | None
    ) -> jnp.ndarray:
        orig_sign, orig_logpsi = self.phase_logpsi(
            params, data, precision=self.wavefunction_precision
        )
        if state is not None:
            min_idx = self._minority_idx[state % self._n_min]
            return self._n_min * self._ratio_sum_one_electron(
                params, data, orig_sign, orig_logpsi, min_idx
            )
        per_minority = jax.vmap(
            self._ratio_sum_one_electron,
            in_axes=(None, None, None, None, 0),
        )(params, data, orig_sign, orig_logpsi, self._minority_idx)
        return jnp.sum(per_minority)

    def _ratio_sum_one_electron(
        self,
        params: Params,
        data: Data,
        orig_sign: jnp.ndarray,
        orig_logpsi: jnp.ndarray,
        min_idx: jnp.ndarray,
    ) -> jnp.ndarray:
        r"""One minority electron's contribution to :math:`S_- S_+`.

        Computes :math:`1 - \sum_j \Psi(\text{swap}_{ij})/\Psi`
        for minority electron *i*.  The ``+1`` accounts for this
        electron's share of the :math:`+n_\text{minority}` identity
        term.

        Returns:
            Scalar contribution from minority electron ``min_idx``.
        """
        swapped_data = self._swap_with_majority(data, min_idx)
        swap_sign, swap_logpsi = self._vmapped_phase_logpsi(params, swapped_data)

        log_ratio = swap_logpsi - orig_logpsi
        ratio_phases = swap_sign * jnp.conj(orig_sign)
        # Write the stabilized sum explicitly instead of ``logsumexp(b=...)``
        # so ``jax.grad`` can differentiate through complex/signed phases.
        max_log = jax.lax.stop_gradient(jnp.max(log_ratio))
        ratio_sum = jnp.sum(ratio_phases * jnp.exp(log_ratio - max_log)) * jnp.exp(
            max_log
        )

        return 1.0 - ratio_sum.real

    def _swap_with_majority(self, data: Data, min_idx: jnp.ndarray) -> Data:
        """Build data with ``min_idx`` swapped with each majority electron.

        Returns:
            Stacked data with shape ``[n_majority, ...]``.
        """
        electrons: jnp.ndarray = data[self.data_field]

        def _swap_one(maj_idx: jnp.ndarray) -> Data:
            perm = self._idx_all.at[min_idx].set(maj_idx)
            perm = perm.at[maj_idx].set(min_idx)
            return data.merge({self.data_field: electrons[perm]})

        return jax.vmap(_swap_one)(self._majority_idx)
