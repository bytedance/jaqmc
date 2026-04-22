# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

r"""Spin-squared (:math:`S^2`) estimator.

Computes the local value of :math:`S^2` using the identity
:math:`S^2 = S_z(S_z+1) + S_- S_+` with wavefunction coordinate swaps.

.. seealso:: :doc:`/guide/estimators/spin` for the full derivation
   and implementation notes.
"""

from collections.abc import Mapping
from typing import Any

import jax
from jax import numpy as jnp

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import Data
from jaqmc.estimator.base import LocalEstimator
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.wiring import runtime_dep


@configurable_dataclass
class SpinSquared(LocalEstimator):
    r"""Estimator for the total spin operator :math:`S^2`.

    Computes the local value of :math:`S^2` for a single walker using

    .. math::

        S^2_\text{local}
            = S_z(S_z + 1)
            + n_\text{minority}
            - \sum_{i \in \text{minority}} \sum_{j \in \text{majority}}
              \frac{\Psi(\mathbf{r}_{i \leftrightarrow j})}
                   {\Psi(\mathbf{r})}

    where :math:`S_z = |n_\uparrow - n_\downarrow| / 2` and minority
    is the spin channel with fewer (or equal) electrons.

    Args:
        n_up: Number of spin-up electrons.
        n_down: Number of spin-down electrons.
        phase_logpsi: Wavefunction evaluate function returning
            ``(sign, log|psi|)`` (runtime dep).
    """

    n_up: int = runtime_dep()
    n_down: int = runtime_dep()
    phase_logpsi: Any = runtime_dep()

    def init(self, data: Data, rngs: PRNGKey) -> None:
        n_up, n_down = self.n_up, self.n_down
        n_elec = n_up + n_down

        # Ties: minority = up.
        if n_up > n_down:
            self._majority_idx = jnp.arange(n_up)
            self._minority_idx = jnp.arange(n_up, n_elec)
        else:
            self._majority_idx = jnp.arange(n_up, n_elec)
            self._minority_idx = jnp.arange(n_up)

        self._idx_all = jnp.arange(n_elec)
        self._sz = jnp.abs(n_up - n_down) * 0.5
        self._vmapped_phase_logpsi = jax.vmap(self.phase_logpsi, in_axes=(None, 0))
        return None

    def evaluate_local(
        self,
        params: Params,
        data: Data,
        prev_local_stats: Mapping[str, Any],
        state: None,
        rngs: PRNGKey,
    ) -> tuple[dict[str, Any], None]:
        del prev_local_stats, rngs
        s2 = self._sz * (self._sz + 1) + self._sum_wf_ratios(params, data)
        return {"spin:s2": s2}, state

    def _sum_wf_ratios(self, params: Params, data: Data) -> jnp.ndarray:
        r"""Compute :math:`n_\text{min} - \sum_{i,j} \Psi(\text{swap})/\Psi`.

        Evaluates the original wavefunction once and vmaps the inner
        sum over all minority electrons.

        Returns:
            The :math:`S_- S_+` (or :math:`S_+ S_-`) contribution.
        """
        orig_sign, orig_logpsi = self.phase_logpsi(params, data)
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
        lse, sum_sign = jax.scipy.special.logsumexp(
            log_ratio, b=ratio_phases, return_sign=True
        )
        ratio_sum = sum_sign * jnp.exp(lse)  # = sum_j psi(swap_ij)/psi

        return 1.0 - ratio_sum

    def _swap_with_majority(self, data: Data, min_idx: jnp.ndarray) -> Data:
        """Build data with ``min_idx`` swapped with each majority electron.

        Returns:
            Stacked data with shape ``[n_majority, ...]``.
        """
        electrons: jnp.ndarray = data["electrons"]

        def _swap_one(maj_idx: jnp.ndarray) -> Data:
            perm = self._idx_all.at[min_idx].set(maj_idx)
            perm = perm.at[maj_idx].set(min_idx)
            return data.merge({"electrons": electrons[perm]})

        return jax.vmap(_swap_one)(self._majority_idx)
