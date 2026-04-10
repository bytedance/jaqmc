# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol, runtime_checkable

from jaqmc.array_types import ArrayTree, Params


class OptimizerInit(Protocol):
    def __call__(self, params: Params, **kwargs) -> ArrayTree:
        """Initialize optimizer state.

        Args:
            params: Wavefunction parameters.
            **kwargs: Optimizer-specific arguments (e.g., ``batched_data``,
                ``rngs``). Runtime deps like ``f_log_psi`` are wired via
                :func:`~jaqmc.utils.wiring.runtime_dep` fields, not passed here.

        Returns:
            Initial optimizer state.
        """


class OptimizerUpdate(Protocol):
    def __call__(
        self,
        grads: Params,
        state: ArrayTree,
        params: Params,
        **kwargs,
    ) -> tuple[Params, ArrayTree]:
        """Apply optimizer update.

        Args:
            grads: Gradient updates.
            state: Current optimizer state.
            params: Current wavefunction parameters.
            **kwargs: Optimizer-specific arguments (e.g., ``batched_data``,
                ``rngs``).

        Returns:
            Tuple of (processed updates, new optimizer state).
        """


@runtime_checkable
class OptimizerLike(Protocol):
    """Protocol for optimizers.

    All optimizers expose ``init`` and ``update`` methods. The only
    positional argument required by both is ``params`` (for ``init``)
    or ``grads, state, params`` (for ``update``). Runtime deps like
    ``f_log_psi`` are wired via :func:`~jaqmc.utils.wiring.runtime_dep` fields before
    ``init`` is called. Call-time args (``batched_data``, ``rngs``)
    are passed via ``**kwargs``.
    """

    init: OptimizerInit
    update: OptimizerUpdate
