# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Protocol, runtime_checkable

from jaqmc.array_types import Params, PRNGKey, PyTree
from jaqmc.data import BatchedData


class OptimizerInit(Protocol):
    def __call__(
        self,
        params: Params,
        *,
        batched_data: BatchedData,
        rngs: PRNGKey,
        **kwargs: Any,
    ) -> PyTree:
        """Initialize optimizer state.

        Args:
            params: Wavefunction parameters.
            batched_data: Batched walker configurations.
            rngs: Random key for state initialization.
            **kwargs: Optimizer-specific arguments. Runtime deps like
                ``f_log_psi`` are wired via
                :func:`~jaqmc.utils.wiring.runtime_dep` fields, not passed here.

        Returns:
            Initial optimizer state.
        """


class OptimizerUpdate(Protocol):
    def __call__(
        self,
        grads: Params,
        state: PyTree,
        params: Params,
        *,
        batched_data: BatchedData,
        rngs: PRNGKey,
        **kwargs: Any,
    ) -> tuple[Params, PyTree]:
        """Apply optimizer update.

        Args:
            grads: Gradient updates.
            state: Current optimizer state.
            params: Current wavefunction parameters.
            batched_data: Batched walker configurations.
            rngs: Random key for this step.
            **kwargs: Optimizer-specific arguments.

        Returns:
            Tuple of (processed updates, new optimizer state).
        """


@runtime_checkable
class OptimizerLike(Protocol):
    """Protocol for optimizers.

    All optimizers expose ``init`` and ``update`` callables. The training
    stage always passes ``batched_data`` and ``rngs`` as keyword arguments;
    implementations that do not need them may absorb them with ``**kwargs``.
    Runtime deps like ``f_log_psi`` are wired via
    :func:`~jaqmc.utils.wiring.runtime_dep` fields before ``init`` is called.
    """

    @property
    def init(self) -> OptimizerInit: ...

    @property
    def update(self) -> OptimizerUpdate: ...
