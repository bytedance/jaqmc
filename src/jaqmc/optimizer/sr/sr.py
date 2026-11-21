# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from dataclasses import field
from functools import partial
from typing import Annotated, Any, Literal

import optax

from jaqmc.array_types import Params
from jaqmc.data import BatchedData
from jaqmc.optimizer.schedule import Standard
from jaqmc.utils import parallel_jax
from jaqmc.utils.config import ModuleConfig, configurable_dataclass
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction.base import NumericWavefunctionEvaluate

from .jasr import DotPrecision, SolvePrecision, robust_sr

__all__ = ["SROptimizer"]


@configurable_dataclass
class SROptimizer:
    """Robust stochastic reconfiguration optimizer.

    Stochastic reconfiguration (SR) is a natural-gradient method that updates
    parameters in wavefunction space rather than raw parameter space. This is
    often more stable than first-order optimizers for variational Monte Carlo,
    especially when small parameter changes can produce uneven changes in the
    wavefunction.

    JaQMC's SR optimizer uses a robust SR update together with two optional
    extensions:

    * SPRING adds momentum through ``spring_mu``.
    * MARCH adds an adaptive metric through ``march_beta`` and ``march_mode``.

    Use SR when you want an SR-style natural-gradient update instead of KFAC's
    structured approximation. The chunking options trade speed for lower memory
    use on larger systems.

    Args:
        wavefunction_precision: Wavefunction-defined precision mode passed to
            wavefunction evaluation. ``None`` inherits the wavefunction default.
        learning_rate: Step size (scalar or schedule).
        max_norm: Constrained update norm ``C`` (scalar or schedule). If
            ``"fixed"``, the update is normalized to the norm set by
            ``learning_rate``. If ``None``, only the learning-rate scaling is
            applied.
        damping: Damping ``lambda`` (scalar or schedule).
        max_cond_num: Maximum condition number for adaptive damping. If
            ``None``, adaptive damping is disabled.
        robust_gamma: Gamma factor for the robust update. ``"sqrt"`` uses
            ``1 / sqrt(damping)``, ``None`` uses standard SR
            ``1 / damping``, and scalar or schedule values are clipped into the
            stable range by the underlying solver.
        spring_mu: SPRING momentum coefficient ``mu`` (scalar or schedule).
            If ``None``, SPRING momentum is disabled.
        march_beta: Decay factor for the MARCH variance accumulator (scalar or
            schedule). If ``None``, the MARCH metric is disabled.
        march_mode: MARCH variance mode. ``"diff"`` uses update differences and
            ``"var"`` uses score variance along the batch axis.
        eps: Small numerical constant for stability.
        score_chunk_size: Chunk size for score computation. If ``None``,
            full-batch score computation is used.
        score_norm_clip: Optional clip value for the mean absolute score per
            batch row. If ``None``, score clipping is disabled.
        gram_num_chunks: Number of parameter chunks for Gram matrix
            computation. If ``None``, the chunk count is chosen automatically
            based on parameter count and device count (``1`` sets unchunked).
        gram_dot_prec: Precision mode for Gram matrix dot products (``"F64"``,
            ``"F32"``, ``"F16"``, or ``"F16_X3"``).
        gram_solve_prec: Precision mode for accumulating, regularizing, and
            inverting the Gram matrix (``"F64"``, ``"F32_PCG"``, or ``"F32"``).
    """

    learning_rate: Annotated[
        float | Callable[..., Any],
        ModuleConfig(package="jaqmc.optimizer"),
    ] = field(default_factory=Standard)
    max_norm: Annotated[
        float | Literal["fixed"] | Callable[..., Any] | None,
        ModuleConfig(package="jaqmc.optimizer"),
    ] = "fixed"
    damping: Annotated[
        float | Callable[..., Any],
        ModuleConfig(package="jaqmc.optimizer"),
    ] = 1e-3
    max_cond_num: float | None = 1e7
    robust_gamma: Annotated[
        float | Literal["sqrt"] | Callable[..., Any] | None,
        ModuleConfig(package="jaqmc.optimizer"),
    ] = "sqrt"
    spring_mu: Annotated[
        float | Callable[..., Any] | None,
        ModuleConfig(package="jaqmc.optimizer"),
    ] = 0.9
    march_beta: Annotated[
        float | Callable[..., Any] | None,
        ModuleConfig(package="jaqmc.optimizer"),
    ] = 0.995
    march_mode: Literal["var", "diff"] = "var"
    eps: float = 1e-8
    score_chunk_size: int | None = 128
    score_norm_clip: float | None = None
    gram_num_chunks: int | None = 4
    gram_dot_prec: DotPrecision = "F64"
    gram_solve_prec: SolvePrecision = "F64"
    wavefunction_precision: str | None = None
    f_log_psi: NumericWavefunctionEvaluate = runtime_dep()

    def init(
        self,
        params: Params,
        *,
        batched_data: BatchedData,
        **extra,
    ) -> optax.OptState:
        """Initialize the SR optimizer state.

        Returns:
            Initial optimizer state produced by the underlying Optax transform.
        """
        del extra
        self._sr_base = robust_sr(
            log_psi_fn=partial(self.f_log_psi, precision=self.wavefunction_precision),
            learning_rate=self.learning_rate,
            max_norm=self.max_norm,
            damping=self.damping,
            max_cond_num=self.max_cond_num,
            robust_gamma=self.robust_gamma,
            spring_mu=self.spring_mu,
            march_beta=self.march_beta,
            march_mode=self.march_mode,
            eps=self.eps,
            score_in_axes=(batched_data.vmap_axis,),
            score_chunk_size=self.score_chunk_size,
            score_norm_clip=self.score_norm_clip,
            gram_num_chunks=self.gram_num_chunks,
            gram_dot_prec=self.gram_dot_prec,
            gram_solve_prec=self.gram_solve_prec,
            axis_name=parallel_jax.BATCH_AXIS_NAME,
            prune_inactive=False,
        )
        return self._sr_base.init(params)

    def update(
        self,
        grads: optax.Updates,
        state: optax.OptState,
        params: Params,
        *,
        batched_data: BatchedData,
        **extra,
    ) -> tuple[optax.Updates, optax.OptState]:
        """Apply one SR update.

        Returns:
            Tuple of processed updates and the new optimizer state.

        Raises:
            RuntimeError: If :meth:`init` has not been called yet.
        """
        del extra
        if not hasattr(self, "_sr_base"):
            raise RuntimeError("SROptimizer.init must be called before update")
        return self._sr_base.update(grads, state, params, data=batched_data.data)


SROptimizer.__module__ = __package__
