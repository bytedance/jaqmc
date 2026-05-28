# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

import optax

from jaqmc.array_types import Params
from jaqmc.data import BatchedData
from jaqmc.optimizer.schedule import Standard
from jaqmc.utils import parallel_jax
from jaqmc.utils.config import configurable_dataclass, module_config
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction.base import NumericWavefunctionEvaluate

from .jasr import robust_sr

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
        mixed_precision: Whether to use mixed precision for Gram factorization.
        score_chunk_size: Chunk size for score computation. If ``None``,
            full-batch score computation is used.
        score_norm_clip: Optional clip value for the mean absolute score per
            batch row. If ``None``, score clipping is disabled.
        gram_num_chunks: Number of chunks for Gram matrix computation. If
            ``None``, full-batch Gram computation is used.
        gram_dot_prec: Precision mode for Gram matrix dot products.
    """

    learning_rate: Any = module_config(Standard, direct_value_type=float)
    max_norm: Any = module_config(
        "fixed",
        direct_value_type=float | Literal["fixed"],
        module_import_base="jaqmc.optimizer",
    )
    damping: Any = module_config(
        1e-3, direct_value_type=float, module_import_base="jaqmc.optimizer"
    )
    max_cond_num: float | None = 1e7
    robust_gamma: Any = module_config(
        "sqrt",
        direct_value_type=float | Literal["sqrt"],
        module_import_base="jaqmc.optimizer",
    )
    spring_mu: Any = module_config(
        0.9, direct_value_type=float, module_import_base="jaqmc.optimizer"
    )
    march_beta: Any = module_config(
        0.995, direct_value_type=float, module_import_base="jaqmc.optimizer"
    )
    march_mode: Literal["var", "diff"] = "var"
    eps: float = 1e-8
    mixed_precision: bool = True
    score_chunk_size: int | None = 128
    score_norm_clip: float | None = None
    gram_num_chunks: int | None = 4
    gram_dot_prec: str | None = "F64"
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
            log_psi_fn=self.f_log_psi,
            learning_rate=self.learning_rate,
            max_norm=self.max_norm,
            damping=self.damping,
            max_cond_num=self.max_cond_num,
            robust_gamma=self.robust_gamma,
            spring_mu=self.spring_mu,
            march_beta=self.march_beta,
            march_mode=self.march_mode,
            eps=self.eps,
            mixed_precision=self.mixed_precision,
            score_in_axes=(batched_data.vmap_axis,),
            score_chunk_size=self.score_chunk_size,
            score_norm_clip=self.score_norm_clip,
            gram_num_chunks=self.gram_num_chunks,
            gram_dot_prec=self.gram_dot_prec,
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
