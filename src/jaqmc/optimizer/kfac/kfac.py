# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from typing import Any, NamedTuple

import jax
import kfac_jax
import optax

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import BatchedData
from jaqmc.optimizer.schedule import Standard
from jaqmc.utils.config import configurable_dataclass, module_config
from jaqmc.utils.parallel_jax import BATCH_AXIS_NAME
from jaqmc.utils.wiring import runtime_dep
from jaqmc.wavefunction import NumericWavefunctionEvaluate

from .complex_support import patch_block_diagonal_curvature
from .curvature_blocks import make_tag_to_block_ctor
from .tag_registration import make_graph_patterns

__all__ = ["KFACOptimizer"]


class KFACState(NamedTuple):
    optax: optax.OptState
    kfac: kfac_jax.OptaxPreconditionState


@configurable_dataclass
class KFACOptimizer:
    r"""Kronecker-Factored Approximate Curvature (KFAC) optimizer.

    KFAC `[arXiv:1503.05671] <https://arxiv.org/abs/1503.05671>`_ is a second-order
    optimization technique. It employs a Kronecker product structure to approximate the
    natural gradient descent, which considers the geometry of the parameter space during
    optimization. For more details of the Kronecker product structure, please refer to
    `[arXiv:2507.05127] <http://arxiv.org/abs/2507.05127>`_ for comprehensive tutorial.

    Natural gradient descent updates for optimizing loss :math:`\mathcal{L}` with
    respect to parameters :math:`\theta` have the form :math:`\delta \theta \propto
    \mathcal{F}^{-1} \nabla_\theta \mathcal{L}(\theta)`, where :math:`\mathcal{F}` is
    the Fisher Information Matrix (FIM):

    .. math::
        \mathcal{F}_{i j}=\mathbb{E}_{p(\mathbf{X})}\left[
            \frac{\partial \log p(\mathbf{X})}{\partial \theta_i}
            \frac{\partial \log p(\mathbf{X})}{\partial \theta_j}
        \right].

    For real-valued wavefunctions, :math:`p(\mathbf{X}) \propto \psi^2(\mathbf{X})`,
    it gives the same formula as stochastic reconfiguration (Appendix C of `[Phys. Rev.
    Research 2, 033429 (2020)] <https://doi.org/10.1103/PhysRevResearch.2.033429>`_).

    However, for complex-valued wavefunctions, the natural gradient descent is deviating
    from stochastic reconfiguration, but KFAC can still be used to approximate
    stochastic reconfigurations. In stochastic reconfigurations, the parameters updates
    follow :math:`\delta \theta \propto \operatorname{Re} [S]^{-1} \operatorname{Re}
    \left[\nabla_\theta \mathcal{L}(\theta)\right]`, where

    .. math::
        S_{ij} = \mathbb{E}_{|\psi|^2(\mathbf{X})}\left[
            \frac{\partial \log \psi^*}{\partial \theta_i}
            \frac{\partial \log \psi}{\partial \theta_j}
        \right].

    We only show the uncentered version (i.e. assuming normalized wavefunction) for
    simplicity. The reason why we are using :math:`\operatorname{Re}` for :math:`S` and
    :math:`\nabla_\theta \mathcal{L}(\theta)` is that parameter updates must be real.
    See "Complex neural networks" paragraph of `[Nat. Phys. 20, 1476 (2024)]
    <https://doi.org/10.1038/s41567-024-02566-1>`_.

    The key difference between the :math:`S` matrix and FIM is the complex conjugate on
    the left. The version of KFAC included in JaQMC accounts for this by patching some
    internal parts of original `kfac_jax <https://github.com/google-deepmind/kfac-jax>`_.

    Args:
        learning_rate: The learning rate.
        norm_constraint: The update is scaled down so that its approximate squared
            Fisher norm :math:`v^T F v` is at most the specified value.
        curvature_ema: Decay factor used when calculating the covariance estimate
            moving averages.
        l2_reg: Tell the optimizer what L2 regularization coefficient you are using.
        inverse_update_period: Number of steps in between updating the inverse curvature
            approximation.
        damping: Fixed damping parameter.
    """

    learning_rate: Any = module_config(Standard)
    norm_constraint: float = 1e-3
    curvature_ema: float = 0.95
    l2_reg: float = 0.0
    inverse_update_period: int = 1
    damping: float = 1e-3
    f_log_psi: NumericWavefunctionEvaluate = runtime_dep()

    def init(
        self,
        params: Params,
        *,
        batched_data: BatchedData,
        rngs: PRNGKey,
        **extra,
    ) -> KFACState:
        """Initialize the KFAC optimizer state.

        Args:
            params: Wavefunction parameters.
            batched_data: Batched electron configurations.
            rngs: Random key for preconditioner initialization.
            **extra: Ignored extra keyword arguments.

        Returns:
            Initial KFAC optimizer state.
        """
        del extra
        self._build(self.f_log_psi)
        return KFACState(
            self._tx.init(params),
            self._preconditioner.init((params, batched_data), rngs),
        )

    def update(
        self,
        grads: optax.Updates,
        state: optax.OptState,
        params: Params,
        *,
        batched_data: BatchedData,
        rngs: PRNGKey,
        **extra,
    ) -> tuple[optax.Updates, KFACState]:
        """Apply KFAC update.

        Args:
            grads: Gradient updates.
            state: Current optimizer state.
            params: Current wavefunction parameters.
            batched_data: Batched electron configurations.
            rngs: Random key for preconditioner update.
            **extra: Ignored extra keyword arguments.

        Returns:
            Tuple of (processed updates, new optimizer state).

        Raises:
            RuntimeError: If ``init`` has not been called.
        """
        del extra
        if not hasattr(self, "_tx"):
            raise RuntimeError("KFACOptimizer.init must be called before update")
        opt_state, precond_state = state
        precond_state = self._preconditioner.maybe_update(
            precond_state, (params, batched_data), rngs
        )
        precond_state = self._preconditioner.increment_count(precond_state)
        grads, opt_state = self._tx.update(
            grads, opt_state, precond_state=precond_state
        )
        return grads, KFACState(opt_state, precond_state)

    def _build(self, f_log_psi: NumericWavefunctionEvaluate):
        """Build the preconditioner and optax chain from config fields.

        Args:
            f_log_psi: The wavefunction returning logpsi.
        """

        def value_func(params: Params, batched_data: BatchedData):
            batch_log_psi = jax.vmap(f_log_psi, in_axes=(None, batched_data.vmap_axis))
            logpsi = batch_log_psi(params, batched_data.data)
            kfac_jax.register_normal_predictive_distribution(logpsi[:, None])
            return logpsi

        self._preconditioner = kfac_jax.OptaxPreconditioner(
            value_func,
            l2_reg=self.l2_reg,
            damping=self.damping,
            norm_constraint=self.norm_constraint,
            curvature_ema=self.curvature_ema,
            inverse_update_period=self.inverse_update_period,
            estimation_mode="fisher_exact",
            pmap_axis_name=BATCH_AXIS_NAME,
            layer_tag_to_block_ctor=make_tag_to_block_ctor(),
            auto_register_kwargs=dict(graph_patterns=make_graph_patterns()),
            # The default batch_size_extractor is not smart enough. Since we already
            # know the batch_size now, we just hard code it into the preconditioner.
            batch_size_extractor=lambda batched_data: batched_data.batch_size,  # type: ignore
        )
        patch_block_diagonal_curvature(self._preconditioner.estimator)
        # We have to put the scale_by_learning_rate in front of the preconditioner,
        # because the preconditioner puts Fisher norm constaints according to the
        # update norm, in assumption that the learning rate is already applied.
        self._tx = optax.chain(
            optax.scale_by_learning_rate(self.learning_rate),
            self._preconditioner.as_gradient_transform(),
        )


KFACOptimizer.__module__ = __package__
