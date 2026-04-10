# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Variational Monte Carlo work stage for sampling and training."""

from dataclasses import dataclass, replace
from operator import itemgetter
from typing import Any, ClassVar

import jax
import optax
from jax import numpy as jnp

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import BatchedData
from jaqmc.estimator.base import Estimator
from jaqmc.estimator.loss_grad import LossAndGrad
from jaqmc.optimizer.base import OptimizerLike
from jaqmc.optimizer.kfac import KFACOptimizer
from jaqmc.utils import parallel_jax
from jaqmc.utils.config import configurable_dataclass
from jaqmc.utils.wiring import check_wired, wire
from jaqmc.wavefunction.base import NumericWavefunctionEvaluate
from jaqmc.writer import Writers

from .base import StageAbort, WorkStageConfig
from .sampling import SamplingStageBuilder, SamplingState, SamplingWorkStage


@configurable_dataclass
class VMCWorkStageConfig(WorkStageConfig):
    """Configuration for VMC work stages.

    Args:
        stop_on_nan: Abort training when NaN is detected in step statistics.
            ``True`` checks all stat keys, ``False`` disables the check,
            or pass a comma-separated string of specific keys to monitor
            (e.g. ``"loss"``).
    """

    stop_on_nan: bool | str = "loss"
    burn_in: int = 100


class VMCState(SamplingState):
    """State for VMC work stages."""

    opt_state: Any


class VMCStageBuilder(SamplingStageBuilder):
    """Builder for VMC work stages.

    Extends :class:`~jaqmc.workflow.stage.sampling.SamplingStageBuilder`
    with optimizer and loss gradient
    configuration. Call :meth:`build` to create a fully-configured
    :class:`VMCWorkStage`.
    """

    config_class: ClassVar[type[VMCWorkStageConfig]] = VMCWorkStageConfig
    config: VMCWorkStageConfig

    def configure_optimizer(
        self, *, default: str | type, f_log_psi: NumericWavefunctionEvaluate, **kwargs
    ) -> None:
        """Configure the optimizer.

        Args:
            default: Object or its module path for the default optimizer.
            f_log_psi: Log wavefunction for optimizer wiring.
            **kwargs: Additional keyword arguments for optimizer wiring.

        Raises:
            TypeError: If configured optimizer is not an
                :class:`~jaqmc.optimizer.base.OptimizerLike`.
        """
        opt = self.cfg.get_module("optim", default)
        if not isinstance(opt, OptimizerLike):
            raise TypeError(f"{self.name}: optimizer must be an OptimizerLike.")
        wire(opt, f_log_psi=f_log_psi, **kwargs)
        self.optimizer = opt

    def configure_loss_grads(
        self,
        loss_grads: type[Estimator] | Estimator | None = LossAndGrad,
        *,
        f_log_psi: NumericWavefunctionEvaluate,
    ) -> None:
        """Configure loss gradient estimator and add to estimator dict.

        Args:
            f_log_psi: Log wavefunction for loss gradient wiring.
            loss_grads: A class (resolved from config and wired), an
                already-wired instance, or None (no loss grads).
                Defaults to :class:`~jaqmc.estimator.loss_grad.LossAndGrad`.

        Raises:
            ValueError: If :meth:`configure_estimators` was not called
                before this method.
        """
        if not hasattr(self, "estimators"):
            raise ValueError(
                f"{self.name}: configure_estimators() must be called "
                "before configure_loss_grads()."
            )
        if loss_grads is None:
            return
        if isinstance(loss_grads, type):
            est = self.cfg.get("grads", loss_grads)
            wire(est, f_log_psi=f_log_psi)
            self.estimators["grads"] = est
        else:
            wire(loss_grads, f_log_psi=f_log_psi)
            check_wired(loss_grads, label="loss_grads")
            self.estimators["grads"] = loss_grads

    def ensure_configured(self) -> None:
        """Auto-resolve defaults and validate.

        Raises:
            ValueError: If required configure methods were not called.
        """
        super().ensure_configured()
        if not hasattr(self, "optimizer"):
            raise ValueError(f"{self.name}: configure_optimizer() was not called.")
        if "grads" not in self.estimators:
            raise ValueError(
                f"{self.name}: configure_loss_grads() was not called. "
                "A VMC training stage requires loss gradients. Either call "
                "configure_loss_grads() or pass a 'grads' estimator to "
                "configure_estimators()."
            )

    def build(self) -> "VMCWorkStage":
        """Build a fully-configured :class:`VMCWorkStage`.

        Returns:
            A fully-configured :class:`VMCWorkStage`.
        """
        self.ensure_configured()
        return VMCWorkStage(
            config=self.config,
            name=self.name,
            wavefunction=self.wavefunction,
            sample_plan=self.sample_plan,
            estimators=self.estimators,
            writers=self.writers,
            optimizer=self.optimizer,
        )


@dataclass(kw_only=True, eq=False)
class VMCWorkStage(SamplingWorkStage):
    """Variational Monte Carlo work stage for sampling and training.

    Performs MCMC sampling, observable estimation, and parameter optimization.
    For evaluation without training, use
    :class:`~jaqmc.workflow.stage.evaluation.EvaluationWorkStage` instead.

    Usage::

        builder = VMCWorkStage.builder(cfg.scoped("train"), wf)
        sampler = cfg.get("sampler", MCMCSampler)
        builder.configure_sample_plan(wf.logpsi, {"electrons": sampler})
        builder.configure_optimizer(default="jaqmc.optimizer.kfac", f_log_psi=wf.logpsi)
        builder.configure_estimators(kinetic=..., potential=..., total=...)
        builder.configure_loss_grads(f_log_psi=wf.logpsi)
        train = builder.build()
    """

    config: VMCWorkStageConfig
    name: str
    writers: Writers
    optimizer: OptimizerLike

    builder: ClassVar[type[VMCStageBuilder]] = VMCStageBuilder

    def create_state(  # type: ignore[override]
        self,
        rngs: PRNGKey,
        *,
        params: Params | None = None,
        batched_data: BatchedData,
        sampler_state: Any | None = None,
    ) -> VMCState:
        """Create sharded VMC state.

        Args:
            rngs: Initial random seed for all random operations.
            params: Pre-existing wavefunction parameters (already sharded).
            batched_data: Sharded electron configurations.
            sampler_state: Pre-existing sampler state (already sharded).

        Returns:
            Sharded VMC state.
        """
        base_rngs, opt_rngs = jax.random.split(rngs)
        base = super().create_state(
            base_rngs,
            params=params,
            batched_data=batched_data,
            sampler_state=sampler_state,
        )

        local_data = jax.tree.map(parallel_jax.addressable_data, batched_data)
        # KFAC requires device-local parameters to construct graphs
        # Optax optimizer don't need this and just shard-in shard-out
        opt_state = jax.device_put(
            self.optimizer.init(
                jax.tree.map(parallel_jax.addressable_data, base.params)
                if isinstance(self.optimizer, KFACOptimizer)
                else base.params,
                batched_data=local_data,
                rngs=opt_rngs,
            ),
            parallel_jax.make_sharding(parallel_jax.SHARE_PARTITION),
        )

        return VMCState(
            params=base.params,
            batched_data=base.batched_data,
            sampler_state=base.sampler_state,
            estimator_state=base.estimator_state,
            opt_state=opt_state,
        )

    def compute_step(
        self, state: VMCState, rngs: PRNGKey
    ) -> tuple[dict[str, Any], VMCState]:
        """Sample, estimate, compute gradients, and update parameters.

        Returns:
            Tuple of (finalized stats, updated state with new params).

        Raises:
            ValueError: If no estimator provides ``grads``.
        """
        sampler_rngs, est_rngs, opt_rngs = jax.random.split(rngs, 3)
        data, sampler_stats, sampler_state = self.sample_plan.step(
            state.params, state.batched_data, state.sampler_state, sampler_rngs
        )
        step_stats, estimator_state = self.estimators.evaluate(
            state.params, data, state.estimator_state, est_rngs
        )
        batched = jax.tree.map(itemgetter(None), step_stats)
        final_stats = self.estimators.finalize_stats(batched, estimator_state)
        grads = final_stats.pop("grads", None)
        if grads is None:
            raise ValueError("None of the estimators provides `grads` stats.")
        updates, opt_state = self.optimizer.update(
            grads,
            state.opt_state,
            params=state.params,
            batched_data=data,
            rngs=opt_rngs,
        )
        params = optax.apply_updates(state.params, updates)
        return {**final_stats, **sampler_stats}, replace(
            state,
            params=params,
            batched_data=data,
            sampler_state=sampler_state,
            estimator_state=estimator_state,
            opt_state=opt_state,
        )

    def _has_nan(self, stats: dict[str, Any]) -> bool:
        if not self.config.stop_on_nan:
            return False
        return any(
            jnp.isnan(stats[key].real).any()
            for key in stats
            if self.config.stop_on_nan is True
            or key in self.config.stop_on_nan.split(",")
        )

    def loop(self, state: VMCState, initial_step: int, rngs):
        """Yield ``(step, state)`` for each VMC iteration.

        Handles KFAC check_vma override, JIT compilation, burn-in,
        then yields after each compute step. Writes stats and raises
        :class:`~jaqmc.workflow.stage.base.StageAbort` on NaN detection.

        Raises:
            StageAbort: When NaN is detected in stats.
        """
        # KFAC is incompatible with check_vma
        check_vma = self.config.check_vma
        if isinstance(self.optimizer, KFACOptimizer) and check_vma:
            self.logger.warning("Disabling check_vma (incompatible with KFAC).")
            check_vma = False

        partition = state.partition()
        split_rngs = parallel_jax.jit_sharded(
            lambda r: tuple(jax.random.split(r)),
            in_specs=parallel_jax.DATA_PARTITION,
            out_specs=parallel_jax.DATA_PARTITION,
        )
        compute = parallel_jax.jit_sharded(
            self.compute_step,
            in_specs=(partition, parallel_jax.DATA_PARTITION),
            out_specs=(parallel_jax.SHARE_PARTITION, partition),
            check_vma=check_vma,
            donate_argnums=0,
        )

        if initial_step == 0:
            state, rngs = self.burn_in(state, rngs)

        # Main loop
        for step in range(initial_step, self.config.iterations):
            rngs, sub_rngs = split_rngs(rngs)
            stats, state = compute(state, sub_rngs)
            self.writers.write(step, stats)
            if self._has_nan(stats):
                raise StageAbort(step, state)
            yield step, state
