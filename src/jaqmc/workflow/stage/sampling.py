# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Base builder and stage for sampling-based work stages."""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Self

import jax
from upath import UPath

from jaqmc.array_types import Params, PRNGKey
from jaqmc.data import BatchedData
from jaqmc.estimator import Estimator, EstimatorPipeline
from jaqmc.estimator.base import EstimatorCollection
from jaqmc.sampler.base import SamplePlan, SamplerLike
from jaqmc.utils import parallel_jax
from jaqmc.utils.checkpoint import NumPyCheckpointManager
from jaqmc.utils.jax_dataclass import JAXDataclassMeta
from jaqmc.wavefunction import WavefunctionLike
from jaqmc.wavefunction.base import NumericWavefunctionEvaluate
from jaqmc.writer import Writer, Writers

from .base import StageState, WorkStage, WorkStageConfig

logger = logging.getLogger(__name__)


class SamplingState(StageState, metaclass=JAXDataclassMeta):
    """Base state for sampling-based work stages.

    Contains the common fields shared across VMC and evaluation stages.
    Subclasses can add extra fields
    (e.g. :class:`~jaqmc.workflow.stage.vmc.VMCState` adds ``opt_state``).
    """

    params: Params
    batched_data: BatchedData
    sampler_state: Any
    estimator_state: Any

    def partition(self) -> Self:
        """Return a matching pytree of :class:`~jax.sharding.PartitionSpec`."""
        return replace(
            self,
            **{  # type: ignore[arg-type]
                name: jax.sharding.PartitionSpec()
                if not isinstance(s, BatchedData)
                else s.partition_spec
                for name, s in self.__dict__.items()
            },
        )

    def process_allgather(self) -> Self:
        return replace(self, batched_data=self.batched_data.process_allgather())


class SamplingStageBuilder:
    """Base builder for sampling-based work stages.

    Provides the progressive ``configure_*`` API. After calling
    :meth:`build`, the resulting stage is guaranteed fully configured.

    Usage::

        builder = VMCWorkStage.builder(run_config, wf, name="train")
        builder.configure_sample_plan(wf.logpsi, {"electrons": sampler})
        builder.configure_optimizer(optimizer, f_log_psi=wf.logpsi)
        builder.configure_estimators({"kinetic": ..., "potential": ..., "total": ...})
        builder.configure_loss_grads(loss_grads, f_log_psi=wf.logpsi)
        builder.configure_writers(writers)
        stage = builder.build()

    Args:
        config: Resolved stage run configuration.
        wavefunction: Wavefunction instance.
        name: Stage name.
    """

    def __init__(
        self,
        config: WorkStageConfig,
        wavefunction: WavefunctionLike,
        *,
        name: str,
    ):
        self.name = name
        self.config = config
        self.wavefunction = wavefunction

    def configure_sample_plan(
        self,
        f_log_amplitude: NumericWavefunctionEvaluate,
        samplers: Mapping[str | tuple[str, ...], SamplerLike] | None = None,
    ) -> None:
        """Configure MCMC sampling.

        Args:
            f_log_amplitude: Log amplitude for sampling.
            samplers: Mapping from data field to corresponding sampler.
        """
        self.sample_plan = SamplePlan(f_log_amplitude, samplers)

    def configure_estimators(
        self, estimators: EstimatorCollection | Mapping[str, Estimator]
    ) -> None:
        """Configure estimators.

        Args:
            estimators: Named configured estimators.
        """
        self.estimators: dict[str, Estimator] = dict(estimators)

    def configure_writers(self, writers: Mapping[str, Writer | None]) -> None:
        """Configure writers from resolved writer instances.

        Args:
            writers: Named writers, where ``None`` disables an entry.
        """
        self.writers = Writers([writer for writer in writers.values() if writer])

    def ensure_configured(self) -> None:
        """Validate that every required configure method was called.

        Called by :meth:`build`.

        Raises:
            ValueError: If a required configure method was not called.
        """
        if not hasattr(self, "writers"):
            raise ValueError(f"{self.name}: configure_writers() was not called.")
        if not hasattr(self, "sample_plan"):
            raise ValueError(f"{self.name}: configure_sample_plan() was not called.")
        if not hasattr(self, "estimators"):
            raise ValueError(f"{self.name}: configure_estimators() was not called.")

    def build(self) -> WorkStage:
        """Build the fully-configured work stage.

        Raises:
            NotImplementedError: Subclasses must override this method.
        """
        raise NotImplementedError


@dataclass(kw_only=True, eq=False)
class SamplingWorkStage(WorkStage):
    """Base class for sampling-based work stages.

    Provides shared burn-in logic. Subclasses implement
    :meth:`compute_step` and :meth:`loop`.
    """

    sample_plan: SamplePlan
    estimators: EstimatorPipeline
    wavefunction: WavefunctionLike

    def restore_checkpoint(
        self,
        checkpoint_path: str | Path | UPath,
        template: Any,
        *,
        prefix: str = "",
        strict: bool = False,
    ):
        """Restore state from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file or directory.
            template: Template state for deserialization.
            prefix: Checkpoint filename prefix to match.
            strict: If ``True``, raise when no matching checkpoint can be
                restored.

        Returns:
            Restored state.
        """
        checkpoint_path = UPath(checkpoint_path)
        ckpt = NumPyCheckpointManager(checkpoint_path, checkpoint_path, prefix=prefix)
        _, state = ckpt.restore(template, strict=strict)
        return state

    def create_state(  # type: ignore[override]
        self,
        rngs: PRNGKey,
        *,
        params: Params | None = None,
        batched_data: BatchedData,
        sampler_state: Any | None = None,
    ) -> SamplingState:
        """Create sharded sampling state.

        Args:
            rngs: Initial random seed for all random operations.
            params: Pre-existing wavefunction parameters (already sharded).
            batched_data: Sharded electron configurations.
            sampler_state: Pre-existing sampler state (already sharded).

        Returns:
            Sharded sampling state.
        """
        sampler_rngs, params_rngs, est_rngs = jax.random.split(rngs, 3)

        local_data = jax.tree.map(parallel_jax.addressable_data, batched_data)

        shared = parallel_jax.make_sharding(parallel_jax.SHARE_PARTITION)

        if params is None:
            local_params = self.wavefunction.init_params(
                local_data.unbatched_example(), params_rngs
            )
            params = jax.device_put(local_params, shared)

        local_sampler_state = self.sample_plan.init(local_data, sampler_rngs)
        if sampler_state is None:
            sampler_state = jax.device_put(local_sampler_state, shared)

        estimator_state = jax.device_put(
            self.estimators.init(local_data, est_rngs),
            parallel_jax.make_sharding(parallel_jax.SHARE_PARTITION),
        )

        return SamplingState(
            params=params,
            batched_data=batched_data,
            sampler_state=sampler_state,
            estimator_state=estimator_state,
        )

    def _run_burn_in(self, state: Any, rngs: PRNGKey) -> tuple[Any, PRNGKey]:
        """Execute the burn-in fori_loop (called inside jit_sharded).

        Returns:
            Tuple of (state, rngs) after burn-in steps.
        """
        sample_step = self.sample_plan.step

        def body(_, carry):
            state, rngs = carry
            rngs, sub_rngs = jax.random.split(rngs)
            data, _, sampler_state = sample_step(
                state.params,
                state.batched_data,
                state.sampler_state,
                sub_rngs,
            )
            return replace(state, batched_data=data, sampler_state=sampler_state), rngs

        return jax.lax.fori_loop(0, self.config.burn_in, body, (state, rngs))

    def burn_in(
        self,
        state: Any,
        rngs: PRNGKey,
    ) -> tuple[Any, PRNGKey]:
        """Run MCMC burn-in steps. No-op if ``config.burn_in <= 0``.

        Uses ``jax.lax.fori_loop`` to execute all burn-in steps in a
        single JIT-compiled call.

        Args:
            state: Sharded state.
            rngs: Sharded random keys.

        Returns:
            Tuple of (state, rngs) updated after burn-in.
        """
        if self.config.burn_in <= 0:
            return state, rngs

        partition = state.partition()
        jitted_burn_in = parallel_jax.jit_sharded(
            self._run_burn_in,
            in_specs=(partition, parallel_jax.DATA_PARTITION),
            out_specs=(partition, parallel_jax.DATA_PARTITION),
            check_vma=True,
            donate_argnums=0,
        )

        n_steps = self.config.burn_in
        logger.info("Start burn in %s steps.", n_steps)
        state, rngs = jitted_burn_in(state, rngs)
        logger.info("Burn in %s steps complete.", n_steps)

        return state, rngs
