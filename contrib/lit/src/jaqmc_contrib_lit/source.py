# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code="attr-defined"

"""Source construction and distillation for the molecular LIT workflow."""

from __future__ import annotations

import logging
from dataclasses import replace

import jax
import numpy as np
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree
from upath import UPath

from jaqmc.data import BatchedData
from jaqmc.sampler.base import SamplePlan
from jaqmc.utils import parallel_jax
from jaqmc_contrib_lit.common import (
    _ATOM_PARITY_PENDING_SECTOR_LABEL,
    _AXIS_NAMES,
    _two_spin_tuple,
)
from jaqmc_contrib_lit.optimization import (
    _apply_updates,
    _finite_source_distillation_stats,
    _log_spring_optimizer_diagnostics,
    _source_distillation_stats_from_log_ratios,
)
from jaqmc_contrib_lit.pool import (
    _batched_data_chunks,
    _concat_batched_data,
    _copy_matching_parameters,
    _cyclic_batched_data_chunk,
    _flatten_batched_tree,
    _indexed_batched_data_chunk,
    _load_batched_pool,
    _replicate_across_local_devices,
    _require_pool_walker_count,
    _save_batched_pool,
    _shard_batched_data_across_local_devices,
    _shard_rng_across_local_devices,
    _shuffled_batched_data_chunk,
)
from jaqmc_contrib_lit.response import (
    LITResponseFermiNet,
    molecular_electronic_dipole,
    parity_log_amplitude_loss,
    parity_project_log_amplitude,
)
from jaqmc_contrib_lit.sector import (
    SourceSector,
    _is_atom_hard_parity_sector,
    _is_atom_parity_sector,
    _is_identity_operation,
    _response_parity_character,
    discover_source_sector,
    transform_molecule_data,
)
from jaqmc_contrib_lit.state import (
    _AtomicParityResolution,
    _FidelityPlateauTracker,
    _SourceDistillationStats,
    _SpringState,
)

logger = logging.LoggerAdapter(
    logging.getLogger(__name__), extra={"category": "response"}
)


class SourceStageMixin:
    def _configured_source_sector(self, geometry_data) -> SourceSector:
        """Resolve the restricted atomic-hard-parity or molecular-C1 policy.

        Returns:
            ``atom_parity_pending`` with identity and inversion for exactly one
            nucleus, or the identity-only ``C1`` sector for a multi-nuclear
            geometry with no discovered nontrivial operation.  The atomic
            ground checkpoint later selects the opposite response parity.

        Raises:
            NotImplementedError: If a multi-nuclear geometry is not C1.
            RuntimeError: If atomic geometry discovery omits inversion.
        """
        sector = discover_source_sector(
            geometry_data.atoms,
            geometry_data.charges,
            tolerance=float(self.lit_config.ansatz.sector_tolerance),
        )
        atom_count = int(np.asarray(geometry_data.atoms).shape[0])
        if atom_count == 1:
            identity = next(
                operation
                for operation in sector.operations
                if _is_identity_operation(operation)
            )
            inversion = next(
                (
                    operation
                    for operation in sector.operations
                    if np.allclose(
                        np.asarray(operation),
                        -np.eye(3),
                        rtol=0.0,
                        atol=self.lit_config.ansatz.sector_tolerance,
                    )
                ),
                None,
            )
            if inversion is None:
                msg = "Atomic source-sector discovery did not contain inversion."
                raise RuntimeError(msg)
            resolved = replace(
                sector,
                operations=(identity, inversion),
                label=_ATOM_PARITY_PENDING_SECTOR_LABEL,
            )
        elif sector.is_trivial:
            resolved = replace(sector, label="C1")
        else:
            msg = (
                "LIT currently supports only one-center atoms with an "
                "automatically selected hard response parity and multi-center "
                "C1 molecules without "
                f"spatial symmetry; discovered n_atoms={atom_count}, "
                f"sector={sector.label!r}, order={sector.order}."
            )
            raise NotImplementedError(msg)

        return resolved

    def _resolve_atomic_parity(
        self,
        ground_logpsi,
        ground_params,
        batched_data: BatchedData,
        source_sector: SourceSector,
    ) -> _AtomicParityResolution:
        """Diagnose atomic ground parity and select the opposite response parity.

        C1 molecules have no hard spatial projector and return zero parity
        labels with non-applicable losses.  An atom is accepted only when the
        checkpoint is a clean inversion eigenstate on the held-out batch.

        Returns:
            The diagnosed ground parity, opposite response parity, and both
            held-out parity losses.  C1 returns zero characters and NaN losses.

        Raises:
            RuntimeError: If neither atomic inversion parity passes the hard
                admission threshold, the losses are invalid, or the result is
                ambiguous because both amplitudes vanish.
        """
        if not _is_atom_parity_sector(source_sector):
            return _AtomicParityResolution(
                0, 0, float("nan"), float("nan"), float("nan")
            )

        evaluation_batch = _cyclic_batched_data_chunk(
            batched_data,
            min(
                batched_data.batch_size,
                int(self.lit_config.ansatz.parity_eval_batch_size),
            ),
            0,
        )
        inversion = jnp.asarray(
            -np.eye(3),
            dtype=evaluation_batch.data.electrons.dtype,
        )
        symmetry_center = jnp.asarray(
            source_sector.center,
            dtype=evaluation_batch.data.electrons.dtype,
        )

        @jax.jit
        def evaluate(local_ground_params, local_batch):
            def paired_logs(data):
                return (
                    ground_logpsi(local_ground_params, data),
                    ground_logpsi(
                        local_ground_params,
                        transform_molecule_data(data, inversion, symmetry_center),
                    ),
                )

            log_amplitudes, inverted_log_amplitudes = jax.vmap(
                paired_logs,
                in_axes=(local_batch.vmap_axis,),
            )(local_batch.data)
            return (
                parity_log_amplitude_loss(
                    log_amplitudes,
                    inverted_log_amplitudes,
                    1,
                ),
                parity_log_amplitude_loss(
                    log_amplitudes,
                    inverted_log_amplitudes,
                    -1,
                ),
            )

        even_loss_array, odd_loss_array = jax.device_get(
            evaluate(ground_params, evaluation_batch)
        )
        even_loss = float(even_loss_array)
        odd_loss = float(odd_loss_array)
        ground_parity = 1 if even_loss <= odd_loss else -1
        selected_loss = even_loss if ground_parity == 1 else odd_loss
        opposite_loss = odd_loss if ground_parity == 1 else even_loss
        maximum = float(self.lit_config.ansatz.atomic_ground_parity_max_loss)
        logger.info(
            "Atomic ground parity diagnosis even_loss=%.6e odd_loss=%.6e "
            "selected=%s response=%s maximum=%.6e",
            even_loss,
            odd_loss,
            "even" if ground_parity == 1 else "odd",
            "odd" if ground_parity == 1 else "even",
            maximum,
        )
        if (
            not np.isfinite(even_loss)
            or not np.isfinite(odd_loss)
            or selected_loss > maximum
            or opposite_loss < 2.0 - maximum
        ):
            msg = (
                "Atomic ground checkpoint is not a clean inversion-parity "
                f"eigenstate: even_loss={even_loss:.6e}, "
                f"odd_loss={odd_loss:.6e}, required selected_loss <= "
                f"lit.ansatz.atomic_ground_parity_max_loss={maximum:.6e} and "
                "opposite_loss >= 2 - maximum. Retrain or explicitly project "
                "the ground state before computing its dipole response."
            )
            raise RuntimeError(msg)
        return _AtomicParityResolution(
            ground_parity,
            -ground_parity,
            even_loss,
            odd_loss,
            selected_loss,
        )

    def _validate_atomic_source_parity(
        self,
        ground_logpsi,
        ground_params,
        eval_pool: BatchedData,
        source_sector: SourceSector,
        source_centers,
        *,
        axis: int,
        response_parity: int = 0,
    ) -> float:
        """Validate an atomic dipole source against its diagnosed hard parity.

        C1 molecules have no spatial-sector preflight and return ``NaN``.

        Returns:
            The held-out atomic source parity loss, or ``NaN`` for C1.

        Raises:
            ValueError: If ``source_centers`` is not a Cartesian vector.
            RuntimeError: If the atomic parity loss is non-finite or exceeds
                the configured maximum.
        """
        if not _is_atom_hard_parity_sector(source_sector):
            return float("nan")

        evaluation_batch = _cyclic_batched_data_chunk(
            eval_pool,
            min(
                eval_pool.batch_size,
                int(self.lit_config.ansatz.parity_eval_batch_size),
            ),
            0,
        )
        if self._lit_data_parallel_enabled():
            self._validate_data_parallel_batch(
                evaluation_batch,
                purpose="atomic source parity evaluation",
            )
            ground_params = _replicate_across_local_devices(ground_params)
            evaluation_batch = _shard_batched_data_across_local_devices(
                evaluation_batch
            )
        centers = jnp.asarray(
            source_centers,
            dtype=evaluation_batch.data.electrons.dtype,
        )
        if centers.shape != (3,):
            msg = f"source_centers must have shape (3,), got {centers.shape}."
            raise ValueError(msg)
        expected_parity = _response_parity_character(source_sector)
        if response_parity != expected_parity:
            msg = (
                "Atomic source validation requires resolved response parity "
                f"{expected_parity:+d}, got {response_parity!r}."
            )
            raise ValueError(msg)
        active_operations = tuple(
            operation
            for operation in source_sector.operations
            if not _is_identity_operation(operation)
        )
        if len(active_operations) != 1 or not np.allclose(
            np.asarray(active_operations[0]),
            -np.eye(3),
            rtol=0.0,
            atol=self.lit_config.ansatz.sector_tolerance,
        ):
            msg = (
                "A resolved atomic source sector must contain exactly one "
                "non-identity inversion operation."
            )
            raise RuntimeError(msg)

        inversion = jnp.asarray(
            active_operations[0],
            dtype=evaluation_batch.data.electrons.dtype,
        )
        symmetry_center = jnp.asarray(
            source_sector.center,
            dtype=evaluation_batch.data.electrons.dtype,
        )

        def pure_source_scalar_apply(local_ground_params, data):
            ground_log_amplitude = ground_logpsi(local_ground_params, data)
            complex_dtype = jnp.result_type(ground_log_amplitude, jnp.complex64)
            source_factor = jnp.asarray(
                molecular_electronic_dipole(data, axis) - centers[axis],
                dtype=complex_dtype,
            )
            return jnp.asarray(
                ground_log_amplitude,
                dtype=complex_dtype,
            ) + jnp.log(source_factor)

        @jax.jit
        def evaluate_parity(local_ground_params, local_evaluation_batch):
            def paired_logs(data):
                return (
                    pure_source_scalar_apply(local_ground_params, data),
                    pure_source_scalar_apply(
                        local_ground_params,
                        transform_molecule_data(data, inversion, symmetry_center),
                    ),
                )

            source_logs, inverted_source_logs = jax.vmap(
                paired_logs,
                in_axes=(local_evaluation_batch.vmap_axis,),
            )(local_evaluation_batch.data)
            return parity_log_amplitude_loss(
                source_logs,
                inverted_source_logs,
                response_parity,
            )

        loss_array = jax.device_get(evaluate_parity(ground_params, evaluation_batch))
        loss = float(loss_array)
        maximum = float(self.lit_config.ansatz.atomic_source_parity_max_loss)
        logger.info(
            "axis=%s pure_source_heldout_parity=%+.0f loss=%.6e maximum=%.6e",
            _AXIS_NAMES[axis],
            float(response_parity),
            loss,
            maximum,
        )
        if not np.isfinite(loss) or loss > maximum:
            msg = (
                f"axis={_AXIS_NAMES[axis]} pure-source held-out parity loss "
                f"{loss:.6e} exceeds "
                "lit.ansatz.atomic_source_parity_max_loss="
                f"{maximum:.6e} for response parity {response_parity:+d}. "
                "The sampled (D-D0)Psi0 source is outside the diagnosed "
                "atomic response sector; check the ground checkpoint, source "
                "center, and source-pool equilibration."
            )
            raise RuntimeError(msg)
        return loss

    def _make_response_ansatz(  # noqa: C901
        self,
        example,
        response_rng,
        ground_params,
        *,
        source_sector: SourceSector | None = None,
        response_parity: int = 0,
    ):
        """Create the independent PRL-style response NQS.

        The production parameter tree is exactly the raw response-network
        parameter tree.  In a hard atomic sector the raw network is initialized
        independently rather than copied from the ground state: copying a
        nearly pure ground-state parity and projecting onto the opposite parity
        produces a numerically singular zero state.  The subsequent fixed
        ``pi_Phi`` distillation supplies the physically useful initialization.
        For a symmetry-free C1 molecule, matching ground-state parameters are
        still a useful nonsingular starting point.

        Returns:
            Scalar response apply function and the direct response-network
            parameter tree.

        Raises:
            ValueError: If the resolved atom/C1 symmetry policy is inconsistent.
        """
        if source_sector is None:
            msg = (
                "A response symmetry policy resolved from the physical fixed "
                "geometry is required."
            )
            raise ValueError(msg)
        hard_atomic_parity = _is_atom_hard_parity_sector(source_sector)
        if _is_atom_parity_sector(source_sector) and not hard_atomic_parity:
            msg = "Atomic response parity must be diagnosed before ansatz creation."
            raise ValueError(msg)
        if hard_atomic_parity:
            expected_parity = _response_parity_character(source_sector)
            if response_parity != expected_parity:
                msg = (
                    "Atomic response ansatz requires resolved parity "
                    f"{expected_parity:+d}, got {response_parity!r}."
                )
                raise ValueError(msg)
        elif response_parity != 0:
            msg = "A C1 response must not receive a hard parity character."
            raise ValueError(msg)
        response = LITResponseFermiNet(
            nspins=_two_spin_tuple(self.system_config.electron_spins),
            ndets=int(self.lit_config.ansatz.determinants),
            hidden_dims_single=tuple(self.lit_config.ansatz.hidden_dims_single),
            hidden_dims_double=tuple(self.lit_config.ansatz.hidden_dims_double),
            use_last_layer=bool(self.lit_config.ansatz.use_last_layer),
            envelope=self.lit_config.ansatz.envelope,
            orbitals_spin_split=bool(self.lit_config.ansatz.orbitals_spin_split),
        )
        raw_params = response.init(response_rng, example)
        if not hard_atomic_parity:
            raw_params = _copy_matching_parameters(raw_params, ground_params)

        inversion = jnp.asarray(-np.eye(3), dtype=example.electrons.dtype)
        symmetry_center = jnp.asarray(
            source_sector.center,
            dtype=example.electrons.dtype,
        )

        def inverted_data(data):
            return transform_molecule_data(data, inversion, symmetry_center)

        def raw_apply(params, data):
            return response.apply(params, data)

        def projected_raw_apply(params, data):
            raw_logpsi = raw_apply(params, data)
            if not hard_atomic_parity:
                return raw_logpsi
            return parity_project_log_amplitude(
                raw_logpsi,
                raw_apply(params, inverted_data(data)),
                response_parity,
            )

        return projected_raw_apply, raw_params

    def _prepare_source_sampler(
        self,
        sampler,
        batched_data,
        ground_params,
        ground_logpsi,
        rng,
        *,
        axis: int,
        source_center: float,
    ):
        source_plan = SamplePlan(
            self._make_source_log_amplitude(axis, source_center, ground_logpsi),
            {"electrons": sampler},
        )
        rng, source_rng = jax.random.split(rng)
        source_state = source_plan.init(batched_data, source_rng)
        if self._lit_data_parallel_enabled():
            self._validate_data_parallel_batch(
                batched_data,
                purpose="source sampling",
            )
            source_step = self._source_sample_step_kernel(
                source_plan,
                batched_data,
            )
            ground_params = _replicate_across_local_devices(ground_params)
            batched_data = _shard_batched_data_across_local_devices(batched_data)
            source_state = _replicate_across_local_devices(source_state)
            for _ in range(self.lit_config.source.burn_in):
                rng, source_rng = jax.random.split(rng)
                source_rngs = _shard_rng_across_local_devices(source_rng)
                batched_data, _, source_state = source_step(
                    ground_params,
                    batched_data,
                    source_state,
                    source_rngs,
                )
            return source_plan, source_state, batched_data, rng

        for _ in range(self.lit_config.source.burn_in):
            rng, source_rng = jax.random.split(rng)
            batched_data, _, source_state = source_plan.step(
                ground_params,
                batched_data,
                source_state,
                source_rng,
            )
        return source_plan, source_state, batched_data, rng

    def _collect_sample_pool(
        self,
        sample_plan: SamplePlan,
        params,
        batched_data,
        sampler_state,
        rng,
        *,
        batches: int,
        stride: int | None = None,
    ):
        pool = []
        stride = max(
            1,
            int(self.lit_config.source.pool_stride if stride is None else stride),
        )
        if self._lit_data_parallel_enabled():
            self._validate_data_parallel_batch(
                batched_data,
                purpose="source sampling",
            )
            sample_step = self._source_sample_step_kernel(sample_plan, batched_data)
            params = _replicate_across_local_devices(params)
            batched_data = _shard_batched_data_across_local_devices(batched_data)
            sampler_state = _replicate_across_local_devices(sampler_state)
            for _ in range(max(1, int(batches))):
                for _ in range(stride):
                    rng, sample_rng = jax.random.split(rng)
                    sample_rngs = _shard_rng_across_local_devices(sample_rng)
                    batched_data, _, sampler_state = sample_step(
                        params,
                        batched_data,
                        sampler_state,
                        sample_rngs,
                    )
                pool.append(batched_data)
            return _concat_batched_data(pool), batched_data, sampler_state, rng

        for _ in range(max(1, int(batches))):
            for _ in range(stride):
                rng, sample_rng = jax.random.split(rng)
                batched_data, _, sampler_state = sample_plan.step(
                    params,
                    batched_data,
                    sampler_state,
                    sample_rng,
                )
            pool.append(batched_data)
        return _concat_batched_data(pool), batched_data, sampler_state, rng

    def _source_sample_step_kernel(
        self,
        sample_plan: SamplePlan,
        batched_data: BatchedData,
    ):
        """Compile one source MCMC step over process-local devices.

        Returns:
            A kernel with sharded walkers and RNGs, replicated parameters and
            sampler state, and globally reduced sampler statistics.
        """
        cache = getattr(self, "_source_sample_step_kernel_cache", None)
        if cache is None:
            cache = {}
            self._source_sample_step_kernel_cache = cache
        cache_key = id(sample_plan)
        kernel = cache.get(cache_key)
        if kernel is not None:
            return kernel

        device_count = jax.local_device_count()
        logger.info(
            "Compiling LIT source sampling data parallelism devices=%d "
            "global_batch=%d local_batch=%d",
            device_count,
            int(batched_data.batch_size),
            int(batched_data.batch_size) // device_count,
        )
        kernel = parallel_jax.jit_sharded(
            sample_plan.step,
            in_specs=(
                parallel_jax.SHARE_PARTITION,
                batched_data.partition_spec,
                parallel_jax.SHARE_PARTITION,
                parallel_jax.DATA_PARTITION,
            ),
            out_specs=(
                batched_data.partition_spec,
                parallel_jax.SHARE_PARTITION,
                parallel_jax.SHARE_PARTITION,
            ),
            check_vma=True,
        )
        cache[cache_key] = kernel
        return kernel

    def _load_or_collect_source_pool(
        self,
        sample_plan: SamplePlan,
        params,
        batched_data,
        sampler_state,
        rng,
        *,
        axis: int,
        source_center: float,
        target_sha256: str,
        split: str,
        batches: int,
        pool_root: UPath | None = None,
    ):
        pool_path = self._source_pool_path(axis, split, root=pool_root)
        expected_walkers = self._expected_source_pool_walkers(batches)
        metadata = self._source_pool_metadata(
            axis,
            source_center,
            target_sha256=target_sha256,
            expected_walkers=expected_walkers,
        )
        if self.lit_config.source.reuse_pool and pool_path.exists():
            try:
                pool = _load_batched_pool(pool_path, batched_data, metadata=metadata)
                _require_pool_walker_count(
                    pool,
                    expected_walkers=expected_walkers,
                    split=split,
                )
                logger.info(
                    "Loaded %s source pool for axis=%s from %s",
                    split,
                    _AXIS_NAMES[axis],
                    pool_path,
                )
                return pool, batched_data, sampler_state, rng
            except (KeyError, ValueError, OSError) as exc:
                logger.warning(
                    "Ignoring incompatible %s source pool %s: %s",
                    split,
                    pool_path,
                    exc,
                )

        pool, batched_data, sampler_state, rng = self._collect_sample_pool(
            sample_plan,
            params,
            batched_data,
            sampler_state,
            rng,
            batches=batches,
        )
        _require_pool_walker_count(
            pool,
            expected_walkers=expected_walkers,
            split=split,
        )
        if self.lit_config.source.save_pool:
            _save_batched_pool(pool_path, pool, metadata=metadata)
            logger.info(
                "Saved %s source pool for axis=%s to %s",
                split,
                _AXIS_NAMES[axis],
                pool_path,
            )
        return pool, batched_data, sampler_state, rng

    def _try_load_source_pools(
        self,
        batched_data,
        *,
        axis: int,
        source_center: float,
        target_sha256: str,
        pool_root: UPath | None = None,
    ):
        if not self.lit_config.source.reuse_pool:
            return None
        loaded = []
        split_batches = (
            ("train", self.lit_config.source.train_pool_batches),
            ("eval", self.lit_config.source.eval_pool_batches),
        )
        for split, batches in split_batches:
            expected_walkers = self._expected_source_pool_walkers(batches)
            metadata = self._source_pool_metadata(
                axis,
                source_center,
                target_sha256=target_sha256,
                expected_walkers=expected_walkers,
            )
            pool_path = self._source_pool_path(axis, split, root=pool_root)
            if not pool_path.exists():
                return None
            try:
                pool = _load_batched_pool(pool_path, batched_data, metadata=metadata)
                _require_pool_walker_count(
                    pool,
                    expected_walkers=expected_walkers,
                    split=split,
                )
            except (KeyError, ValueError, OSError) as exc:
                logger.warning(
                    "Ignoring incompatible %s source pool %s: %s",
                    split,
                    pool_path,
                    exc,
                )
                return None
            logger.info(
                "Loaded %s source pool for axis=%s from %s",
                split,
                _AXIS_NAMES[axis],
                pool_path,
            )
            loaded.append(pool)
        return tuple(loaded)

    def _source_pool_path(
        self, axis: int, split: str, *, root: UPath | None = None
    ) -> UPath:
        if root is None:
            root = (
                UPath(self.lit_config.source.pool_dir)
                if self.lit_config.source.pool_dir
                else self.save_path / "source_pools"
            )
        return root / f"axis_{_AXIS_NAMES[axis]}_{split}.npz"

    def _source_pool_metadata(
        self,
        axis: int,
        source_center: float,
        *,
        target_sha256: str,
        expected_walkers: int,
    ) -> dict[str, object]:
        return {
            "axis": float(axis),
            "source_center": float(source_center),
            "source_floor": float(self.lit_config.source.floor),
            "walker_count": float(expected_walkers),
            # A pi_Phi pool is distributed according to the exact ground
            # checkpoint and Hamiltonian geometry.  Binding both here avoids
            # silently reusing statistically incompatible walkers after a
            # checkpoint or system change that leaves the pool shape intact.
            "target_sha256": str(target_sha256),
        }

    def _expected_source_pool_walkers(self, batches: int) -> int:
        return int(self.config.batch_size) * int(batches)

    def _source_distillation_log_ratios(
        self,
        response_apply,
        response_params,
        ground_logpsi,
        ground_params,
        batched_data,
        *,
        axis: int,
        source_center: float,
    ):
        """Return ``log(Psi_response/Phi)`` and exact-pi_Phi weights."""
        score_eps = jnp.asarray(
            self.lit_config.solver.sr_score_epsilon,
            dtype=batched_data.data.electrons.dtype,
        )
        source_floor = jnp.asarray(
            self.lit_config.source.floor,
            dtype=batched_data.data.electrons.dtype,
        )

        def one(data):
            response_log = response_apply(response_params, data)
            ground_log = ground_logpsi(ground_params, data)
            dipole = molecular_electronic_dipole(data, axis)
            source = dipole - jnp.asarray(source_center, dtype=dipole.dtype)
            safe_abs_source = jnp.maximum(jnp.abs(source), score_eps)
            source_phase = jnp.where(
                source < 0.0,
                jnp.asarray(jnp.pi, dtype=source.dtype),
                jnp.asarray(0.0, dtype=source.dtype),
            )
            complex_dtype = jnp.result_type(response_log, ground_log, jnp.complex64)
            source_log = (
                jnp.asarray(ground_log, dtype=complex_dtype)
                + jnp.log(safe_abs_source).astype(complex_dtype)
                + 1j * source_phase.astype(complex_dtype)
            )
            sampled_abs_source = jnp.maximum(jnp.abs(source), source_floor)
            source_weight = (
                jnp.abs(source) / jnp.maximum(sampled_abs_source, score_eps)
            ) ** 2
            return (
                jnp.asarray(response_log, dtype=complex_dtype) - source_log,
                source_weight,
            )

        return jax.vmap(one, in_axes=(batched_data.vmap_axis,))(batched_data.data)

    def _source_distillation_scores(
        self,
        response_apply,
        response_params,
        ground_logpsi,
        ground_params,
        batched_data,
        *,
        axis: int,
        source_center: float,
        axis_name: str | None = None,
    ):
        """Return response log-scores and stable response/source ratios."""
        score_eps = float(self.lit_config.solver.sr_score_epsilon)

        def one(params, data):
            def split_log_ratio(local_params):
                response_log = response_apply(local_params, data)
                ground_log = ground_logpsi(ground_params, data)
                dipole = molecular_electronic_dipole(data, axis)
                source = dipole - jnp.asarray(source_center, dtype=dipole.dtype)
                safe_abs_source = jnp.maximum(jnp.abs(source), score_eps)
                source_phase = jnp.where(source < 0.0, jnp.pi, 0.0)
                complex_dtype = jnp.result_type(
                    response_log,
                    ground_log,
                    jnp.complex64,
                )
                log_ratio = (
                    jnp.asarray(response_log, dtype=complex_dtype)
                    - jnp.asarray(ground_log, dtype=complex_dtype)
                    - jnp.log(safe_abs_source).astype(complex_dtype)
                    - 1j * jnp.asarray(source_phase, dtype=complex_dtype)
                )
                return jnp.stack((jnp.real(log_ratio), jnp.imag(log_ratio))), (
                    log_ratio,
                    source,
                )

            jacobian, (log_ratio, source) = jax.jacrev(
                split_log_ratio,
                has_aux=True,
            )(params)
            score_tree = jax.tree.map(
                lambda leaf: leaf[0] + 1j * leaf[1],
                jacobian,
            )
            return log_ratio, source, score_tree

        log_ratio, source, score_tree = jax.vmap(
            lambda data: one(response_params, data),
            in_axes=(batched_data.vmap_axis,),
        )(batched_data.data)
        score = _flatten_batched_tree(score_tree, log_ratio.shape[0])
        source_floor = jnp.asarray(
            self.lit_config.source.floor,
            dtype=source.dtype,
        )
        eps = jnp.asarray(score_eps, dtype=source.dtype)
        sampled_abs_source = jnp.maximum(jnp.abs(source), source_floor)
        source_weight = (jnp.abs(source) / jnp.maximum(sampled_abs_source, eps)) ** 2
        finite = (
            jnp.isfinite(jnp.real(log_ratio))
            & jnp.isfinite(jnp.imag(log_ratio))
            & jnp.isfinite(source_weight)
            & jnp.all(
                jnp.isfinite(jnp.real(score)) & jnp.isfinite(jnp.imag(score)),
                axis=1,
            )
        )
        safe_log_real = jnp.where(finite, jnp.real(log_ratio), -jnp.inf)
        log_scale = jnp.max(safe_log_real)
        if axis_name is not None:
            log_scale = jax.lax.pmax(log_scale, axis_name=axis_name)
        log_scale = jnp.where(jnp.isfinite(log_scale), log_scale, 0.0)
        log_scale = jax.lax.stop_gradient(log_scale)
        ratio = jnp.where(
            finite,
            jnp.exp(log_ratio - log_scale),
            jnp.asarray(0.0, dtype=log_ratio.dtype),
        )
        source_weight = jnp.where(finite, source_weight, 0.0)
        score = jnp.where(finite[:, None], score, 0.0)
        return score, ratio, source_weight, log_ratio

    def _evaluate_source_distillation(
        self,
        response_apply,
        response_params,
        ground_logpsi,
        ground_params,
        eval_pool,
        *,
        axis: int,
        source_center: float,
    ) -> _SourceDistillationStats:
        """Evaluate initialization fidelity on the independent held-out pool.

        Returns:
            Held-out normalized-overlap, reverse-KL, ESS, and health statistics.

        Raises:
            ValueError: If the evaluation pool is empty or cannot be sharded.
        """
        cache = getattr(self, "_source_distillation_eval_kernel_cache", None)
        if cache is None:
            cache = {}
            self._source_distillation_eval_kernel_cache = cache
        chunk_size = self._lit_eval_batch_size()
        cache_key = (
            self._lit_data_parallel_mode(),
            id(response_apply),
            id(ground_logpsi),
            int(axis),
            float(source_center),
            min(int(eval_pool.batch_size), int(chunk_size)),
        )
        data_parallel = self._lit_data_parallel_enabled()
        kernel = cache.get(cache_key)
        if kernel is None:

            def evaluate(local_params, local_ground_params, local_pool):
                return self._source_distillation_log_ratios(
                    response_apply,
                    (
                        parallel_jax.pvary(local_params)
                        if data_parallel
                        else local_params
                    ),
                    ground_logpsi,
                    local_ground_params,
                    local_pool,
                    axis=axis,
                    source_center=source_center,
                )

            if data_parallel:
                kernel = parallel_jax.jit_sharded(
                    evaluate,
                    in_specs=(
                        parallel_jax.SHARE_PARTITION,
                        parallel_jax.SHARE_PARTITION,
                        eval_pool.partition_spec,
                    ),
                    out_specs=(
                        parallel_jax.DATA_PARTITION,
                        parallel_jax.DATA_PARTITION,
                    ),
                    check_vma=True,
                )
            else:
                kernel = jax.jit(evaluate)
            cache[cache_key] = kernel

        kernel_params = response_params
        kernel_ground_params = ground_params
        if data_parallel:
            kernel_params = _replicate_across_local_devices(response_params)
            kernel_ground_params = _replicate_across_local_devices(ground_params)

        log_ratios = []
        source_weights = []
        for chunk in _batched_data_chunks(eval_pool, chunk_size):
            kernel_chunk = chunk
            if data_parallel:
                self._validate_data_parallel_batch(
                    chunk,
                    purpose="distillation evaluation chunk",
                )
                kernel_chunk = _shard_batched_data_across_local_devices(chunk)
            local_log_ratio, local_source_weight = kernel(
                kernel_params,
                kernel_ground_params,
                kernel_chunk,
            )
            log_ratios.append(local_log_ratio)
            source_weights.append(local_source_weight)
        if not log_ratios:
            raise ValueError("Cannot evaluate source distillation on an empty pool.")
        return _source_distillation_stats_from_log_ratios(
            jnp.concatenate(log_ratios),
            jnp.concatenate(source_weights),
            reverse_kl_weight=self.lit_config.solver.reverse_kl_weight,
        )

    def _distill_response_from_source(  # noqa: C901
        self,
        response_apply,
        initial_params,
        ground_logpsi,
        ground_params,
        train_pool,
        eval_pool,
        rng,
        *,
        axis: int,
        source_center: float,
    ):
        """Fit the direct response NQS to ``Phi`` before action optimization.

        Returns:
            Best independently selected response parameters and the unchanged
            workflow random key.

        Raises:
            RuntimeError: If the initial held-out estimator is invalid.
        """
        iterations = int(self.lit_config.source.distillation_iterations)
        data_parallel = self._lit_data_parallel_enabled()
        device_count = jax.local_device_count() if data_parallel else 1

        def update_impl(params, local_ground_params, batch, spring_previous):
            score, ratio, source_weight, log_ratio = self._source_distillation_scores(
                response_apply,
                parallel_jax.pvary(params) if data_parallel else params,
                ground_logpsi,
                local_ground_params,
                batch,
                axis=axis,
                source_center=source_center,
                axis_name=(parallel_jax.BATCH_AXIS_NAME if data_parallel else None),
            )
            spring_state = _SpringState(spring_previous)
            if data_parallel:
                updates, spring_state, _, diagnostics = (
                    self._weighted_sr_updates_from_scores_data_parallel(
                        params,
                        score,
                        ratio,
                        source_weight,
                        spring_state,
                        device_count=device_count,
                    )
                )
            else:
                updates, spring_state, _, diagnostics = (
                    self._weighted_sr_updates_from_scores(
                        params,
                        score,
                        ratio,
                        source_weight,
                        spring_state,
                    )
                )
            stats = _source_distillation_stats_from_log_ratios(
                log_ratio,
                source_weight,
                reverse_kl_weight=self.lit_config.solver.reverse_kl_weight,
                axis_name=(parallel_jax.BATCH_AXIS_NAME if data_parallel else None),
            )
            return (
                _apply_updates(params, updates),
                stats,
                spring_state.previous_direction,
                diagnostics,
            )

        if data_parallel:
            sample_batch = _indexed_batched_data_chunk(
                train_pool,
                self._lit_train_update_batch_size(),
                0,
            )
            self._validate_data_parallel_batch(
                sample_batch,
                purpose="distillation training",
            )
            update_kernel = parallel_jax.jit_sharded(
                update_impl,
                in_specs=(
                    parallel_jax.SHARE_PARTITION,
                    parallel_jax.SHARE_PARTITION,
                    sample_batch.partition_spec,
                    parallel_jax.SHARE_PARTITION,
                ),
                out_specs=parallel_jax.SHARE_PARTITION,
                check_vma=True,
            )
        else:
            update_kernel = jax.jit(update_impl)

        def evaluate(params):
            return jax.device_get(
                self._evaluate_source_distillation(
                    response_apply,
                    params,
                    ground_logpsi,
                    ground_params,
                    eval_pool,
                    axis=axis,
                    source_center=source_center,
                )
            )

        params = initial_params
        flat_params, _ = ravel_pytree(params)
        spring_previous = jnp.zeros_like(flat_params)
        initial_stats = evaluate(params)
        if not _finite_source_distillation_stats(initial_stats):
            raise RuntimeError(
                f"axis={_AXIS_NAMES[axis]} source distillation has invalid "
                "initial held-out statistics."
            )
        best_params = params
        best_stats = initial_stats
        best_iteration = 0
        plateau = _FidelityPlateauTracker(
            start_iteration=self.lit_config.solver.plateau_start_iteration,
            patience_iterations=(self.lit_config.solver.plateau_patience_iterations),
            min_delta=self.lit_config.solver.plateau_min_delta,
        )
        if plateau.start_iteration == 0:
            plateau.observe(0, float(best_stats.fidelity))
        executed_iterations = iterations
        stop_reason = "max_budget"
        selection_interval = int(self.lit_config.solver.selection_interval)
        forced_evaluations = {iterations}
        if plateau.enabled and 0 < plateau.start_iteration <= iterations:
            forced_evaluations.add(plateau.start_iteration)
        shuffle_seed = self._training_shuffle_seed(
            axis=axis,
            stage="source_distillation",
        )

        for iteration in range(iterations):
            update_batch = _shuffled_batched_data_chunk(
                train_pool,
                self._lit_train_update_batch_size(),
                iteration,
                seed=shuffle_seed,
            )
            if data_parallel:
                kernel_params = _replicate_across_local_devices(params)
                kernel_ground_params = _replicate_across_local_devices(ground_params)
                kernel_batch = _shard_batched_data_across_local_devices(update_batch)
                kernel_spring = _replicate_across_local_devices(spring_previous)
            else:
                kernel_params = params
                kernel_ground_params = ground_params
                kernel_batch = update_batch
                kernel_spring = spring_previous
            params, train_stats, spring_previous, diagnostics = update_kernel(
                kernel_params,
                kernel_ground_params,
                kernel_batch,
                kernel_spring,
            )
            completed = iteration + 1
            candidate_stats = None
            if completed % selection_interval == 0 or completed in forced_evaluations:
                candidate_stats = evaluate(params)
                if _finite_source_distillation_stats(candidate_stats) and float(
                    candidate_stats.loss
                ) < float(best_stats.loss):
                    best_params = params
                    best_stats = candidate_stats
                    best_iteration = completed
            if (
                self.lit_config.solver.log_interval > 0
                and completed % self.lit_config.solver.log_interval == 0
            ):
                host_train, host_diagnostics = jax.device_get(
                    (train_stats, diagnostics)
                )
                logger.info(
                    "axis=%s stage=source_distillation iter=%d "
                    "train_loss=%.6e train_fidelity=%.6f "
                    "train_reverse_kl=%.6e train_ess=%.3f best_iter=%d "
                    "best_loss=%.6e best_fidelity=%.6f best_reverse_kl=%.6e "
                    "best_ess=%.3f",
                    _AXIS_NAMES[axis],
                    completed,
                    float(host_train.loss),
                    float(host_train.fidelity),
                    float(host_train.reverse_kl),
                    float(host_train.reweight_ess_fraction),
                    best_iteration,
                    float(best_stats.loss),
                    float(best_stats.fidelity),
                    float(best_stats.reverse_kl),
                    float(best_stats.reweight_ess_fraction),
                )
                _log_spring_optimizer_diagnostics(
                    host_diagnostics,
                    axis=axis,
                    stage="source_distillation",
                    omega=float("nan"),
                    iteration=completed,
                )
            if (
                candidate_stats is not None
                and _finite_source_distillation_stats(candidate_stats)
                and plateau.observe(completed, float(best_stats.fidelity))
            ):
                executed_iterations = completed
                stop_reason = "fidelity_plateau"
                break

        logger.info(
            "axis=%s stage=source_distillation selected_iter=%d/%d "
            "heldout_loss=%.6e fidelity=%.6f reverse_kl=%.6e ess=%.3f "
            "invalid=%.3e initial_fidelity=%.6f fidelity_gain=%+.6e "
            "stop_reason=%s",
            _AXIS_NAMES[axis],
            best_iteration,
            executed_iterations,
            float(best_stats.loss),
            float(best_stats.fidelity),
            float(best_stats.reverse_kl),
            float(best_stats.reweight_ess_fraction),
            float(best_stats.invalid_sample_fraction),
            float(initial_stats.fidelity),
            float(best_stats.fidelity) - float(initial_stats.fidelity),
            stop_reason,
        )
        return best_params, rng
