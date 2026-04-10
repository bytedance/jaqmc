# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging
import time

import jax

from jaqmc.utils.config import configurable_dataclass

try:
    from jax import shard_map
except ImportError:
    from jax.experimental.shard_map import shard_map as exp_shard_map

    def shard_map(f, mesh, in_specs, out_specs, check_vma=True):  # type: ignore
        return exp_shard_map(f, mesh, in_specs, out_specs, check_rep=check_vma)


logger = logging.LoggerAdapter(
    logging.getLogger(__name__), extra={"category": "parallel"}
)

BATCH_AXIS_NAME = "qmc_batch_axis"

DATA_PARTITION = jax.sharding.PartitionSpec(BATCH_AXIS_NAME)
SHARE_PARTITION = jax.sharding.PartitionSpec()


def make_mesh():
    """Create a 1-D device mesh along the batch axis.

    Returns:
        A JAX mesh spanning all available devices.
    """
    return jax.make_mesh((jax.device_count(),), (BATCH_AXIS_NAME,))


def make_sharding(partition):
    """Convert partition specs into NamedSharding objects on the default mesh.

    Returns:
        A pytree of NamedSharding matching the input structure.
    """
    mesh = make_mesh()
    return jax.tree.map(lambda spec: jax.sharding.NamedSharding(mesh, spec), partition)


def jit_sharded(fn, *, in_specs, out_specs, check_vma=True, donate_argnums=None):
    """JIT-compile a function with shard_map in one call.

    Args:
        fn: Function to compile.
        in_specs: Input partition specs for ``shard_map``.
        out_specs: Output partition specs for ``shard_map``.
        check_vma: Whether to enable validity checks during shard_map.
        donate_argnums: Argument indices to donate (passed to ``jax.jit``).

    Returns:
        JIT-compiled, shard-mapped function.
    """
    mesh = make_mesh()
    jit_kwargs = {}
    if donate_argnums is not None:
        jit_kwargs["donate_argnums"] = donate_argnums
    sharded = shard_map(
        fn,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=check_vma,
    )
    return jax.jit(sharded, **jit_kwargs)


def pvary[ValueT](x: ValueT) -> ValueT:
    """Mark ``x`` as varying across the batch axis inside ``shard_map``.

    Returns:
        The input value annotated as varying, or unchanged outside shard_map.

    Type Parameters:
        ValueT: Arbitrary pytree-like value type preserved across the call.
    """
    if hasattr(jax.lax, "pcast"):
        try:
            return jax.lax.pcast(x, BATCH_AXIS_NAME, to="varying")
        except ValueError:
            # Outside a shard_map context (e.g. interactive notebook use).
            logger.warning("Using pvary() outside a shard_map context.")
            return x
    if hasattr(jax.lax, "pvary"):
        try:
            return jax.lax.pvary(x, BATCH_AXIS_NAME)
        except ValueError:
            logger.warning("Using pvary() outside a shard_map context.")
            return x
    return x


def pmean[ValueT](x: ValueT) -> ValueT:
    """Average ``x`` across devices along the batch axis.

    Returns:
        The mean of ``x`` across all devices, or ``x`` unchanged outside shard_map.

    Type Parameters:
        ValueT: Arbitrary pytree-like value type preserved across the call.
    """
    try:
        return jax.lax.pmean(x, axis_name=BATCH_AXIS_NAME)
    except NameError:
        # Outside a shard_map context (e.g. interactive notebook use).
        # With a single shard the mean is the identity.
        return x


def all_gather[ValueT](x: ValueT) -> ValueT:
    """Gather arrays from all devices along the batch axis.

    Collects arrays sharded across devices and materializes the complete
    array on each device. This is useful for checkpointing or when you need
    to access the full dataset on each process.

    This function should mimic the behavior of
    `jax.experimental.multihost_utils.process_allgather(x, tiled=True)`,
    which is only available for JAX >= 0.8.X

    Args:
        x: Array or pytree of arrays to gather. Each array should be sharded
            along the leading dimension corresponding to BATCH_AXIS_NAME.

    Returns:
        Gathered array or pytree with the same structure and shape as input.
        The sharding is changed from sharded to replicated - each device now
        has a complete copy of the full array instead of just a shard.

    Type Parameters:
        ValueT: Arbitrary pytree-like value type preserved across the call.
    """
    mesh = make_mesh()

    def gather_fn(y):
        return jax.lax.all_gather(y, axis_name=BATCH_AXIS_NAME, tiled=True)

    # Determine partition specs based on input structure
    in_spec = jax.tree.map(lambda _: jax.sharding.PartitionSpec(BATCH_AXIS_NAME), x)
    out_spec = jax.tree.map(lambda _: jax.sharding.PartitionSpec(), x)

    return shard_map(
        gather_fn,
        mesh=mesh,
        in_specs=in_spec,
        out_specs=out_spec,
        check_vma=False,
    )(x)


def addressable_data(x):
    """Return the process-local shard of a potentially sharded array.

    For any ``jax.Array`` (sharded or replicated), returns the addressable
    (local) portion as a concrete array without sharding metadata. This is
    needed for init functions (KFAC, samplers) that trace the computation
    and cannot handle arrays with sharding information.

    Args:
        x: Input value, possibly a sharded ``jax.Array``.

    Returns:
        The local (addressable) portion of the array, or *x* unchanged
        if it is not a ``jax.Array``.
    """
    if isinstance(x, jax.Array):
        return x.addressable_data(0)
    return x


@configurable_dataclass
class DistributedConfig:
    """Configuration for initializing JAX distributed runtime."""

    coordinator_address: str | None = None
    "IP address and port of the coordinator process (e.g., 192.168.1.10:1234)."

    num_processes: int = 1
    "Total number of processes in the distributed run."

    process_id: int = 0
    "ID of the current process (0 to num_processes - 1)."

    initialization_timeout: int = 300
    "Timeout in seconds for distributed runtime initialization."

    wait_second_before_connect: float = 10.0
    "Seconds to wait before non-master processes connecting to the coordinator."

    def init_runtime(self):
        """Initialize JAX distributed runtime for multi-host training.

        If coordinator_address is None or num_processes is 1, the distributed
        runtime will not be initialized.

        """
        if self.coordinator_address is None or self.num_processes == 1:
            logger.info("Initialize a local runtime.")
            return

        logger.info("server_addr=%s", self.coordinator_address)
        logger.info("num_hosts=%s", self.num_processes)
        logger.info("host_idx=%s", self.process_id)

        if self.process_id > 0:
            logger.info(
                "Sleeping %s seconds before connecting to server...",
                self.wait_second_before_connect,
            )
            time.sleep(self.wait_second_before_connect)
        # This may fail if waiting for too long, controlled by `initialization_timeout`
        # in xla_client_self.
        jax.distributed.initialize(
            self.coordinator_address,
            num_processes=self.num_processes,
            process_id=self.process_id,
            initialization_timeout=self.initialization_timeout,
        )
