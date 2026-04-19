# Multi-Device

If you have multiple GPUs on one machine, JaQMC will use all of them automatically — no configuration needed. This page explains what happens under the hood, and how to scale to multiple machines when you need to.

## Single Node, Multiple GPUs

Just run your workflow as usual:

```bash
jaqmc molecule train
```

JaQMC logs what it finds at startup:

```
Starting QMC with 4 local XLA devices across 1 processes.
```

If you want to use only specific GPUs, set `CUDA_VISIBLE_DEVICES` before running:

```bash
CUDA_VISIBLE_DEVICES=0,2 jaqmc molecule train
```

### How walkers are distributed

`workflow.batch_size` is the total number of walkers. They are split evenly across GPUs — so with `batch_size=4096` on 4 GPUs, each GPU handles 1024 walkers.

**Don't increase the batch size just because you have more GPUs.** The batch size controls the statistical quality of each VMC step (more walkers = lower variance in the energy estimate). Picking a batch size is a physics decision, not a hardware one. More GPUs make each step *faster*, but the batch size should stay the same.

## Multi-Host (Multi-Node) Training

When a single machine isn't enough, you can spread the work across multiple machines. Each machine runs one JaQMC process, and they coordinate through JAX's distributed runtime. This is most useful when you need more GPUs than a single node has — additional machines add inter-node communication overhead (synchronizing gradients and statistics after every step), which can cancel out the speedup for systems that already fit on one machine.

### Setup

Tell each process how to find the others via the `distributed.*` config keys:

```yaml
distributed:
  coordinator_address: "192.168.1.10:1234"  # IP:port of process 0
  num_processes: 4                           # total across all machines
  process_id: 0                              # different on each machine
```

The same works from the CLI:

```bash
jaqmc molecule train \
  distributed.coordinator_address=192.168.1.10:1234 \
  distributed.num_processes=4 \
  distributed.process_id=0
```

| Key | Default | Description |
|-----|---------|-------------|
| `distributed.coordinator_address` | `null` | IP:port of process 0. Set this to enable multi-host. |
| `distributed.num_processes` | `1` | Total number of processes across all machines. |
| `distributed.process_id` | `0` | ID of the current process (0 to N-1). |
| `distributed.initialization_timeout` | `300` | Timeout (seconds) waiting for all processes to connect. |
| `distributed.wait_second_before_connect` | `10.0` | Seconds non-coordinator processes wait before connecting, giving the coordinator time to start listening. |

`batch_size` must be divisible by `num_processes` (raises `ValueError` otherwise). Each process gets `batch_size / num_processes` walkers, which are then [split further across its local GPUs](#how-walkers-are-distributed).

### Checkpoints are portable

Before saving, JaQMC gathers all data onto process 0 and writes a single checkpoint file. On restore, the data is redistributed to match the current device layout. So you can:

- Train on 4 nodes, resume on 2 (or 1).
- Switch between GPU counts without converting checkpoints.

Only process 0 writes checkpoints, but all processes read from the same directory on restore — so the checkpoint directory must be on shared storage (e.g. NFS) visible to every node.

## Launching on Clusters

### SLURM

A typical `sbatch` script for 2 nodes with 4 GPUs each. We run one JaQMC process per node, and each process uses all 4 GPUs on that node:

```bash
#!/bin/bash
#SBATCH --job-name=jaqmc-train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00

# First node in the allocation becomes the coordinator
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=29500

srun jaqmc molecule train --yml my_system.yml \
  distributed.coordinator_address="${MASTER_ADDR}:${MASTER_PORT}" \
  distributed.num_processes="${SLURM_NTASKS}" \
  distributed.process_id="${SLURM_PROCID}"
```

```{note}
The script above uses `--ntasks-per-node=1`, so each SLURM task is one JaQMC process that sees all GPUs on its node. If your cluster requires one task per GPU instead, set `--ntasks-per-node` to the GPU count and use `CUDA_VISIBLE_DEVICES` to give each task its own GPU.
```

### MPI

With Open MPI, you can use `$OMPI_COMM_WORLD_RANK` for the process ID:

```bash
# Resolve coordinator address once on the launch host
MASTER_ADDR=$(head -1 hosts.txt)

mpirun -np 4 --hostfile hosts.txt -x MASTER_ADDR bash -c '
  jaqmc molecule train --yml my_system.yml \
    distributed.coordinator_address="${MASTER_ADDR}:29500" \
    distributed.num_processes=4 \
    distributed.process_id=$OMPI_COMM_WORLD_RANK
'
```

### Manual launch

If you don't have a cluster manager, run the command on each machine yourself with a different `process_id`:

```bash
# Machine 0
jaqmc molecule train --yml my_system.yml \
  distributed.coordinator_address=192.168.1.10:1234 \
  distributed.num_processes=2 \
  distributed.process_id=0

# Machine 1
jaqmc molecule train --yml my_system.yml \
  distributed.coordinator_address=192.168.1.10:1234 \
  distributed.num_processes=2 \
  distributed.process_id=1
```

(programmatic-multi-host)=
## Programmatic (API) Usage

The CLI handles distributed initialization automatically, but if you're calling a workflow from Python, create a {class}`jaqmc.utils.config.ConfigManager` and initialize the distributed runtime yourself:

```python
from jaqmc.utils.config import ConfigManager
from jaqmc.utils.runtime import configure_runtime

cfg = ConfigManager(my_config_dict)

configure_runtime(cfg)  # no-op for single-process runs

my_workflow(cfg)
```

JAX and distributed settings can both be configured from the root config:

```yaml
jax:
  enable_x64: true
  default_matmul_precision: highest

distributed:
  coordinator_address: 192.168.1.10:1234
  num_processes: 2
  process_id: 1
```

JAX requires distributed initialization to happen before process/device-dependent runtime setup. In JaQMC, {func}`jaqmc.utils.runtime.configure_runtime` applies logging and JAX-global flags first, then initializes the distributed runtime so startup configuration is active consistently before workflow execution.

Without {func}`jaqmc.utils.runtime.configure_runtime`, multi-host config keys are silently ignored and the workflow runs on a single process, and any configured JAX global flags are not applied.

## Simulating Multiple Devices in Tests

To test multi-device behavior on a CPU-only machine, use the `--n-cpu-devices` flag:

```bash
pytest --n-cpu-devices=4
```

This creates 4 logical CPU devices that behave like separate accelerators, so you can verify sharding and communication without needing real GPUs.
