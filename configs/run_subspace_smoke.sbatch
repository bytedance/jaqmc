#!/usr/bin/env bash

# One launcher for 1/2/4-GPU correctness smoke runs. Select GPU resources in
# the sbatch command and select memory sizing through CONFIG_HW.
set -euo pipefail

CONFIG_SYSTEM="${CONFIG_SYSTEM:-configs/systems/h24_base.yml}"
CONFIG_WORKFLOW="${CONFIG_WORKFLOW:-configs/workflows/subspace_smoke.yml}"
CONFIG_HW="${CONFIG_HW:-configs/hardware/single_v100.yml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
JAQMC_BIN="${JAQMC_BIN:-jaqmc}"

git branch --show-current
git rev-parse HEAD
git status --short
"${PYTHON_BIN}" --version
"${PYTHON_BIN}" -c 'import jax, jaxlib; print("jax", jax.__version__); print("jaxlib", jaxlib.__version__); print("devices", jax.devices()); print("process_count", jax.process_count()); print("local_device_count", jax.local_device_count())'
printenv CUDA_VISIBLE_DEVICES || true
nvidia-smi || true

"${JAQMC_BIN}" solid subspace-train \
  --yml "${CONFIG_SYSTEM}" \
  --yml "${CONFIG_WORKFLOW}" \
  --yml "${CONFIG_HW}"
