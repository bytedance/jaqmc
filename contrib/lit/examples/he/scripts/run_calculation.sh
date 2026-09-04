#!/usr/bin/env bash
set -euo pipefail

repository_root="$(git rev-parse --show-toplevel)"
example_root="${repository_root}/contrib/lit/examples/he"
run_root="${JAQMC_LIT_RUN_ROOT:-${repository_root}/runs/he_lit}"
ground_dir="${run_root}/ground"
lit_dir="${run_root}/lit"
jaqmc_command="${repository_root}/.venv/bin/jaqmc"

if [[ ! -x "${jaqmc_command}" ]]; then
  jaqmc_command=jaqmc
fi

if [[ -e "${run_root}" ]]; then
  echo "Refusing to overwrite existing run: ${run_root}" >&2
  exit 1
fi

mkdir -p "${ground_dir}" "${lit_dir}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_DEFAULT_MATMUL_PRECISION=float32
export NVIDIA_TF32_OVERRIDE=0

"${jaqmc_command}" molecule train \
  --yml "${example_root}/config/ground.yml" \
  workflow.save_path="${ground_dir}" \
  pretrain.writers.console.interval=1000 \
  train.writers.console.interval=1000

"${jaqmc_command}" lit run \
  --yml "${example_root}/config/lit.yml" \
  workflow.save_path="${lit_dir}" \
  lit.ground.checkpoint_path="${ground_dir}"

test -s "${lit_dir}/lit_spectrum.npz"
echo "Raw LIT spectrum: ${lit_dir}/lit_spectrum.npz"
