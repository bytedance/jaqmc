#!/usr/bin/env bash
set -euo pipefail

repository_root="$(git rev-parse --show-toplevel)"
example_root="${repository_root}/contrib/lit/examples/he"
run_root="${JAQMC_LIT_RUN_ROOT:-${repository_root}/runs/he_lit}"
raw_path="${run_root}/lit/lit_spectrum.npz"
post_dir="${run_root}/post"
jaqmc_command="${repository_root}/.venv/bin/jaqmc"
python_command="${repository_root}/.venv/bin/python"

if [[ ! -x "${jaqmc_command}" ]]; then
  jaqmc_command=jaqmc
  python_command=python
fi

if [[ ! -s "${raw_path}" ]]; then
  echo "Missing LIT spectrum: ${raw_path}" >&2
  echo "Run scripts/run_calculation.sh first." >&2
  exit 1
fi

cd "${repository_root}"
"${jaqmc_command}" lit invert \
  --yml "${example_root}/config/invert.yml" \
  "inversion.input_paths=[${raw_path}]" \
  "inversion.output_path=${post_dir}/fit.npz"

export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/jaqmc-contrib-lit-matplotlib}"
"${python_command}" "${example_root}/scripts/post.py" \
  --raw-path "${raw_path}" \
  --fit-path "${post_dir}/fit.npz" \
  --output-dir "${post_dir}" \
  --figure-dir "${post_dir}/fig"
