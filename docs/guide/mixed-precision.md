(wavefunction-precision)=
# Mixed Precision

You can significantly reduce QMC training step time by running expensive wavefunction operations in reduced precision while keeping master parameters, checkpoints, and determinant factorizations in full precision.

JaQMC separates precision control into a wavefunction-wide default (`wf.precision`) and optional per-consumer overrides (`*.wavefunction_precision`, which default to `null` and inherit `wf.precision`). Built-in neural wavefunctions (FermiNet, LapNet, Psiformer, solid and electron-gas FermiNet, and Hall MHPO) provide three named modes:

| Mode | Forward pass | Forward Laplacian contractions | Determinant factorization |
|------|--------------|--------------------------------|---------------------------|
| `standard` (default) | Normal dtype promotion between inputs and parameters (FP32 by default, or FP64 when `jax.enable_x64=true`). | Standard dtype promotion. | Follows orbital dtype (at least FP32). |
| `fp16_jac` | FP32 neural evaluation. | Power-of-two scaled FP16 operands with FP32 accumulation. | FP32 (real or complex). |
| `fp16_mixed` | FP16 neural layers with scoped `bfloat16` matmul precision and architecture-specific FP32 protections (envelopes, Bloch phases, Jastrows, and Psiformer residual streams stay in FP32, and molecular LapNet requires `wf.use_layernorm=true`, see <project:wavefunction.md>). | Power-of-two scaled FP16 operands with FP32 accumulation. | FP32 (real or complex). |

When a mode is selected, the wavefunction configures its internal layer dtypes and JAX matmul precision scopes automatically without changing root `jax.*` flags (`fp16_jac` and `fp16_mixed` pin neural evaluation to FP32 or FP16 even if `jax.enable_x64=true` is set).

These modes match the two types of wavefunction evaluations in a QMC workflow:

1. **Second-derivative Laplacian contractions** in Forward Laplacian local energy estimators (such as kinetic energy or Pseudo-Hamiltonian energy). Setting `wf.precision=fp16_jac` accelerates these Jacobian contractions while keeping every forward pass in FP32.
2. **Forward and first-order passes** in consumers that evaluate the wavefunction or its parameter derivatives without second derivatives (such as MCMC samplers, forward-only nonlocal estimators, and optimizer score or curvature passes). Because these consumers do not run `forward_laplacian`, `fp16_jac` leaves them in FP32, while overriding `<scope>.wavefunction_precision` to `fp16_mixed` accelerates their neural forward and backward passes on FP16 Tensor Cores:

| Consumer Role | Consumer Scope | Recommended Mode |
|---|---|---|
| **Laplacian local energy** | `estimators.energy.kinetic`, `estimators.energy.ph` | `null` (inherits `wf.precision=fp16_jac`) |
| **MCMC sampling** | `sampler` | `fp16_mixed` |
| **Forward-only estimators** (when present) | e.g., `estimators.energy.ecp` | `fp16_mixed` (or a [tiered schedule](#ecp-nonlocal-precision) for `ecp`) |
| **Loss gradients** | `train.grads` | `null` (inherits `wf.precision=fp16_jac`) |
| **Natural-gradient optimizer pass** | `train.optim` | `null` (conservative FP32) or `fp16_mixed` (faster score/curvature pass) |

Because stored parameters and checkpoint `.npz` files never change layout or dtype across modes, you can mix precision modes across consumers during training and switch back to `standard` precision for final evaluation.

## Speedup Workflow

### 1. Accelerate Laplacian contractions (`wf.precision`)

Start by setting `wf.precision=fp16_jac` on your training command:

```bash
jaqmc <app> train ... \
  wf.precision=fp16_jac \
  workflow.save_path=./runs/<name>-prod
```

If you want a conservative speedup with zero change to the forward pass or sampling distribution, stop here. With all consumer `wavefunction_precision` keys left at `null`, Forward Laplacian estimators use scaled FP16 Jacobian contractions while every forward pass across sampling, energy evaluation, gradients, and preconditioners remains in FP32.

### 2. Accelerate forward-heavy consumers (`*.wavefunction_precision`)

Any component that evaluates the wavefunction (`sampler`, wavefunction-dependent `estimators.*`, `train.grads`, and natural-gradient `train.optim`) exposes a `wavefunction_precision` setting (`null` inherits `wf.precision`). Because `fp16_jac` only accelerates second-derivative Laplacian contractions, forward-only and first-order consumers remain in FP32 unless overridden to `fp16_mixed`:

```bash
jaqmc <app> train ... \
  wf.precision=fp16_jac \
  sampler.wavefunction_precision=fp16_mixed \
  workflow.save_path=./runs/<name>-prod
```

Leaving local energy estimators and `train.grads` at `null` keeps local energy derivatives and parameter gradients at `fp16_jac` (FP32 forward pass) for optimization stability, while the sampler's repeated proposals run in `fp16_mixed`. Depending on which components your workflow enables, you can apply the same override to other forward-only or first-order consumers:

- **Forward-only nonlocal estimators** (when present, such as ECPs): Pass `estimators.energy.ecp.wavefunction_precision=fp16_mixed` or configure a [tiered nonlocal ECP precision schedule](#ecp-nonlocal-precision) in YAML.
- **Natural-gradient optimizers** (`train.optim`): Pass `train.optim.wavefunction_precision=fp16_mixed` to accelerate the optimizer's first-order score or curvature wavefunction pass. Some optimizers also expose dedicated precision settings for their own internal linear algebra (for example, `gram_dot_prec` and `gram_solve_prec` in Stochastic Reconfiguration, see [Gram Matrix Precision](#sr-gram-precision)).

:::{admonition} Example: Molecular LapNet with Stochastic Reconfiguration
:class: note

For a molecular LapNet calculation trained with SR (`train.optim.module=sr`), set `wf.use_layernorm=true` in your system config (so attention and MLP activations stay well-scaled in `fp16_mixed`, and keep it set during `evaluate` since it adds layer-norm parameters) and combine `fp16_jac`, `fp16_mixed`, and SR's Gram precision settings:

```bash
# Add train.optim.gram_solve_prec=F32_PCG on GPUs with limited FP64 throughput (e.g., H20)
jaqmc molecule train ... \
  wf.module=lapnet wf.use_layernorm=true \
  wf.precision=fp16_jac \
  sampler.wavefunction_precision=fp16_mixed \
  train.optim.module=sr \
  train.optim.wavefunction_precision=fp16_mixed \
  train.optim.gram_dot_prec=F16 \
  workflow.save_path=./runs/<name>-prod
```
:::

If training with `fp16_mixed` introduces energy spikes or NaNs on a challenging system, reset consumer overrides to `null` (starting with `train.optim.wavefunction_precision=null` if set), fall back to `wf.precision=fp16_jac` with `*.wavefunction_precision=null`, or test with `wf.precision=standard` (and `*.wavefunction_precision=null` if also testing `jax.enable_x64=true`, see <project:troubleshooting.md>).

(mixed-precision-evaluation-and-resuming)=
### 3. Verify final observables in standard precision

Using different precision modes for MCMC sampling (`fp16_mixed`) and energy evaluation (`fp16_jac`) introduces a small sampling-distribution approximation, because walkers are drawn from the sampler's mode while local energies are evaluated in the estimator's mode. Always verify final reported observables with `evaluate` using the same mode for sampling and energy, typically `wf.precision=standard` with no consumer overrides:

```bash
jaqmc <app> evaluate ... \
  wf.precision=standard \
  workflow.source_path=./runs/<name>-prod \
  workflow.save_path=./runs/<name>-eval \
  run.burn_in=100 \
  run.iterations=5000
```

Set `run.burn_in` (which defaults to `0` in `evaluate`) when switching `sampler.wavefunction_precision` between training and evaluation so walkers and proposal widths loaded from `workflow.source_path` re-equilibrate to the `standard` sampling distribution before statistics are accumulated. If the early evaluation steps in `evaluation_stats.h5` still show initial drift in `total_energy` or `pmove`, increase `run.burn_in`. Then extract final energies and error bars as described in <project:analyzing-evaluations.md>.

Precision settings are recorded in the resolved configuration file (`train_config.yaml` or `evaluation_config.yaml`), not inside checkpoint `.npz` files:

- **Shared YAML overrides during evaluation or `x64` debugging**: Explicit consumer overrides (`*.wavefunction_precision`) take precedence over `wf.precision`. If you placed training overrides such as `sampler.wavefunction_precision` or `estimators.energy.ecp.wavefunction_precision` inside a YAML file shared by `train` and `evaluate`, setting `wf.precision=standard` (or `jax.enable_x64=true`) alone will not clear them. Either keep training precision overrides on the CLI (or in a separate training YAML) as shown above, or reset overridden consumers with `*.wavefunction_precision=null`.
- **Resuming or branching training**: Because checkpoints do not store precision settings, a resumed or branched training command reverts to `standard` unless you pass your `wf.precision` and `*.wavefunction_precision` settings again. Conversely, you can freely change precision modes when resuming or branching from an earlier checkpoint as long as architecture flags (such as `wf.use_layernorm` on LapNet) remain unchanged.

## See Also

- <project:wavefunction.md>: Architecture-specific precision details for FermiNet, LapNet, and Psiformer.
- <project:estimators/ecp.md>: Rank-fraction precision schedules for nonlocal ECP integrals.
- [Gram Matrix Precision](#sr-gram-precision): Optimizer-internal Gram contraction and linear-solve precision for Stochastic Reconfiguration.
- <project:running-workflows.md>: Shared training, resume, and evaluation workflow mechanics.
- <project:analyzing-evaluations.md>: Extracting final observables and uncertainties from evaluation runs.
- <project:troubleshooting.md>: Diagnosing NaNs, energy spikes, and train-versus-evaluation discrepancies.
- [Implementing evaluation precision in custom wavefunctions](#wavefunction-evaluation-precision): Extending `Wavefunction` subclasses and wiring custom consumers in the developer guide (wavefunctions without configurable modes accept only `null` and raise a `ValueError` for explicit modes).
