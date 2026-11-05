(wavefunction-precision)=
# Mixed Precision

You can significantly reduce QMC training step time by running expensive wavefunction operations in reduced precision while keeping master parameters, checkpoints, and determinant factorizations in full precision.

JaQMC separates precision control into a wavefunction-wide default (`wf.precision`) and optional per-consumer overrides (`*.wavefunction_precision`, which default to `null` and inherit `wf.precision`). Built-in neural wavefunctions (FermiNet, LapNet, Psiformer, solid and electron-gas FermiNet, and Hall MHPO) provide three named modes:

| Mode | Forward pass | Forward Laplacian contractions | Determinant factorization |
|------|--------------|--------------------------------|---------------------------|
| `standard` (default) | Normal dtype promotion between inputs and parameters (FP32 by default, or FP64 when `jax.enable_x64=true`). | Standard dtype promotion. | Follows orbital dtype (at least FP32). |
| `fp16_jac` | FP32 neural evaluation. | Power-of-two scaled FP16 operands with FP32 accumulation. | FP32 (real or complex). |
| `fp16_mixed` | FP16 neural layers with scoped `bfloat16` matmul precision and architecture-specific FP32 protections (envelopes, Bloch phases, Jastrows, and Psiformer residual streams stay in FP32). | Power-of-two scaled FP16 operands with FP32 accumulation. | FP32 (real or complex). |

When a mode is selected, the wavefunction configures its internal layer dtypes and JAX matmul precision scopes automatically without changing root `jax.*` flags (`fp16_jac` and `fp16_mixed` pin neural evaluation to FP32 or FP16 even if `jax.enable_x64=true` is set).

These modes match the two workloads that dominate QMC runtime:

1. **Second-derivative contractions** in Forward Laplacian local energy estimators (the default `forward_laplacian` backend for kinetic energy and Pseudo-Hamiltonian energy), which propagate $O(N_{\text{elec}})$ Jacobian tangents through every neural layer. Local energy derivatives, parameter gradients, and natural-gradient preconditioners are sensitive to forward-pass rounding, so they use `fp16_jac` to accelerate Jacobian contractions while keeping forward passes in FP32.
2. **Repeated forward passes** in Markov Chain Monte Carlo (MCMC) sampling and nonlocal estimators such as effective core potentials (ECPs), which evaluate $\log\psi$ many times per walker without second derivatives. Ratio-based proposals and quadratures tolerate FP16 neural forward passes well, so they can override their mode to `fp16_mixed`.

Because stored parameters and checkpoint `.npz` files never change layout or dtype across modes, you can mix precision modes across consumers during training and switch back to `standard` precision for final evaluation.

:::{important}
**LapNet prerequisite for `fp16_mixed`**: If you use molecular **LapNet** (`wf.module=lapnet`) and plan to enable `fp16_mixed` (either initially or when resuming a run), set `wf.use_layernorm=true` in your system config before starting training so intermediate attention and MLP activations stay well-scaled in FP16. Unlike `wf.precision`, `wf.use_layernorm` alters model parameters and must remain set for both training and evaluation. FermiNet, Psiformer, and Hall MHPO (including `wf.backbone=lapnet`, which enables LayerNorm automatically) require no architecture changes (see <project:wavefunction.md>).
:::

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

Any component that evaluates the wavefunction (`sampler`, wavefunction-dependent `estimators.*`, `train.grads`, and natural-gradient `train.optim`) exposes a `wavefunction_precision` setting (`null` inherits `wf.precision`). On runs with many Metropolis-Hastings sub-steps or expensive nonlocal evaluations, override `wavefunction_precision` on those forward-heavy consumers to `fp16_mixed`:

```bash
jaqmc <app> train ... \
  wf.precision=fp16_jac \
  sampler.wavefunction_precision=fp16_mixed \
  workflow.save_path=./runs/<name>-prod
```

Leaving local energy estimators, `train.grads`, and `train.optim` at `null` keeps local energy derivatives, parameter gradients, and natural-gradient preconditioner curvature at `fp16_jac` (FP32 forward pass) for optimization stability, while the sampler's repeated proposals run in `fp16_mixed`. On ECP systems, you can similarly pass `estimators.energy.ecp.wavefunction_precision=fp16_mixed` or configure a [tiered nonlocal ECP precision schedule](#ecp-nonlocal-precision) in YAML.

If training with `fp16_mixed` introduces energy spikes or NaNs on a challenging system, fall back to `wf.precision=fp16_jac` with `*.wavefunction_precision=null`, or test with `wf.precision=standard` (and `*.wavefunction_precision=null` if also testing `jax.enable_x64=true`, see <project:troubleshooting.md>).

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
- <project:running-workflows.md>: Shared training, resume, and evaluation workflow mechanics.
- <project:analyzing-evaluations.md>: Extracting final observables and uncertainties from evaluation runs.
- <project:troubleshooting.md>: Diagnosing NaNs, energy spikes, and train-versus-evaluation discrepancies.
- [Implementing evaluation precision in custom wavefunctions](#wavefunction-evaluation-precision): Extending `Wavefunction` subclasses and wiring custom consumers in the developer guide (wavefunctions without configurable modes accept only `null` and raise a `ValueError` for explicit modes).
