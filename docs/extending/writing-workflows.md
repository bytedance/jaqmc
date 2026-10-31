# Writing Workflows

This tutorial walks through building a complete QMC workflow or a new system/app from scratch, using the hydrogen atom example ({ghsrc}`src/jaqmc/app/hydrogen_atom.py`) as a guide.

This page is for framework developers writing workflow code. For day-to-day execution,
resume/evaluate recipes, and output interpretation, see <project:../guide/running-workflows.md>.

## Prerequisites

Before starting, make sure you have:

- Basic proficiency in Python
- Working familiarity with the JAX concepts JaQMC relies on. If you are new to JAX, read <project:../extending/jax-for-jaqmc.md> first, then come back here.
- Basic familiarity with [Flax Linen](inv:flax:*:doc#index)
- Familiarity with core quantum mechanics concepts (wavefunctions, potential energy, observables)
- JaQMC installed (follow the instructions in <project:../getting-started/quick-start.md>)

## Step 1: Define Data Structures

Define a typed data container for your runtime inputs:

```{literalinclude} ../../src/jaqmc/app/hydrogen_atom.py
:pyobject: HydrogenAtomData
```

{class}`~jaqmc.data.Data` subclasses are automatically registered as JAX
PyTrees, so they flow through `jit`, `grad`, and `vmap` seamlessly. On the
common path, define `Data` from the single-walker point of view: per-walker
particle coordinates live in fields such as `electrons`, and batching is added
later by the framework. The full built-in convention is documented in
<project:runtime-data-conventions.md>; you can treat its
advanced batching section as optional on a first pass.

:::{admonition} What goes in Data?
:class: tip

Put values that **change at runtime** in `Data` — like particle coordinates.
Values that **determine array shapes or control flow** (for example,
`n_particles` and `ndim`) belong in the config, because JAX needs to know them
at compile time.
:::

## Step 2: Implement the Wavefunction

Define a variational wavefunction — a parameterized model that returns $\log|\psi|$ for numerical stability. For hydrogen, $\psi(r) = \exp(\alpha \cdot r)$:

```{literalinclude} ../../src/jaqmc/app/hydrogen_atom.py
:pyobject: HydrogenAtom
```

The {class}`~jaqmc.wavefunction.Wavefunction` base class is a Flax/Linen [`nn.Module`](inv:flax:py:class#flax.linen.Module). You implement `__call__(data)` as the model definition, while the framework calls `evaluate(params, data)`, a JaQMC wrapper over Flax's {meth}`~flax.linen.Module.apply` that takes explicit `params` so it can be differentiated with respect to them.

## Step 3: Create the Potential Energy Estimator

An estimator subclasses `PerWalkerEstimator` and implements
`evaluate_single_walker`:

```{literalinclude} ../../src/jaqmc/app/hydrogen_atom.py
:pyobject: PotentialEnergy
```

`evaluate_single_walker` receives the wavefunction parameters, one walker,
values from earlier estimators, the estimator state, and a random key. Values
from earlier estimators support derived quantities such as total energy, while
the state carries data across iterations. A simple estimator can ignore inputs
it does not need.

The ``energy:`` prefix is a naming convention: any estimator that returns a key starting with ``energy:`` contributes to the total energy, which becomes the VMC optimization target. A {class}`~jaqmc.estimator.total_energy.TotalEnergy` estimator auto-sums all ``energy:``-prefixed keys into a ``total_energy`` value — you'll use it in Step 5.

## Step 4: Initialize Walker Data

The data initializer generates the starting runtime data for the whole local
walker batch. In the hydrogen example that means electron positions. The
workflow calls it with ``size`` (the number of walkers sampled in parallel) and
a JAX PRNG key:

```{literalinclude} ../../src/jaqmc/app/hydrogen_atom.py
:pyobject: data_init
```

This is one place where the single-walker mental model needs one extra detail:
`data_init` does not return one walker at a time. In simple examples it returns
a plain {class}`~jaqmc.data.Data` object whose sampled field already has a
leading batch axis, and JaQMC wraps that with the default batching metadata.
Return explicit {class}`~jaqmc.data.BatchedData` yourself when your workflow
needs non-default batched fields or shared fields. The reference page
<project:runtime-data-conventions.md> covers that contract.

The default batching strategy is:

- `size` is the local batch size for this process, not the total global batch size.
- If `data_init` returns plain `Data`, JaQMC assumes the field named
  `electrons` is already batched with a leading walker axis.
  Any other fields in that `Data` object are treated as shared across walkers.

## Step 5: Build the Training Workflow

With the individual pieces ready, the workflow dataclass defines the available
options and builds its runtime stage when it is invoked:

```{literalinclude} ../../src/jaqmc/app/hydrogen_atom.py
:pyobject: HydrogenAtomTrainWorkflow
```

The class extends {class}`~jaqmc.workflow.vmc.VMCWorkflow`. Configuration is
loaded before the workflow wires its selected components and builds the stage.
``VMCWorkflow`` handles output directory creation, configuration saving, and
stage execution. Field defaults define the command defaults, and YAML or
dotlist values override them. `HydrogenEstimators` groups the workflow's
estimators in declaration order, and `EstimatorPipeline` runs only the enabled
ones (see <project:custom-components/estimators.md>).

``wf.evaluate`` is the connection point between the wavefunction and everything else. The workflow passes it to ``configure_sample_plan`` (along with an explicit sampler mapping such as ``{"electrons": sampler}``), ``configure_optimizer`` uses it for curvature estimation, and ``TotalEnergy`` uses it for parameter updates. ``build()`` produces a fully-wired {class}`~jaqmc.workflow.stage.vmc.VMCWorkStage`.

Here ``wf.evaluate`` works as the log-amplitude function because this wavefunction's ``__call__`` returns $\log|\psi|$ directly. Production wavefunctions expose dedicated methods like ``logpsi`` and ``phase_logpsi`` for different consumers — see <project:wavefunctions.md>.

:::{admonition} Wiring rule
:class: note

**The builder wires the optimizer and log-psi gradients. The workflow wires everything else.** The hydrogen workflow calls ``wire(self.estimators.energy.kinetic, f_log_psi=self.wf.evaluate)`` before ``configure_estimators(self.estimators)``. <project:custom-components/index.md> covers the mechanism.
:::

``data_init`` is set on the workflow rather than on individual stages, since the initial electron configurations are shared across all stages (pretrain, train, evaluation).

## Step 6: Build an Evaluation Workflow

Evaluation uses {class}`~jaqmc.workflow.evaluation.EvaluationWorkflow` instead of ``VMCWorkflow``. The stage needs no optimizer or loss gradients — it only samples and evaluates:

```{literalinclude} ../../src/jaqmc/app/hydrogen_atom.py
:pyobject: HydrogenAtomEvalWorkflow
```

The structure mirrors the training workflow but without optimizer and loss
gradient configuration. The optional density estimator defaults to
`estimators.density.enabled=false`, and `estimators.density.enabled=true` turns it on.

``EvaluationWorkflow`` creates fresh state before deciding how to initialize
the run. When ``workflow.source_path`` is set, it loads ``params``,
``batched_data``, and ``sampler_state`` from that training checkpoint. Without
one, evaluation can proceed only when the initialized parameter PyTree is
empty; it then uses the fresh parameters, walkers, and sampler state.
Wavefunctions with parameter leaves therefore require ``workflow.source_path``.
Point evaluation at a training output directory by setting it in your config:

```yaml
workflow:
  source_path: ./my-training-output  # directory containing train_ckpt_*.npz files
```

## Step 7: Add CLI Support

``make_cli`` provides a CLI with YAML configuration files (``--yml``),
dotlist overrides, and dry-run mode (``--dry-run``):

```python
from jaqmc.app.hydrogen_atom import HydrogenAtomTrainWorkflow
from jaqmc.utils.cli import make_cli

@make_cli
def main(configs, dotlist, dry_run):
    workflow = HydrogenAtomTrainWorkflow.from_config(configs, dotlist)
    workflow.initialize_runtime(dry_run=dry_run)
    return workflow(dry_run=dry_run)

if __name__ == "__main__":
    main()
```

Run with ``python my_workflow.py``, or ``jaqmc hydrogen-atom train`` for the built-in example. See <project:../guide/running-workflows.md> for CLI options and YAML configuration.

::::{admonition} Multiple subcommands (train + evaluate)
:class: dropdown

To support both train and evaluate subcommands, use a Click group:

```python
import click
from jaqmc.utils.cli import make_cli

@click.group()
def cli():
    pass

@cli.add_command
@make_cli(name="train", help="Train the model.")
def train(configs, dotlist, dry_run):
    from my_module import MyTrainWorkflow
    workflow = MyTrainWorkflow.from_config(configs, dotlist)
    workflow.initialize_runtime(dry_run=dry_run)
    return workflow(dry_run=dry_run)

@cli.add_command
@make_cli(name="evaluate", help="Evaluate a trained model.")
def evaluate(configs, dotlist, dry_run):
    from my_module import MyEvalWorkflow
    workflow = MyEvalWorkflow.from_config(configs, dotlist)
    workflow.initialize_runtime(dry_run=dry_run)
    return workflow(dry_run=dry_run)

if __name__ == "__main__":
    cli()
```
::::

## Programmatic Usage

``from_config`` constructs a {class}`~jaqmc.workflow.base.Workflow` instance,
and calling the workflow executes it. Passing ``dry_run=True`` to
``initialize_runtime`` and the workflow call prepares the configuration without
initializing the distributed runtime or running the stages:

```python
from jaqmc.app.hydrogen_atom import HydrogenAtomTrainWorkflow

config_dict = {
    "workflow": {"seed": 42, "save_path": "./output"},
    "train": {
        "run": {"iterations": 100},
        "optim": {"learning_rate": {"rate": 0.05}},
    },
}

wf = HydrogenAtomTrainWorkflow.from_config(config_dict)
wf.initialize_runtime()
wf()

# dry_run=True configures without executing (useful for inspecting resolved config)
wf = HydrogenAtomTrainWorkflow.from_config(config_dict)
wf.initialize_runtime(dry_run=True)
wf(dry_run=True)
```

## Production Workflows

Production workflows (see {ghsrc}`src/jaqmc/app/molecule/` and
{ghsrc}`src/jaqmc/app/solid/`) follow the same structure as the hydrogen
workflow, with options declared as dataclass fields and stages constructed in
``build_runtime()``. Placing runtime-only work such as SCF calculations in
``build_runtime()`` ensures that dry runs inspect the configuration without
performing it.

## Next Steps

- <project:custom-components/index.md> — make estimators and other components user-tunable via YAML
- <project:wavefunctions.md> — build neural network ansatzes with Flax/Linen
- <project:configuration.md> — how config resolution and overrides work under the hood
- {ghsrc}`src/jaqmc/app/molecule/` — a more complex workflow example
