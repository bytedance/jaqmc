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

The simplest estimator is a plain function with signature `(params, data, stats, state, rngs) -> (dict, state)`:

```{literalinclude} ../../src/jaqmc/app/hydrogen_atom.py
:pyobject: potential_energy
```

The signature ``(params, data, stats, state, rngs)`` is the standard estimator interface. ``stats`` (called ``prev_walker_stats`` in the {class}`~jaqmc.estimator.base.Estimator` class interface) contains values from other estimators in the current step (for derived quantities like total energy), and ``state`` carries mutable state across iterations. Simple estimators like this one can ignore most parameters — ``del`` marks them as intentionally unused.

The ``energy:`` prefix is a naming convention: any estimator that returns a key starting with ``energy:`` contributes to the total energy, which becomes the VMC optimization target. A {class}`~jaqmc.estimator.total_energy.TotalEnergy` estimator auto-sums all ``energy:``-prefixed keys into a ``total_energy`` value — you'll use it in Step 5.

Pass the function directly in the estimators dict — JaQMC wraps it automatically. For estimators that need configuration fields, subclass {class}`~jaqmc.estimator.base.Estimator` instead — see <project:custom-components/index.md>.

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

With the individual pieces ready, the workflow function assembles them into a training stage — this is the configure phase from the <project:index.md>:

```{literalinclude} ../../src/jaqmc/app/hydrogen_atom.py
:pyobject: hydrogen_atom_train_workflow
```

The function creates a {class}`~jaqmc.workflow.vmc.VMCWorkflow`, configures a training stage, and returns the assembled workflow. ``VMCWorkflow`` handles output directory creation, configuration saving, and stage execution. Three patterns in this function deserve explanation:

- ``cfg.use_preset(...)`` sets default config values that the user's YAML can override. Here it configures the console writer to show acceptance rate, energy, and variance.
- ``cfg.get("energy.kinetic", EuclideanKinetic(...))`` provides a default estimator that users can replace via config. This is how JaQMC makes components swappable — see <project:custom-components/index.md>.
- ``cfg.scoped("train")`` gives the builder a config view restricted to the ``train`` section, so config reads like ``sampler.steps`` resolve to ``train.sampler.steps`` in the full config.

Estimators can be plain functions (like ``potential_energy``) or class instances. Use a class when the estimator needs configurable parameters or a wavefunction reference — ``EuclideanKinetic`` takes ``wf.evaluate`` so it can differentiate the wavefunction for the Laplacian.

``wf.evaluate`` is the connection point between the wavefunction and everything else. The workflow passes it to ``configure_sample_plan`` (along with an explicit sampler mapping such as ``{"electrons": sampler}``), ``configure_optimizer`` uses it for curvature estimation, and ``configure_loss_grads`` uses it for parameter updates. ``build()`` produces a fully-wired {class}`~jaqmc.workflow.stage.vmc.VMCWorkStage`.

Here ``wf.evaluate`` works as the log-amplitude function because this wavefunction's ``__call__`` returns $\log|\psi|$ directly. Production wavefunctions expose dedicated methods like ``logpsi`` and ``phase_logpsi`` for different consumers — see <project:wavefunctions.md>.

(wiring-principles)=
:::{admonition} Wiring rule
:class: note

**The builder wires what it creates; you wire everything else.** Components loaded from config by builder methods (optimizer, writers) receive runtime deps through ``configure_*`` methods. For sampling, resolve or construct a sampler in the workflow and pass it via ``configure_sample_plan(wf.evaluate, {"electrons": sampler})``. Components you construct yourself (like estimators) receive dependencies through their constructor arguments. <project:custom-components/index.md> covers the wiring mechanism in detail.
:::

``data_init`` is set on the workflow rather than on individual stages, since the initial electron configurations are shared across all stages (pretrain, train, evaluation).

## Step 6: Build an Evaluation Workflow

Evaluation uses {class}`~jaqmc.workflow.evaluation.EvaluationWorkflow` instead of ``VMCWorkflow``. The stage needs no optimizer or loss gradients — it only samples and evaluates:

```{literalinclude} ../../src/jaqmc/app/hydrogen_atom.py
:pyobject: hydrogen_atom_eval_workflow
```

The structure mirrors the training workflow but without ``configure_optimizer`` and ``configure_loss_grads``. The conditional block at the end adds an optional density estimator when enabled via config — ``cfg.get("estimators.enabled.density", False)`` reads a boolean flag, and the ``CartesianDensity`` default defines the histogram grid.

``EvaluationWorkflow`` loads ``params``, ``batched_data``, and ``sampler_state`` from the training checkpoint. Point it to the training output directory by setting ``workflow.source_path`` in your config:

```yaml
workflow:
  source_path: ./my-training-output  # directory containing train_ckpt_*.npz files
```

## Step 7: Add CLI Support

Wrap your workflow function with ``make_cli`` to get a CLI with YAML config files (``--yml``), dotlist overrides, and dry-run mode (``--dry-run``):

```python
from jaqmc.utils.cli import make_cli

if __name__ == "__main__":
    make_cli(hydrogen_atom_train_workflow)()
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
def train(cfg, dry_run):
    from my_module import my_train_workflow
    my_train_workflow(cfg, dry_run=dry_run)

@cli.add_command
@make_cli(name="evaluate", help="Evaluate a trained model.")
def evaluate(cfg, dry_run):
    from my_module import my_eval_workflow
    my_eval_workflow(cfg, dry_run=dry_run)

if __name__ == "__main__":
    cli()
```
::::

## Programmatic Usage

Workflow functions return a {class}`~jaqmc.workflow.base.Workflow` instance. Call it to execute, or pass ``dry_run=True`` to validate the configuration without running:

```python
from jaqmc.utils.config import ConfigManager

config_dict = {
    "workflow": {"seed": 42, "save_path": "./output"},
    "train": {
        "run": {"iterations": 100},
        "optim": {"learning_rate": {"rate": 0.05}},
    },
}

cfg = ConfigManager(config_dict)
wf = hydrogen_atom_train_workflow(cfg)
wf()

# dry_run=True configures without executing (useful for inspecting resolved config)
wf = hydrogen_atom_train_workflow(cfg)
wf(dry_run=True)
```

## Class-Based Workflows

Production workflows (see {ghsrc}`src/jaqmc/app/molecule/` and {ghsrc}`src/jaqmc/app/solid/`) subclass ``VMCWorkflow`` directly. This gives you control over the execution lifecycle — you can override ``run()`` to separate cheap configuration from expensive pre-computation (like SCF calculations):

```python
class MyWorkflow(VMCWorkflow):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.scf = MySCF(...)  # cheap: just builds the object
        wf = MyWavefunction(...)
        sampler = cfg.get("sampler", MCMCSampler)
        train = VMCWorkStage.builder(cfg.scoped("train"), wf)
        train.configure_sample_plan(wf.evaluate, {"electrons": sampler})
        train.configure_optimizer(default="jaqmc.optimizer.kfac", f_log_psi=wf.evaluate)
        train.configure_estimators(...)
        train.configure_loss_grads(f_log_psi=wf.evaluate)
        self.train_stage = train.build()
        self.data_init = data_init

    def run(self):
        self.scf.run()  # expensive: only runs during actual execution
        super().run()
```

This ensures ``--dry-run`` skips expensive computations while still resolving the full config. The function pattern from Step 5 is simpler when you don't need to override ``run()``.

## Next Steps

- <project:custom-components/index.md> — make estimators and other components user-tunable via YAML
- <project:wavefunctions.md> — build neural network ansatzes with Flax/Linen
- <project:configuration.md> — how config resolution and overrides work under the hood
- {ghsrc}`src/jaqmc/app/molecule/` — a more complex workflow example
