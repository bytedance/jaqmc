# Configuration System

JaQMC configuration is a statically defined dataclass tree. A concrete
workflow declares its accepted fields, component selections remain explicit,
and parsing rejects unknown keys before runtime construction begins. For YAML
files, command-line overrides, and dry runs, see
<project:../guide/configuration.md>.

## Parsing and Executing a Workflow

Concrete workflows provide {meth}`~jaqmc.workflow.base.Workflow.from_config`,
which merges one mapping or an ordered sequence of YAML mappings with optional
`key=value` dotlist overrides, and {meth}`~jaqmc.workflow.base.Workflow.to_yaml`,
which serializes the resolved configuration:

```python
from jaqmc.app.molecule.workflow import MoleculeTrainWorkflow

workflow = MoleculeTrainWorkflow.from_config(
    [{"workflow": {"batch_size": 1024}}],
    dotlist=["train.run.iterations=5000"],
)
print(workflow.to_yaml())
```

Parsing constructs the configuration tree and runs `__post_init__` hooks to
validate and normalize fields across the completed configuration. Heavy startup
work such as SCF calculations, runtime stage construction, and opening output
writers is deferred until execution.

Executing a parsed workflow is a two-step process:

```python
workflow.initialize_runtime()
workflow()
```

{meth}`~jaqmc.workflow.base.Workflow.initialize_runtime` applies logging, JAX
global flags, and distributed runtime initialization. Calling the workflow then
writes the resolved configuration snapshot, builds runtime stages, and runs
them. Passing `dry_run=True` to both calls applies logging and JAX flags,
validates paths, and logs the resolved YAML while skipping distributed
initialization, snapshot writing, SCF, stage construction, and execution.

Libraries that parse a standalone configurable dataclass without a workflow
can call {func}`~jaqmc.utils.config.parse_config` and
{func}`~jaqmc.utils.config.config_to_yaml` directly.

## Defining Configuration Schemas

{deco}`~jaqmc.utils.config.configurable_dataclass` decorates a configurable
workflow root or component. Ordinary dataclass fields define fixed values and
nested sections:

```python
from dataclasses import field

from jaqmc.utils.config import configurable_dataclass

@configurable_dataclass
class RunConfig:
    iterations: int = 1_000

@configurable_dataclass
class MyWorkflowConfig:
    run: RunConfig = field(default_factory=RunConfig)
```

Defaults belong on the dataclass field or its `default_factory`, and field
annotations must resolve to runtime types when the class is defined (do not use
`from __future__ import annotations`). During `parse_config`, YAML layers are
merged from left to right and dotlist overrides take highest precedence. Every
ordinary field must be known to the schema, and unknown keys or incompatible
values raise `pydantic.ValidationError`. `__post_init__` runs only after all
layers have been merged, so it can validate or normalize the completed
configuration.

Runtime dependencies declared with {func}`~jaqmc.utils.wiring.runtime_dep` are
excluded from both parsing and YAML serialization (see
<project:custom-components/index.md>).

When a field accepts a closed set of alternative schemas, a union of
configurable dataclasses selects between them based on the provided keys and
field types:

```python
@configurable_dataclass
class LatticeVectors:
    a: tuple[float, float, float]
    b: tuple[float, float, float]
    c: tuple[float, float, float]

@configurable_dataclass
class LatticeParameters:
    a: float
    b: float
    c: float
    alpha: float = 90
    beta: float = 90
    gamma: float = 90

@configurable_dataclass
class SolidConfig:
    lattice: LatticeVectors | LatticeParameters = field(
        default_factory=lambda: LatticeParameters(a=2, b=2, c=2)
    )
```

```yaml
system:
  lattice:
    a: [2.0, 0.0, 0.0]
    b: [0.0, 2.0, 0.0]
    c: [0.0, 0.0, 2.0]
```

Consecutive layers that validate against the active union variant merge field by
field, whereas switching to another variant resets to that variant's defaults so
fields from `LatticeParameters` (such as `alpha`) never carry over to
`LatticeVectors`.

(swappable-modules-extending)=
## Swappable Module Fields

{class}`~jaqmc.utils.config.ModuleConfig` annotates a field whose implementation
can be selected at runtime (such as a wavefunction, optimizer, or sampler).
Because `module` is the config key used to select an implementation, a
`@configurable_dataclass` cannot declare an ordinary field named `module`.

The field annotation provides both the static type seen by workflow code and
the runtime contract (any class or {func}`typing.runtime_checkable` protocol in
the annotated type) that the resolved object must satisfy during parsing:

```python
from dataclasses import field
from typing import Annotated

from jaqmc.optimizer.base import OptimizerLike
from jaqmc.optimizer.kfac import KFACOptimizer
from jaqmc.utils.config import ModuleConfig, configurable_dataclass

@configurable_dataclass
class TrainConfig:
    optim: Annotated[
        OptimizerLike, ModuleConfig(package="jaqmc.optimizer")
    ] = field(default_factory=KFACOptimizer)
```

Selected targets must be decorated with `@configurable_dataclass`. If a Python
module defines `__all__`, JaQMC resolves a bare module path to `__all__[0]`, so
users can omit the `:ClassName` suffix (see <project:#swappable-modules>). The
field's `default_factory` supplies the default component instance (for example,
`field(default_factory=lambda: adam(learning_rate=Constant(rate=3e-4)))`).

In YAML or dotlist overrides, a mapping with `module` selects an implementation
and a mapping without `module` updates the active one. Keeping the same module
across layers merges settings, while switching to a different module resets to
that module's defaults (see <project:#switching-modules-across-files>):

```yaml
train:
  optim:
    module: optax:adam
    learning_rate:
      rate: 0.001
```

Relative module paths resolve against `package`. When omitted, only
absolute module paths are accepted. Fields that also accept a plain scalar
shorthand (such as a float learning rate in place of a schedule object) include
the scalar type in the annotated union (for example,
`Annotated[float | Callable[..., Any], ModuleConfig(package="jaqmc.optimizer")]`).
When the field's default is a direct scalar value (such as `0.9`), a mapping
must include `module` explicitly to select an implementation.

## Named Module Collections

A dictionary of `ModuleConfig`-annotated values defines named, independently
selectable components such as stage writers:

```python
from dataclasses import field
from typing import Annotated

from jaqmc.utils.config import ModuleConfig, configurable_dataclass
from jaqmc.writer import Writer
from jaqmc.writer.console import ConsoleWriter

@configurable_dataclass
class TrainConfig:
    writers: dict[
        str, Annotated[Writer | None, ModuleConfig(package="jaqmc.writer")]
    ] = field(default_factory=lambda: {"console": ConsoleWriter()})
```

In configuration layers, each named entry merges and switches modules like an
individual `ModuleConfig` field. Setting an entry to `null` disables it. Adding
an entry or re-enabling a disabled one in a later layer requires a `module` key
(see <project:#replacing-module-components>).
