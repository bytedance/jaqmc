# Workflows

Workflows orchestrate the full training or evaluation pipeline. A workflow composes one or more <project:stages.md> — for example, a training workflow might run a pretrain stage followed by a VMC training stage.

Workflows are configurable dataclasses whose fields define the run. Concrete workflows subclass {class}`~jaqmc.workflow.vmc.VMCWorkflow` or {class}`~jaqmc.workflow.evaluation.EvaluationWorkflow` and implement `build_runtime` to construct their stages.

```{eval-rst}
.. autoclass:: jaqmc.workflow.base.Workflow
   :members: from_config, to_yaml, initialize_runtime, run, prepare

.. autoclass:: jaqmc.workflow.vmc.VMCWorkflow
   :members: run, restore_checkpoint

.. autoclass:: jaqmc.workflow.evaluation.EvaluationWorkflow
   :members: run
```

## Workflow configuration

```{eval-rst}
.. autoclass:: jaqmc.workflow.base.WorkflowConfig
.. autoclass:: jaqmc.workflow.evaluation.EvaluationWorkflowConfig
```
