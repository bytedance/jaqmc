# Workflows

Workflows orchestrate the full training or evaluation pipeline. A workflow composes one or more <project:stages.md> — for example, a training workflow might run a pretrain stage followed by a VMC training stage.

Use the built-in {class}`~jaqmc.workflow.vmc.VMCWorkflow` and {class}`~jaqmc.workflow.evaluation.EvaluationWorkflow` base classes, or write a plain function that creates and returns a workflow instance.

```{eval-rst}
.. autoclass:: jaqmc.workflow.base.Workflow
   :members: run, prepare

.. autoclass:: jaqmc.workflow.vmc.VMCWorkflow
   :members: run, restore_checkpoint

.. autoclass:: jaqmc.workflow.evaluation.EvaluationWorkflow
   :members: run
```

## Workflow configuration

```{eval-rst}
.. autoclass:: jaqmc.workflow.base.WorkflowConfig
.. autoclass:: jaqmc.workflow.base.ConfigCheck
.. autoclass:: jaqmc.workflow.evaluation.EvaluationWorkflowConfig
```
