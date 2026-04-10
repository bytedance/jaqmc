# Work Stages

Work stages are the execution units within a <project:workflows.md>. Each stage encapsulates its own loop with, for example, optimizer, sampler, estimators, writers, etc.

Stages are created through the **builder pattern**. For example, call ``VMCWorkStage.builder(cfg, wf)`` for training, or ``EvaluationWorkStage.builder(cfg, wf)`` for evaluation, then configure with ``configure_*`` methods and call ``build()``.

## VMC stage

```{eval-rst}
.. autoclass:: jaqmc.workflow.stage.vmc.VMCStageBuilder
   :members: configure_sample_plan, configure_estimators, configure_writers, configure_optimizer, configure_loss_grads, build

.. autoclass:: jaqmc.workflow.stage.sampling.SamplingStageBuilder
   :members:

.. autoclass:: jaqmc.workflow.stage.vmc.VMCWorkStage

.. autoclass:: jaqmc.workflow.stage.vmc.VMCWorkStageConfig
```

## Evaluation stage

```{eval-rst}
.. autoclass:: jaqmc.workflow.stage.evaluation.EvalStageBuilder
   :members: configure_sample_plan, configure_estimators, configure_writers, build

.. autoclass:: jaqmc.workflow.stage.evaluation.EvaluationWorkStage

.. autoclass:: jaqmc.workflow.stage.evaluation.EvaluationWorkStageConfig
```

## State classes

```{eval-rst}
.. autoclass:: jaqmc.workflow.stage.base.RunContext
   :members:

.. autoclass:: jaqmc.workflow.stage.base.StageAbort
   :members:

.. autoclass:: jaqmc.workflow.stage.base.WorkStage
   :members:

.. autoclass:: jaqmc.workflow.stage.base.WorkStageConfig

.. autoclass:: jaqmc.workflow.stage.sampling.SamplingState
   :members:

.. autoclass:: jaqmc.workflow.stage.vmc.VMCState
   :members:
```
