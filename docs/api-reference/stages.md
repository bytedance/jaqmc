# Work Stages

Work stages are the execution units within a <project:workflows.md>. Each stage encapsulates its own loop with, for example, optimizer, sampler, estimators, writers, etc.

Stages are created through the **builder pattern**. ``VMCWorkStage.builder(run_config, wf, name="train")`` creates a training stage builder, and ``EvaluationWorkStage.builder(run_config, wf, name="evaluation")`` creates an evaluation stage builder. Calling the ``configure_*`` methods and ``build()`` produces the configured stage.

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
