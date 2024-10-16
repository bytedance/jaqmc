# JaQMC: JAX accelerated Quantum Monte Carlo

A collection of GPU-friendly and neural-network-friendly scalable 
Quantum Monte Carlo (QMC) implementations in JAX.

Currently supported functionalities:
- Diffusion Monte Carlo (DMC)
- Spin Symmetry Enforcement 

## Installation
JaQMC can be installed via the supplied setup.py file.
```shell
pip3 install -e .
```

## Introduction

JaQMC is modularizely designed for easier integration with various Neural
Network based Quantum Monte Carlo (NNQMC) projects. 

The functionalities are developed in `jaqmc` module, while we provide a number
of scripts integrating with different NNQMC projects in `example` directory.

### Diffusion Monte Carlo (DMC)
The fixed-node DMC implementation introduced in 
[Towards the ground state of molecules via diffusion Monte Carlo on neural networks](https://www.nature.com/articles/s41467-023-37609-3)

See [DMC](#dmc) section for more details. 

### Spin Symmetry Enforcement with $\hat{S}_+$ penalties
The spin symmetry enforced solution introduced in [Symmetry enforced solution of the many-body Schrödinger equation with deep neural network](https://arxiv.org/abs/2406.01222)

See [Spin Symmetry](#spin-symmetry) section for more details. 

## DMC
The fixed-node diffusion Monte Carlo (FNDMC) implementation here has a simple interface.
In the simplest case, it requires only a (real-valued) trial wavefunction, taking in a dim-3N electron configuration and producing two outputs: 
the sign of the wavefunction value and the logarithm of its absolute value.
In more sophisticated cases, users can also provide the implementation of local energy and quantum force, for instance, when ECP is considered.

Several examples are provided integrating with neural-network-based trial wavefunctions. The DMC related config can be found in the `examples/dmc_config.py`.
See [here](https://github.com/google/ml_collections/tree/master#config-flags) for instructions on how to play with those config or flags.

### Integration with FermiNet
Please first install FermiNet following instructions in https://github.com/deepmind/ferminet. 
Then train FermiNet for your favorite atom / molecule and generate a checkpoint to be reused in DMC as the trial wavefunction.
```shell
python3 examples/dmc/ferminet/run.py --config $YOUR_FERMINET_CONFIG_FILE --config.log.save_path $YOUR_FERMINET_CKPT_DIRECTORY --dmc_config.iterations 100 --dmc_config.fix_size --dmc_config.block_size 10 --dmc_config.log.save_path $YOUR_DMC_CKPT_DIRECTORY
```

### Integration with LapNet
Please first install LapNet following instructions in https://github.com/bytedance/lapnet.
Then train LapNet for your favorite atom / molecule and generate a checkpoint to be reused in DMC as the trial wavefunction.
```shell
python3 examples/dmc/lapnet/run.py --config $YOUR_LAPNET_CONFIG_FILE --config.log.save_path $YOUR_LAPNET_CKPT_DIRECTORY --dmc_config.iterations 100 --dmc_config.fix_size --dmc_config.block_size 10 --dmc_config.log.save_path $YOUR_DMC_CKPT_DIRECTORY
```

### Integration with DeepErwin
Please first install DeepErwin following instructions in https://mdsunivie.github.io/deeperwin/. 
Then train DeepErwin for your favorite atom / molecule and generate a checkpoint to be reused in DMC as the trial wavefunction.
```shell
python3 examples/dmc/deeperwin/run.py --deeperwin_ckpt $YOUR_DEEPERVIN_CKPT_FILE --dmc_config.iterations 100 --dmc_config.fix_size --dmc_config.block_size 10 --dmc_config.log.save_path $YOUR_DMC_CKPT_DIRECTORY
```

### Do Your Own Integration
The entry point for DMC integration is the `run` function in `jaqmc/dmc/dmc.py`, which is quite heavily commented.
Basically you only need to construct your favorite trial wavefunction in JAX, then simply pass it to this `run` function and it should work smoothly.
Please don't hesitate to file an issue if you need help to integrate with your favorite (JAX-implemented) trial wavefunction.

Note that our DMC implementation is "multi-node calculation ready" in the sense that if you initialize the distributed JAX runtime 
 on a multi-node cluster, then our DMC implementation can do multi-node calculation correctly, i.e. aggregation across
 different computing nodes. See [here](https://jax.readthedocs.io/en/latest/multi_process.html?highlight=multi-node) for instructions on initialization of the distributed JAX runtime.


### Output
The data at each checkpoint step will be stored in the specified path (namely `$YOUR_DMC_CKPT_DIRECTORY` in the examples above) with the naming pattern
```
dmc_data_{step}.tgz
```
which contains a csv file with the metric produced from each DMC step up to the checkpoint step.
The columns of the metric file are
1. step: The step index in DMC
2. estimator: The mixed estimator calculated at each step, calculated and smoothed within a certain time window.
3. offset: The energy offset used to update DMC walker weights. 
4. average: The local energy weighted average calculated at each DMC step.
5. num_walkers: The total number of walkers across all the computing nodes.
6. old_walkers: The number of walkers got rejected for too many times in the process.
7. total_weight: The total weight of all walkers across all the computing nodes.
8. acceptance_ratio: The acceptence ratio of the acceptence-rejection action.
9. effective_time_step: The effective time step
10. num_cutoff_updated, num_cutoff_orig: Debug related, indicating the number of outliers in terms of local energy.

## Spin Symmetry
We enforce the spin symmetry with two steps:
1. Set the spin magnetic spin number to be the target spin value $s_z = s$, by setting the number of spin-up and spin-down electrons in the input of the neural network wavefunction.
2. Integrate $\hat{S}_+$  penalty into the loss function to enforce the spin symmetry.

We implement `loss` module in JaQMC for that purpose.
For each component of loss, such as VMC energy and spin related penalties, we build a 
factory method to produce losses with the same interface:
```
class Loss(Protocol):
  def __call__(self,
               params: ParamTree,
               func_state: BaseFuncState,
               key: chex.PRNGKey,
               data: jnp.ndarray) -> Tuple[jnp.ndarray, Tuple[BaseFuncState, BaseAuxData]]:
    """
    Args:
      params: network parameters.
      func_state: function state passed to the loss function to control its behavior.
      key: JAX PRNG state.
      data: QMC walkers with electronic configuration to evaluate.
    Returns:
      (loss value, (updated func_state, auxillary data)
    """
```
This loss interface works well with [KFAC optimizer](https://github.com/google-deepmind/kfac-jax).
It is also flexible enough to work with optimizers in [optax](https://github.com/google-deepmind/optax),
[SPRING](https://github.com/jeffminlin/vmcnet/blob/master/vmcnet/updates/spring.py) and etc.

We also provide user-facing entry points in `jaqmc/loss/factory.py`. 
One for building `func_state`, one of the inputs to the loss function, and
another one for building the loss function.
```
def build_func_state(step=None) -> FuncState:
    '''
    Helper function to create parent FuncState from actual data.
    '''
    ......

```

### Integration with LapNet
Please first install LapNet following instructions in https://github.com/bytedance/lapnet.
To simulate singlet state for Oxygen atom with LapNet and spin symmetry enforced, simply turn on `loss_config.enforce_spin.with_spin` flag
as follows.
```shell
python3 $JAQMC_PATH/examples/loss/lapnet/run.py --config $JAQMC_PATH/examples/loss/lapnet/atom_spin_state.py:O,0 
--loss_config.enforce_spin.with_spin --config.$OTHER_LAPNET_CONFIGS --loss_config.enforce_spin.$OTHER_SPIN_CONFIGS
```
Note that this example script is by no means "production-ready". It is just a
show case on how to integrate the `loss` module with exisiting NNQMC projects. 
For instance, it's not including the pretrain phase since it has nothing to do
with the `loss` module.

## Giving Credit
If you use certain functionalities of JaQMC in your work, please consider citing the corresponding papers.
### DMC paper
```
@article{ren2023towards,
  title={Towards the ground state of molecules via diffusion Monte Carlo on neural networks},
  author={Ren, Weiluo and Fu, Weizhong and Wu, Xiaojie and Chen, Ji},
  journal={Nature Communications},
  volume={14},
  number={1},
  pages={1860},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```

### Spin Symmetry paper
```
@article{li2024symmetry,
  title={Symmetry enforced solution of the many-body Schr$\backslash$" odinger equation with deep neural network},
  author={Li, Zhe and Lu, Zixiang and Li, Ruichen and Wen, Xuelan and Li, Xiang and Wang, Liwei and Chen, Ji and Ren, Weiluo},
  journal={arXiv preprint arXiv:2406.01222},
  year={2024}
}
```
