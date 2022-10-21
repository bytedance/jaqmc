# JaQMC: JAX accelerated Quantum Monte Carlo

A collection of GPU-friendly and neural-network-friendly scalable QMC implementations in JAX.

## Installation
JaQMC can be installed via the supplied setup.py file.
```shell
pip3 install -e .
```

## DMC
The fixed-node diffusion Monte Carlo (FNDMC) implementation here has a simple interface.
In the simplest case, it requires only a (real-valued) trial wavefunction, taking in a dim-3N electron configuration and producing two outputs: 
the sign of the wavefunction value and the logarithm of its absolute value.
In more sophisticated cases, users can also provide the implementation of local energy and quantum force, for instance, when ECP is considered.

Two examples are provided integrating with neural-network-based trial wavefunctions. The DMC related config can be found in the `examples/dmc_config.py`.
See [here](https://github.com/google/ml_collections/tree/master#config-flags) for instructions on how to play with those config / flags.

### Integration with FermiNet
Please first install FermiNet following instructions in https://github.com/deepmind/ferminet. 
Then train FermiNet for your favorite atom / molecule and generate a checkpoint to be reused in DMC as the trial wavefunction.
```shell
python3 examples/dmc/ferminet/run.py --config $YOUR_FERMINET_CONFIG_FILE --config.log.save_path $YOUR_FERMINET_CKPT_DIRECTORY --dmc_config.iterations 100 --dmc_config.fix_size --dmc_config.block_size 10 --dmc_config.log.save_path $YOUR_DMC_CKPT_DIRECTORY
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

### Giving Credit
If you use this FNDMC implementation in your work, please cite the associated paper.
```
@article{ren2022towards,
  title={Towards the ground state of molecules via diffusion Monte Carlo on neural networks},
  author={Ren, Weiluo and Fu, Weizhong and Chen, Ji},
  journal={arXiv preprint arXiv:2204.13903},
  year={2022}
}
```
