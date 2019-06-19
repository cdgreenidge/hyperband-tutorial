# Hyperband implementation

This repository contains an implementation of the Hyperband hyperparameter tuning
algorithm found in

> Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A. & Talwalkar, A. Hyperband: A Novel
> Bandit-Based Approach to Hyperparameter Optimization.
> J. Mach. Learn. Res. 18, 1–52 (2018).

The code is well-documented, so you should be able to easily adapt it for your
application. It uses [dask-jobqueue](https://jobqueue.dask.org/en/latest/) to enable
parallelism on SLURM-based clusters like *scotty*. The main class of interest is
`hyperband.Hyperband`.

To run a local toy hyperparameter search, run the following shell command:

```
$ python3 hyperband_demo.py
```

To run a search on the cluster, change the command to

```
$ python3 hyperband_demo.py --use_slurm=True
```

Note that you will have to submit the command via `sbatch` so that the scheduler and
worker processes can communicate with each other. An example sbatch script is provided,
`submit.sh`, though you will likely have to increase the requested time limit for a real
application.

For a real application, you will also have to modify the following two functions in
`hyperband_demo.py`:

1. `get_hyperparameter_configuration(n)`, which randomly samples `n` hyperparameter
   configurations from the search space.
2. `run_then_return_val_loss(config, resources)`, which trains a hyperparameter
   configuration using the specified amount of resources and returns the validation loss.

Be sure to tune the cluster settings to use the correct amount of
memory, cores, processes and walltime for your task. For more
information, see the [dask-jobqueue
documentation](https://jobqueue.dask.org/en/latest/install.html).