"""A quick demo of hyperband."""
# For some reason the yaml package throws a spurious warning, so let's disable it before
# we do anything else. This requires us to disable flake8 on the subsequent imports
import yaml

yaml.warnings({"YAMLLoadWarning": False})


import argparse  # noqa: E402
import random  # noqa: E402
import typing  # noqa: E402
from typing import Iterable  # noqa: E402

import dask.distributed  # noqa: E402
import dask_jobqueue  # noqa: E402

import hyperband  # noqa: E402


class Config(typing.NamedTuple):
    """A toy hyperparameter configuration."""

    rho: float

    def __str__(self) -> str:  # noqa: D105
        return "Config(rho={0:.2f})".format(self.rho)


def get_hyperparameter_configuration(n: int) -> Iterable[Config]:
    """Return an iterable of `n` random hyperparameter configurations."""
    return [Config(random.uniform(-100, 100)) for _ in range(n)]


def run_then_return_val_loss(config: Config, resources: float) -> float:
    """Sample a noisy quadratic with minimum at 0.

    If resources is significantly below 40.0, the sample will be
    noisy, if it is above, it will be more precise.

    """
    loss = random.normalvariate(config.rho ** 2, 40.0 / resources)
    return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a hyperparameter search.")
    parser.add_argument(
        "--use_slurm",
        type=bool,
        default=False,
        help=(
            "set to True to use a SLURM cluster, False to run locally. If True, make"
            + " sure you run this script with sbatch, so the scheduler is on the same "
            + "network as the worker nodes."
        ),
    )
    args = parser.parse_args()

    if args.use_slurm:
        cluster = dask_jobqueue.SLURMCluster(
            cores=4,
            processes=4,
            memory="2GB",
            walltime="0:00:05",
            queue="all",
            local_directory="/tmp/",
            interfacestr="em2",
        )
        cluster.scale(16)  # Ask the cluster for 16 worker processes, wait until arrival

        # Print a link to the HTTP diagnostics server
        print("Dashboard link: {0}".format(cluster.dashboard_link))
    else:
        cluster = dask.distributed.LocalCluster(processes=False, dashboard_address=None)
    client = dask.distributed.Client(cluster)

    tuner = hyperband.Hyperband(
        get_hyperparameter_configuration,
        run_then_return_val_loss,
        R=81.0,
        eta=3.0,
        client=client,
    )
    best_config = tuner.run()
    cluster.close()

    print("Best config: {0}".format(best_config))
