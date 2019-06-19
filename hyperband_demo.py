"""A quick demo of hyperband."""
import random
import typing
from typing import Iterable

import dask.distributed
import dask_jobqueue

import hyperband


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

    If resources is significantly below 100, the sample will be noisy, if it is above, it
    will be more precise.

    """
    loss = random.normalvariate(config.rho ** 2, 40.0 / resources)
    return loss


if __name__ == "__main__":
    use_slurm = True
    if use_slurm:
        cluster = dask_jobqueue.SLURMCluster(
            cores=4,
            processes=4,
            memory="2GB",
            walltime="0:00:05",
            queue="all",
            local_directory="/tmp/",
            interfacestr="em2",
        )
        cluster.scale(16)  # Can be 10, 100, ...
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
    print("Best config: {0}".format(best_config))
    cluster.close()
