"""A quick demo of hyperband."""
import random
import typing
from typing import Iterable

import hyperband


class Config(typing.NamedTuple):
    """A toy hyperparameter configuration."""

    rho: float


def get_hyperparameter_configuration(n: int) -> Iterable[Config]:
    """Returns an iterable of `n` random hyperparameter configurations."""
    return [Config(random.uniform(-100, 100)) for _ in range(n)]


def run_then_return_val_loss(config: Config, resources: float) -> float:
    """Samples a noisy quadratic with minimum at 0.

    If resources is significantly below 100, the sample will be noisy, if it is above, it
    will be more precise.

    """
    loss = random.normalvariate(config.rho ** 2, 1.0 / resources)
    return loss


if __name__ == "__main__":

    tuner = hyperband.Hyperband(
        get_hyperparameter_configuration, run_then_return_val_loss, R=81.0, eta=3.0
    )
    best_config = tuner.run()
    print("Best config: {0}".format(best_config))
