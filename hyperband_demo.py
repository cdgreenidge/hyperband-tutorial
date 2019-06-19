"""A quick demo of hyperband."""
import random
import typing
from typing import Iterable

import hyperband


class Config(typing.NamedTuple):
    """A toy hyperparameter configuration."""

    rho: float

    def __str__(self) -> str:  # noqa: D105
        return "Config(rho={0:.2f})".format(self.rho)


def get_hyperparameter_configuration(n: int) -> Iterable[Config]:
    """Return an iterable of `n` random hyperparameter configurations."""
    return [Config(random.uniform(-100, 100)) for _ in range(n)]


resources_used = 0


def run_then_return_val_loss(config: Config, resources: float) -> float:
    """Sample a noisy quadratic with minimum at 0.

    If resources is significantly below 100, the sample will be noisy, if it is above, it
    will be more precise.

    """
    global resources_used
    resources_used += resources

    loss = random.normalvariate(config.rho ** 2, 40.0 / resources)
    return loss


resources_used = 0
R = 81.0
tuner = hyperband.Hyperband(
    get_hyperparameter_configuration, run_then_return_val_loss, R=R, eta=3.0
)
best_config = tuner.run()
print("Best config: {0}".format(best_config))
print("Resources: {0:.2f}".format(resources_used / R))
