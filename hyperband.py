"""An implementation of the Hyperband hyperparameter tuning algorithm.

See Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A. & Talwalkar,
A. Hyperband: A Novel Bandit-Based Approach to Hyperparameter
Optimization. J. Mach. Learn. Res. 18, 1–52 (2018).

Author: C. Daniel Greenidge <dev@danielgreenidge.com>
Date: 2019-06-17

"""
import heapq
import math
import typing
from typing import Any, Callable, Generator, Sequence, Tuple, TypeVar

Config = TypeVar("Config")
"""A generic type variable representing a hyperparameter configuration."""


class ConfigEvaluation(typing.NamedTuple):
    """Contains the results of evaluating a hyperparameter configuration."""

    config: Any  # Using the Config generic causes a metaclass conflict before py37
    loss: float


def _top_k(
    configs: Sequence[Config], losses: Sequence[float], k: int
) -> Tuple[Tuple[Config, ...], Tuple[float, ...]]:
    """Return the k configs with the best (lowest) losses."""
    assert k >= 1
    losses, configs = zip(*heapq.nsmallest(k, zip(losses, configs)))
    return configs, losses


class Hyperband:
    """An implementation of Hyperband.

    Variable names follow the original paper for clarity.

    Args:
        get_hyperparameter_configuration: A function that takes a number n and returns
            n randomly sampled hyperparameter configurations.

        run_then_return_val_loss: A function that takes a hyperparameter configuration
            and a resource allocation and returns the validation loss after training
            using the amount of specified resources.

        R: The maximum resources allocated to any hyperparameter configuration. Must be
            greater than or equal to 1.0.

        eta: A factor controlling the poportion of hyperparameter configurations to cull
            in each iteration of SuccessiveHalving. The optimal value is e (2.718...),
            typical values are 3 or 4. Must be strictly positive.

    Raises:
        ValueError: if R is less than 1.0, or if eta is not strictly positive.

    """

    def __init__(  # noqa: D107
        self,
        get_hyperparameter_configuration: Callable[[int], Sequence[Config]],
        run_then_return_val_loss: Callable[[Config, float], float],
        R: float,
        eta: float = 3.0,
    ) -> None:
        self.get_hyperparameter_configuration = get_hyperparameter_configuration
        self.run_then_return_val_loss = run_then_return_val_loss

        if R < 1.0:
            raise ValueError("R is {0:.2f}, but it must be >= 1.0.".format(R))
        self.R = R

        if eta <= 0.0:
            raise ValueError("eta is {0:.2f}, but it must be > 0.".format(eta))
        self.eta = eta

        self.s_max = math.floor(math.log(self.R, self.eta))
        self.B = (self.s_max + 1) * self.R

    def step_generator(self) -> Generator[ConfigEvaluation, None, None]:
        """Return a generator that takes steps of Hyperband.

        On each step, it will run one bracket of SuccessiveHalving and yield a named
        tuple `(config, loss)` containing the best hyperparameter config seen so far, and
        its associated loss.

        """
        best_config = None
        for s in range(self.s_max, -1, -1):
            n = math.ceil((self.B * (self.eta ** s)) / (self.R * (s + 1)))
            r = self.R * (self.eta ** -s)
            T = self.get_hyperparameter_configuration(n)

            for i in range(s + 1):
                n_i = math.floor(n * (self.eta ** -i))
                r_i = r * (self.eta ** i)
                L = [self.run_then_return_val_loss(t, r_i) for t in T]
                T, losses = _top_k(T, L, max(1, math.floor(n_i / self.eta)))

            if best_config is None or losses[0] < best_config.loss:
                best_config = ConfigEvaluation(T[0], losses[0])

            yield best_config  # type: ignore

    def run(self) -> ConfigEvaluation:
        """Run Hyperband and returns the best config.

        Returns:
            A NamedTuple `(config, loss)` containing the best hyperparameter config and
            its associated loss.

        Raises:
            RuntimeError: if no hyperparameter configs were evaluated due to
            mis-specification of Hyperband's parameters.

        """
        best_config = None
        for best_config in self.step_generator():  # noqa: B007
            pass

        if best_config is None:
            raise RuntimeError(
                "No hyperparameter configs were evaluated. Double-check your settings?"
            )
        return best_config
