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
from typing import Callable, Generator, Iterable, List, Optional, TypeVar

Config = TypeVar("Config")
"""A generic type variable representing a hyperparameter configuration."""


class ConfigEvaluation(typing.NamedTuple):
    """Contains the results of evaluating a hyperparameter configuration."""

    config: Config
    loss: float


def _top_k(configs: Iterable[Config], losses: Iterable[float], k: int) -> List[Config]:
    """Returns the k configs with the best (lowest) losses."""
    assert k > 0.0
    return zip(*heapq.nsmallest(k, zip(losses, configs)))[1]


class HyperBand:
    """An implementation of HyperBand.

    Variable names follow the original paper for clarity.

    Args:
        get_hyperparameter_configuration: A function that takes a number n and returns
            n randomly sampled hyperparameter configurations.

        run_then_return_val_loss: A function that takes a hyperparameter configuration
            and a resource allocation and returns the validation loss after training
            using the amount of specified resources.

        R: The maximum resources allocated to any hyperparameter configuration. Must be
            greater than 1.0.

        eta: A factor controlling the poportion of hyperparameter configurations to cull
            in each iteration of SuccessiveHalving. The optimal value is e (2.718...),
            typical values are 3 or 4. Must be nonnegative.

    Raises:
        ValueError: if R is less than 1.0, or if eta is not strictly positive.

    """

    def __init__(
        self,
        get_hyperparameter_configuration: Callable[[int], Iterable[Config]],
        run_then_return_val_loss: Callable[[Config, float], float],
        R: float,
        eta: float = 3.0,
    ) -> None:
        self.get_hyperparameter_configuration = get_hyperparameter_configuration
        self.run_then_return_val_loss = run_then_return_val_loss

        if R < 1.0:
            raise ValueError("R is {0:.2f}, but it must be greater than 1.0.".format(R))
        self.R = R

        if eta <= 0.0:
            raise ValueError("eta is {0:.2f}, but it must be strictly positive.")
        self.eta = eta

        self.best_config: Optional[Config] = None
        self.s_max = math.floor(math.log(self.R, base=self.eta))
        self.B = (self.s_max + 1) * self.R

    def step_generator(self) -> Generator[ConfigEvaluation, None, None]:
        """Returns a generator that takes steps of HyperBand.

        On each step, it will run one bracket of SuccessiveHalving and yield a named
        tuple `(config, loss)` containing the best hyperparameter config seen so far, and
        its associated loss.

        """
        for s in range(self.s_max, -1, -1):
            n = math.ceil((self.B * math.pow(self.eta, s)) / (self.R * (s + 1)))
            r = self.R * math.pow(self.eta, -s)
            T = self.get_hyperparameter_configuration(n)

            for i in range(s + 1):
                n_i = math.floor(n * math.pow(self.eta, -i))
                r_i = r * math.pow(self.eta, i)
                L = [self.run_then_return_val_loss(t, r_i) for t in T]
                T = _top_k(L, T, math.floor(n_i / self.eta))

            self.best_config = T[0]
            yield self.best_config

    def run(self) -> ConfigEvaluation:
        """Runs Hyperband and returns the best config.

        Returns:
            A NamedTuple `(config, loss)` containing the best hyperparameter config and
            its associated loss.

        Raises:
            RuntimeError: if no hyperparameter configs were evaluated due to
            mis-specification of Hyperband's parameters.

        """
        generator = self.step_generator()
        best_config = None
        for best_config in generator:
            pass

        if best_config is None:
            raise RuntimeError(
                "No hyperparameter configs were evaluated. Double-check your settings?"
            )
        return best_config
