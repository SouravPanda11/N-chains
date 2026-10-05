"""Independent N-chain implementation; no dependency on the reference notebook."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ChainConfig:
    n: int = 5
    slip: float = 0.0
    small: float = 2.0
    large: float = 10.0
    horizon: int = 100

    def __post_init__(self):
        if self.n < 2 or self.horizon < 1:
            raise ValueError("n must be >= 2 and horizon >= 1")
        if not 0 <= self.slip <= 1:
            raise ValueError("slip must be in [0, 1]")
        if not np.isfinite([self.small, self.large]).all():
            raise ValueError("rewards must be finite")


class NChain:
    """Action 0 advances; action 1 resets. Slip flips the chosen action.

    Entering the last state pays zero. Advancing while already there pays large.
    The chain never terminates naturally; horizon is a sampling time limit.
    """

    def __init__(self, config=ChainConfig(), seed=0):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.state = 0
        self.elapsed = 0
        return self.state

    def step(self, action):
        if action not in (0, 1):
            raise ValueError("action must be 0 or 1")
        if self.elapsed >= self.config.horizon:
            raise RuntimeError("reset after the time limit")
        slipped = self.rng.random() < self.config.slip
        executed = 1 - action if slipped else action
        if executed == 1:
            self.state, reward = 0, self.config.small
        elif self.state == self.config.n - 1:
            reward = self.config.large
        else:
            self.state += 1
            reward = 0.0
        self.elapsed += 1
        return self.state, reward, False, self.elapsed == self.config.horizon, {
            "executed_action": executed, "slipped": bool(slipped)
        }


def expected_return(probabilities, config):
    """Exact, bonus-free expected undiscounted horizon return from state zero.

    This diagnostic integrates slip and policy randomness, avoiding evaluation
    Monte Carlo noise. It is never used for training.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (config.n, 2):
        raise ValueError("expected probabilities shaped (n, 2)")
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or not np.allclose(probabilities.sum(axis=1), 1)):
        raise ValueError("invalid action probabilities")
    forward = (1 - config.slip) * probabilities[:, 0] + config.slip * probabilities[:, 1]
    reward = (1 - forward) * config.small
    reward[-1] += forward[-1] * config.large
    distribution = np.zeros(config.n)
    distribution[0] = 1
    total = 0.0
    destinations = np.minimum(np.arange(config.n) + 1, config.n - 1)
    for _ in range(config.horizon):
        total += float(distribution @ reward)
        new = np.zeros(config.n)
        new[0] = distribution @ (1 - forward)
        np.add.at(new, destinations, distribution * forward)
        distribution = new
    return total
