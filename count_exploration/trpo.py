"""Categorical neural policy, linear baseline, and a full TRPO update."""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters


@dataclass(frozen=True)
class TRPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 1.0
    max_kl: float = 0.01
    damping: float = 0.1
    cg_steps: int = 10
    backtracks: int = 10
    hidden: tuple = (32, 32)

    def __post_init__(self):
        if not 0 < self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("invalid discount or GAE lambda")
        if not np.isfinite([self.max_kl, self.damping]).all() or self.max_kl <= 0 or self.damping < 0:
            raise ValueError("invalid KL limit or damping")
        if min(self.cg_steps, self.backtracks) < 1 or not self.hidden or min(self.hidden) < 1:
            raise ValueError("invalid optimizer or network sizes")


class Policy(nn.Module):
    def __init__(self, n_states, hidden=(32, 32)):
        super().__init__()
        self.n_states = n_states
        layers = []
        for index, (in_dim, out_dim) in enumerate(zip((n_states,) + tuple(hidden), tuple(hidden) + (2,))):
            layer = nn.Linear(in_dim, out_dim, dtype=torch.float64)
            nn.init.orthogonal_(layer.weight, gain=0.01 if index == len(hidden) else np.sqrt(2))
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            if index < len(hidden):
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, states):
        x = torch.nn.functional.one_hot(states.long(), self.n_states).to(torch.float64)
        return self.net(x)

    def probabilities(self):
        with torch.no_grad():
            return self(torch.arange(self.n_states)).softmax(-1).numpy()


class LinearBaseline:
    """Ridge-fitted linear value function over one-hot chain states.

    Predicts continuing-task values; time-limit rollouts bootstrap this value.
    """
    def __init__(self, n_states):
        self.weights = np.zeros(n_states)

    def predict(self, states):
        return self.weights[np.asarray(states, dtype=int)]

    def fit(self, states, targets):
        counts = np.bincount(states, minlength=len(self.weights))
        totals = np.bincount(states, weights=targets, minlength=len(self.weights))
        visited = counts > 0
        self.weights[visited] = totals[visited] / (counts[visited] + 1e-5)


def advantages_and_returns(rewards, values, next_values, boundaries, terminated, gamma, lam):
    """GAE stops at rollout/reset boundaries; time limits still bootstrap V(s')."""
    advantages = np.zeros(len(rewards))
    running = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (not terminated[t]) - values[t]
        running = delta + gamma * lam * running * (not boundaries[t]) * (not terminated[t])
        advantages[t] = running
    # Separate bootstrapped return targets from lambda-smoothed advantages.
    returns = np.zeros(len(rewards))
    running = 0.0
    for t in reversed(range(len(rewards))):
        if boundaries[t] or t == len(rewards) - 1:
            running = next_values[t]
        running = rewards[t] + gamma * running * (not terminated[t])
        returns[t] = running
    return advantages, returns


def flat_gradient(scalar, parameters, create_graph=False):
    return torch.cat([g.reshape(-1) for g in torch.autograd.grad(
        scalar, parameters, create_graph=create_graph, retain_graph=True)])


def conjugate_gradient(matrix_vector, b, iterations=10, tolerance=1e-12):
    x, residual, direction = torch.zeros_like(b), b.clone(), b.clone()
    squared = residual @ residual
    for _ in range(iterations):
        if squared < tolerance:
            break
        product = matrix_vector(direction)
        denominator = direction @ product
        if not torch.isfinite(denominator) or denominator <= 0:
            break
        alpha = squared / denominator
        x += alpha * direction
        residual -= alpha * product
        new_squared = residual @ residual
        direction = residual + (new_squared / squared) * direction
        squared = new_squared
    return x


def update_policy(policy, states, actions, advantages, config):
    states = torch.as_tensor(states, dtype=torch.long)
    actions = torch.as_tensor(actions, dtype=torch.long)
    advantages = torch.as_tensor(advantages, dtype=torch.float64)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    params = tuple(policy.parameters())
    old_vector = parameters_to_vector(params).detach().clone()
    # Group repeated chain states: mathematically identical empirical objective,
    # but neural forward passes and Hessian products only need n states.
    all_states = torch.arange(policy.n_states)
    weights = torch.bincount(states, minlength=policy.n_states).double() / len(states)
    coefficients = torch.zeros(policy.n_states * 2, dtype=torch.float64)
    coefficients.scatter_add_(0, states * 2 + actions, advantages / len(states))
    coefficients = coefficients.reshape(policy.n_states, 2)
    old_logp = policy(all_states).log_softmax(-1).detach()
    old_p = old_logp.exp()

    def objective_and_kl():
        logp = policy(all_states).log_softmax(-1)
        objective = (coefficients * (logp - old_logp).exp()).sum()
        kl = (weights * (old_p * (old_logp - logp)).sum(-1)).sum()
        return objective, kl

    objective, kl = objective_and_kl()
    gradient = flat_gradient(objective, params).detach()
    kl_gradient = flat_gradient(kl, params, create_graph=True)

    def fisher_vector(vector):
        hvp = torch.autograd.grad(kl_gradient @ vector, params, retain_graph=True)
        return torch.cat([g.reshape(-1) for g in hvp]).detach() + config.damping * vector

    direction = conjugate_gradient(fisher_vector, gradient, config.cg_steps)
    curvature = direction @ fisher_vector(direction)
    diagnostics = {"kl": 0.0, "surrogate_gain": 0.0, "step_accepted": False}
    if not torch.isfinite(curvature) or curvature <= 1e-16:
        return diagnostics
    step = direction * torch.sqrt(2 * config.max_kl / curvature)
    expected_gain = gradient @ step
    if not torch.isfinite(step).all() or expected_gain <= 0:
        return diagnostics
    old_objective = float(objective.detach())
    with torch.no_grad():
        for i in range(config.backtracks):
            fraction = 0.5 ** i
            vector_to_parameters(old_vector + fraction * step, params)
            new_objective, new_kl = objective_and_kl()
            gain = new_objective - old_objective
            if (torch.isfinite(new_objective) and torch.isfinite(new_kl)
                    and new_kl <= config.max_kl and gain > 0
                    and gain >= 0.1 * fraction * expected_gain):
                return {"kl": float(new_kl), "surrogate_gain": float(gain), "step_accepted": True}
        vector_to_parameters(old_vector, params)
    return diagnostics
