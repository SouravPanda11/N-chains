"""Algorithm 1: fixed hash, persistent counts, post-batch bonuses."""
from collections import Counter
import numpy as np


class StateEncoder:
    def __init__(self, n_states, kind="simhash", bits=32, seed=0):
        if kind not in ("exact", "simhash") or n_states < 2 or bits < 1:
            raise ValueError("invalid encoder configuration")
        self.n_states, self.kind, self.bits = n_states, kind, bits
        self.projection = None
        if kind == "exact":
            self.codes = list(range(n_states))
        else:
            # g(s) = one_hot(s). A @ g(s) selects column s of A.
            self.projection = np.random.default_rng(seed).normal(size=(bits, n_states))
            self.codes = [np.packbits(self.projection[:, s] >= 0).tobytes()
                          for s in range(n_states)]

    def __call__(self, state):
        if not 0 <= int(state) < self.n_states:
            raise ValueError("state out of range")
        return self.codes[int(state)]

    @property
    def unique_codes(self):
        return len(set(self.codes))


class CountBonus:
    def __init__(self, encoder, mode="state", beta=0.01):
        if mode not in ("none", "state", "state_action"):
            raise ValueError("unknown count mode")
        if not np.isfinite(beta) or beta < 0:
            raise ValueError("beta must be finite and nonnegative")
        self.encoder, self.mode, self.beta = encoder, mode, beta
        self.counts = Counter()

    def update_batch(self, states, actions):
        """Count (s_t, chosen a_t), then look up final batch counts for all t.

        Repeated keys in the same batch receive the same bonus. Counts survive
        episode resets; evaluation must never call this method.
        """
        if len(states) != len(actions):
            raise ValueError("states and actions must have equal length")
        if any(a not in (0, 1) for a in actions):
            raise ValueError("actions must be 0 or 1")
        if self.mode == "none":
            return np.zeros(len(states), dtype=float)
        keys = [self.encoder(s) for s in states]
        if self.mode == "state_action":
            keys = [(key, int(a)) for key, a in zip(keys, actions)]
        self.counts.update(keys)
        return np.array([self.beta / np.sqrt(self.counts[key]) for key in keys])
