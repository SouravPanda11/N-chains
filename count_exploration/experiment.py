"""Matched-seed training, independent evaluation, and experiment artifacts."""
from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
import os
import platform
import numpy as np
import torch

from .counting import CountBonus, StateEncoder
from .env import ChainConfig, NChain, expected_return
from .trpo import (LinearBaseline, Policy, TRPOConfig,
                   advantages_and_returns, update_policy)


@dataclass(frozen=True)
class RunConfig:
    iterations: int = 100
    batch_size: int = 5000
    hashing: str = "simhash"
    hash_bits: int = 32

    def __post_init__(self):
        if self.iterations < 1 or self.batch_size < 2 or self.hash_bits < 1:
            raise ValueError("iterations >= 1, batch_size >= 2, hash_bits >= 1 required")
        if self.hashing not in ("exact", "simhash"):
            raise ValueError("invalid hashing method")


def train(chain, optimizer, run, mode, beta, seed):
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    # Separate streams prevent hashing or evaluation from changing training draws.
    action_seed, env_seed, hash_seed = np.random.SeedSequence(seed).spawn(3)
    rng = np.random.default_rng(action_seed)
    env = NChain(chain, seed=env_seed)
    encoder = StateEncoder(chain.n, run.hashing, run.hash_bits, hash_seed)
    bonus = CountBonus(encoder, mode, beta)
    policy = Policy(chain.n, optimizer.hidden)
    baseline = LinearBaseline(chain.n)
    visits = np.zeros((chain.n, 2), dtype=np.int64)
    first_large_reward = None
    rows = []

    def evaluation(iteration, extras):
        probabilities = policy.probabilities()
        greedy = np.eye(2)[probabilities.argmax(axis=1)]
        row = {
            "mode": mode, "beta": beta, "seed": seed,
            "iteration": iteration, "steps": iteration * run.batch_size,
            "eval_return": expected_return(probabilities, chain),
            "greedy_return": expected_return(greedy, chain),
            "state_coverage": float(np.mean(visits.sum(axis=1) > 0)),
            "state_action_coverage": float(np.mean(visits > 0)),
            "first_large_reward_step": first_large_reward,
            "unique_hash_codes": encoder.unique_codes,
        }
        row.update(extras)
        rows.append(row)

    empty = {"train_reward_per_step": 0.0, "bonus_per_step": 0.0,
             "kl": 0.0, "surrogate_gain": 0.0, "step_accepted": False}
    evaluation(0, empty)
    for iteration in range(1, run.iterations + 1):
        probabilities = policy.probabilities()
        states = np.empty(run.batch_size, dtype=np.int64)
        next_states = np.empty_like(states)
        actions = np.empty_like(states)
        rewards = np.empty(run.batch_size)
        boundaries = np.zeros(run.batch_size, dtype=bool)
        terminated = np.zeros(run.batch_size, dtype=bool)
        state = env.reset()
        for t in range(run.batch_size):
            action = int(rng.random() >= probabilities[state, 0])
            next_state, reward, terminal, truncated, info = env.step(action)
            states[t], actions[t], next_states[t], rewards[t] = state, action, next_state, reward
            visits[state, action] += 1
            if first_large_reward is None and state == chain.n - 1 and info["executed_action"] == 0:
                first_large_reward = (iteration - 1) * run.batch_size + t + 1
            boundaries[t], terminated[t] = terminal or truncated, terminal
            state = env.reset() if boundaries[t] else next_state
        boundaries[-1] = True
        intrinsic = bonus.update_batch(states, actions)
        values, next_values = baseline.predict(states), baseline.predict(next_states)
        advantages, returns = advantages_and_returns(
            rewards + intrinsic, values, next_values, boundaries, terminated,
            optimizer.gamma, optimizer.gae_lambda)
        diagnostics = update_policy(policy, states, actions, advantages, optimizer)
        baseline.fit(states, returns)
        evaluation(iteration, {
            "train_reward_per_step": float(rewards.mean()),
            "bonus_per_step": float(intrinsic.mean()), **diagnostics})
    artifacts = {
        "policy": policy,
        "baseline_weights": baseline.weights,
        "visits": visits,
        "hash_projection": encoder.projection,
        "count_table": [{"key": repr(key), "count": value} for key, value in bonus.counts.items()],
    }
    return rows, artifacts


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}


def summarize(rows):
    final = {}
    for row in rows:
        key = (row["mode"], row["beta"], row["seed"])
        final[key] = row
    summaries = []
    for mode, beta in sorted({(key[0], key[1]) for key in final}):
        selected = [row for (m, b, _), row in final.items() if (m, b) == (mode, beta)]
        summary = {"mode": mode, "beta": beta, "seeds": len(selected)}
        for metric in ("eval_return", "greedy_return", "state_coverage", "state_action_coverage"):
            summary[metric] = mean_std([row[metric] for row in selected])
        discoveries = [row["first_large_reward_step"] for row in selected
                       if row["first_large_reward_step"] is not None]
        summary["large_reward_discovered_runs"] = len(discoveries)
        summary["first_large_reward_step_discovered_only"] = mean_std(discoveries) if discoveries else None
        summaries.append(summary)
    paired = []
    for beta in sorted({b for m, b, _ in final if m == "state"}):
        differences = [row["eval_return"] - final[("state", beta, seed)]["eval_return"]
                       for (mode, b, seed), row in final.items()
                       if mode == "state_action" and b == beta and ("state", beta, seed) in final]
        if differences:
            paired.append({"beta": beta, "seeds": len(differences),
                           "state_action_minus_state": mean_std(differences),
                           "per_seed_differences": differences})
    return {"final_results": summaries, "paired_final_return_differences": paired,
            "spread": "sample standard deviation across training seeds, not a confidence interval"}


def plot_results(rows, output):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(output).resolve() / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    betas = sorted({r["beta"] for r in rows if r["mode"] != "none"}) or [0.0]
    colors = {"none": "#777777", "state": "#1677b8", "state_action": "#d86520"}
    fig, axes = plt.subplots(len(betas), 2, squeeze=False, figsize=(12, 4 * len(betas)))
    for index, beta in enumerate(betas):
        for mode in ("none", "state", "state_action"):
            selected = [r for r in rows if r["mode"] == mode and (mode == "none" or r["beta"] == beta)]
            if not selected:
                continue
            steps = sorted({r["steps"] for r in selected})
            for col, (metric, title) in enumerate((
                    ("eval_return", "Expected extrinsic return"),
                    ("state_action_coverage", "State-action coverage"))):
                stats = [mean_std([r[metric] for r in selected if r["steps"] == step]) for step in steps]
                means = np.array([s["mean"] for s in stats])
                std = np.array([s["std"] for s in stats])
                ax = axes[index, col]
                ax.plot(steps, means, label=mode, color=colors[mode])
                ax.fill_between(steps, means - std, means + std, color=colors[mode], alpha=0.15)
                ax.set(title=f"{title} (beta={beta:g})", xlabel="Training environment steps")
                ax.grid(alpha=0.2)
                ax.legend()
    fig.suptitle("TRPO on N-chain: mean +/- 1 sample SD across seeds")
    fig.tight_layout()
    fig.savefig(Path(output) / "comparison.png", dpi=160)
    plt.close(fig)


def run_comparison(output, chain, optimizer, run, modes, betas, seeds):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}; choose a new path")
    if (not modes or not seeds or not betas or len(set(seeds)) != len(seeds)
            or len(set(modes)) != len(modes) or len(set(betas)) != len(betas)):
        raise ValueError("modes, betas and seeds must be nonempty and unique")
    if any(m not in ("none", "state", "state_action") for m in modes):
        raise ValueError("invalid mode")
    if any(s < 0 for s in seeds) or any(not np.isfinite(b) or b < 0 for b in betas):
        raise ValueError("seeds and finite betas must be nonnegative")
    output.mkdir(parents=True, exist_ok=True)
    config = {"environment": asdict(chain), "trpo": asdict(optimizer), "run": asdict(run),
              "modes": modes, "betas": betas, "seeds": seeds,
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
              "count_timing": "current state and chosen action; increment full batch then compute bonuses",
              "evaluation": "exact expected undiscounted extrinsic return from state zero over horizon",
              "time_limit": "bootstrap value at time limits and batch boundaries"}
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    all_rows = []
    for mode in modes:
        for beta in ([0.0] if mode == "none" else betas):
            for seed in seeds:
                print(f"Training {mode}, beta={beta:g}, seed={seed}", flush=True)
                rows, artifacts = train(chain, optimizer, run, mode, beta, seed)
                directory = output / f"{mode}_beta{beta}_seed{seed}"
                directory.mkdir()
                write_csv(directory / "metrics.csv", rows)
                np.save(directory / "state_action_visits.npy", artifacts["visits"])
                if artifacts["hash_projection"] is not None:
                    np.save(directory / "hash_projection.npy", artifacts["hash_projection"])
                torch.save({"policy_state_dict": artifacts["policy"].state_dict(),
                            "n_states": chain.n, "hidden": optimizer.hidden}, directory / "policy.pt")
                np.save(directory / "baseline_weights.npy", artifacts["baseline_weights"])
                (directory / "counts.json").write_text(json.dumps(artifacts["count_table"], indent=2), encoding="utf-8")
                np.savetxt(directory / "action_probabilities.csv", artifacts["policy"].probabilities(),
                           delimiter=",", header="forward,reset", comments="")
                all_rows.extend(rows)
                print(f"  final expected return={rows[-1]['eval_return']:.3f}, "
                      f"coverage={rows[-1]['state_action_coverage']:.0%}", flush=True)
    write_csv(output / "metrics.csv", all_rows)
    summary = summarize(all_rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_results(all_rows, output)
    return summary
