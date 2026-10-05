# Count-based exploration on N-chain

A new implementation of the static-hashing method in Tang et al.,
[*#Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning*](https://arxiv.org/abs/1611.04717),
Section 2.2 / Algorithm 1, adapted to the discrete N-chain environment.
The learner is **TRPO with a neural categorical policy**. The existing
`Q-Learning-Demo-Play-nChain/` reference folder is unchanged and is not imported.

The experiment compares three agents:

| Mode | Training reward |
| --- | --- |
| `none` | Extrinsic reward only |
| `state` | `r(s,a) + beta / sqrt(N(phi(s)))` |
| `state_action` | `r(s,a) + beta / sqrt(N(phi(s),a))` |

The paper itself compares state and state-action counting in Supplementary
Section 5.4, Table 9 (PDF page 17). Its results depend on the hyperparameters;
they do not establish that either choice always wins.

## Run

From this workspace, in PowerShell (activation is unnecessary):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m count_exploration --output results\comparison
```

The local `.venv` is already installed. Default settings: five states, no slip,
100-step rollout horizon, 100 TRPO iterations, 5,000 training transitions per
iteration, beta 0.01, 32-bit SimHash, and seeds 0 through 4. This runs all three
agents. Choose a fresh output directory each time; existing results are never
overwritten.

A short run:

```powershell
.\.venv\Scripts\python.exe -m count_exploration --iterations 5 --batch-size 500 --seeds 0 --output results\quick
```

To focus on **counting rather than hash collisions**, use exact state IDs and
sweep the bonus strength on a longer chain:

```powershell
.\.venv\Scripts\python.exe -m count_exploration --n 10 --hashing exact --betas 0.01 0.1 1 10 --seeds 0 1 2 3 4 --output results\n10_exact
```

Repeat with `--hashing simhash` and another output path to examine hashing.
`--slip 0.2` enables stochastic action flips, matching historical Gym's default
slip; the reference notebook explicitly sets slip to zero. Use `--horizon 10`
to match the notebook's rollout length. `--help` lists all settings.

## Method and counting conventions

1. Collect a batch using the current stochastic policy.
2. Encode each **current state `s_t`**, where the action was chosen.
3. Increment the entire batch's counts, including repeated visits.
4. Compute each bonus from the resulting cumulative count.
5. Fit TRPO using extrinsic plus intrinsic reward; evaluate extrinsic reward only.

This follows Algorithm 1's batch ordering. Repeated keys in a batch receive the
same bonus. Counts start at zero, persist across episodes and iterations, and
start fresh for each independent run. Incrementing before lookup avoids division
by zero. In `state_action`, the action is the **chosen action**, even if the
environment flips it due to slip. It is a tuple `(phi(s_t), a_t)`; actions cannot
collide through hashing. We never count `(s_{t+1}, a_t)`.

`simhash` implements `phi(s) = sign(A @ one_hot(s))`, with a fixed Gaussian
projection A. The projection and policy initialization are matched across agents
for each seed. Directly applying SimHash to scalar chain positions would map all
positive positions to the same code, because its sign is invariant to positive
scaling. One-hot preprocessing avoids that collapse. It treats states as distinct
categories; it does not encode proximity along the chain. `exact` uses `phi(s)=s`.
If SimHash has no collisions, these encodings give identical count bonuses.
`unique_hash_codes` reports the actual number of distinct codes.

The neural policy has two 32-unit tanh hidden layers and a two-action softmax.
TRPO uses a conjugate-gradient natural-gradient direction, an autodifferentiated
KL Hessian-vector product, damping, and backtracking that requires surrogate
improvement and empirical mean KL <= 0.01. Advantages are standardized. The
linear state-value baseline uses one-hot features. No epsilon-greedy schedule,
entropy bonus, replay buffer, or Q-learning is involved.

## Fidelity and adaptations

The attached PDF's Supplementary Table 1 specifies TRPO, gamma 0.99, KL step
size 0.01, beta 0.01, and 32-bit SimHash for its non-image control experiments.
The MountainCar setup also uses two 32-unit tanh layers, a linear baseline, and
a 5,000-transition batch. These are the corresponding defaults here.

This is an N-chain adaptation, **not a reproduction of the paper's benchmark
scores**. Differences are explicit:

- N-chain has discrete actions, so a categorical policy replaces the continuous
  experiments' Gaussian policy. Observations use one-hot chain positions.
- A dictionary stores counts exactly for each hash key. The paper's larger
  experiments use a count-min sketch to save computation/memory.
- The linear baseline is fitted by ridge least squares over one-hot states.
  The supplement describes Adam for baseline optimization. This small discrete
  problem allows a direct fit; it is identical across the counting variants.
- The default horizon is 100 rather than the control experiments' 500. N-chain
  never terminates naturally. We bootstrap the value of the final observation
  at time limits and batch cutoffs, and stop advantage recursion across resets.
  This learns a continuing-task policy; evaluation reports a finite window from
  the start state. `--gae-lambda 1` is the default; this setting, damping 0.1,
  ten CG steps, and ten backtracking attempts are implementation choices.
- We compute the **exact expected** bonus-free, undiscounted return over the
  evaluation horizon from state zero. This integrates policy and slip randomness
  using the small known transition model. The model is used only for evaluation;
  the learner trains entirely from sampled transitions. Greedy-policy return is
  reported separately. Evaluation does not change training counts or RNGs.

N-chain behavior: action 0 advances; action 1 jumps to state zero for reward 2.
Entering the last state gives zero reward; taking action 0 while already there
gives reward 10 and stays there. Slip reverses the selected action. The included
notebook imports this environment from Gym 0.13.1 rather than implementing it;
the new environment independently implements those dynamics without old Gym.

## Reading the results

Each output directory contains:

- `comparison.png`: expected extrinsic returns and state-action coverage, with
  mean and +/- one sample standard deviation across seeds.
- `metrics.csv`: every seed's per-iteration return, coverage, average bonus,
  first large reward step, accepted updates, and actual KL.
- `summary.json`: final seed means and standard deviations, paired
  state-action-minus-state return differences, and discovery counts.
- `config.json`: complete experiment settings, seeds, and runtime versions.
- Per-run folders: metrics, policy checkpoint, action probabilities, baseline
  weights, hash projection, persistent count table, and raw state-action visits.

The same seeds mean identical initial policies, hashes, and random-number
streams; trajectories naturally diverge once policies differ. With beta zero,
all three agents give identical results (covered by a test). The no-bonus
baseline runs once per seed even in a multi-beta sweep.

At a fixed beta, state-action counting generally gives a larger bonus: if the
two actions are equally frequent, `N(s,a) ~= N(s)/2`, so its bonus is about
`sqrt(2)` times the state bonus. A same-beta experiment therefore changes both
where novelty is rewarded and its typical magnitude. Sweep beta for both modes;
an additional scale-matched check uses `beta_state_action = beta_state/sqrt(2)`.
That adjustment is only approximate as the policy becomes nonuniform.

Five-state N-chain is easy enough that all methods can solve it quickly. For a
stronger comparison, vary chain length and slip, inspect learning curves and
time to discover the large reward, and use more seeds. Report unsuccessful
discovery runs too; the mean discovery step in `summary.json` is explicitly
conditional on successful discovery. Shaded standard deviations are not
confidence intervals or significance tests. The small paper beta can have very
little effect against N-chain rewards of 2 and 10.

## Initial experiment

Executed command:

```powershell
.\.venv\Scripts\python.exe -m count_exploration --iterations 50 --batch-size 1000 --betas 0.01 1 10 --seeds 0 1 2 3 4 --output results\initial_comparison
```

This uses five states, slip zero, horizon 100, and 50,000 transitions per run.
The always-forward policy returns 960 over this horizon. Final expected
extrinsic returns (mean +/- sample SD across five seeds):

| Bonus beta | State counts | State-action counts |
| --- | ---: | ---: |
| 0.01 | 959.005 +/- 1.328 | 959.005 +/- 1.328 |
| 1 | 958.789 +/- 1.766 | 959.317 +/- 1.106 |
| 10 | 959.993 +/- 0.004 | 959.375 +/- 0.763 |

No-bonus TRPO: **958.980 +/- 1.305**. All runs reached 100% state-action
coverage. These results establish that the implementation learns this easy
chain; they do not establish an overall advantage for either counting method.
Full local artifacts: `results/initial_comparison/` (ignored by version control).

## Code map

| File | Role |
| --- | --- |
| `count_exploration/env.py` | N-chain and exact evaluation |
| `count_exploration/counting.py` | Static encoders and both bonus definitions |
| `count_exploration/trpo.py` | Policy, baseline, advantages, CG, trust-region update |
| `count_exploration/experiment.py` | Matched-seed runs, logs, checkpoints, plots |
| `count_exploration/__main__.py` | Command-line interface |
| `tests/test_exploration.py` | Dynamics, counts, TRPO, reproducibility, output checks |
