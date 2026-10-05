import json
import numpy as np
import pytest
import torch
from torch.nn.utils import parameters_to_vector

from count_exploration.counting import CountBonus, StateEncoder
from count_exploration.env import ChainConfig, NChain, expected_return
from count_exploration.experiment import RunConfig, run_comparison, train
from count_exploration.trpo import (Policy, TRPOConfig, advantages_and_returns,
                                    conjugate_gradient, update_policy)


def test_chain_reward_timing_and_reset():
    env = NChain(ChainConfig(n=3, horizon=5))
    assert env.step(0)[:2] == (1, 0)
    assert env.step(0)[:2] == (2, 0)
    assert env.step(0)[:2] == (2, 10)
    assert env.step(1)[:2] == (0, 2)
    assert env.step(0)[2:4] == (False, True)
    with pytest.raises(RuntimeError):
        env.step(0)
    assert env.reset() == 0


def test_slip_flips_action():
    env = NChain(ChainConfig(n=3, slip=1))
    assert env.step(0)[:2] == (0, 2)
    assert env.step(1)[:2] == (1, 0)


def test_exact_evaluation_known_policies():
    config = ChainConfig(n=5, horizon=10)
    assert expected_return(np.tile([1, 0], (5, 1)), config) == 60
    assert expected_return(np.tile([0, 1], (5, 1)), config) == 20
    assert expected_return(np.tile([0, 1], (5, 1)), ChainConfig(slip=1, horizon=10)) == 60


def test_batch_counts_and_persistence():
    bonus = CountBonus(StateEncoder(3, "exact"), "state", beta=2)
    np.testing.assert_allclose(bonus.update_batch([0, 0, 1], [0, 1, 0]),
                               [2 / np.sqrt(2), 2 / np.sqrt(2), 2])
    np.testing.assert_allclose(bonus.update_batch([0, 1], [0, 0]), [2 / np.sqrt(3), 2 / np.sqrt(2)])
    assert sum(bonus.counts.values()) == 5


def test_action_counts_separate_chosen_actions():
    encoder = StateEncoder(3, "exact")
    state = CountBonus(encoder, "state", 1)
    pair = CountBonus(encoder, "state_action", 1)
    np.testing.assert_allclose(state.update_batch([0, 0, 0], [0, 0, 1]), np.ones(3) / np.sqrt(3))
    np.testing.assert_allclose(pair.update_batch([0, 0, 0], [0, 0, 1]), [1 / np.sqrt(2), 1 / np.sqrt(2), 1])


def test_simhash_is_fixed_and_does_not_merge_all_positive_states():
    a, b = StateEncoder(20, seed=12), StateEncoder(20, seed=12)
    assert a.codes == b.codes
    assert a.unique_codes == 20
    assert StateEncoder(20, bits=1).unique_codes <= 2


def test_hash_collisions_are_counted_together():
    encoder = StateEncoder(20, bits=1)
    s, t = next((s, t) for s in range(20) for t in range(s + 1, 20) if encoder(s) == encoder(t))
    bonus = CountBonus(encoder)
    np.testing.assert_allclose(bonus.update_batch([s, t], [0, 1]), [0.01 / np.sqrt(2)] * 2)


@pytest.mark.parametrize("mode", ["state", "state_action"])
def test_collision_free_simhash_matches_exact_counting(mode):
    exact = CountBonus(StateEncoder(5, "exact"), mode, 1)
    hashed = CountBonus(StateEncoder(5, "simhash", seed=4), mode, 1)
    assert hashed.encoder.unique_codes == 5
    for states, actions in [([0, 1, 2, 0], [0, 0, 0, 1]),
                             ([3, 4, 4, 2], [0, 0, 1, 1])]:
        np.testing.assert_array_equal(exact.update_batch(states, actions),
                                      hashed.update_batch(states, actions))


def test_gae_bootstraps_time_limit_without_crossing_reset():
    advantages, returns = advantages_and_returns(
        np.array([1., 100.]), np.array([2., 0.]), np.array([3., 50.]),
        np.array([True, True]), np.array([False, True]), 0.9, 1)
    np.testing.assert_allclose(advantages, [1 + 0.9 * 3 - 2, 100])
    np.testing.assert_allclose(returns, [1 + 0.9 * 3, 100])


def test_gae_matches_hand_computed_multistep_returns():
    advantages, returns = advantages_and_returns(
        np.array([1., 2., 3.]), np.zeros(3), np.zeros(3),
        np.array([False, False, True]), np.array([False, False, True]), 0.5, 1)
    np.testing.assert_allclose(returns, [2.75, 3.5, 3])
    np.testing.assert_allclose(advantages, returns)


def test_conjugate_gradient_solves_known_system():
    matrix = torch.tensor([[4., 1.], [1., 3.]], dtype=torch.float64)
    b = torch.tensor([1., 2.], dtype=torch.float64)
    torch.testing.assert_close(conjugate_gradient(lambda x: matrix @ x, b), torch.linalg.solve(matrix, b))


def test_trpo_improves_surrogate_and_respects_kl():
    torch.manual_seed(2)
    torch.set_num_threads(1)
    policy = Policy(3, (8, 8))
    before = policy.probabilities()
    states = np.repeat(np.arange(3), 20)
    actions = np.tile([0, 1], 30)
    advantages = np.where(actions == 0, 1., -1.)
    result = update_policy(policy, states, actions, advantages, TRPOConfig())
    after = policy.probabilities()
    kl = np.mean(np.sum(before * np.log(before / after), axis=1))
    assert result["step_accepted"]
    assert 0 <= kl <= 0.01 + 1e-10
    assert result["surrogate_gain"] > 0
    assert np.mean(after[:, 0]) > np.mean(before[:, 0])


def test_zero_advantages_do_not_change_policy():
    policy = Policy(3)
    before = parameters_to_vector(policy.parameters()).detach().clone()
    result = update_policy(policy, [0, 1], [0, 1], [0., 0.], TRPOConfig())
    assert not result["step_accepted"]
    torch.testing.assert_close(before, parameters_to_vector(policy.parameters()))


def test_rejected_line_search_restores_policy():
    policy = Policy(3)
    before = parameters_to_vector(policy.parameters()).detach().clone()
    # Large trust region gives an unacceptably small actual/expected gain ratio.
    result = update_policy(policy, [0, 0, 1, 1], [0, 1, 0, 1], [1., -1., 1., -1.],
                           TRPOConfig(max_kl=1e8, backtracks=1))
    assert not result["step_accepted"]
    torch.testing.assert_close(before, parameters_to_vector(policy.parameters()))


def test_zero_beta_modes_and_repeat_runs_are_identical():
    chain = ChainConfig(n=3, horizon=10)
    run = RunConfig(iterations=3, batch_size=100)
    outcomes = [train(chain, TRPOConfig(), run, mode, 0, 3)
                for mode in ("none", "state", "state_action", "state")]
    for rows, artifacts in outcomes[1:]:
        np.testing.assert_array_equal(artifacts["visits"], outcomes[0][1]["visits"])
        np.testing.assert_array_equal(artifacts["policy"].probabilities(), outcomes[0][1]["policy"].probabilities())
        assert [r["eval_return"] for r in rows] == [r["eval_return"] for r in outcomes[0][0]]
        assert sum(item["count"] for item in artifacts["count_table"]) == 300


def test_full_output_pipeline_and_reload(tmp_path):
    output = tmp_path / "experiment"
    summary = run_comparison(output, ChainConfig(n=3, horizon=10), TRPOConfig(),
                             RunConfig(iterations=2, batch_size=40),
                             ["none", "state", "state_action"], [0.1], [0])
    assert len(summary["final_results"]) == 3
    assert (output / "comparison.png").stat().st_size > 0
    json.loads((output / "summary.json").read_text())
    checkpoint = torch.load(output / "state_beta0.1_seed0" / "policy.pt", weights_only=True)
    policy = Policy(checkpoint["n_states"], checkpoint["hidden"])
    policy.load_state_dict(checkpoint["policy_state_dict"])
    saved = np.loadtxt(output / "state_beta0.1_seed0" / "action_probabilities.csv", delimiter=",", skiprows=1)
    np.testing.assert_allclose(policy.probabilities(), saved)
    with pytest.raises(ValueError, match="not empty"):
        run_comparison(output, ChainConfig(), TRPOConfig(), RunConfig(), ["state"], [0.1], [0])


@pytest.mark.parametrize("config", [lambda: ChainConfig(n=1), lambda: ChainConfig(slip=1.1),
                                    lambda: TRPOConfig(gamma=2), lambda: RunConfig(batch_size=0)])
def test_invalid_configuration(config):
    with pytest.raises(ValueError):
        config()
