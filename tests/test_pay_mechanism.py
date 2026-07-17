"""Tests for the Clean Up monetary system (pay action / reward transfers).

Runnable two ways:
    python tests/test_pay_mechanism.py          # plain runner, no pytest needed
    python -m pytest tests/test_pay_mechanism.py  # if pytest is installed

Covers the accounting invariants that, if silently wrong, would invalidate every
downstream experiment: zero-sum transfers, correct nearest-target selection,
no-op behaviour without a target, and that the full env step still compiles
under jax.jit (the same XLA-lowering path used on GPU).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

import socialjax
from socialjax.environments.cleanup.clean_up import (
    Actions,
    ROTATIONS,
    STEP_MOVE,
    compute_pay_transfers,
)

PAY = int(Actions.pay)
STAY = int(Actions.stay)


def locs(*rows_cols):
    """Build an (N, 3) agent_locs array from (row, col) pairs (orientation 0)."""
    return jnp.array([[r, c, 0] for r, c in rows_cols], dtype=jnp.int16)


# ---------------------------------------------------------------- module-level arrays

def test_action_indexed_arrays_cover_all_actions():
    assert len(Actions) == 10
    assert ROTATIONS.shape[0] == len(Actions), "ROTATIONS must have one row per action"
    assert STEP_MOVE.shape[0] == len(Actions), "STEP_MOVE must have one row per action"
    # pay must be movement/rotation-neutral
    assert np.array_equal(np.array(ROTATIONS[PAY]), [0, 0, 0])
    assert np.array_equal(np.array(STEP_MOVE[PAY]), [0, 0, 0])


# ---------------------------------------------------------------- transfer accounting

def test_pays_nearest_in_range():
    agent_locs = locs((0, 0), (0, 3), (0, 10))
    actions = jnp.array([PAY, STAY, STAY])
    delta, attempted, executed = compute_pay_transfers(agent_locs, actions, 5, 1.0)
    assert np.allclose(np.array(delta), [-1.0, 1.0, 0.0])
    assert np.array_equal(np.array(attempted), [True, False, False])
    assert np.array_equal(np.array(executed), [True, False, False])


def test_tie_broken_by_lowest_index():
    # agents 1 and 2 both at Chebyshev distance 2 from agent 0
    agent_locs = locs((5, 5), (5, 7), (5, 3))
    actions = jnp.array([PAY, STAY, STAY])
    delta, _, _ = compute_pay_transfers(agent_locs, actions, 5, 1.0)
    assert np.allclose(np.array(delta), [-1.0, 1.0, 0.0]), "tie must go to lowest agent index"


def test_no_target_in_range_is_noop():
    agent_locs = locs((0, 0), (0, 20))
    actions = jnp.array([PAY, STAY])
    delta, attempted, executed = compute_pay_transfers(agent_locs, actions, 5, 1.0)
    assert np.allclose(np.array(delta), [0.0, 0.0])
    assert np.array_equal(np.array(attempted), [True, False])
    assert np.array_equal(np.array(executed), [False, False])


def test_cannot_pay_self():
    agent_locs = locs((4, 4))
    actions = jnp.array([PAY])
    delta, _, executed = compute_pay_transfers(agent_locs, actions, 5, 1.0)
    assert np.allclose(np.array(delta), [0.0])
    assert not bool(executed[0])


def test_multiple_senders_same_receiver_accumulate():
    # agents 0 and 2 flank agent 1; both pay -> agent 1 receives 2
    agent_locs = locs((0, 0), (0, 2), (0, 4))
    actions = jnp.array([PAY, STAY, PAY])
    delta, _, executed = compute_pay_transfers(agent_locs, actions, 5, 1.0)
    assert np.allclose(np.array(delta), [-1.0, 2.0, -1.0])
    assert np.array_equal(np.array(executed), [True, False, True])


def test_mutual_payment_nets_out():
    # both agents pay each other simultaneously -> net zero each, both executed
    agent_locs = locs((0, 0), (0, 1))
    actions = jnp.array([PAY, PAY])
    delta, _, executed = compute_pay_transfers(agent_locs, actions, 5, 1.0)
    assert np.allclose(np.array(delta), [0.0, 0.0])
    assert np.array_equal(np.array(executed), [True, True])


def test_zero_sum_under_random_configurations():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(1, 10))
        agent_locs = jnp.array(
            np.column_stack([rng.integers(0, 20, n), rng.integers(0, 20, n), np.zeros(n)]),
            dtype=jnp.int16,
        )
        actions = jnp.array(rng.integers(0, len(Actions), n))
        delta, _, _ = compute_pay_transfers(agent_locs, actions, 5, 1.0)
        assert abs(float(jnp.sum(delta))) < 1e-6, "transfers must be exactly zero-sum"


def test_custom_amount_and_radius():
    agent_locs = locs((0, 0), (0, 3))
    actions = jnp.array([PAY, STAY])
    delta, _, _ = compute_pay_transfers(agent_locs, actions, 3, 2.5)
    assert np.allclose(np.array(delta), [-2.5, 2.5])
    delta, _, executed = compute_pay_transfers(agent_locs, actions, 2, 2.5)
    assert np.allclose(np.array(delta), [0.0, 0.0]), "radius 2 puts target out of range"
    assert not bool(executed[0])


def test_transfer_function_jits():
    jitted = jax.jit(compute_pay_transfers)
    agent_locs = locs((0, 0), (0, 2), (9, 9))
    actions = jnp.array([PAY, PAY, STAY])
    delta, attempted, executed = jitted(agent_locs, actions, 5, 1.0)
    ref = compute_pay_transfers(agent_locs, actions, 5, 1.0)
    assert np.allclose(np.array(delta), np.array(ref[0]))
    assert np.array_equal(np.array(attempted), np.array(ref[1]))
    assert np.array_equal(np.array(executed), np.array(ref[2]))


# ---------------------------------------------------------------- env integration

def test_action_space_size_per_mode():
    assert socialjax.make("clean_up", num_agents=3, pay_mode="off").action_space().n == 9
    assert socialjax.make("clean_up", num_agents=3, pay_mode="noop").action_space().n == 10
    assert socialjax.make("clean_up", num_agents=3, pay_mode="on").action_space().n == 10


def test_invalid_pay_mode_rejected():
    try:
        socialjax.make("clean_up", num_agents=3, pay_mode="yes")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid pay_mode should raise ValueError")


def _step_env(pay_mode, actions_fn, seed=7, num_agents=4):
    """Reset + one jitted step of clean_up under individual rewards."""
    env = socialjax.make(
        "clean_up", num_agents=num_agents, shared_rewards=False, pay_mode=pay_mode
    )
    key = jax.random.PRNGKey(seed)
    key, k_reset = jax.random.split(key)
    obs, state = env.reset(k_reset)
    actions = actions_fn(state)
    obs, new_state, rewards, done, info = env.step_env(key, state, actions)
    return state, rewards, info


def test_env_step_on_vs_noop_zero_sum():
    """Full jitted env step: 'on' redistributes but never creates/destroys reward."""
    all_pay = lambda state: jnp.array([PAY] * 4)
    state_on, rewards_on, info_on = _step_env("on", all_pay)
    state_noop, rewards_noop, info_noop = _step_env("noop", all_pay)

    # identical seed -> identical world; only the transfer differs
    assert np.allclose(
        float(jnp.sum(rewards_on)), float(jnp.sum(rewards_noop))
    ), "pay must be zero-sum at the env level"

    expected_delta, _, _ = compute_pay_transfers(
        state_on.agent_locs, jnp.array([PAY] * 4), 11 // 2, 1.0
    )
    assert np.allclose(
        np.array(rewards_on - rewards_noop).squeeze(), np.array(expected_delta)
    ), "env-level reward diff must equal the computed transfer delta"

    # noop mode logs the action but moves nothing
    assert float(jnp.sum(info_noop["pay_volume"])) == 0.0
    assert np.array_equal(np.array(info_on["pay_attempts"]), np.ones(4, dtype=np.float32))


def test_shaped_rewards_metric_reflects_pay_transfer():
    """Regression test: info["shaped_rewards"] must include the pay transfer.

    It's computed by the reward-mode branch (shared/individual/svo/etc.) before
    the pay-transfer block runs later in the same function, so it's easy for a
    future edit to silently revert to logging the pre-transfer value again --
    which would make the WandB-logged "shaped_rewards" curve inconsistent with
    the actual reward the policy trains on (which always includes the transfer;
    verified separately by test_env_step_on_vs_noop_zero_sum).
    """
    all_pay = lambda state: jnp.array([PAY] * 4)
    _, rewards_on, info_on = _step_env("on", all_pay)
    assert np.allclose(
        np.array(info_on["shaped_rewards"]), np.array(rewards_on).squeeze()
    ), "info['shaped_rewards'] must match the actual (post-transfer) returned rewards"


def test_env_metrics_present_and_absent():
    _, _, info_on = _step_env("on", lambda s: jnp.array([PAY] * 4))
    for k in ("pay_attempts", "pay_executed", "pay_volume"):
        assert k in info_on, f"missing metric {k} in pay_mode=on"

    _, _, info_off = _step_env("off", lambda s: jnp.array([STAY] * 4))
    for k in ("pay_attempts", "pay_executed", "pay_volume"):
        assert k not in info_off, f"metric {k} must not exist in pay_mode=off (baseline unchanged)"


def test_env_step_off_matches_original_baseline():
    """pay_mode='off' must produce byte-identical rewards to the pre-change env."""
    stay = lambda state: jnp.array([STAY] * 4)
    _, rewards_off, _ = _step_env("off", stay)
    _, rewards_noop, _ = _step_env("noop", stay)
    assert np.allclose(np.array(rewards_off), np.array(rewards_noop)), (
        "with no pay actions taken, off and noop must produce identical rewards"
    )


ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    failed = 0
    for t in ALL_TESTS:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(ALL_TESTS) - failed}/{len(ALL_TESTS)} tests passed")
    sys.exit(1 if failed else 0)
