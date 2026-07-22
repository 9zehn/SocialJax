"""Tests for the Clean Up monetary system (pay action / reward transfers).

Runnable two ways:
    python tests/test_pay_mechanism.py          # plain runner, no pytest needed
    python -m pytest tests/test_pay_mechanism.py  # if pytest is installed

Mechanism under test (Option B): an agent's `pay` action sends pay_amount to the
*most recent OTHER agent that cleaned dirt within pay_clean_window steps*, provided
the payer's balance covers it. Otherwise the attempt is a no-op. Covers the
accounting invariants that, if wrong, would invalidate downstream experiments:
zero-sum transfers, correct recent-cleaner selection, window expiry, self-exclusion,
insufficient-funds no-op, and that the full env step still compiles under jax.jit.
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
    compute_tithe_transfers,
)

PAY = int(Actions.pay)
STAY = int(Actions.stay)


def _pay(actions, last_clean_t, current_t, balance, clean_window=50, pay_amount=1.0):
    """Thin wrapper to keep the tests readable."""
    return compute_pay_transfers(
        jnp.array(actions),
        jnp.array(last_clean_t, dtype=jnp.int32),
        current_t,
        jnp.array(balance, dtype=jnp.float32),
        clean_window,
        pay_amount,
    )


def _tithe(actions, last_clean_t, current_t, share_expiry_t, income,
           clean_window=50, share_fraction=0.5, share_duration=50, split_recipients=False):
    """Thin wrapper to keep the tithe tests readable.

    Returns the original 6-tuple (delta, pledged, executed, target, new_expiry,
    active); the newer `received` element is sliced off so existing tests keep their
    unpacking. Split-recipient tests call compute_tithe_transfers directly for it.
    """
    return compute_tithe_transfers(
        jnp.array(actions),
        jnp.array(last_clean_t, dtype=jnp.int32),
        current_t,
        jnp.array(share_expiry_t, dtype=jnp.int32),
        jnp.array(income, dtype=jnp.float32),
        clean_window,
        share_fraction,
        share_duration,
        split_recipients,
    )[:6]


# ---------------------------------------------------------------- module-level arrays

def test_action_indexed_arrays_cover_all_actions():
    assert len(Actions) == 10
    assert ROTATIONS.shape[0] == len(Actions), "ROTATIONS must have one row per action"
    assert STEP_MOVE.shape[0] == len(Actions), "STEP_MOVE must have one row per action"
    assert np.array_equal(np.array(ROTATIONS[PAY]), [0, 0, 0])
    assert np.array_equal(np.array(STEP_MOVE[PAY]), [0, 0, 0])


# ---------------------------------------------------------------- transfer accounting

def test_pays_most_recent_cleaner():
    # agents 1 and 2 both cleaned recently; 2 more recently -> gets paid
    delta, attempted, executed, target = _pay(
        [PAY, STAY, STAY], last_clean_t=[-10000, 5, 8], current_t=10, balance=[3, 0, 0]
    )
    assert np.allclose(np.array(delta), [-1.0, 0.0, 1.0])
    assert np.array_equal(np.array(attempted), [True, False, False])
    assert np.array_equal(np.array(executed), [True, False, False])
    assert int(target[0]) == 2, "should pay the MOST recent cleaner (agent 2)"


def test_no_recent_cleaner_is_noop():
    # both cleaners are outside the window (age > 50) -> no valid recipient
    delta, _, executed, _ = _pay(
        [PAY, STAY, STAY], last_clean_t=[-10000, 5, 8], current_t=100, balance=[3, 0, 0]
    )
    assert np.allclose(np.array(delta), [0.0, 0.0, 0.0])
    assert not bool(executed[0])


def test_window_boundary_inclusive():
    # age exactly == clean_window is still payable; age == window+1 is not
    _, _, ex_in, _ = _pay([PAY, STAY], last_clean_t=[-10000, 0], current_t=50, balance=[1, 0])
    _, _, ex_out, _ = _pay([PAY, STAY], last_clean_t=[-10000, 0], current_t=51, balance=[1, 0])
    assert bool(ex_in[0]) and not bool(ex_out[0])


def test_insufficient_funds_is_noop():
    delta, attempted, executed, _ = _pay(
        [PAY, STAY], last_clean_t=[-10000, 5], current_t=10, balance=[0.0, 0.0]
    )
    assert np.allclose(np.array(delta), [0.0, 0.0])
    assert bool(attempted[0]) and not bool(executed[0]), "attempted but not executed (no funds)"


def test_partial_funds_below_amount_is_noop():
    # balance 0.5 < pay_amount 1.0 -> can't afford
    _, _, executed, _ = _pay([PAY, STAY], last_clean_t=[-10000, 5], current_t=10, balance=[0.5, 0])
    assert not bool(executed[0])


def test_cannot_pay_self_even_if_only_recent_cleaner():
    # agent 0 is the only recent cleaner AND the payer -> no valid other recipient
    delta, _, executed, _ = _pay([PAY, STAY], last_clean_t=[5, -10000], current_t=10, balance=[3, 0])
    assert np.allclose(np.array(delta), [0.0, 0.0])
    assert not bool(executed[0])


def test_multiple_payers_same_recipient_accumulate():
    # agents 0 and 3 both pay; agent 2 is the most recent cleaner -> receives 2
    delta, _, executed, target = _pay(
        [PAY, STAY, STAY, PAY], last_clean_t=[-10000, 3, 8, -10000], current_t=10, balance=[3, 0, 0, 3]
    )
    assert np.allclose(np.array(delta), [-1.0, 0.0, 2.0, -1.0])
    assert np.array_equal(np.array(executed), [True, False, False, True])
    assert int(target[0]) == 2 and int(target[3]) == 2


def test_zero_sum_under_random_configurations():
    rng = np.random.default_rng(0)
    for _ in range(100):
        n = int(rng.integers(1, 10))
        actions = jnp.array(rng.integers(0, len(Actions), n))
        last_clean_t = jnp.array(rng.integers(-20, 20, n), dtype=jnp.int32)
        balance = jnp.array(rng.uniform(0, 5, n), dtype=jnp.float32)
        delta, _, _, _ = compute_pay_transfers(actions, last_clean_t, 15, balance, 10, 1.0)
        assert abs(float(jnp.sum(delta))) < 1e-5, "transfers must be exactly zero-sum"


def test_custom_pay_amount():
    delta, _, executed, _ = _pay(
        [PAY, STAY], last_clean_t=[-10000, 5], current_t=10, balance=[3, 0], pay_amount=2.5
    )
    assert np.allclose(np.array(delta), [-2.5, 2.5])
    # balance 2.0 < amount 2.5 -> no-op
    _, _, ex2, _ = _pay([PAY, STAY], last_clean_t=[-10000, 5], current_t=10, balance=[2.0, 0], pay_amount=2.5)
    assert not bool(ex2[0])


def test_transfer_function_jits():
    jitted = jax.jit(compute_pay_transfers, static_argnums=())
    args = (jnp.array([PAY, PAY, STAY]), jnp.array([-10000, 2, 9], dtype=jnp.int32),
            12, jnp.array([3.0, 3.0, 0.0]), 50, 1.0)
    out = jitted(*args)
    ref = compute_pay_transfers(*args)
    for a, b in zip(out, ref):
        assert np.allclose(np.array(a), np.array(b))


# ---------------------------------------------------------------- tithe scheme

def test_tithe_pledge_activates_and_expires():
    # pledge at t=10 with duration 5 -> active for t in [10, 14], expired at t=15
    _, pledged, _, _, new_expiry, active = _tithe(
        [PAY, STAY], last_clean_t=[-10000, -10000], current_t=10,
        share_expiry_t=[0, 0], income=[0, 0], share_duration=5,
    )
    assert bool(pledged[0]) and int(new_expiry[0]) == 15 and bool(active[0])
    assert not bool(active[1]), "non-pledger stays inactive"
    # at t=14 (stored expiry 15) still active; at t=15 expired
    for t, expect in ((14, True), (15, False)):
        _, _, _, _, _, act = _tithe(
            [STAY, STAY], last_clean_t=[-10000, -10000], current_t=t,
            share_expiry_t=[15, 0], income=[0, 0], share_duration=5,
        )
        assert bool(act[0]) == expect, f"active at t={t} should be {expect}"


def test_tithe_shares_fraction_of_income_while_active():
    # agent 0 active (expiry 20 > t=10), harvests income 4.0; agent 1 cleaned recently
    delta, _, executed, target, _, _ = _tithe(
        [STAY, STAY], last_clean_t=[-10000, 8], current_t=10,
        share_expiry_t=[20, 0], income=[4.0, 0.0], share_fraction=0.5,
    )
    assert np.allclose(np.array(delta), [-2.0, 2.0]), "half of the 4.0 income moves"
    assert bool(executed[0]) and int(target[0]) == 1
    assert abs(float(jnp.sum(delta))) < 1e-6, "zero-sum"


def test_tithe_no_transfer_without_income_or_when_inactive():
    # active but no income -> nothing moves
    delta, _, executed, _, _, _ = _tithe(
        [STAY, STAY], last_clean_t=[-10000, 8], current_t=10,
        share_expiry_t=[20, 0], income=[0.0, 0.0],
    )
    assert np.allclose(np.array(delta), [0, 0]) and not bool(executed[0])
    # income but inactive (expiry in the past) -> harvester keeps everything
    delta2, _, executed2, _, _, _ = _tithe(
        [STAY, STAY], last_clean_t=[-10000, 8], current_t=10,
        share_expiry_t=[3, 0], income=[4.0, 0.0],
    )
    assert np.allclose(np.array(delta2), [0, 0]) and not bool(executed2[0])


def test_tithe_no_recipient_harvester_keeps_income():
    # active + income but nobody cleaned recently -> no transfer (pledge isn't punished)
    delta, _, executed, _, _, _ = _tithe(
        [STAY, STAY], last_clean_t=[-10000, -10000], current_t=100,
        share_expiry_t=[200, 0], income=[4.0, 0.0],
    )
    assert np.allclose(np.array(delta), [0, 0]) and not bool(executed[0])


def test_tithe_pledge_transfer_same_step_and_repledge_extends():
    # pledging and harvesting the same step: pledge takes effect immediately
    delta, pledged, executed, _, new_expiry, active = _tithe(
        [PAY, STAY], last_clean_t=[-10000, 9], current_t=10,
        share_expiry_t=[0, 0], income=[4.0, 0.0], share_fraction=0.25, share_duration=8,
    )
    assert bool(pledged[0]) and bool(active[0]) and bool(executed[0])
    assert np.allclose(np.array(delta), [-1.0, 1.0])  # 0.25 * 4.0
    assert int(new_expiry[0]) == 18
    # re-pledging while already active extends from current_t, not from old expiry
    _, _, _, _, expiry2, _ = _tithe(
        [PAY, STAY], last_clean_t=[-10000, 9], current_t=12,
        share_expiry_t=[18, 0], income=[0.0, 0.0], share_duration=8,
    )
    assert int(expiry2[0]) == 20


def test_tithe_cannot_share_to_self():
    # sharer is itself the only recent cleaner -> no valid recipient -> keeps income
    delta, _, executed, _, _, _ = _tithe(
        [STAY, STAY], last_clean_t=[9, -10000], current_t=10,
        share_expiry_t=[20, 0], income=[4.0, 0.0],
    )
    assert np.allclose(np.array(delta), [0, 0]) and not bool(executed[0])


def test_tithe_negative_income_never_shared():
    # callers clip at 0, but the function itself must also not move negative income
    delta, _, executed, _, _, _ = _tithe(
        [STAY, STAY], last_clean_t=[-10000, 8], current_t=10,
        share_expiry_t=[20, 0], income=[-3.0, 0.0],
    )
    assert np.allclose(np.array(delta), [0, 0]) and not bool(executed[0])


def test_tithe_split_divides_equally_among_recent_cleaners():
    # agent 0 active, harvests income 4.0; agents 1 and 2 both cleaned recently.
    # split: 0.5*4=2.0 shared, divided equally -> 1.0 each; harvester nets -2.0.
    delta, _, executed, _, _, _, received = compute_tithe_transfers(
        jnp.array([STAY, STAY, STAY]),
        jnp.array([-10000, 8, 9], dtype=jnp.int32), 10,
        jnp.array([20, 0, 0], dtype=jnp.int32),
        jnp.array([4.0, 0.0, 0.0], dtype=jnp.float32),
        50, 0.5, 50, True,
    )
    assert np.allclose(np.array(delta), [-2.0, 1.0, 1.0]), "2.0 split evenly across both cleaners"
    assert np.allclose(np.array(received), [0.0, 1.0, 1.0])
    assert bool(executed[0]) and abs(float(jnp.sum(delta))) < 1e-6


def test_tithe_split_with_one_cleaner_matches_latest():
    # only agent 1 cleaned recently -> split has nothing to divide, same as winner-take-all
    args = (jnp.array([STAY, STAY, STAY]),
            jnp.array([-10000, 8, -10000], dtype=jnp.int32), 10,
            jnp.array([20, 0, 0], dtype=jnp.int32),
            jnp.array([4.0, 0.0, 0.0], dtype=jnp.float32), 50, 0.5, 50)
    split = compute_tithe_transfers(*args, True)
    latest = compute_tithe_transfers(*args, False)
    assert np.allclose(np.array(split[0]), np.array(latest[0]))
    assert np.allclose(np.array(split[0]), [-2.0, 2.0, 0.0])


def test_tithe_split_respects_clean_window_and_self_exclusion():
    # agent 0 active + harvests; agent 1 cleaned within window (age 5), agent 2 too stale
    # (age 40 > window 25); agent 0 itself cleaned recently but can't receive its own tithe.
    delta, _, _, _, _, _, received = compute_tithe_transfers(
        jnp.array([STAY, STAY, STAY]),
        jnp.array([9, 5, -30], dtype=jnp.int32), 10,   # ages: 1, 5, 40
        jnp.array([20, 0, 0], dtype=jnp.int32),
        jnp.array([4.0, 0.0, 0.0], dtype=jnp.float32),
        25, 0.5, 50, True,
    )
    # only agent 1 is an eligible recipient -> all 2.0 to it, none to self (0) or stale (2)
    assert np.allclose(np.array(delta), [-2.0, 2.0, 0.0])
    assert np.allclose(np.array(received), [0.0, 2.0, 0.0])


def test_tithe_split_env_end_to_end():
    # split_recipients propagates through socialjax.make and the full env step
    env = socialjax.make("clean_up", num_agents=4, pay_mode="on", pay_scheme="tithe",
                         shared_rewards=False, split_recipients=True, pay_clean_window=25)
    assert env.split_recipients is True
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)
    # agents 1 and 2 cleaned recently; agent 0 pledged and is about to harvest
    state = state.replace(
        last_clean_t=jnp.array([-10000, 0, 0, -10000], dtype=jnp.int32),
        share_expiry_t=jnp.array([1000, 0, 0, 0], dtype=jnp.int32),
    )
    state, move_act = _place_apple_next_to(state, 0)
    obs, ns, rewards, done, info = env.step_env(key, state, [move_act, STAY, STAY, STAY])
    r = np.array(rewards).squeeze()
    # apple = num_agents = 4; 0.5*4 = 2.0 shared, split across agents 1 and 2 -> 1.0 each
    assert abs(r[0] - 2.0) < 1e-5, f"harvester keeps 4 - 2 = 2.0, got {r[0]}"
    assert abs(r[1] - 1.0) < 1e-5 and abs(r[2] - 1.0) < 1e-5, "each cleaner gets half of 2.0"
    assert np.allclose(np.array(info["pay_received"]), [0.0, 1.0, 1.0, 0.0])
    assert abs(float(np.sum(r)) - 4.0) < 1e-5, "zero-sum: total still one apple's 4.0"


def test_tithe_zero_sum_under_random_configurations():
    rng = np.random.default_rng(1)
    for _ in range(100):
        n = int(rng.integers(1, 10))
        actions = jnp.array(rng.integers(0, len(Actions), n))
        last_clean_t = jnp.array(rng.integers(-20, 20, n), dtype=jnp.int32)
        expiry = jnp.array(rng.integers(0, 40, n), dtype=jnp.int32)
        income = jnp.array(rng.uniform(0, 5, n), dtype=jnp.float32)
        split = bool(rng.integers(0, 2))  # zero-sum must hold for both recipient rules
        delta = compute_tithe_transfers(
            actions, last_clean_t, 15, expiry, income, 10, 0.5, 20, split
        )[0]
        assert abs(float(jnp.sum(delta))) < 1e-4, "tithe transfers must be zero-sum"


def test_tithe_function_jits():
    jitted = jax.jit(compute_tithe_transfers)
    args = (jnp.array([PAY, STAY, STAY]), jnp.array([-10000, 2, 9], dtype=jnp.int32),
            12, jnp.array([0, 0, 0], dtype=jnp.int32),
            jnp.array([4.0, 0.0, 0.0]), 50, 0.5, 20)
    out = jitted(*args)
    ref = compute_tithe_transfers(*args)
    for a, b in zip(out, ref):
        assert np.allclose(np.array(a), np.array(b))


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


def _seed_state(env, key, balance, last_clean_t):
    obs, state = env.reset(key)
    return state.replace(
        agent_balance=jnp.array(balance, dtype=jnp.float32),
        last_clean_t=jnp.array(last_clean_t, dtype=jnp.int32),
    )


def test_env_step_executes_payment_to_recent_cleaner():
    env = socialjax.make("clean_up", num_agents=4, shared_rewards=False, pay_mode="on")
    key = jax.random.PRNGKey(0)
    state = _seed_state(env, key, balance=[5, 0, 0, 0], last_clean_t=[-10000, -10000, 0, -10000])
    actions = [PAY, STAY, STAY, STAY]
    obs, ns, rewards, done, info = env.step_env(key, state, actions)
    assert float(info["pay_executed"][0]) == 1.0
    assert int(info["pay_target"][0]) == 2, "recipient must be the recent cleaner"
    assert np.allclose(np.array(rewards).squeeze(), [-1.0, 0.0, 1.0, 0.0]), "reward moves -1/+1"
    assert abs(float(ns.agent_balance[0]) - 4.0) < 1e-5, "payer balance 5 -> 4"
    assert abs(float(ns.agent_balance[2]) - 1.0) < 1e-5, "recipient balance 0 -> 1"


def test_env_step_on_vs_noop_zero_sum_and_placebo_moves_nothing():
    key = jax.random.PRNGKey(0)
    seed_kwargs = dict(balance=[5, 5, 5, 5], last_clean_t=[-10000, 0, -10000, -10000])
    actions = [PAY, STAY, PAY, PAY]

    env_on = socialjax.make("clean_up", num_agents=4, shared_rewards=False, pay_mode="on")
    st_on = _seed_state(env_on, key, **seed_kwargs)
    _, _, r_on, _, info_on = env_on.step_env(key, st_on, actions)

    env_noop = socialjax.make("clean_up", num_agents=4, shared_rewards=False, pay_mode="noop")
    st_noop = _seed_state(env_noop, key, **seed_kwargs)
    _, _, r_noop, _, info_noop = env_noop.step_env(key, st_noop, actions)

    assert abs(float(jnp.sum(r_on)) - float(jnp.sum(r_noop))) < 1e-5, "pay must be zero-sum"
    # noop logs the attempt but moves no reward
    assert float(jnp.sum(info_noop["pay_volume"])) == 0.0
    assert not np.allclose(np.array(r_on), np.array(r_noop)), "on must actually move reward vs noop"


def test_env_metrics_present_and_absent():
    env_on = socialjax.make("clean_up", num_agents=4, shared_rewards=False, pay_mode="on")
    key = jax.random.PRNGKey(0)
    st = _seed_state(env_on, key, balance=[5, 0, 0, 0], last_clean_t=[-10000, -10000, 0, -10000])
    _, _, _, _, info_on = env_on.step_env(key, st, [PAY, STAY, STAY, STAY])
    for k in ("pay_attempts", "pay_executed", "pay_volume", "pay_target"):
        assert k in info_on, f"missing metric {k} in pay_mode=on"

    env_off = socialjax.make("clean_up", num_agents=4, shared_rewards=False, pay_mode="off")
    _, st_off = env_off.reset(key)
    _, _, _, _, info_off = env_off.step_env(key, st_off, [STAY] * 4)
    for k in ("pay_attempts", "pay_executed", "pay_volume", "pay_target"):
        assert k not in info_off, f"metric {k} must not exist in pay_mode=off (baseline unchanged)"


def test_shaped_rewards_metric_reflects_pay_transfer():
    """info["shaped_rewards"] must include the pay transfer (it's captured before the
    pay block runs, so a future edit could silently revert it to the pre-transfer value)."""
    env = socialjax.make("clean_up", num_agents=4, shared_rewards=False, pay_mode="on")
    key = jax.random.PRNGKey(0)
    st = _seed_state(env, key, balance=[5, 0, 0, 0], last_clean_t=[-10000, -10000, 0, -10000])
    _, _, rewards, _, info = env.step_env(key, st, [PAY, STAY, STAY, STAY])
    assert np.allclose(np.array(info["shaped_rewards"]), np.array(rewards).squeeze())


def test_balance_accumulates_and_resets_on_episode_boundary():
    # A short-episode env: balance should accumulate net reward, and reset to 0 at
    # the inner-episode boundary along with last_clean_t.
    env = socialjax.make("clean_up", num_agents=4, shared_rewards=False, pay_mode="on", num_inner_steps=2)
    key = jax.random.PRNGKey(0)
    st = _seed_state(env, key, balance=[3, 0, 0, 0], last_clean_t=[-10000, 0, -10000, -10000])
    # step 1 (inner_t 0 -> 1): agent 0 pays agent 1 -> balances 3->2 (payer), 0->1 (recipient)
    _, st, _, _, _ = env.step_env(key, st, [PAY, STAY, STAY, STAY])
    assert abs(float(st.agent_balance[0]) - 2.0) < 1e-5
    # step 2 hits the inner-episode reset -> everything back to reset defaults
    _, st, _, _, _ = env.step(key, st, [STAY] * 4)
    assert np.allclose(np.array(st.agent_balance), [0, 0, 0, 0]), "balance resets each episode"
    assert np.all(np.array(st.last_clean_t) < 0), "last_clean_t resets each episode"


def test_env_step_off_matches_original_baseline():
    env_off = socialjax.make("clean_up", num_agents=4, shared_rewards=False, pay_mode="off")
    env_noop = socialjax.make("clean_up", num_agents=4, shared_rewards=False, pay_mode="noop")
    key = jax.random.PRNGKey(3)
    _, st_off = env_off.reset(key)
    _, st_noop = env_noop.reset(key)
    _, _, r_off, _, _ = env_off.step_env(key, st_off, [STAY] * 4)
    _, _, r_noop, _, _ = env_noop.step_env(key, st_noop, [STAY] * 4)
    assert np.allclose(np.array(r_off), np.array(r_noop)), (
        "with no pay actions taken, off and noop must produce identical rewards"
    )


# ---------------------------------------------------------------- tithe env integration

def _place_apple_next_to(state, agent_idx):
    """Put an apple on an empty cell adjacent to the agent; return (state, move_action).

    Lets a test force a deterministic harvest: the agent takes move_action and
    steps onto the apple. Direction/action mapping mirrors STEP_MOVE rows.
    """
    from socialjax.environments.cleanup.clean_up import Items

    locs = np.array(state.agent_locs)
    r, c = int(locs[agent_idx, 0]), int(locs[agent_idx, 1])
    grid = np.array(state.grid)
    H, W = grid.shape
    for dr, dc, act in ((1, 0, 4), (-1, 0, 5), (0, 1, 2), (0, -1, 3)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < H and 0 <= cc < W and grid[rr, cc] == int(Items.empty):
            return state.replace(grid=state.grid.at[rr, cc].set(jnp.int16(Items.apple))), act
    raise AssertionError("no empty neighbor found for apple placement")


def _tithe_env(pay_mode, **kwargs):
    return socialjax.make(
        "clean_up", num_agents=4, shared_rewards=False,
        pay_mode=pay_mode, pay_scheme="tithe", **kwargs,
    )


def test_env_tithe_shares_harvest_with_recent_cleaner():
    env = _tithe_env("on", share_fraction=0.5)
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)
    # agent 1 cleaned recently; agent 0 has an active pledge and is about to harvest
    state = state.replace(
        last_clean_t=jnp.array([-10000, 0, -10000, -10000], dtype=jnp.int32),
        share_expiry_t=jnp.array([1000, 0, 0, 0], dtype=jnp.int32),
    )
    state, move_act = _place_apple_next_to(state, 0)
    obs, ns, rewards, done, info = env.step_env(key, state, [move_act, STAY, STAY, STAY])
    r = np.array(rewards).squeeze()
    assert abs(r[0] - 2.0) < 1e-5, f"harvester keeps 4*(1-0.5)=2.0, got {r[0]}"
    assert abs(r[1] - 2.0) < 1e-5, f"cleaner receives the shared 2.0, got {r[1]}"
    assert float(info["pay_executed"][0]) == 1.0
    assert int(info["pay_target"][0]) == 1
    assert float(info["share_active"][0]) == 1.0
    assert abs(float(np.array(info["pay_volume"])[0]) - 2.0) < 1e-5


def test_env_tithe_noop_placebo_toggles_but_moves_nothing():
    env = _tithe_env("noop", share_fraction=0.5)
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)
    state = state.replace(
        last_clean_t=jnp.array([-10000, 0, -10000, -10000], dtype=jnp.int32),
        share_expiry_t=jnp.array([1000, 0, 0, 0], dtype=jnp.int32),
    )
    state, move_act = _place_apple_next_to(state, 0)
    obs, ns, rewards, done, info = env.step_env(key, state, [move_act, STAY, STAY, STAY])
    r = np.array(rewards).squeeze()
    assert abs(r[0] - 4.0) < 1e-5, "noop: harvester keeps the full apple"
    assert abs(r[1]) < 1e-5, "noop: cleaner receives nothing"
    assert float(np.array(info["pay_volume"])[0]) == 0.0
    # but the pledge dynamics still ran (placebo keeps action semantics identical)
    assert "share_active" in info


def test_env_tithe_pledge_sets_expiry_in_state():
    env = _tithe_env("on", share_duration=30)
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)  # inner_t = 0
    obs, ns, rewards, done, info = env.step_env(key, state, [PAY, STAY, STAY, STAY])
    assert int(ns.share_expiry_t[0]) == 30, "pledge at t=0 with duration 30 -> expiry 30"
    assert int(ns.share_expiry_t[1]) == 0, "non-pledgers unchanged"
    assert float(info["pay_attempts"][0]) == 1.0


def test_env_tithe_invalid_configs_rejected():
    for bad_kwargs in (dict(pay_scheme="foo"), dict(share_fraction=0.0), dict(share_fraction=1.5)):
        try:
            socialjax.make("clean_up", num_agents=3, pay_mode="on", **bad_kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad_kwargs} should raise ValueError")


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
