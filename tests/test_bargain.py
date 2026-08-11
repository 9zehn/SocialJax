"""Tests for the Rubinstein bargaining stage.

The invariants here are the ones that, if wrong, produce a plausible-looking run
that measures the wrong thing: a quorum that lets the proposer vote for its own
offer, credit assigned to rounds where no decision was made, or a semi-MDP reward
that drops the payoff of agreeing.

Runnable two ways:
    python tests/test_bargain.py
    python -m pytest tests/test_bargain.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

from algorithms.MOCA import bargain
from algorithms.MOCA.networks import BargainingActorCritic


# ------------------------------------------------------------------- protocol

def test_quorum_sizes():
    assert bargain.quorum_size("all", 7) == 6, "every non-proposer"
    assert bargain.quorum_size("majority", 7) == 4, "strict majority of 6 responders"
    assert bargain.quorum_size(3, 7) == 3
    for bad in ("nope", 0, 7):
        try:
            bargain.quorum_size(bad, 7)
        except ValueError:
            pass
        else:
            raise AssertionError(f"quorum {bad!r} should raise")


def test_rotation_gives_every_agent_equal_turns():
    """The point of rotating is symmetry -- if turns were uneven, so is the split."""
    counts = np.zeros(5, dtype=int)
    for r in range(20):
        p = np.array(bargain.proposer_for_round(r, 5, 3, "rotate"))
        assert (p == p[0]).all(), "rotation must be identical across envs"
        counts[p[0]] += 1
    assert (counts == 4).all(), counts


def test_contribution_proposer_favours_the_cleaners():
    """Proposal power should accrue to whoever provisions the public good."""
    contributions = jnp.array([[9.0] * 400, [0.1] * 400, [0.1] * 400])   # (N, E)
    p = np.array(bargain.proposer_for_round(
        0, 3, 400, "contribution", key=jax.random.PRNGKey(0),
        contributions=contributions))
    assert (p == 0).mean() > 0.9, "the heavy contributor should usually propose"
    # All-zero contributions (the state at every episode start) must stay uniform
    # rather than divide by zero.
    p0 = np.array(bargain.proposer_for_round(
        0, 3, 400, "contribution", key=jax.random.PRNGKey(1),
        contributions=jnp.zeros((3, 400))))
    assert set(np.unique(p0)) == {0, 1, 2}


def test_proposer_never_votes_on_its_own_offer():
    """Otherwise the proposer could carry a contract nobody else wanted."""
    num_agents, quorum = 4, 3          # unanimity among the 3 responders
    # Everyone accepts EXCEPT one responder; with the proposer's own vote wrongly
    # counted this would still reach 3 and pass.
    votes = jnp.array([[True], [True], [True], [False]])       # (N, E)
    proposer = jnp.array([0])
    passed, n = bargain.accepted(votes, proposer, quorum, num_agents)
    assert not bool(passed[0]) and int(n[0]) == 2

    # And a proposer voting against itself must not block its own offer.
    votes = jnp.array([[False], [True], [True], [True]])
    passed, n = bargain.accepted(votes, proposer, quorum, num_agents)
    assert bool(passed[0]) and int(n[0]) == 3


# ------------------------------------------------------------------- features

def test_feature_tiers_are_nested_and_maskable():
    n = 7
    dim = bargain.feature_dim(n)
    masks = {lvl: np.array(bargain.feature_mask(lvl, n))
             for lvl in bargain.FEATURE_LEVELS}
    assert all(m.shape == (dim,) for m in masks.values())
    # Strictly nested, so an ablation only ever removes information.
    assert masks["protocol"].sum() < masks["private"].sum() < masks["public"].sum()
    for a, b in (("protocol", "private"), ("private", "public")):
        assert np.all(masks[a] <= masks[b]), f"{a} must be a subset of {b}"
    assert masks["public"].sum() == dim


def test_private_tier_hides_commons_aggregates():
    """The default tier must not leak the mean-contribution signal.

    Handing an agent the average contribution is close to handing it the inequality
    the mechanism is supposed to discover, so a fair split at "private" must not be
    attributable to it.
    """
    n = 3
    common = dict(round_idx=0, num_rounds=4, num_agents=n,
                  proposer_idx=jnp.zeros((2,), jnp.int32),
                  last_theta_norm=jnp.zeros((2,)), had_offer=jnp.zeros((2,), bool),
                  n_reject=jnp.zeros((2,), jnp.int32),
                  own_return=jnp.ones((n, 2)), own_cleaning=jnp.ones((n, 2)))
    quiet = bargain.bargaining_features(
        river_stock=jnp.zeros((2,)), mask=bargain.feature_mask("private", n), **common)
    loud = bargain.bargaining_features(
        river_stock=jnp.full((2,), 99.0), mask=bargain.feature_mask("private", n),
        **common)
    assert np.allclose(np.array(quiet), np.array(loud)), \
        "river stock reached the policy at the 'private' tier"
    seen = bargain.bargaining_features(
        river_stock=jnp.full((2,), 99.0), mask=bargain.feature_mask("public", n),
        **common)
    assert not np.allclose(np.array(quiet), np.array(seen)), \
        "'public' must actually expose it"


def test_proposer_sees_that_it_is_the_proposer():
    n = 3
    feats = np.array(bargain.bargaining_features(
        round_idx=1, num_rounds=4, proposer_idx=jnp.array([2, 0]), num_agents=n,
        last_theta_norm=jnp.zeros((2,)), had_offer=jnp.zeros((2,), bool),
        n_reject=jnp.zeros((2,), jnp.int32), own_return=jnp.zeros((n, 2)),
        own_cleaning=jnp.zeros((n, 2)), river_stock=jnp.zeros((2,)),
        mask=bargain.feature_mask("protocol", n)))
    # feature 1 is "am I the proposer"; env 0 -> agent 2, env 1 -> agent 0.
    assert feats[2, 0, 1] == 1.0 and feats[0, 0, 1] == 0.0
    assert feats[0, 1, 1] == 1.0 and feats[2, 1, 1] == 0.0


# ------------------------------------------------------------ credit over rounds

def _run_gae(rewards, active, terminal, values=None, gamma=1.0, lam=1.0):
    K = len(rewards)
    r = jnp.array(rewards, dtype=jnp.float32)[:, None]
    v = jnp.zeros((K, 1)) if values is None else jnp.array(values, jnp.float32)[:, None]
    a = jnp.array(active, bool)[:, None]
    t = jnp.array(terminal, bool)[:, None]
    adv, targets = bargain.round_gae(r, v, a, t, gamma, lam)
    return np.array(adv)[:, 0], np.array(targets)[:, 0]


def test_rounds_after_agreement_get_no_credit():
    """No decision is made there, so training on them would credit inert actions."""
    # Agreed at round 1: rounds 0-1 active, 1 terminal, 2-3 inactive.
    adv, _ = _run_gae(rewards=[0.0, 10.0, 3.0, 3.0],
                      active=[True, True, False, False],
                      terminal=[False, True, False, False])
    assert adv[2] == 0.0 and adv[3] == 0.0
    assert adv[1] != 0.0


def test_agreement_round_carries_the_whole_remaining_payoff():
    """With zero baselines and no discount, the advantage of agreeing at round 1 is
    the reward booked there -- which the caller has already made the sum of every
    remaining segment."""
    adv, targets = _run_gae(rewards=[0.0, 30.0, 0.0, 0.0],
                            active=[True, True, False, False],
                            terminal=[False, True, False, False])
    assert np.isclose(adv[1], 30.0)
    assert np.isclose(adv[0], 30.0), "the rejected round sees the payoff it led to"
    assert np.isclose(targets[1], 30.0)


def test_never_agreeing_terminates_at_the_last_round():
    """No bootstrap past the end of the episode."""
    adv, _ = _run_gae(rewards=[1.0, 1.0, 1.0, 1.0],
                      active=[True] * 4, terminal=[False, False, False, True])
    assert np.isclose(adv[3], 1.0)
    assert np.isclose(adv[0], 4.0), "undiscounted sum of the remaining rounds"


def test_discounting_rounds_shrinks_the_delayed_payoff():
    """Guard on BARGAIN_GAMMA actually being wired through (it should stay 1.0)."""
    adv_1, _ = _run_gae([0.0, 10.0], [True, True], [False, True], gamma=1.0)
    adv_h, _ = _run_gae([0.0, 10.0], [True, True], [False, True], gamma=0.5)
    assert np.isclose(adv_1[0], 10.0) and np.isclose(adv_h[0], 5.0)


# --------------------------------------------------------------------- network

def test_accept_bias_lifts_unanimity_off_the_floor():
    """Six responders at an unbiased init agree 1.6% of the time, which starves the
    proposal head. The bias is the cheap fix; this pins that it works."""
    F = bargain.feature_dim(7)
    x = jnp.zeros((1, F))
    probs = {}
    for bias in (0.0, 1.0):
        net = BargainingActorCritic(accept_bias=bias)
        p = net.init(jax.random.PRNGKey(0), x)
        _, pi_vote, _ = net.apply(p, x)
        probs[bias] = float(jnp.exp(pi_vote.log_prob(jnp.ones((1,), int)))[0])
    assert abs(probs[0.0] - 0.5) < 0.05
    assert probs[1.0] > 0.7
    # What matters is the joint event, not the single vote: unanimity has to clear
    # the floor by enough that phase-2 signal exists at all. 0.5**6 = 1.6% does not.
    assert probs[0.0] ** 6 < 0.02 < 0.10 < probs[1.0] ** 6


def test_heads_are_shaped_for_their_decisions():
    F = bargain.feature_dim(4)
    net = BargainingActorCritic()
    x = jnp.zeros((5, F))
    p = net.init(jax.random.PRNGKey(0), x)
    pi_theta, pi_vote, v = net.apply(p, x)
    theta = pi_theta.sample(seed=jax.random.PRNGKey(1))
    assert theta.shape == (5, 1), "one scalar contract per env"
    assert pi_theta.log_prob(theta).shape == (5,)
    vote = pi_vote.sample(seed=jax.random.PRNGKey(2))
    assert vote.shape == (5,) and set(np.unique(np.array(vote))) <= {0, 1}
    assert v.shape == (5,)


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
