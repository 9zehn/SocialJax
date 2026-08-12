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


def test_random_start_spreads_the_first_move_but_keeps_alternation():
    """Rotation alone is not symmetric.

    The SPE of this game is agreement in round 0, so with a fixed order agent 0
    proposes on every path actually taken and keeps the first-mover premium
    permanently. The per-episode offset has to move the FIRST move around while
    leaving the alternating structure inside an episode intact.
    """
    n, envs = 5, 400
    offset = jax.random.randint(jax.random.PRNGKey(0), (envs,), 0, n)

    first = np.array(bargain.proposer_for_round(0, n, envs, "rotate",
                                                start_offset=offset))
    counts = np.bincount(first, minlength=n)
    assert (counts > envs / n * 0.7).all(), f"first move is not spread: {counts}"

    # Within an episode the order must still advance by exactly one agent a round.
    prev = first
    for r in range(1, 2 * n):
        cur = np.array(bargain.proposer_for_round(r, n, envs, "rotate",
                                                  start_offset=offset))
        assert np.array_equal(cur, (prev + 1) % n), f"round {r} broke alternation"
        prev = cur

    # And every agent still gets exactly one turn per full cycle, per env.
    cycle = np.stack([np.array(bargain.proposer_for_round(r, n, envs, "rotate",
                                                          start_offset=offset))
                      for r in range(n)])
    assert (np.sort(cycle, axis=0) == np.arange(n)[:, None]).all()

    # offset=None is the fixed-order ablation: agent 0 always opens.
    assert (np.array(bargain.proposer_for_round(0, n, envs, "rotate")) == 0).all()


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

def _feats(n, envs=2, level="protocol", **over):
    """bargaining_features with every argument at a neutral default."""
    kw = dict(round_idx=0, num_rounds=4, num_agents=n,
              proposer_idx=jnp.zeros((envs,), jnp.int32),
              last_theta_norm=jnp.zeros((envs,)),
              had_offer=jnp.zeros((envs,), bool),
              n_reject=jnp.zeros((envs,), jnp.int32),
              live_theta_norm=jnp.zeros((envs,)),
              offer_live=jnp.zeros((envs,)),
              last_votes=jnp.zeros((n, envs)),
              last_n_accept=jnp.zeros((envs,)),
              own_return=jnp.zeros((n, envs)), own_cleaning=jnp.zeros((n, envs)),
              river_stock=jnp.zeros((envs,)))
    kw.update(over)
    kw.setdefault("mask", bargain.feature_mask(level, n))
    return np.array(bargain.bargaining_features(**kw))


def test_the_vote_sees_the_offer_it_is_voting_on():
    """The defect this whole two-pass structure exists to fix.

    With the offer absent from the state, the only voting strategies expressible are
    "accept whoever proposed" and "always accept" -- a reservation value is not in
    the policy class at all. Both degenerate runs in findings.md are exactly those
    two. So: the live offer must reach the policy, and it must do so at the LOWEST
    tier, since `protocol` is meant to be sufficient for the SPE on its own.
    """
    n, envs = 4, 2
    low = _feats(n, envs, live_theta_norm=jnp.full((envs,), -0.9),
                 offer_live=jnp.ones((envs,)))
    high = _feats(n, envs, live_theta_norm=jnp.full((envs,), 0.9),
                  offer_live=jnp.ones((envs,)))
    assert not np.allclose(low, high), "theta on the table never reached the policy"
    # Exactly one slot may differ: the offer's value. Anything else moving would mean
    # the two passes disagree about the history as well as the phase.
    differing = np.where(np.abs(low - high).max(axis=(0, 1)) > 0)[0]
    assert list(differing) == [6], differing


def test_offer_live_flag_separates_no_offer_from_an_offer_of_zero():
    """theta=0 normalises to a real value, so without the flag "the midpoint of the
    range is on the table" and "nothing is on the table" are the same vector."""
    n = 4
    none_yet = _feats(n, offer_live=jnp.zeros((2,)), live_theta_norm=jnp.zeros((2,)))
    zero_offer = _feats(n, offer_live=jnp.ones((2,)), live_theta_norm=jnp.zeros((2,)))
    assert not np.allclose(none_yet, zero_offer)


def test_last_rounds_votes_and_count_are_visible():
    """The concession signal. "5 of 6 accepted" and "0 of 6" call for very different
    next offers, and WHICH agent refused says whom to appease."""
    n = 4
    base = _feats(n)
    votes = np.zeros((n, 2), np.float32)
    votes[2] = 1.0
    with_votes = _feats(n, last_votes=jnp.asarray(votes),
                        last_n_accept=jnp.ones((2,)))
    assert not np.allclose(base, with_votes)
    # The count is one slot, the per-agent votes are N slots, and they are distinct:
    # a change in who accepted must be visible even at a fixed count.
    other = np.zeros((n, 2), np.float32)
    other[3] = 1.0
    assert not np.allclose(with_votes, _feats(n, last_votes=jnp.asarray(other),
                                              last_n_accept=jnp.ones((2,))))
    # Every agent reads the same public record, so the vote block is identical
    # across rows -- only the "am I the proposer" flag is private to a row.
    block = with_votes[:, :, 8 + n:8 + 2 * n]
    assert np.allclose(block, block[0]), "last round's votes must be public"


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
    common = dict(own_return=jnp.ones((n, 2)), own_cleaning=jnp.ones((n, 2)))
    quiet = _feats(n, level="private", river_stock=jnp.zeros((2,)), **common)
    loud = _feats(n, level="private", river_stock=jnp.full((2,), 99.0), **common)
    assert np.allclose(quiet, loud), \
        "river stock reached the policy at the 'private' tier"
    seen = _feats(n, level="public", river_stock=jnp.full((2,), 99.0), **common)
    assert not np.allclose(quiet, seen), "'public' must actually expose it"


def test_proposer_sees_that_it_is_the_proposer():
    n = 3
    feats = _feats(n, round_idx=1, proposer_idx=jnp.array([2, 0]))
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


def test_advantage_is_standardised_over_active_rounds_only():
    """Agreement in round 0 is the SPE, so most of the K x E block is structural
    zeros. Normalising over those would make the update size track how fast the
    agents agreed rather than how good the decision was."""
    a = jnp.array([[10.0, 12.0], [0.0, 0.0], [0.0, 0.0]])
    w = jnp.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    out = np.array(bargain.masked_standardise(a, w))
    assert np.isclose(out[0].mean(), 0.0, atol=1e-5)
    assert np.isclose(out[0].std(), 1.0, atol=1e-3)
    # The unmasked version is dragged off zero-mean by the inert rounds.
    naive = np.array((a - a.mean()) / (a.std() + 1e-8))
    assert not np.isclose(naive[0].mean(), 0.0, atol=1e-2)
    # All-inactive must not divide by zero.
    assert np.isfinite(np.array(bargain.masked_standardise(a, jnp.zeros_like(w)))).all()


# ------------------------------------------------------------ the vote's floor

def _vote_rate(bias, eps, draws=4000):
    net = BargainingActorCritic(accept_bias=bias)
    x = jnp.zeros((draws, bargain.feature_dim(3)))
    p = net.init(jax.random.PRNGKey(0), x)
    _, pi_vote, _ = net.apply(p, x)
    vote, log_p = bargain.floored_vote(pi_vote, eps, jax.random.PRNGKey(1))
    return np.array(vote), np.array(log_p), np.array(pi_vote.probs[..., 1])


def test_vote_floor_keeps_both_branches_sampled():
    """Saturation is self-sealing: an agent that never refuses never learns what
    refusing would have bought it. ENT_COEF=0.01 did not stop either run saturating.
    """
    # A large accept bias is a saturated always-accept policy.
    vote, _, prob = _vote_rate(bias=8.0, eps=0.1)
    assert prob.mean() > 0.99, "the unfloored policy should be saturated here"
    assert 0.05 < 1.0 - vote.mean() < 0.15, \
        f"floor did not hold the reject branch open: {1 - vote.mean():.3f}"
    # ...and symmetrically for a saturated always-REJECT policy, which is the other
    # degenerate run (seed 42's veto dictator).
    vote, _, prob = _vote_rate(bias=-8.0, eps=0.1)
    assert prob.mean() < 0.01
    assert 0.05 < vote.mean() < 0.15


def test_floored_log_prob_is_the_distribution_actually_sampled_from():
    """PPO's ratio is only a correct importance weight if the stored log-prob is the
    behaviour policy's, so it must reflect the floor rather than the raw logits."""
    vote, log_p, prob = _vote_rate(bias=8.0, eps=0.1, draws=64)
    p_floor = np.clip(prob, 0.1, 0.9)
    want = np.where(vote == 1, np.log(p_floor), np.log1p(-p_floor))
    assert np.allclose(log_p, want, atol=1e-5)
    assert not np.allclose(log_p, np.where(vote == 1, np.log(prob),
                                           np.log1p(-prob)), atol=1e-3)
    # eps=0 has to be exactly the policy again, so evaluation replays the policy.
    vote, log_p, prob = _vote_rate(bias=1.0, eps=0.0, draws=64)
    want = np.where(vote == 1, np.log(prob), np.log1p(-prob))
    assert np.allclose(log_p, want, atol=1e-5)


def test_vote_floor_anneals_to_zero():
    """It is exploration, not part of the mechanism: the checkpointed policy has to
    be the one a replay reproduces."""
    assert np.isclose(float(bargain.vote_eps_at(0.05, 0, 100)), 0.05)
    assert np.isclose(float(bargain.vote_eps_at(0.05, 50, 100)), 0.025)
    assert float(bargain.vote_eps_at(0.05, 100, 100)) == 0.0
    assert float(bargain.vote_eps_at(0.05, 150, 100)) == 0.0     # never negative


def test_vote_floor_can_end_above_zero():
    """Rejection is a policing strategy: it is only maintained while occasionally
    sampled, so the anneal must be able to stop at a persistent floor instead of
    extinguishing it (the fixesV1 run annealed to 0, and proposers began walking
    theta back down as the last sampled rejections disappeared)."""
    assert np.isclose(float(bargain.vote_eps_at(0.05, 0, 100, end=0.02)), 0.05)
    assert np.isclose(float(bargain.vote_eps_at(0.05, 50, 100, end=0.02)), 0.035)
    assert np.isclose(float(bargain.vote_eps_at(0.05, 100, 100, end=0.02)), 0.02)
    assert np.isclose(float(bargain.vote_eps_at(0.05, 150, 100, end=0.02)), 0.02)


# ------------------------------------------------------ checkpoint compatibility

def test_stale_bargaining_checkpoints_are_refused_not_misread():
    """A feature layout change makes old weights unreadable. The failure to prevent
    is the silent one -- replaying a checkpoint as a mechanism it was never trained
    on, which has already invalidated one comparison in this project."""
    n = 7
    net = BargainingActorCritic()
    good = net.init(jax.random.PRNGKey(0), jnp.zeros((1, bargain.feature_dim(n))))
    bargain.check_params_compatible(good, n, bargain.FEATURE_VERSION)   # no raise

    stale = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 9 + n)))      # version 1
    for args in ((stale, n, bargain.FEATURE_VERSION), (stale, n, None),
                 (good, n, bargain.FEATURE_VERSION - 1)):
        try:
            bargain.check_params_compatible(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"should have refused: {args[2]!r}")
    # A width mismatch is the same class of error and equally silent.
    try:
        bargain.check_params_compatible(good, n, bargain.FEATURE_VERSION, hidden=32)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong BARGAIN_HIDDEN should be refused")


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


def test_a_reservation_value_is_representable():
    """The end-to-end version of the two-pass fix: p(accept) has to be a FUNCTION of
    the theta on the table, all the way through the network to the vote head.

    Under the one-pass round this was flat by construction, whatever the weights --
    which is why `evaluate_bargain`'s accept-rate-by-theta table could only ever
    have come out constant.
    """
    n = 3
    net = BargainingActorCritic()
    params = net.init(jax.random.PRNGKey(0),
                      jnp.zeros((1, bargain.feature_dim(n))))

    def p_accept(theta_norm, params):
        feats = jnp.asarray(_feats(n, envs=1, live_theta_norm=jnp.array([theta_norm]),
                                   offer_live=jnp.ones((1,))))
        _, pi_vote, _ = net.apply(params, feats[1])          # a responder's row
        return float(pi_vote.probs[0, 1])

    assert p_accept(-1.0, params) != p_accept(1.0, params), \
        "the vote head cannot see the offer at all"

    # And the dependence can be made SHARP -- a threshold is inside the policy
    # class, not just a numerical wobble at init. Slot 6 is the live offer; Dense_3
    # is the vote head, whose orthogonal(0.01) init is what keeps the swing small
    # until something has been learned.
    sharp = dict(params)
    layers = dict(params["params"])
    k0 = layers["Dense_0"]["kernel"]
    layers["Dense_0"] = {**layers["Dense_0"], "kernel": k0.at[6].set(k0[6] * 50.0)}
    layers["Dense_3"] = {**layers["Dense_3"],
                         "kernel": layers["Dense_3"]["kernel"] * 50.0}
    sharp["params"] = layers
    lo, hi = p_accept(-1.0, sharp), p_accept(1.0, sharp)
    assert abs(hi - lo) > 0.4, f"no threshold reachable: {lo:.3f} -> {hi:.3f}"


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
