"""Rubinstein-style alternating-offers bargaining over the contract space.

Replaces the take-it-or-leave-it stage of Christoffersen et al. (arXiv:2208.10469).
Their negotiation is one shot: agent 0 proposes, nu=2 sampled non-proposers gate the
contract with the PRODUCT of their accept probabilities, and rejection nulls the
contract for the whole episode. The paper is explicit that this is a limitation --
"Formal contracting does not have the dynamic structure of a negotiation and lets a
proposing agent make a take-it-or-leave-it offer" -- and Proposition 4.5 spells out
the consequence: "All agents except for the proposing agent are compensated exactly
to the point of indifference."

Prop 4.5 is not the obstacle here, it is the mechanism. The proposer holds responders
to their CONTINUATION VALUE. One-shot, that continuation is "no contract for the rest
of the episode", which in Clean Up is close to worthless, so the proposer takes
everything. Give the responder a credible next move and the same extraction logic
yields an equitable split -- with no fairness axiom added anywhere.

The game, over an episode of T steps split into K segments of `segment` steps:

    round r = 0..K-1, while no contract is in force:
        proposer p(r) offers theta_r                         (forward pass 1)
        every other agent sees theta_r and votes accept/reject   (forward pass 2)
        if #accept >= quorum:  theta_r binds for ALL REMAINING segments, done
        else:                  theta = 0 for segment r, continue to round r+1
    never agreed -> the null contract for the whole episode

That is `BARGAIN_BINDING=episode`. Under `segment` and `sticky` there is no "while
no contract is in force" -- every segment is renegotiated from scratch, so an offer
that carries governs its own segment only. See `apply_binding` for why, and for
what it costs: the shrinking pie and the delay cost that make this Rubinstein are
exactly what those modes remove.

The round is TWO passes over the same network, not one, and that is load-bearing: a
vote produced in the same pass as the proposal cannot condition on the offer, which
leaves "reject anything below my reservation value" outside the policy class
entirely. See the feature layout note below.

Three deliberate departures from the reference, each with a reason:

  * ROTATING proposer, one per round. The paper warns that "if two or more agents may
    propose in a game, SPEs may be socially suboptimal" (Appendix A) -- but that
    concerns SIMULTANEOUS competing proposers. Alternating offers has exactly one
    proposer per round, so the structure that result needs survives; what changes is
    that rejection is no longer fatal.
  * UNANIMITY among non-proposers by default, not nu=2. With majority and 7 agents
    this becomes Baron-Ferejohn: the proposer buys a minimal winning coalition and
    excludes the rest -- and in Clean Up the excluded minority is exactly the
    cleaners the contract exists to compensate. Unanimity is affordable only because
    rejection now costs one segment instead of the episode.
  * COUNTED votes, not a product of probabilities. The reference's accept
    "probability" is a clipped coordinate of a Gaussian action, so its log-prob is
    the Gaussian's rather than a Bernoulli's, and the product over nu voters
    conflates "I accept" with "the contract passes". Here each responder emits a real
    Bernoulli vote and the contract passes on a count.
"""
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp

PROPOSER_MODES = ("rotate", "random", "contribution", "holdout")
# Which contracting game each round plays. "alternating" is everything this module
# was built for: one proposer, a vote, a quorum. "median" replaces the round with a
# single simultaneous move -- every agent names a theta and the median of the N asks
# binds for the segment, no vote at all. See `median_offer` for why.
PROTOCOL_MODES = ("alternating", "median")
# Which slice of the bargaining state the policy may see. Tiered so the headline
# result can be shown not to depend on handing agents the inequality signal --
# see `feature_mask`.
FEATURE_LEVELS = ("protocol", "private", "public")
# How long an accepted contract binds for. See `apply_binding`.
BINDING_MODES = ("episode", "segment", "sticky")

# Bumped whenever the feature layout or the round protocol changes, and recorded in
# every run's .run.yaml sidecar. Weights are shaped by the feature dimension, so a
# checkpoint from an older version cannot be loaded by this code at all -- but a
# version number turns that into a clear message instead of a shape error six frames
# down, and catches the case where the dimension coincidentally survives a change.
#
#   1  round / turn / standing (rejected) offer / rejection count / proposer one-hot,
#      one forward pass per agent per round. The vote never saw the offer it was
#      voting on, so a reservation-value strategy was unrepresentable.
#   2  two-pass round (propose, then vote on the live offer), a live-offer slot with
#      its own 0/1 flag, and last round's votes and accept count.
FEATURE_VERSION = 2


# ------------------------------------------------------------------ protocol

def quorum_size(spec, num_agents: int) -> int:
    """How many of the N-1 non-proposers must accept for a contract to bind."""
    responders = num_agents - 1
    if spec == "all":
        return responders
    if spec == "majority":
        return responders // 2 + 1
    q = int(spec)
    if not 1 <= q <= responders:
        raise ValueError(
            f"BARGAIN_QUORUM must be 'all', 'majority', or an int in "
            f"[1, {responders}] for {num_agents} agents; got {spec!r}"
        )
    return q


def proposer_for_round(round_idx, num_agents: int, num_envs: int, mode: str,
                       key=None, contributions=None, start_offset=None,
                       holdouts=None) -> jnp.ndarray:
    """(num_envs,) index of the proposing agent, per env.

    rotate       round r is agent (r + start_offset) mod N -- alternating offers.
                 `start_offset` is drawn ONCE PER EPISODE, per env, and is what keeps
                 the protocol symmetric: without it agent 0 proposes in round 0 of
                 every single episode, and since the SPE of this game is agreement in
                 round 0, agent 0 would hold a permanent first-mover advantage on
                 every path actually taken. Rotation would then shape only off-path
                 continuation values while one fixed agent captured the premium --
                 and if that agent happened to emerge as a cleaner, the fairness
                 result would flatter itself. Pass None to recover the fixed order
                 (the ablation that measures how large that advantage is).
    random       uniform each round (Baron-Ferejohn's random recognition). Included
                 because it is the standard multilateral baseline, and because with
                 majority quorum it should reproduce minority exclusion.
    contribution sampled proportional to cumulative cleaning, so proposal power
                 accrues to whoever provisions the public good (Ostrom's
                 proportional equivalence). The only way to game it is to clean more.
    holdout      drawn uniformly among LAST round's rejecters (`holdouts`, (N, E)),
                 falling back to uniform-over-all when there are none (round 0, or
                 a null offer that passed). This is the multilateral analogue of
                 Rubinstein's alternation, where the player who refuses is exactly
                 the one who speaks next: it collapses the credit path from
                 "reject, then hope the rotation reaches you" -- two coordinated
                 moves -- to "reject and you may hold the pen". The strategic risk
                 is its point: rejecting to seize proposal power is priced by the
                 segment the rejection burns, so the pen goes to whoever values
                 changing the offer more than a tenth of the episode.
    """
    if mode == "rotate":
        if start_offset is None:
            return jnp.full((num_envs,), round_idx % num_agents, dtype=jnp.int32)
        return ((start_offset + round_idx) % num_agents).astype(jnp.int32)
    if mode == "random":
        return jax.random.randint(key, (num_envs,), 0, num_agents).astype(jnp.int32)
    if mode == "contribution":
        # (N, E) -> (E, N) logits; +eps so an all-zero column stays uniform rather
        # than NaN, which is the state at the very start of every episode.
        w = jnp.transpose(contributions) + 1e-6
        return jax.random.categorical(key, jnp.log(w), axis=-1).astype(jnp.int32)
    if mode == "holdout":
        w = jnp.transpose(jnp.asarray(holdouts, jnp.float32))        # (E, N)
        # No holdouts (round 0, or a passed null offer) -> random recognition.
        has_any = w.sum(axis=-1, keepdims=True) > 0
        w = jnp.where(has_any, w, jnp.ones_like(w))
        return jax.random.categorical(key, jnp.log(w + 1e-9), axis=-1).astype(jnp.int32)
    raise ValueError(f"unknown BARGAIN_PROPOSER {mode!r} "
                     f"(available: {', '.join(PROPOSER_MODES)})")


def apply_binding(mode: str, passed, offer_null, theta_offer, agreed, standing,
                  null_theta) -> Tuple:
    """How long an accepted offer binds. The one place the three protocols differ.

    `episode` is the original game: the first offer to carry binds for every
    remaining segment and bargaining ENDS. That makes the stake attached to a single
    vote the whole rest of the episode, while the threat backing that vote is one
    segment -- and the measured lock-continue gaps (+60 to +250 against a per-agent
    episode return around 390) say that ratio is why nobody can afford to refuse.
    It also hands the entire episode to whoever happens to propose in round 0.

    `segment` renegotiates from scratch every segment: an offer that carries governs
    ITS OWN segment and nothing more, and a failed round plays uncontracted. Stake
    and threat are then the same size -- one segment either way -- and proposing in
    round r captures round r rather than the episode. This stops being Rubinstein
    (there is no shrinking pie and no delay cost) and becomes a repeated contracting
    game; the literature to check it against is repeated games and relational
    contracts rather than alternating offers.

    `sticky` renegotiates too, but a failed round leaves the INCUMBENT contract in
    force instead of falling back to null. Rejection then costs a responder who
    likes the current deal nothing at all, which is the strongest responder position
    of the three -- at the price that round 0 still bargains against a null
    disagreement point and whatever it settles becomes the default thereafter.

    Under every mode an offer of exactly the null contract never takes force: it is
    a formal pass, so the fallback applies and the vote on it is outcome-free.

    Args:
        passed: (E,) bool, did the vote reach quorum.
        offer_null: (E,) bool, was the offer exactly the null contract.
        theta_offer: (E,) the offer on the table.
        agreed: (E,) bool, has a contract already bound (episode mode only; stays
            False forever under the renegotiating modes, which is what keeps every
            round `active` with no special-casing downstream).
        standing: (E,) the contract in force before this round -- the locked one
            under `episode`, last segment's effective theta under `sticky`.
        null_theta: the null contract's value.

    Returns:
        in_force: (E,) bool, did THIS round's offer take force.
        theta_eff: (E,) the contract this segment actually plays under.
        next_agreed, next_standing: the carry for the following round.
    """
    if mode not in BINDING_MODES:
        raise ValueError(f"BARGAIN_BINDING must be one of "
                         f"{', '.join(BINDING_MODES)}; got {mode!r}")
    live = passed & ~offer_null
    if mode == "episode":
        in_force = live & ~agreed
        theta_eff = jnp.where(
            agreed, standing, jnp.where(in_force, theta_offer, null_theta))
        return (in_force, theta_eff, agreed | in_force,
                jnp.where(in_force, theta_offer, standing))
    fallback = jnp.float32(null_theta) if mode == "segment" else standing
    theta_eff = jnp.where(live, theta_offer, fallback)
    # `agreed` is passed through untouched: nothing is ever absorbing here, so every
    # round stays a decision point and `active` needs no mode-specific handling.
    return live, theta_eff, agreed, theta_eff


def is_proposer_mask(proposer_idx, num_agents: int) -> jnp.ndarray:
    """(N, E) bool: which agent proposes in each env."""
    return jnp.arange(num_agents)[:, None] == proposer_idx[None, :]


def accepted(vote_accept, proposer_idx, quorum: int, num_agents: int) -> Tuple:
    """Does the offer bind, per env?

    Args:
        vote_accept: (N, E) bool, every agent's vote (the proposer's is ignored).
        proposer_idx: (E,) int, who proposed.
        quorum: how many non-proposers must accept.

    Returns:
        (E,) bool passed, and (E,) int32 the accept count, for logging.
    """
    counted = vote_accept & ~is_proposer_mask(proposer_idx, num_agents)
    n_accept = counted.sum(axis=0)
    return n_accept >= quorum, n_accept.astype(jnp.int32)


# ------------------------------------------------------------------ features
#
# Layout, for N agents (F = 12 + 2N). The tiers are contiguous and in this order so
# `feature_mask` can select a prefix and the ablation stays a one-value change.
#
#   protocol (8 + 2N)
#     0            rounds left, / num_rounds
#     1            am I the proposer this round
#     2            has any offer been made yet
#     3            the standing REJECTED offer, normalised to [-1, 1]
#     4            rejections so far, / num_rounds
#     5            is there an offer on the table RIGHT NOW  (0 in the proposal pass)
#     6            that live offer, normalised to [-1, 1]    (0 when none)
#     7            last round's accept count, / (N - 1)
#     8 .. 8+N-1   proposer one-hot
#     8+N .. 7+2N  last round's per-agent accept votes (the proposer's slot is 0:
#                  it casts no counted vote)
#   private (2)
#     8+2N         own accumulated return, scaled
#     9+2N         own accumulated cleaning, scaled
#   public (2)
#     10+2N        river stock, scaled
#     11+2N        mean cleaning across agents, scaled
#
# Slots 5 and 6 are the fix for the structural defect in version 1: a vote that
# cannot see the offer can only condition on WHO proposed and WHEN, so "reject
# anything below my reservation value" -- the strategy the whole mechanism rests on
# -- was not in the policy class. Slot 5 is separate from slot 6 so the network can
# distinguish "no offer yet" from "an offer of value 0".

def feature_dim(num_agents: int) -> int:
    return 12 + 2 * num_agents


def normalise_theta(theta, low: float, high: float):
    """A contract value on [low, high] -> [-1, 1], the scale the policy reads.

    The same mapping `negotiate.unsquash` inverts, kept here so the training loop,
    the evaluator and the viewer cannot drift apart on it.
    """
    return 2.0 * (theta - low) / (high - low) - 1.0


def feature_mask(level: str, num_agents: int) -> jnp.ndarray:
    """(F,) 0/1 mask selecting which tier of the bargaining state is visible.

    The tiers exist because "handcrafted features" is a fair criticism to level at a
    learned bargainer, and the tiers separate the fair part from the contentious one.

    protocol  The extensive-form game itself: round, whose turn, the offer on the
              table, the standing rejected offer, how many rounds have failed, and
              how last round's votes fell. Every bargaining model in the literature
              assumes players know these, and the SPE of a finite alternating-offers
              game is Markov in (round, proposer, offer) -- so this tier alone is in
              principle sufficient to represent the equilibrium.
    private   ...plus the agent's OWN accumulated return and cleaning. Standard: you
              know your own payoff history.
    public    ...plus commons-level aggregates (river stock, mean cleaning). This is
              where the objection bites -- telling an agent the average contribution
              is close to handing it the inequality signal the mechanism is supposed
              to discover on its own. Kept available, but as an ablation rather than
              the default, so a fair split at `private` cannot be attributed to it.

    Masking rather than resizing keeps the network shape fixed across levels, so the
    ablation changes one config value and nothing else.
    """
    if level not in FEATURE_LEVELS:
        raise ValueError(f"BARGAIN_FEATURES must be one of "
                         f"{', '.join(FEATURE_LEVELS)}; got {level!r}")
    n_protocol = 8 + 2 * num_agents
    mask = jnp.zeros((feature_dim(num_agents),), dtype=jnp.float32)
    mask = mask.at[:n_protocol].set(1.0)
    if level in ("private", "public"):
        mask = mask.at[n_protocol:n_protocol + 2].set(1.0)
    if level == "public":
        mask = mask.at[n_protocol + 2:].set(1.0)
    return mask


def bargaining_features(round_idx, num_rounds: int, proposer_idx, num_agents: int,
                        last_theta_norm, had_offer, n_reject,
                        live_theta_norm, offer_live, last_votes, last_n_accept,
                        own_return, own_cleaning, river_stock, mask) -> jnp.ndarray:
    """(N, E, F) bargaining state, one row per agent.

    Called TWICE per round, which is the point (see the layout note above). Once
    before anyone has moved, with `offer_live=0`, to produce the proposal; then again
    with the proposer's theta in `live_theta_norm` and `offer_live=1`, to produce the
    responders' votes and each responder's critic value. Every other argument is the
    same across the two passes, so the network reads one consistent history and only
    the phase changes.

    Everything is pre-scaled to roughly unit range; a raw episode return of ~500
    alongside a 0/1 turn flag would make the first layer's job needlessly hard.

    Args:
        last_theta_norm: (E,) the standing (rejected) offer on [-1, 1], 0 if none.
        had_offer: (E,) bool, whether any offer has been made yet.
        n_reject: (E,) how many rounds have failed so far.
        live_theta_norm: (E,) the offer currently on the table on [-1, 1]; 0 in the
            proposal pass, where nothing has been offered yet.
        offer_live: (E,) 0/1, whether that slot holds a real offer. Separate from the
            value so "no offer" is distinguishable from "an offer of 0".
        last_votes: (N, E) last round's accept votes, 0/1, zeros in round 0. Who
            refused is what tells a proposer whom it has to appease.
        last_n_accept: (E,) last round's RAW accept count; scaled here by the number
            of responders. "5 of 6 accepted" and "0 of 6" call for very different
            concessions and are otherwise indistinguishable.
        own_return: (N, E) each agent's accumulated episode return so far, scaled.
        own_cleaning: (N, E) accumulated cells cleaned, scaled.
        river_stock: (E,) clear cells in the river, scaled.
        mask: (F,) from `feature_mask`.
    """
    num_envs = last_theta_norm.shape[0]
    rounds_left = jnp.full((num_envs,), (num_rounds - round_idx) / num_rounds,
                           dtype=jnp.float32)
    prop_onehot = jax.nn.one_hot(proposer_idx, num_agents, dtype=jnp.float32)  # (E, N)
    mine = is_proposer_mask(proposer_idx, num_agents).astype(jnp.float32)      # (N, E)
    n_responders = max(num_agents - 1, 1)
    last_votes = jnp.asarray(last_votes, jnp.float32)
    accept_frac = jnp.asarray(last_n_accept, jnp.float32) / n_responders

    def per_agent(i):
        shared = [
            rounds_left,
            mine[i],
            had_offer.astype(jnp.float32),
            last_theta_norm,
            n_reject.astype(jnp.float32) / num_rounds,
            jnp.asarray(offer_live, jnp.float32),
            live_theta_norm,
            accept_frac,
        ]
        cols = shared + [prop_onehot[:, a] for a in range(num_agents)]
        cols += [last_votes[a] for a in range(num_agents)]
        cols += [own_return[i], own_cleaning[i], river_stock,
                 own_cleaning.mean(axis=0)]
        return jnp.stack(cols, axis=-1)                                        # (E, F)

    feats = jnp.stack([per_agent(i) for i in range(num_agents)])               # (N,E,F)
    return feats * mask


# ------------------------------------------------------------ median protocol
#
# Why a protocol with no vote exists at all: under per-segment renegotiation the
# accept/reject game has a degenerate SPE that the counterfactual credit machinery
# learns FAITHFULLY. A responder's vote changes exactly one thing -- whether this
# segment plays at theta or at null -- so the true counterfactual is a per-segment
# individual-rationality test, and in Clean Up the null segment is bad enough that
# every theta in range passes it for everyone. Accept-everything is then correct,
# proposers bid their ideal points unopposed, and the outcome is a random
# dictatorship per segment. No estimator fixes that; it is a property of the game.
#
# The median mechanism swaps IR-gated acceptance for preference aggregation.
# Everyone asks, the median binds. With single-peaked preferences over a scalar
# theta -- which Clean Up's contract space gives every agent: too little buys no
# cleaning, too much funds rent-seeking entry -- the median mechanism is
# STRATEGYPROOF (Moulin 1980's generalised median voter schemes): no agent can move
# the outcome toward its peak by misreporting, so each agent's whole learning
# problem collapses to "find my ideal point", a stationary target that plain
# REINFORCE can hit. No threats, no reservation values, no co-learning
# chicken-and-egg between proposers and voters.

def median_offer(theta_all) -> jnp.ndarray:
    """(E,) the theta that binds: the median of every agent's simultaneous ask.

    With an odd number of agents this is the middle order statistic, so the winning
    ask is always one an agent actually made; with an even number jnp.median
    averages the two middle asks, which is still inside the asked range but belongs
    to nobody -- prefer odd N when it matters. A single extremist cannot drag the
    outcome: the median moves only when the middle of the distribution does, which
    is exactly the property the alternating-offers game lacked (whoever held the
    pen set the number).

    Args:
        theta_all: (N, E) every agent's ask, already unsquashed onto [low, high].
    """
    return jnp.median(jnp.asarray(theta_all, jnp.float32), axis=0)


def median_round_features(round_idx, num_rounds: int, num_agents: int,
                          last_median_norm, had_offer,
                          own_return, own_cleaning, river_stock,
                          mask) -> jnp.ndarray:
    """(N, E, F) decision state for a median round: everyone asks, nobody votes.

    Reuses the alternating-offers layout (`bargaining_features`) rather than
    defining its own, so the network shape, the checkpoint tooling and the feature
    tiers are shared unchanged. The mapping onto that layout:

      * every agent is its own proposer -- the `mine` flag is 1 for all, and the
        proposer one-hot carries the agent's own identity;
      * the standing-offer slot carries LAST round's median, which is the whole
        public history this protocol has;
      * the live-offer slots stay empty (one simultaneous pass -- nothing is ever
        "on the table" when the decision is made), and the vote-history and
        rejection slots stay zero, because votes and rejections do not exist.

    The structurally-zero slots cost nothing: a constant input is folded into the
    first layer's bias. What matters is that a median-trained checkpoint reads a
    feature vector whose live slots mean what they meant in training, which this
    guarantees by construction.

    Args:
        last_median_norm: (E,) last round's median on [-1, 1], 0 in round 0.
        had_offer: (E,) bool, whether any round has resolved yet.
        own_return, own_cleaning: (N, E) per-agent accumulations, pre-scaled.
        river_stock: (E,) pre-scaled.
        mask: (F,) from `feature_mask`.
    """
    num_envs = last_median_norm.shape[0]
    zeros_i = jnp.zeros((num_envs,), jnp.int32)
    zeros_f = jnp.zeros((num_envs,), jnp.float32)
    zeros_votes = jnp.zeros((num_agents, num_envs), jnp.float32)
    rows = []
    for i in range(num_agents):
        feats_i = bargaining_features(
            round_idx, num_rounds, jnp.full((num_envs,), i, jnp.int32), num_agents,
            last_median_norm, had_offer, zeros_i, zeros_f, zeros_f,
            zeros_votes, zeros_f, own_return, own_cleaning, river_stock, mask)
        rows.append(feats_i[i])
    return jnp.stack(rows)


# -------------------------------------------------------------- the vote itself

def floored_vote(pi_vote, eps, key) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sample an accept/reject vote from an eps-floored copy of `pi_vote`.

    Both bargaining runs so far saturated -- one agent that always rejected in the
    first, every agent always accepting in the second -- and BARGAIN_ENT_COEF=0.01
    did not stop it. A saturated vote is self-sealing: once accept probability is
    1.0 the policy never observes the outcome of refusing, so nothing can teach it
    to. The floor keeps both branches sampled at rate `eps` regardless of what the
    logits say, and anneals to 0 so the final policy is the learned one.

    The returned log-prob is under the FLOORED distribution, because that is the
    distribution the action was actually drawn from: PPO's ratio is then a proper
    importance weight of the policy against the behaviour distribution. The loss
    still evaluates the new log-prob under the UNfloored policy, so gradient reaches
    a saturated agent instead of dying in the clip.

    Args:
        pi_vote: distrax.Categorical over [reject, accept].
        eps: scalar floor on both branches; 0 recovers plain sampling.
        key: PRNG key.

    Returns:
        (E,) int32 vote, (E,) float32 log-prob under the floor.
    """
    p = jnp.clip(pi_vote.probs[..., 1], eps, 1.0 - eps)
    vote = (jax.random.uniform(key, p.shape) < p).astype(jnp.int32)
    log_p = jnp.where(vote == 1, jnp.log(p), jnp.log1p(-p))
    return vote, log_p


def vote_eps_at(base: float, update_step, num_updates: int, end: float = 0.0):
    """The vote floor at `update_step`, annealed linearly from `base` to `end`.

    Exploration early, when saturation would be permanent. `end` is the floor that
    REMAINS at the end of training, and it should usually not be 0: the fixesV1 run
    annealed to 0 and showed why. Rejection is a policing strategy that pays only
    when someone lowballs, so it is only ever maintained by being occasionally
    sampled -- and the run's proposers began walking theta back down at almost
    exactly the point the anneal extinguished the last rejections that would have
    punished them. A persistent `end` keeps the threat alive for as long as
    proposers are still learning. The floor only shapes the training rollouts; the
    checkpointed weights are the un-floored policy either way.
    """
    frac = 1.0 - jnp.asarray(update_step, jnp.float32) / max(int(num_updates), 1)
    return jnp.float32(end) + (jnp.float32(base) - jnp.float32(end)) * jnp.clip(
        frac, 0.0, 1.0)


# ------------------------------------------- counterfactual credit for the vote
#
# The vote head was trained on the shared round-GAE advantage, and that advantage
# answers the wrong question. It is dominated by the lump-sum agreement reward --
# the round that locks books every remaining segment at once -- so it says "this
# round went well", not "MY vote made it go well". A responder whose vote changed
# nothing collects the same credit as the one who was actually pivotal.
#
# It is worse than merely noisy, because the noise has a sign. The eps floor
# samples rejections uniformly, but the offers on the table are not uniform: most
# rounds carry an offer good enough to be accepted by everyone else, so a floored
# rejection usually lands on a GOOD offer, delays it by a segment, and is punished.
# Correlational credit therefore teaches "rejecting is bad" from the exploration
# that was supposed to teach when rejecting is good -- which is what the theta
# sweep saw: every slope the right sign, all of them ~5-10x too flat to discipline
# a proposer.
#
# The counterfactual is exact and cheap here, because the vote is binary and the
# quorum is a counting rule: agent i changes the outcome iff exactly quorum-1 of
# the OTHERS accepted, and what it changes is the whole branch -- lock this offer
# now, or play a null segment and reopen. Two learned branch values give the
# difference directly. This is COMA's counterfactual baseline, specialised: with a
# binary action and a known pivot rule the marginalisation is a single comparison
# rather than a sum over the action space.

def pivotal_mask(n_accept, vote, is_proposer, quorum: int) -> jnp.ndarray:
    """(N, E) bool: whose vote actually decided the round.

    Agent i is pivotal iff the other counted votes sit exactly one short of the
    quorum -- then i accepting carries the offer and i rejecting kills it. Under
    `BARGAIN_QUORUM=all` that means every other responder accepted, which is the
    common case precisely because unanimity makes everyone a veto.

    The proposer is never pivotal: its vote is not counted (see `accepted`).

    Args:
        n_accept: (..., E) counted accepts in the round, from `accepted`.
        vote: (..., N, E) each agent's vote, 0/1.
        is_proposer: (..., N, E) bool, from `is_proposer_mask`.
        quorum: how many non-proposers must accept.

    Returns:
        (..., N, E) bool. Leading axes broadcast, so this takes one round or a
        whole (K, N, E) block of them.
    """
    counted = jnp.asarray(vote, jnp.int32) * (~is_proposer).astype(jnp.int32)
    others = jnp.expand_dims(jnp.asarray(n_accept, jnp.int32), -2) - counted
    return (others == quorum - 1) & ~is_proposer


def vote_credit_mask(active, is_proposer, offer_null, pivotal=None):
    """Weight on each agent's vote in the policy gradient. Zero means EXCLUDED.

    Every exclusion here is a case where the vote had no consequence, so training on
    it would fit noise:

      inactive round   the contract was already agreed; nobody decided anything.
      proposer         its vote is not counted (see `accepted`).
      null offer       accepting and rejecting lead to the same place -- one
                       uncontracted segment, negotiation reopens -- so the vote is
                       outcome-free by construction.
      not pivotal      (counterfactual credit only) the outcome is identical either
                       way, so the true counterfactual is exactly zero.

    Everything is elementwise and broadcasts, so this serves both the per-agent
    (K, E) slice the loss works in and a whole (K, N, E) block for diagnostics.
    """
    w = (jnp.asarray(active, jnp.float32)
         * (1.0 - jnp.asarray(is_proposer, jnp.float32))
         * (1.0 - jnp.asarray(offer_null, jnp.float32)))
    if pivotal is not None:
        w = w * jnp.asarray(pivotal, jnp.float32)
    return w


def counterfactual_vote_advantage(vote, lock_value, cont_value,
                                  pivotal) -> jnp.ndarray:
    """What each agent's vote was worth to it, in its own returns.

    A_i = 1{pivotal} * direction * (lock - cont), where direction is +1 if i voted
    accept (its action selected the lock branch) and -1 if it rejected (it selected
    the continue branch). Reinforcing an action by its own branch difference is the
    whole point: accepting a lowball where lock < cont now yields a NEGATIVE
    advantage, so the accept probability falls -- the gradient the correlational
    advantage never supplied.

    Non-pivotal votes are zeroed rather than merely down-weighted. Their true
    counterfactual really is zero (the outcome is identical either way), so any
    non-zero credit they carry is pure noise fitted to other agents' choices.

    Both branch values are stop-gradiented: this is a policy gradient for the vote,
    and the heads are fitted separately by their own regression targets. Letting it
    flow back would let the policy lower its loss by moving its beliefs.
    """
    gap = jax.lax.stop_gradient(lock_value - cont_value)
    direction = 2.0 * jnp.asarray(vote, jnp.float32) - 1.0
    return direction * gap * jnp.asarray(pivotal, jnp.float32)


def return_to_go(rewards, active):
    """(K, ...) realised remaining return from each round on, gamma = 1.

    The regression target for both branch heads. The semi-MDP structure makes this
    exact rather than an approximation at the only rounds that matter: at a round
    that LOCKS, `rewards` already carries every remaining segment and every later
    round is inactive, so the sum from there is that round's reward alone -- a
    realised sample of the lock branch. At a round that does not lock, the sum is
    the realised continuation.

    Args:
        rewards: (K, ...) per-round reward, from the semi-MDP construction.
        active: (K, ...) broadcastable mask of rounds where a decision was made.
    """
    masked = rewards * jnp.asarray(active, rewards.dtype)
    return jnp.flip(jnp.cumsum(jnp.flip(masked, axis=0), axis=0), axis=0)


# ------------------------------------------------------- checkpoint compatibility

AUX_HEAD_KEY = "lock_out"     # the named Dense that only exists with aux heads


def params_have_aux_heads(params) -> bool:
    """Does this checkpoint carry the counterfactual branch value heads?

    Read from the parameter tree rather than from a config flag or the sidecar, so
    a checkpoint replays correctly with no user input either way -- flax needs the
    module structure to match the params it is given, and getting that from the
    params themselves is the only way that cannot be told a lie.
    """
    try:
        return AUX_HEAD_KEY in params["params"]
    except (KeyError, TypeError):
        return False


def params_shape(params) -> Tuple[Optional[int], Optional[int]]:
    """(input width, hidden width) of a saved bargaining policy's first layer."""
    try:
        kernel = params["params"]["Dense_0"]["kernel"]
        return int(kernel.shape[0]), int(kernel.shape[1])
    except (KeyError, IndexError, TypeError):
        return None, None


def check_params_compatible(params: Any, num_agents: int, recorded_version=None,
                            hidden: Optional[int] = None, label: str = "") -> None:
    """Refuse to replay a bargaining checkpoint this code cannot read.

    The failure being prevented is not a crash -- it is the version of this that
    does not crash. A stale checkpoint whose feature vector happens to line up gets
    replayed as a mechanism that was never trained, and every number comes out
    looking ordinary. That has already invalidated one full comparison here (the
    contract-range episode in findings.md), so this is deliberately fatal.
    """
    where = f"{label}: " if label else ""
    if recorded_version is not None and int(recorded_version) != FEATURE_VERSION:
        raise ValueError(
            f"{where}bargaining checkpoint was trained at feature/protocol version "
            f"{int(recorded_version)}, this code is version {FEATURE_VERSION}. The "
            f"round structure and feature layout both changed, so the weights cannot "
            f"be read. Replay it by checking out the commit it was trained at (the "
            f"run's .run.yaml sidecar records the commit), or retrain."
        )
    got, got_hidden = params_shape(params)
    want = feature_dim(num_agents)
    if got is not None and got != want:
        raise ValueError(
            f"{where}bargaining policy expects a {got}-dim feature vector, this code "
            f"builds {want} for {num_agents} agents (feature version "
            f"{FEATURE_VERSION}). The checkpoint predates the current layout -- "
            f"check out the commit in its .run.yaml sidecar to replay it, or retrain."
        )
    if hidden is not None and got_hidden is not None and got_hidden != hidden:
        raise ValueError(
            f"{where}bargaining policy has a width-{got_hidden} hidden layer, but "
            f"the network here was built with BARGAIN_HIDDEN={hidden}. Pass the "
            f"width the run was trained at."
        )
    if recorded_version is None and got is not None:
        print(f"  [warning] {where}no feature version in the run's .run.yaml sidecar "
              f"(the run predates it). The parameter shapes match version "
              f"{FEATURE_VERSION}, which is the strongest check available here.")


# ------------------------------------------------- credit over the round MDP

def masked_standardise(x, weights):
    """Zero-mean, unit-variance over the entries `weights` selects.

    The K x E advantage block is mostly STRUCTURAL zeros: `round_gae` zeroes every
    round after agreement, and when agreement lands in round 0 -- which is the SPE,
    and what both runs so far did -- that is nearly the whole block. Standardising
    over all of it drags the mean toward 0 and inflates the scale by the fraction of
    rounds that happened to be live, so the size of the update would track how
    quickly the agents agreed rather than how good the decision was.

    Entries outside the mask come out meaningless and must stay masked downstream.
    """
    total = weights.sum() + 1e-8
    mean = (x * weights).sum() / total
    var = (jnp.square(x - mean) * weights).sum() / total
    return (x - mean) / (jnp.sqrt(var) + 1e-8)


def masked_scale(x, weights):
    """Unit RMS over the entries `weights` selects -- scaling WITHOUT centering.

    For the counterfactual vote advantage, and deliberately not for anything else.
    A GAE advantage needs centering because its baseline is an estimate; the
    counterfactual advantage lock - continue is already measured AGAINST its
    baseline -- the other branch -- so its mean over a batch is not an artefact to
    remove, it is the signal. An agent whose policy is systematically wrong has a
    one-signed advantage; subtracting the batch mean strips exactly that level and
    leaves only the slope. The cf-vote run showed the resulting failure precisely:
    harvesters holding 'locking is worse than bargaining on' beliefs at every
    theta (gap < 0 across the grid) while their acceptance LEVEL sat untrained at
    ~0.9, with only a slope forming. Scale to unit RMS so the PPO clip sees a
    well-sized update, and leave the sign structure alone.
    """
    w = weights.astype(jnp.float32)
    total = w.sum() + 1e-8
    mean_sq = (jnp.square(x) * w).sum() / total
    return x / (jnp.sqrt(mean_sq) + 1e-8)


def round_gae(rewards, values, active, terminal, gamma: float, gae_lambda: float):
    """GAE over the K-round bargaining MDP, which is a SEMI-Markov process.

    Rounds are decision points, not fixed-length steps. A rejected round pays only
    its own (null-contract) segment; the round where the offer is accepted pays
    every remaining segment at once and ENDS the bargaining episode. Rounds after
    agreement contain no decision at all and must not enter the loss -- training on
    them would credit actions that could not have mattered.

    `gamma` should be 1.0. Rubinstein's discount factor models impatience, but here
    impatience is physically realised: disagreement burns real reward in the
    environment. Discounting on top would count the same delay cost twice and
    inflate the proposer's advantage.

    Args:
        rewards, values, active, terminal: (K, ...) with matching trailing shape.
            `active` marks rounds where a decision was actually made; `terminal`
            marks rounds after which nothing more is decided (agreement, or the last
            round of the episode).

    Returns:
        advantages, targets, each (K, ...).
    """
    def step(carry, x):
        gae, next_value = carry
        reward, value, act, term = x
        not_done = 1.0 - term.astype(jnp.float32)
        delta = reward + gamma * next_value * not_done - value
        gae = delta + gamma * gae_lambda * not_done * gae
        act_f = act.astype(jnp.float32)
        # An inactive round neither accumulates nor propagates: it is not part of
        # the decision sequence. It also must not overwrite the bootstrap value,
        # since the terminal round it follows ignores it anyway.
        gae = gae * act_f
        return (gae, value), gae

    zeros = jnp.zeros_like(values[0])
    _, advantages = jax.lax.scan(
        step, (zeros, zeros), (rewards, values, active, terminal), reverse=True
    )
    return advantages, advantages + values
