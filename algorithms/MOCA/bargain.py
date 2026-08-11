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
        proposer p(r) offers theta_r
        every other agent votes accept/reject
        if #accept >= quorum:  theta_r binds for ALL REMAINING segments, done
        else:                  theta = 0 for segment r, continue to round r+1
    never agreed -> the null contract for the whole episode

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
from typing import Tuple

import jax
import jax.numpy as jnp

PROPOSER_MODES = ("rotate", "random", "contribution")
# Which slice of the bargaining state the policy may see. Tiered so the headline
# result can be shown not to depend on handing agents the inequality signal --
# see `feature_mask`.
FEATURE_LEVELS = ("protocol", "private", "public")


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
                       key=None, contributions=None) -> jnp.ndarray:
    """(num_envs,) index of the proposing agent, per env.

    rotate       round r is agent r mod N. Deterministic and identical across envs,
                 so every agent gets the same number of turns -- the symmetric
                 Rubinstein protocol, and the arm to read first.
    random       uniform each round (Baron-Ferejohn's random recognition). Included
                 because it is the standard multilateral baseline, and because with
                 majority quorum it should reproduce minority exclusion.
    contribution sampled proportional to cumulative cleaning, so proposal power
                 accrues to whoever provisions the public good (Ostrom's
                 proportional equivalence). The only way to game it is to clean more.
    """
    if mode == "rotate":
        return jnp.full((num_envs,), round_idx % num_agents, dtype=jnp.int32)
    if mode == "random":
        return jax.random.randint(key, (num_envs,), 0, num_agents).astype(jnp.int32)
    if mode == "contribution":
        # (N, E) -> (E, N) logits; +eps so an all-zero column stays uniform rather
        # than NaN, which is the state at the very start of every episode.
        w = jnp.transpose(contributions) + 1e-6
        return jax.random.categorical(key, jnp.log(w), axis=-1).astype(jnp.int32)
    raise ValueError(f"unknown BARGAIN_PROPOSER {mode!r} "
                     f"(available: {', '.join(PROPOSER_MODES)})")


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

def feature_dim(num_agents: int) -> int:
    return 9 + num_agents


def feature_mask(level: str, num_agents: int) -> jnp.ndarray:
    """(F,) 0/1 mask selecting which tier of the bargaining state is visible.

    The tiers exist because "handcrafted features" is a fair criticism to level at a
    learned bargainer, and the tiers separate the fair part from the contentious one.

    protocol  The extensive-form game itself: round, whose turn, the standing offer,
              how many rounds have failed. Every bargaining model in the literature
              assumes players know these, and the SPE of a finite alternating-offers
              game is Markov in (round, proposer) -- so this tier alone is in
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
    n_protocol = 5 + num_agents
    mask = jnp.zeros((feature_dim(num_agents),), dtype=jnp.float32)
    mask = mask.at[:n_protocol].set(1.0)
    if level in ("private", "public"):
        mask = mask.at[n_protocol:n_protocol + 2].set(1.0)
    if level == "public":
        mask = mask.at[n_protocol + 2:].set(1.0)
    return mask


def bargaining_features(round_idx, num_rounds: int, proposer_idx, num_agents: int,
                        last_theta_norm, had_offer, n_reject,
                        own_return, own_cleaning, river_stock, mask) -> jnp.ndarray:
    """(N, E, F) bargaining state, one row per agent.

    Everything is pre-scaled to roughly unit range; a raw episode return of ~500
    alongside a 0/1 turn flag would make the first layer's job needlessly hard.

    Args:
        last_theta_norm: (E,) the standing (rejected) offer on [-1, 1], 0 if none.
        had_offer: (E,) bool, whether any offer has been made yet.
        n_reject: (E,) how many rounds have failed so far.
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

    def per_agent(i):
        shared = [
            rounds_left,
            mine[i],
            had_offer.astype(jnp.float32),
            last_theta_norm,
            n_reject.astype(jnp.float32) / num_rounds,
        ]
        cols = shared + [prop_onehot[:, a] for a in range(num_agents)]
        cols += [own_return[i], own_cleaning[i], river_stock,
                 own_cleaning.mean(axis=0)]
        return jnp.stack(cols, axis=-1)                                        # (E, F)

    feats = jnp.stack([per_agent(i) for i in range(num_agents)])               # (N,E,F)
    return feats * mask


# ------------------------------------------------- credit over the round MDP

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
