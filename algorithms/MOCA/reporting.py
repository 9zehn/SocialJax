"""Claims and audits: contract payments on REPORTED cleaning, enforced by audit.

Everywhere else in this codebase, transfers are perfectly enforced: the contract
pays theta per cell actually cleaned, read off the environment's ground truth
(contracts.CleanupContract.compute_transfer). That assumes away the enforcement
problem that makes real contracts hard -- work is observed by the worker, not the
counterparty. This module relaxes it, in the smallest strategic step that still
contains the whole question (Townsend's costly state verification, Becker's
crime-and-punishment calculus):

    Each window (one BARGAIN_SEGMENT), each agent files a CLAIM for cleaning done
    that window. The claim is parameterised as truth plus a chosen OVERCLAIM
    o_i in [0, REPORT_MAX_OVERCLAIM] -- underclaiming is strictly dominated (true
    claims are costless and verifiable by the claimant), so letting agents report
    below truth would add a learning burden with zero strategic content. The TRUE
    portion keeps flowing per step exactly as before; only the overclaim is at
    stake. With probability REPORT_AUDIT_P the claim is audited against ground
    truth: the overclaim payment is voided and a fine of REPORT_FINE_MULT x the
    overclaimed amount is levied. The true portion is NOT forfeited -- the fine is
    proportional to the crime, not to how much honest work the liar happened to
    do, which keeps the cleaner/harvester comparison uncontaminated and the
    analytics exact.

The expected value of one unit of overclaim at contract theta is

    theta * (1 - p) - p * lam * theta  =  theta * (1 - p * (1 + lam))

so honesty is optimal exactly when lam > (1 - p) / p (`honesty_threshold`). The
experiment is whether learning finds that boundary -- the Becker curve, learned.

Audits here are EXOGENOUS (rung 1): a coin flip, not a decision. That makes the
claim a CONTEXTUAL BANDIT -- the settlement lands immediately, nothing carries
over, no reputation -- so it trains with plain REINFORCE against a batch-mean
baseline and needs none of the semi-MDP machinery the bargaining stage does.
Rung 2 (harvesters choosing to audit at a cost, fines flowing to the auditor --
the inspection game) drops into this skeleton as one more Bernoulli head and a
routing change in `settle_claims`.
"""
import distrax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from algorithms.MOCA.bargain import normalise_theta

# [own cleaning this window (scaled), theta in force (normalised), in-force flag]
FEATURE_DIM = 3


class ClaimPolicy(nn.Module):
    """One agent's overclaim distribution, a Gaussian over normalised [-1, 1].

    Unsquashed onto [0, REPORT_MAX_OVERCLAIM] by negotiate.unsquash exactly as the
    bargaining proposal is. The mean's bias is initialised at -1: the clip at the
    lower bound maps that to an overclaim of exactly 0, so agents START HONEST and
    must discover lying through the Gaussian's exploration -- the interesting
    direction. Initialising at the range midpoint would start every agent lying
    at half the cap and make "lying emerged" unreadable.
    """
    hidden: int = 32

    @nn.compact
    def __call__(self, x):
        h = nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(x)
        h = nn.relu(h)
        mean = nn.Dense(1, kernel_init=orthogonal(0.01),
                        bias_init=constant(-1.0))(h)
        log_std = self.param("log_std", nn.initializers.zeros, (1,))
        return distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))


def claim_features(window_cleaning, theta_eff, segment: int,
                   low: float, high: float) -> jnp.ndarray:
    """(N, E, FEATURE_DIM) claim state, one row per agent.

    Args:
        window_cleaning: (N, E) cells each agent cleaned THIS window.
        theta_eff: (E,) the contract in force this window (0 if none).
        segment: window length in steps, the scale for cleaning.
        low, high: the contract range, for normalising theta.
    """
    num_agents = window_cleaning.shape[0]
    tn = normalise_theta(theta_eff, low, high)                       # (E,)
    in_force = (theta_eff > 1e-6).astype(jnp.float32)
    cols = [
        jnp.stack([window_cleaning[i] / float(segment), tn, in_force], axis=-1)
        for i in range(num_agents)
    ]
    return jnp.stack(cols)                                           # (N, E, 3)


def settle_claims(theta_eff, overclaim, audited, fine_mult: float,
                  num_agents: int):
    """Window settlement of the overclaims: who nets what, zero-sum overall.

    Agent i's own position is

        own_i = theta * o_i * (1 - a_i * (1 + lam))

    i.e. the overclaim is paid in full when unaudited, and when audited the
    payment is voided AND a fine of lam * theta * o_i is levied. Funding and
    distribution follow compute_transfer's convention: each agent's net position
    is financed evenly by the other N-1, which makes fines flow to the group the
    same way payments are drawn from it, and the whole settlement exactly
    zero-sum (asserted in tests, not just claimed here).

    Args:
        theta_eff: (E,) contract in force; 0 makes the settlement identically 0.
        overclaim: (N, E) chosen overclaims, already in cell units.
        audited: (N, E) bool audit draws.
        fine_mult: lam, the fine per unit of overclaim caught.
        num_agents: N.

    Returns:
        (transfer, own): both (N, E). `transfer` is the zero-sum reward vector to
        apply; `own` is each agent's own-caused position -- the RIGHT reward for
        the claim bandit, since the funding share of OTHERS' claims does not
        depend on the agent's own action and would only add variance.
    """
    a = audited.astype(jnp.float32)
    own = theta_eff[None, :] * overclaim * (1.0 - a * (1.0 + fine_mult))
    total = own.sum(axis=0, keepdims=True)
    transfer = own - (total - own) / (num_agents - 1)
    return transfer, own


def honesty_threshold(audit_p: float) -> float:
    """The fine multiple above which overclaiming has negative expected value."""
    return (1.0 - audit_p) / max(audit_p, 1e-9)
