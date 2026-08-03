"""Contract spaces for MOCA (Mutually Optimal Contract Algorithm).

Implements the formal-contracting framework of Christoffersen et al., "Formal
Contracts Mitigate Social Dilemmas in Multi-Agent RL" (arXiv:2208.10469 / AAMAS
2023), following the reference implementation at
github.com/Algorithmic-Alignment-Lab/contracts.

A *contract* is a function theta: Omega -> R^N mapping observable outcomes to a
ZERO-SUM vector of reward transfers (sum_i theta_i = 0). Agents play the base
game with rewards R'_i = R_i + theta_i, so a contract cannot create or destroy
welfare -- it only redistributes it. That zero-sum property is what makes the
mechanism a pure redistribution channel rather than an exogenous subsidy, and it
is asserted in the tests.

CleanupContract mirrors the paper's Cleanup contract space: a single scalar
theta = "payment per waste cell cleaned, paid for evenly by the other agents".
"""
from typing import Tuple

import jax
import jax.numpy as jnp

# Contract-state indicators, matching the reference implementation's
# `self.contract_state` values (two_stage_train.py). They ride along in the
# contract observation so a single policy can distinguish the negotiation stages
# from ordinary play.
SUBGAME = 0.0    # playing the game with a contract in force
PROPOSE = 2.0    # agent 0 is choosing a contract to offer
AGREE = 3.0      # the offered contract awaits accept/reject


class CleanupContract:
    """Scalar contract space for Clean Up: pay `theta` per waste cell cleaned.

    Reference implementation (contract/contract_list.py::CleanupContract) defines
    this as "theta in [0, 0.2], which correspond to a payment per waste cell
    cleaned, paid for evenly by the other agents", conditioned on the per-agent
    `cleaned_squares` info field. Here the corresponding env signal is
    info["cleaned_by_agent"] from clean_up.py.

    The transfer for agent i on a step where agents cleaned c = (c_1..c_N) is

        receive_i = theta * c_i                       # paid for what you cleaned
        pay_i     = theta * (sum_j c_j - c_i)/(N-1)   # you fund everyone else's
        transfer_i = receive_i - pay_i

    which is exactly zero-sum:
        sum_i transfer_i = theta*C - theta*(N-1)*C/(N-1) = 0,  C = sum_j c_j.

    Note the sign convention: a *cleaner* is a net receiver and a pure harvester
    is a net payer, so the contract subsidises exactly the under-provided public
    good (river cleaning) at the expense of those free-riding on it.

    Args:
        num_agents: N, the number of agents (>= 2; a transfer needs a counterparty).
        low: minimum contract value (the null contract theta=0 must be included).
        high: maximum contract value.
    """

    def __init__(self, num_agents: int, low: float = 0.0, high: float = 0.2):
        if num_agents < 2:
            raise ValueError(f"contracts need >= 2 agents, got {num_agents}")
        if not high > low:
            raise ValueError(f"need high > low, got low={low}, high={high}")
        self.num_agents = int(num_agents)
        self.low = float(low)
        self.high = float(high)
        # Contract feature vector handed to the policy: [theta_normalised, stage].
        # The reference implementation builds this as
        # np.concatenate((self.params[key], np.array([0]))) -- the contract
        # parameters followed by a stage indicator -- so the shape/layout is kept
        # identical here. `stage` is always 0 (subgame) for us because the
        # contracting stage is played by dedicated proposal/voting networks rather
        # than by the gameplay policy acting in augmented states; it is retained so
        # the interface matches theirs and stays extensible.
        self.obs_dim = 2

    # ---------------------------------------------------------------- sampling

    def sample(self, key: jnp.ndarray, shape: Tuple[int, ...] = (), null_prob: float = 0.0):
        """Sample contracts uniformly from [low, high], with a null-contract mass.

        MOCA Phase 1 trains the gameplay policy on contracts drawn from a fixed
        distribution P(Theta) that does NOT depend on the (still-learning) proposal
        policy -- that independence is what keeps the learned value estimates
        V_i(s_0, theta) unbiased across the contract space, and hence what makes the
        Phase-2 contract choice subgame-perfect.

        `null_prob` mirrors the reference implementation's `null_prob`: with that
        probability the null contract (theta=low, i.e. no transfers) is drawn
        instead, so the policy also keeps experience of playing uncontracted.
        """
        key_u, key_null = jax.random.split(key)
        theta = jax.random.uniform(key_u, shape=shape, minval=self.low, maxval=self.high)
        if null_prob > 0.0:
            is_null = jax.random.uniform(key_null, shape=shape) < null_prob
            theta = jnp.where(is_null, jnp.float32(self.low), theta)
        return theta.astype(jnp.float32)

    def grid(self, num_points: int) -> jnp.ndarray:
        """Evenly spaced contract values spanning [low, high].

        Phase 2's proposal policy is a categorical distribution over this grid.
        Discretising keeps the proposal a plain Categorical (stable under the
        single-decision-per-episode REINFORCE signal of Phase 2) and makes the
        learned contract distribution directly plottable; the gameplay policy is
        still trained on *continuous* theta in Phase 1, as in the reference
        implementation, so it interpolates across the space.
        """
        return jnp.linspace(self.low, self.high, num_points, dtype=jnp.float32)

    # ------------------------------------------------------------- observation

    def to_obs(self, theta: jnp.ndarray, stage: float = SUBGAME) -> jnp.ndarray:
        """Contract feature vector [theta_normalised, stage] for the policy.

        theta is min-max normalised onto [0, 1] so the input is well scaled for the
        network regardless of the configured contract range (raw theta can be as
        small as 0.2, which would be a negligible input next to CNN activations).
        This is a numerical change only -- the ordering and semantics of the
        contract space are untouched.

        `stage` is the reference implementation's contract-state indicator, carried
        raw (0/2/3) as it is there -- its observation Box runs to high=3.0. It is
        what lets one policy tell "propose a contract" from "accept or reject this
        contract" from "play the game under this contract". Defaults to SUBGAME, so
        every existing caller keeps the behaviour it had.
        """
        theta = jnp.asarray(theta, dtype=jnp.float32)
        theta_norm = (theta - self.low) / (self.high - self.low)
        stage_arr = jnp.full_like(theta_norm, jnp.float32(stage))
        return jnp.stack([theta_norm, stage_arr], axis=-1)

    # --------------------------------------------------------------- transfers

    def compute_transfer(self, theta: jnp.ndarray, cleaned: jnp.ndarray) -> jnp.ndarray:
        """Zero-sum per-agent reward transfers for one step.

        Args:
            theta: scalar contract value, broadcastable against `cleaned`'s leading
                dims (e.g. shape (num_envs,) against cleaned (num_envs, N)).
            cleaned: (..., N) per-agent cleaning credit this step
                (info["cleaned_by_agent"]).

        Returns:
            (..., N) float32 transfers summing to zero along the agent axis.
        """
        cleaned = jnp.asarray(cleaned, dtype=jnp.float32)
        theta = jnp.asarray(theta, dtype=jnp.float32)[..., None]  # broadcast over agents
        total = jnp.sum(cleaned, axis=-1, keepdims=True)
        receive = theta * cleaned
        pay = theta * (total - cleaned) / (self.num_agents - 1)
        return receive - pay


def make_contract(name: str, num_agents: int, low: float, high: float):
    """Contract-space factory, so the space is selectable from config."""
    if name != "cleanup":
        raise ValueError(f"unknown contract space {name!r} (available: 'cleanup')")
    return CleanupContract(num_agents, low=low, high=high)
