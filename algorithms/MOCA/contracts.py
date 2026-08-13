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

# The null contract: zero transfers by construction, since compute_transfer is
# linear in theta. It is the disagreement point every acceptance rule measures
# against (solver.select_contract, protocols.accepts), so it is a property of the
# mechanism rather than of the configured range -- see CleanupContract on why the
# two are now separate.
NULL_THETA = 0.0
_NULL_TOL = 1e-6


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

    The contract space is {NULL_THETA} u [low, high]. `low` used to double as the
    null contract, which is only correct when low == 0: with low > 0 the "no
    contract" fallback would itself move reward, silently corrupting every
    disagreement value V_i(s, 0) the acceptance rules compare against. They are now
    separate, so the range can exclude weak contracts -- theta below ~0.2 is too
    small to change behaviour against a unit apple, and those samples only blur the
    boundary between contracted and uncontracted play -- while the null contract
    stays available and genuinely null.

    Args:
        num_agents: N, the number of agents (>= 2; a transfer needs a counterparty).
        low: minimum NON-NULL contract value (>= 0).
        high: maximum contract value.
    """

    def __init__(self, num_agents: int, low: float = 0.0, high: float = 0.2):
        if num_agents < 2:
            raise ValueError(f"contracts need >= 2 agents, got {num_agents}")
        if not high > low:
            raise ValueError(f"need high > low, got low={low}, high={high}")
        if low < NULL_THETA:
            raise ValueError(
                f"low must be >= the null contract {NULL_THETA}, got low={low}"
            )
        self.num_agents = int(num_agents)
        self.low = float(low)
        self.high = float(high)
        self.null = float(NULL_THETA)
        # Contract feature vector handed to the policy:
        # [theta_normalised, is_null, stage]. The reference builds the first and
        # last of these as np.concatenate((self.params[key], np.array([0]))) -- the
        # contract parameters followed by a stage indicator. `stage` is always 0
        # (subgame) for us because the contracting stage is played by dedicated
        # proposal/voting networks rather than by the gameplay policy acting in
        # augmented states; it is retained so the interface matches theirs.
        #
        # `is_null` is an addition; see to_obs for why it is load-bearing.
        self.obs_dim = 3

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
            theta = jnp.where(is_null, jnp.float32(self.null), theta)
        return theta.astype(jnp.float32)

    def sample_batch(self, key: jnp.ndarray, num_envs: int, null_frac: float = 0.0):
        """One phase-1 update's contracts: an exact null block plus a stratified rest.

        Preferred over `sample` for training. Two differences, both mattering more
        as null mass is raised:

        * The null count is EXACT (round(null_frac * num_envs)) rather than
          Binomial(num_envs, null_prob). At 128 envs and p=0.1 the i.i.d. draw
          varies by +-3.4 envs per update, which is noise on the one quantity every
          downstream acceptance rule is measured against.
        * The non-null draws are STRATIFIED -- one jittered sample per equal-width
          bin -- so every update covers the whole range. That matters because
          raising null mass shrinks the budget left for theta > 0, and phase 2 takes
          an argmax over theta, so gaps in coverage become contract-selection error.

        Reweighting P(Theta) toward the null is legitimate under MOCA: the
        unbiasedness of V_i(s_0, theta) needs P(Theta) fixed, independent of the
        proposal policy, and full-support -- NOT uniform. It reallocates estimation
        accuracy toward the disagreement point.

        The result is permuted so no env slot is systematically the null one.
        """
        if not 0.0 <= null_frac <= 1.0:
            raise ValueError(f"null_frac must be in [0, 1], got {null_frac}")
        # Static under jit: null_frac is a config value, not a traced array.
        n_null = int(round(null_frac * num_envs))
        n_draw = num_envs - n_null
        key_j, key_perm = jax.random.split(key)

        nulls = jnp.full((n_null,), self.null, dtype=jnp.float32)
        if n_draw == 0:
            return nulls
        edges = jnp.linspace(self.low, self.high, n_draw + 1, dtype=jnp.float32)
        u = jax.random.uniform(key_j, (n_draw,))
        drawn = edges[:-1] + u * (edges[1:] - edges[:-1])
        return jax.random.permutation(
            key_perm, jnp.concatenate([nulls, drawn]).astype(jnp.float32)
        )

    def is_null(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Boolean mask of which contracts are the null contract."""
        return jnp.asarray(theta, dtype=jnp.float32) <= self.null + _NULL_TOL

    def grid(self, num_points: int) -> jnp.ndarray:
        """Contract values for phase 2, with the NULL CONTRACT ALWAYS AT INDEX 0.

        Phase 2's proposal policy is a categorical distribution over this grid.
        Discretising keeps the proposal a plain Categorical (stable under the
        single-decision-per-episode REINFORCE signal of Phase 2) and makes the
        learned contract distribution directly plottable; the gameplay policy is
        still trained on *continuous* theta in Phase 1, as in the reference
        implementation, so it interpolates across the space.

        When low > null the grid is the null contract followed by num_points - 1
        evenly spaced values over [low, high], so the mechanism can still decline.
        protocols.py and solver.py both index the null as row 0.
        """
        if self.low <= self.null + _NULL_TOL:
            return jnp.linspace(self.low, self.high, num_points, dtype=jnp.float32)
        return jnp.concatenate([
            jnp.array([self.null], dtype=jnp.float32),
            jnp.linspace(self.low, self.high, num_points - 1, dtype=jnp.float32),
        ])

    # ------------------------------------------------------------- observation

    def to_obs(self, theta: jnp.ndarray, stage: float = SUBGAME) -> jnp.ndarray:
        """Contract feature vector [theta_normalised, is_null, stage] for the policy.

        theta is normalised onto [-1, 1] over [low, high] so the input is well scaled
        for the network regardless of the configured range (raw theta can be as small
        as 0.2, a negligible input next to CNN activations) AND so that no contract in
        force encodes as the zero vector -- see below.

        `is_null` is +1 for the null contract and -1 otherwise. It is not cosmetic. The
        contract vector is concatenated onto the CNN embedding and fed to a Dense
        layer (networks.py), whose weight gradient is (upstream grad) (x) (input). The
        previous encoding put the null contract at exactly [0, 0]:

          * forward, the contract weights contributed nothing, and
          * backward, they received exactly zero gradient,

        so the contract pathway was functionally absent at theta=0 in both
        directions. Null episodes could then only move the CNN trunk and biases,
        which are SHARED with every other theta, and the whole theta-response
        collapsed to a rank-1 additive shift that is continuous in theta -- making
        behaviour at theta=0 necessarily the limit of behaviour at theta=eps. That is
        the wrong inductive bias here: "no contract" and "contract in force" are
        qualitatively different regimes (clean vs. free-ride), not two points on a
        smooth ramp, and V_i(s, 0) is the disagreement point every acceptance rule
        compares against, so contamination of it by nearby contracts is fatal to the
        mechanism rather than merely inaccurate.

        A dedicated indicator gives null episodes their own weights to push and lets
        the network place a genuine discontinuity at theta=0. The reference
        implementation has the same degeneracy (its contract_obs is
        np.concatenate((params, [0])) over a raw [0, 0.2] Box, so the null contract
        is literally the zero vector); this deviates from it deliberately.

        `stage` is the reference's contract-state indicator, carried raw (0/2/3) as
        it is there -- its observation Box runs to high=3.0. It is what lets one
        policy tell "propose a contract" from "accept or reject this contract" from
        "play the game under this contract". Defaults to SUBGAME, so every existing
        caller keeps the behaviour it had.
        """
        theta = jnp.asarray(theta, dtype=jnp.float32)
        null = self.is_null(theta)
        # +/-1 rather than 1/0 so the feature vector is never all-zero for ANY input.
        # With a 0/1 flag the midpoint of the contracted range still encodes as
        # [0, 0, 0] and reintroduces exactly the degeneracy this exists to remove --
        # only at an interior theta instead of at the null contract.
        is_null = jnp.where(null, jnp.float32(1.0), jnp.float32(-1.0))
        theta_norm = 2.0 * (theta - self.low) / (self.high - self.low) - 1.0
        # The null contract may sit below `low`, and its position on the contracted
        # scale is meaningless anyway -- `is_null` carries it. Pinning it to 0 keeps
        # the feature inside [-1, 1] for every input.
        theta_norm = jnp.where(null, jnp.float32(0.0),
                               jnp.clip(theta_norm, -1.0, 1.0))
        stage_arr = jnp.full_like(theta_norm, jnp.float32(stage))
        return jnp.stack([theta_norm, is_null, stage_arr], axis=-1)

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


class HarvestTaxContract(CleanupContract):
    """The bargained scalar is a TAX RATE on harvesting, not a wage for cleaning.

    Same scalar space, same observation encoding, same null contract -- what changes
    is where the money comes from and how it is shared out. Each step:

        every agent pays  tau * (its harvest income this step)
        the pot           tau * sum_i harvest_i  is paid out the SAME step,
        in shares         w_j / sum_k w_k,  w = cells cleaned over the trailing window

    Two differences from the clean-wage contract are the point of the design rather
    than incidental to it.

    The BASE is what an agent takes out of the commons, not what it fails to put in.
    Under `CleanupContract` a pure harvester's bill is set by everyone else's
    cleaning: it pays theta*(C - c_i)/(N-1) whether it harvested nothing or stripped
    the orchard. Here the bill is proportional to its own take, so an agent that
    sits idle owes nothing and one that harvests hard owes a lot. That makes tau a
    price on appropriation, which is the side of Ostrom's provision/appropriation
    pair the wage contract leaves untouched.

    The PAYOUT is a share of a pot rather than a per-cell wage, so total
    redistribution is capped by what was actually harvested and cannot outrun the
    surplus that funds it. The realised per-cell wage (pot / cells cleaned) then
    floats with how many agents are cleaning, and is worth logging next to tau: it
    is the quantity directly comparable to a clean-wage theta.

    The TRAILING WINDOW is what keeps a cleaner's income from vanishing on the steps
    it happens not to land a cleaning beam. Cleaning is bursty -- an agent walks to
    the river, clears several cells, walks back -- so a payout weighted by cleaning
    THIS step alone would pay a working cleaner nothing on most of its steps and
    everything on a few. It is only a payout weighting, so it is reset at episode
    start and carries across segment boundaries.

    If nobody has cleaned in the window, NO TAX IS LEVIED at all: harvesters keep
    their income and nothing is burned. Taxing with no one to pay would make the
    contract destroy welfare rather than redistribute it, and the zero-sum property
    is what makes this a contract rather than a penalty.

    The mechanism is inert under the null contract (tau = 0 zeroes both sides), so
    the disagreement point is unchanged and every acceptance rule still compares
    against genuine uncontracted play.
    """

    def compute_transfer(self, theta, cleaned):
        raise NotImplementedError(
            "HarvestTaxContract needs the harvest income and the trailing cleaning "
            "window: call tax_transfer(theta, harvest, window). compute_transfer's "
            "signature belongs to the clean-wage contract, and silently falling "
            "back to it would pay a cleaning wage in a run that levied a tax -- "
            "wrong numbers that look entirely ordinary."
        )

    def tax_transfer(self, theta, harvest, window) -> jnp.ndarray:
        """Zero-sum per-agent transfers for one step.

        Args:
            theta: (...,) the tax rate tau, broadcastable against `harvest`'s
                leading dims.
            harvest: (..., N) each agent's harvest income this step
                (info["original_rewards"] -- reward from apples, before any
                transfer). A rate on income rather than a flat fee per apple, which
                is the same thing at the apple_reward=1.0 every config here sets,
                and the sane reading of "pays tau per apple" at any other.
            window: (..., W, N) cells cleaned per step over the trailing window,
                inclusive of this step.

        Returns:
            (..., N) float32 transfers summing to exactly zero along the agent axis.
        """
        harvest = jnp.asarray(harvest, dtype=jnp.float32)
        window = jnp.asarray(window, dtype=jnp.float32)
        tau = jnp.asarray(theta, dtype=jnp.float32)[..., None]

        w = jnp.sum(window, axis=-2)                          # (..., N) recent work
        total_w = jnp.sum(w, axis=-1, keepdims=True)          # (..., 1)
        levied = total_w > 0.0
        paid = tau * harvest                                  # (..., N)
        pot = jnp.sum(paid, axis=-1, keepdims=True)           # (..., 1)
        # Guard the denominator INSIDE the where as well: a 0/0 produces NaN whose
        # gradient poisons the branch that discards it.
        share = w / jnp.where(levied, total_w, 1.0)
        # Nobody cleaned recently -> no levy at all, rather than a pot with no one
        # to receive it. Both sides go to zero together, so the step stays zero-sum.
        return jnp.where(levied, pot * share - paid, 0.0)


def new_tax_window(window_steps: int, num_agents: int, batch=()) -> jnp.ndarray:
    """A zeroed trailing-cleaning buffer, (*batch, W, N).

    The window axis is second-to-last so the buffer broadcasts against the (..., N)
    per-agent quantities everywhere else -- `batch` is the env axis in training and
    empty when reasoning about one env.

    Built fresh at each episode start. It is a payout weighting rather than part of
    the contract, so it deliberately does NOT reset when a segment ends or when the
    contract is renegotiated -- a cleaner that worked through the last segment is
    still owed for that work by the next one.
    """
    return jnp.zeros(tuple(batch) + (int(window_steps), int(num_agents)),
                     dtype=jnp.float32)


def push_tax_window(window: jnp.ndarray, cleaned: jnp.ndarray) -> jnp.ndarray:
    """Drop the oldest step, append this one's cleaning. Returns the new buffer.

    Rolls along the window axis (-2), so it takes a batched buffer or a bare one.
    """
    return jnp.roll(window, -1, axis=-2).at[..., -1, :].set(
        jnp.asarray(cleaned, dtype=jnp.float32))


class LegacyCleanupContract(CleanupContract):
    """The contract observation as it was BEFORE the is_null flag: [theta_norm, stage].

    Kept solely so pre-fix checkpoints stay loadable. Policies trained under the old
    encoding have a first Dense layer sized for 2 contract features, so the current
    CleanupContract cannot be used with them at all -- the widths simply do not match
    and the checkpoint is unopenable without this.

    Do not train anything new with it. theta_norm ran over [0, 1], so the null
    contract encoded as the all-zero vector, which is the defect described at length
    in CleanupContract.to_obs: it made the contract pathway functionally absent at
    theta=0 and left V_i(s, 0) contaminated by nearby contracts.

    The old code used `low` as the null contract, so this only reproduces the old
    behaviour when low == 0 -- which every pre-fix run had, since splitting the null
    out from the range floor arrived with the flag.
    """

    def __init__(self, num_agents: int, low: float = 0.0, high: float = 0.2):
        super().__init__(num_agents, low=low, high=high)
        if abs(low - self.null) > _NULL_TOL:
            raise ValueError(
                f"the legacy encoding used `low` as the null contract, so it is only "
                f"faithful with low == {self.null}; got low={low}. Pre-fix runs were "
                f"trained on a range starting at 0 (typically [0.0, 0.5])."
            )
        self.obs_dim = 2

    def to_obs(self, theta: jnp.ndarray, stage: float = SUBGAME) -> jnp.ndarray:
        theta = jnp.asarray(theta, dtype=jnp.float32)
        theta_norm = (theta - self.low) / (self.high - self.low)
        stage_arr = jnp.full_like(theta_norm, jnp.float32(stage))
        return jnp.stack([theta_norm, stage_arr], axis=-1)


def contract_obs_dim_of(params) -> int:
    """How many contract features a saved ContractActorCritic expects.

    Read off the weights rather than guessed from a flag, because getting it wrong is
    an unopenable checkpoint rather than a wrong number. The gameplay network
    concatenates the contract vector onto the CNN embedding before its first Dense
    layer, so the width is (that Dense's input) - (the CNN's output).
    """
    p = params.get("params", params)
    try:
        embedding = p["CNN_0"]["Dense_0"]["kernel"].shape[1]
        head_input = p["Dense_0"]["kernel"].shape[0]
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError(
            "these params do not look like a ContractActorCritic gameplay policy "
            "(expected CNN_0/Dense_0 and Dense_0 kernels)"
        ) from exc
    return int(head_input - embedding)


def contract_for_params(params, num_agents: int, low: float, high: float):
    """The contract space matching the encoding a checkpoint was actually trained with.

    Use this instead of constructing CleanupContract directly anywhere a checkpoint
    from disk is involved: pre- and post-fix runs need different contract observation
    widths, and mixing them is a shape error several frames from the real cause.
    """
    dim = contract_obs_dim_of(params)
    current = CleanupContract(num_agents, low=low, high=high)
    if dim == current.obs_dim:
        return current
    if dim == 2:
        if abs(low - current.null) > _NULL_TOL:
            raise SystemExit(
                f"this checkpoint predates the is_null contract flag ({dim} contract "
                f"features). It was trained on a range starting at 0, so pass "
                f"--contract-low 0.0 (and the run's own --contract-high, typically "
                f"0.5); got --contract-low {low}."
            )
        print(f"[MOCA] pre-fix checkpoint ({dim} contract features): using the legacy "
              f"contract encoding [theta_norm, stage] over [{low}, {high}]")
        return LegacyCleanupContract(num_agents, low=low, high=high)
    raise SystemExit(
        f"checkpoint expects {dim} contract features, which matches neither the "
        f"current encoding ({current.obs_dim}) nor the legacy one (2)"
    )


# What the bargained scalar MEANS. The space, the observation encoding and the null
# contract are identical across kinds; only the transfer differs.
CONTRACT_KINDS = ("clean_wage", "harvest_tax")


def make_contract(name: str, num_agents: int, low: float, high: float,
                  kind: str = "clean_wage"):
    """Contract-space factory, so the space is selectable from config."""
    if name != "cleanup":
        raise ValueError(f"unknown contract space {name!r} (available: 'cleanup')")
    if kind not in CONTRACT_KINDS:
        raise ValueError(f"unknown CONTRACT_KIND {kind!r} "
                         f"(available: {', '.join(CONTRACT_KINDS)})")
    cls = CleanupContract if kind == "clean_wage" else HarvestTaxContract
    return cls(num_agents, low=low, high=high)
