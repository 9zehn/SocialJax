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

Three spaces live here, one per environment. All are a single scalar theta over
{0} u [low, high] with the same observation encoding; they differ ONLY in which
observable outcome the transfer reads and where the money goes.

    CleanupContract    theta per waste cell cleaned, funded evenly by the others.
                       Reference: `CleanupContract`, theta in [0, 0.2].
    HarvestContract    theta charged to an agent that eats an apple in a
                       low-density patch, split evenly among the others.
                       Reference: `HarvestFeaturemodLocalContract`, theta in [0, 10].
    CoinGameContract   theta per coin taken of another agent's colour, paid to the
                       agent whose colour it was. NOT in the reference -- see the
                       class docstring for what it is built from instead.

Note the direction of the first two. Cleanup SUBSIDISES an under-provided public
good; Harvest TAXES an over-used common resource. Both are the authors', and the
pair is the reason these environments are worth running together: they are the two
halves of Ostrom's provision/appropriation problem, and a mechanism that fixes one
need not fix the other.
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


class ScalarContract:
    """A contract space parameterised by one scalar theta over {0} u [low, high].

    Everything except the transfer itself lives here: the sampling distribution
    P(Theta) phase 1 draws from, the discretised grid phase 2 proposes over, and the
    feature vector the gameplay policy conditions on. All three environments share
    them, which is what makes an arm run on Harvest or the Coin Game mean the same
    thing it means on Clean Up.

    Subclasses supply two things:

        SIGNAL_KEYS       the env info fields the transfer reads, so a run fails
                          loudly on an environment that does not emit them rather
                          than silently contracting on nothing.
        compute_transfer  theta and those signals -> a zero-sum vector of per-agent
                          reward transfers.

    The contract space is {NULL_THETA} u [low, high]. `low` used to double as the
    null contract, which is only correct when low == 0: with low > 0 the "no
    contract" fallback would itself move reward, silently corrupting every
    disagreement value V_i(s, 0) the acceptance rules compare against. They are now
    separate, so the range can exclude weak contracts -- a theta too small to change
    behaviour against a unit reward only blurs the boundary between contracted and
    uncontracted play -- while the null contract stays available and genuinely null.

    Args:
        num_agents: N, the number of agents (>= 2; a transfer needs a counterparty).
        low: minimum NON-NULL contract value (>= 0).
        high: maximum contract value.
    """

    #: The paper's range for this space, for configs and tests to state exactly.
    DEFAULT_RANGE: Tuple[float, float] = (0.0, 0.2)
    #: Env info fields `transfer_from_info` reads. Checked against the env's actual
    #: info dict at run start.
    SIGNAL_KEYS: Tuple[str, ...] = ()

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

    def compute_transfer(self, theta: jnp.ndarray, *signals) -> jnp.ndarray:
        raise NotImplementedError

    def transfer_from_info(self, theta: jnp.ndarray, info: dict) -> jnp.ndarray:
        """Transfers for one step, reading the signals straight out of the env info.

        The training loops call this rather than `compute_transfer`, so the mapping
        from environment to contract lives in ONE place -- the contract, which is by
        definition the function from observable outcomes to transfers. A caller that
        had to know which info key a given space reads would have to be updated for
        every new space, and getting it wrong is a run that trains perfectly happily
        while redistributing on the wrong quantity.
        """
        missing = [k for k in self.SIGNAL_KEYS if k not in info]
        if missing:
            raise KeyError(
                f"{type(self).__name__} conditions on env info fields {missing}, "
                f"which this environment does not emit. Available: {sorted(info)}"
            )
        return self.compute_transfer(theta, *(info[k] for k in self.SIGNAL_KEYS))


class CleanupContract(ScalarContract):
    """Clean Up: pay `theta` per waste cell cleaned, funded evenly by the others.

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
    """

    DEFAULT_RANGE = (0.0, 0.2)
    SIGNAL_KEYS = ("cleaned_by_agent",)

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


class HarvestContract(ScalarContract):
    """Harvest: an agent that eats an apple in a THIN PATCH pays `theta` to the rest.

    The authors' space verbatim (contract/contract_list.py::
    HarvestFeaturemodLocalContract): "Contracts are parameterized by theta in
    [0, 10]. When an agent eats an apple in a low-density region, defined as an apple
    having less than 4 neighboring apples within a radius of 5, they transfer theta to
    the other agents, which is equally distributed to the other agents."

    Their transfer is

        transfers[i] = theta   if feature_obs[8] < 4 and eaten_close_apples > 0
                       0       otherwise

    with the surrounding machinery doing `rews[i] -= transfers[i]` and
    `rews[j] += transfers[i]/(N-1)` for every other j (two_stage_train.py). So a
    POSITIVE entry means agent i pays, which is why the sign here is the mirror of
    Clean Up's: this space prices a harm rather than subsidising a benefit.

        pay_i      = theta * d_i                        # d_i in {0, 1}
        receive_i  = theta * (sum_j d_j - d_i)/(N-1)    # your share of the fines
        transfer_i = receive_i - pay_i

    Why the local-density condition rather than a flat tax on harvesting: apples
    regrow at a rate that depends on how many apples remain nearby, and stop
    regrowing altogether once a patch is stripped. Eating the last apples of a patch
    is therefore the one act with a lasting external cost, and eating from a full
    patch has almost none. A flat tax on all harvesting would price the two the same
    and suppress the behaviour the commons is FOR; this prices only the depletion.
    The env computes the predicate -- see harvest_open.py's `low_density_*` kwargs
    for the neighbourhood, which is the reference's, quirks included.

    An INDICATOR times theta, not a count: the reference charges the same theta
    however many qualifying apples were eaten. The two coincide in both
    implementations anyway, since an agent moves onto at most one cell per step.

    The range is 50x Clean Up's because it is a different quantity against a
    different base. Clean Up prices a cell of cleaning, of which a working cleaner
    does several per step; this prices one apple-eating event, which a single agent
    does at most once per step and only rarely in a thin patch. theta must be worth
    more than the apple it deters -- at theta < 1 against a unit apple, eating the
    last apple of a patch is still profitable and the contract cannot bind at all.
    """

    DEFAULT_RANGE = (0.0, 10.0)
    SIGNAL_KEYS = ("low_density_eaten",)

    def compute_transfer(self, theta: jnp.ndarray, low_density_eaten) -> jnp.ndarray:
        """Zero-sum per-agent reward transfers for one step.

        Args:
            theta: the fine, broadcastable against `low_density_eaten`'s leading dims.
            low_density_eaten: (..., N) 1.0 where the agent ate an apple in a
                low-density patch this step (info["low_density_eaten"]).

        Returns:
            (..., N) float32 transfers summing to zero along the agent axis.
        """
        charged = (jnp.asarray(low_density_eaten, dtype=jnp.float32) > 0.0
                   ).astype(jnp.float32)
        theta = jnp.asarray(theta, dtype=jnp.float32)[..., None]
        total = jnp.sum(charged, axis=-1, keepdims=True)
        pay = theta * charged
        receive = theta * (total - charged) / (self.num_agents - 1)
        return receive - pay


class CoinGameContract(ScalarContract):
    """Coin Game: `theta` per coin taken of another agent's colour, paid to its owner.

    THE AUTHORS DEFINE NO COIN GAME CONTRACT. Their release covers Cleanup, Harvest
    and a self-driving domain only, so this space is built from their design rules
    rather than transcribed:

      * one scalar theta, a per-unit price on the observable outcome that carries the
        externality (as in all three of theirs);
      * conditioned on a per-agent info field the env already attributes to an
        individual (`cleaned_squares`, `eaten_close_apples`, and here
        `stolen_by_agent`);
      * zero-sum, so the contract redistributes and cannot create welfare.

    Where it follows `SelfdriveContractDistprop` rather than the two grid-world
    spaces: the money goes to the agent actually harmed, not evenly to everyone else.
    That contract pays each car "theta times its distance behind the ambulance",
    i.e. in proportion to the harm each bore, and here the harm is exactly
    attributable -- a taken coin has precisely one owner. At the 2 agents this
    environment supports the two rules are the same transfer anyway; the difference
    only appears if the env is ever widened, where paying the victim is the rule that
    keeps meaning what it means.

        pay_i      = theta * (coins i took that were not i's colour)
        receive_i  = theta * (coins of i's colour that others took)
        transfer_i = receive_i - pay_i

    Zero-sum because every taken coin appears once on each side.

    The range [0, 2] is set by the payoff matrix, which pays +1 for any coin and -2
    to the owner of a stolen one. So:

        theta = 0    no contract
        theta = 1    stealing nets the taker nothing (+1 - 1); the point below which
                     the contract cannot deter and above which it can
        theta = 2    the owner is made whole (-2 + 2 = 0), and theft is a straight
                     loss to the taker

    Above 2 the contract would over-compensate and pay agents to be robbed, which is
    an incentive to be careless with one's own coins rather than a fix for the
    dilemma -- so the ceiling is a property of the game, not a tuning choice. Set it
    against `coin_reward=1.0`; the env's default scales the whole payoff matrix by
    num_agents and the range would then be off by that factor.
    """

    DEFAULT_RANGE = (0.0, 2.0)
    SIGNAL_KEYS = ("stolen_by_agent", "stolen_from_agent")

    def compute_transfer(self, theta: jnp.ndarray, stolen_by, stolen_from) -> jnp.ndarray:
        """Zero-sum per-agent reward transfers for one step.

        Args:
            theta: the price of a stolen coin, broadcastable against the signals'
                leading dims.
            stolen_by: (..., N) coins of another agent's colour that agent i took.
            stolen_from: (..., N) coins of agent i's colour that others took.

        Returns:
            (..., N) float32 transfers summing to zero along the agent axis.
        """
        stolen_by = jnp.asarray(stolen_by, dtype=jnp.float32)
        stolen_from = jnp.asarray(stolen_from, dtype=jnp.float32)
        theta = jnp.asarray(theta, dtype=jnp.float32)[..., None]
        return theta * (stolen_from - stolen_by)


class HarvestTaxContract(CleanupContract):
    """CLEAN UP's harvest tax. Not the Harvest environment -- see `HarvestContract`.

    (The names are close and the mechanisms are not. This one is a tax on eating
    apples in CLEAN UP, selected by CONTRACT_KIND=harvest_tax and paid out by recent
    river cleaning. `HarvestContract` is the paper's contract space for the Harvest
    environment, selected by CONTRACT_SPACE=harvest.)

    The bargained scalar is a TAX RATE on harvesting, not a wage for cleaning.

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


def contract_for_params(params, num_agents: int, low: float, high: float,
                        space: str = "cleanup"):
    """The contract space matching the encoding a checkpoint was actually trained with.

    Use this instead of constructing a contract directly anywhere a checkpoint from
    disk is involved: pre- and post-fix runs need different contract observation
    widths, and mixing them is a shape error several frames from the real cause.

    `space` selects which environment's contract to build. The legacy 2-feature
    fallback is Clean Up only -- it predates the other two spaces existing, so no
    checkpoint outside Clean Up can be in that encoding.
    """
    dim = contract_obs_dim_of(params)
    current = make_contract(space, num_agents, low=low, high=high)
    if dim == current.obs_dim:
        return current
    if dim == 2:
        if space != "cleanup":
            raise SystemExit(
                f"a {dim}-feature checkpoint is the pre-2026-08 Clean Up encoding, "
                f"which no {space!r} run can have been trained in -- that space did "
                f"not exist yet. Check --contract-space against the run's sidecar."
            )
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
# contract are identical across kinds; only the transfer differs. Clean Up only:
# the other two environments have one contract space each, the authors' (or, for the
# Coin Game, the one built to their rules).
CONTRACT_KINDS = ("clean_wage", "harvest_tax")

#: CONTRACT_SPACE -> the class implementing it. One per environment.
CONTRACT_SPACES = {
    "cleanup": CleanupContract,
    "harvest": HarvestContract,
    "coin_game": CoinGameContract,
}


def make_contract(name: str, num_agents: int, low: float, high: float,
                  kind: str = "clean_wage"):
    """Contract-space factory, so the space is selectable from config."""
    if name not in CONTRACT_SPACES:
        raise ValueError(
            f"unknown contract space {name!r} "
            f"(available: {', '.join(sorted(CONTRACT_SPACES))})")
    if kind != "clean_wage" and name != "cleanup":
        raise ValueError(
            f"CONTRACT_KIND is a Clean Up setting -- it chooses between the paper's "
            f"cleaning wage and this repo's harvest tax, both of which are Clean Up "
            f"mechanisms. CONTRACT_SPACE={name!r} has one space. Got kind={kind!r}.")
    if kind not in CONTRACT_KINDS:
        raise ValueError(f"unknown CONTRACT_KIND {kind!r} "
                         f"(available: {', '.join(CONTRACT_KINDS)})")
    cls = (HarvestTaxContract if kind == "harvest_tax"
           else CONTRACT_SPACES[name])
    return cls(num_agents, low=low, high=high)
