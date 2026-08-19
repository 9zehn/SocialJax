"""Formal contracting: the shared training loop, over any registered environment.

Which environment a run plays is a config choice (ENV_NAME) resolved through
algorithms/MOCA/envs.py, which supplies the contract space, the behaviour metrics
and the theta range. The per-env entry points -- moca_cnn_cleanup.py,
moca_cnn_harvest.py, moca_cnn_coins.py -- are the Hydra bindings; the algorithm is
here, once.

Two ways to train the contracting stage, both from Christoffersen et al., "Formal
Contracts Mitigate Social Dilemmas in Multi-Agent RL" (arXiv:2208.10469 / AAMAS
2023):

  TRAINING_MODE=two_phase   MOCA, Algorithm 1. The headline algorithm.
  TRAINING_MODE=combined    Single-stage contracting (the reference's
                            `SeparateContractCombinedStage`): a contract is
                            negotiated afresh at the start of EVERY episode and
                            gameplay learns under it, with nothing frozen and no
                            fixed P(Theta). The baseline MOCA's two-phase
                            construction is meant to beat.

Algorithm 1 itself:

    for t = 1 .. (9/10) num_episodes:          # PHASE 1 -- subgame
        theta ~ P(Theta)
        train_subgame_episode(pi(s_0, theta))
    freeze pi|_{S x Theta}                     # gameplay policy frozen
    for t = 1 .. (1/10) num_episodes:          # PHASE 2 -- contracting
        theta ~ pi_i(i, 0)
        accept if all voters accept
        R <- sample_episode_reward(pi, contract)
        train_with_rewards(pi_contracting, R)

Why the two phases matter (and why this differs from a hand-designed transfer
rule): Phase 1 trains the gameplay policy on contracts drawn from a FIXED
distribution that is independent of the proposal policy. The gameplay policy
therefore learns the whole family {pi(.|s, theta)}, giving an unbiased estimate of
each agent's value V_i(s_0, theta) for every theta, rather than only for the
contracts a co-adapting proposer happened to favour. Freezing that policy before
Phase 2 means the contracting stage optimises against a fixed, already-solved
subgame -- which is what makes the selected contract a subgame-perfect equilibrium
choice rather than an artefact of joint learning dynamics.

What `combined` gives up, stated plainly, because it is the whole point of running
both: the contract policy chases a gameplay policy that is still moving, and the
gameplay policy only ever meets contracts near whatever is currently being proposed.
So V_i(s_0, theta) is estimated on-distribution rather than across the space, the
subgame-perfection argument does not apply, and the null contract is seen only as
often as the negotiation happens to fail. That is not a defect in the
implementation -- it is what MOCA's two phases exist to fix, and measuring the gap
requires running the thing they fix.

Gameplay rewards are R'_i = R_i + theta_i(s,a) with sum_i theta_i = 0, so contracts
redistribute welfare without creating it (see contracts.py).

Only PARAMETER_SHARING=False is supported: MOCA gives every agent its own proposal
and voting policy, and the role specialisation these environments are studied for
requires distinct gameplay policies too.
"""
import time
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
import socialjax
from socialjax.wrappers.baselines import LogWrapper
import wandb

from algorithms.utils import checkpoint_filename, load_params, save_params, save_train_state
from algorithms.MOCA import bargain, contracts, envs, negotiate, reporting, solver
from algorithms.MOCA.contracts import AGREE, PROPOSE, make_contract
from algorithms.MOCA.networks import (
    BargainingActorCritic, ContractActorCritic, NegotiationActorCritic,
    ProposalPolicy, VotingPolicy,
)


def episode_stats(traj_batch, num_agents):
    """Welfare / equality / transfer-volume for one batch of episodes.

    These are the outcome measures the formal-contracting results are stated in
    (the reference implementation tracks the corresponding 'transfers' and
    'transfer_equality' metrics):

    * welfare -- summed episode return across agents. Contracts are zero-sum, so
      they cannot raise welfare directly; welfare only moves if the contract
      changes BEHAVIOUR (more cleaning -> more apples). It is therefore the metric
      that says whether contracting actually mitigated the dilemma.
    * equality -- 1 - Gini over per-agent episode returns (the standard measure in
      the sequential-social-dilemma literature, e.g. Hughes et al. 2018).
    * transfer_volume -- total reward actually moved per episode. The "is the
      mechanism doing anything at all" diagnostic: zero volume means the contract
      is inert regardless of what was signed.
    """
    returns = jnp.stack([traj_batch[i].reward.sum(axis=0) for i in range(num_agents)])
    welfare = returns.sum(axis=0)                      # (NUM_ENVS,)

    # 1 - Gini, computed on the pairwise-absolute-difference form.
    diffs = jnp.abs(returns[:, None, :] - returns[None, :, :]).sum(axis=(0, 1))
    denom = 2.0 * num_agents * jnp.abs(returns).sum(axis=0) + 1e-8
    equality = 1.0 - diffs / denom

    # Zero-sum transfers: the total moved equals the sum of the positive side.
    transfers = jnp.stack(
        [traj_batch[i].info["contract_transfer"].squeeze(-1) for i in range(num_agents)]
    )                                                   # (N, NUM_STEPS, NUM_ENVS)
    transfer_volume = jnp.maximum(transfers, 0.0).sum(axis=(0, 1))

    return {
        "welfare": welfare.mean(),
        "equality": equality.mean(),
        "transfer_volume": transfer_volume.mean(),
        "returns": returns,
    }


# Metrics forwarded to wandb, one namespaced set per phase. Deliberately short:
# logging the whole info dict crossed with _mean/_std produces 30+ stage-1 and 49
# stage-2 series, which buries the handful that decide whether a run worked.
# Anything dropped here is still recoverable from the saved checkpoints.
#
# The behaviour series are per-environment (envs.EnvSpec): "did cleaning rise" on
# Clean Up, "did depleting harvests fall" on Harvest, "did theft fall" on the Coin
# Game. Everything else is the same question in every environment, so the builders
# below differ only in that block.


def _behaviour_metrics(spec) -> Tuple[str, ...]:
    """The per-env "did behaviour actually change" block.

    Transfers are zero-sum, so welfare can only move if behaviour does; without
    these, a welfare change is unattributable and a flat welfare curve cannot be
    told from an inert mechanism.
    """
    out = []
    for name in spec.behaviour_metrics:
        out.append(f"{name}_mean")
    # Spread ACROSS agents on the contracted act: is the burden (or the subsidy)
    # concentrated in a few agents? That IS the role specialisation these
    # environments are studied for, and pooled means hide it completely.
    out.append(f"{spec.contracted_act}_std")
    out.append(f"{spec.commons_metric}_mean")
    return tuple(out)


def stage1_metrics(spec) -> Tuple[str, ...]:
    return (
        # Learning curve and the emergent division of labour it produces.
        "returned_episode_returns_mean",   # per-agent episode return
        "returned_episode_returns_std",    # spread ACROSS agents = specialisation
        f"{spec.progress_metric}_mean",    # read by progress_callback
    ) + _behaviour_metrics(spec) + (
        # Outcome measures the contracting results are stated in.
        "welfare",
        "equality",
        "transfer_volume",
        # PPO health.
        "loss_mean",
        "value_loss_mean",
        "entropy_mean",
        # Sanity check on P(Theta): phase 1 must see the whole contract space.
        "contract_theta_sampled",
        "contract_null_frac",
        # Does the policy CONDITION on theta? The gap must open up; if it stays at
        # ~0 the contract is inert and every phase-2 comparison downstream is moot.
        f"{spec.act_label}_null",
        f"{spec.act_label}_contracted",
        f"{spec.act_label}_gap",
        # V_i(s, 0), the disagreement point phase 2 measures acceptance against.
        "welfare_null",
        "welfare_contracted",
    )


def stage2_metrics(spec) -> Tuple[str, ...]:
    return (
        # What is offered, what ends up in force, and whether it is signed at all.
        "contract_theta_proposed",
        "contract_theta_effective",
        "contract_accept_rate",
        # Convergence diagnostic. Starts at log(NUM_CONTRACT_BINS) and MUST fall: a
        # run that ends at the maximum has learned nothing, whatever its argmax says.
        "contract_proposal_entropy",
    ) + _behaviour_metrics(spec) + (
        "transfer_volume",
        # Headline outcomes.
        "welfare",
        "equality",
    )


def stage2_solver_metrics(spec) -> Tuple[str, ...]:
    """Solver phase 2 has no proposal or acceptance policy to track, so its
    diagnostics are about what the sampling search settled on instead."""
    return (
        "contract_theta_effective",
        "contract_theta_std",        # do the per-env negotiations agree?
        "solver_null_rate",          # how often nothing beat the null contract
        "solver_accept_count",       # agents preferring the chosen contract to null
        "solver_predicted_welfare",  # critic's estimate; compare against `welfare`
    ) + _behaviour_metrics(spec) + (
        "welfare",                   # realised
        "equality",
    )


def stage2_negotiate_metrics(spec) -> Tuple[str, ...]:
    """Learned negotiation stage: a proposal game again, so the diagnostics are what
    agent 0 offers and whether the polled agents sign it."""
    return (
        "contract_theta_proposed",
        "contract_theta_effective",
        "contract_accept_rate",      # realised signings
        "contract_accept_prob",      # the product of the polled agents' probabilities
        "negotiate_policy_entropy",  # falling entropy = the proposal is converging
    ) + _behaviour_metrics(spec) + (
        "welfare",
        "equality",
    )


def combined_metrics(spec) -> Tuple[str, ...]:
    """Single-stage contracting: one loop, so one set of series covering both halves.

    Everything phase 1 tracks about the gameplay policy AND everything the
    negotiation stage tracks, because here they are learning at the same time and
    the interesting failures are in how they interact -- a proposal that converges
    while gameplay is still moving, or gameplay that never meets a null contract
    because acceptance saturated early.
    """
    return (
        "returned_episode_returns_mean",
        "returned_episode_returns_std",
        f"{spec.progress_metric}_mean",
    ) + _behaviour_metrics(spec) + (
        # The negotiation half.
        "contract_theta_proposed",
        "contract_theta_effective",
        "contract_accept_rate",
        "contract_accept_prob",
        "negotiate_policy_entropy",
        # How often gameplay actually sees the null contract. Under MOCA this is
        # NULL_CONTRACT_FRAC and fixed by construction; here it is whatever the
        # negotiation happens to reject, which is the quantity that decides whether
        # V_i(s, 0) means anything at all in this arm.
        "contract_null_frac",
        f"{spec.act_label}_null",
        f"{spec.act_label}_contracted",
        f"{spec.act_label}_gap",
        "welfare_null",
        "welfare_contracted",
        # Outcomes and PPO health.
        "welfare",
        "equality",
        "transfer_volume",
        "loss_mean",
        "value_loss_mean",
        "entropy_mean",
    )


# Joint Rubinstein bargaining. Efficiency here is SPEED OF AGREEMENT, not welfare:
# above the contract threshold welfare is flat, so it cannot discriminate between
# contracts, whereas every round of disagreement burns a measurable slice of the
# episode. Equity is where in the range theta lands.
#
# Grouped into wandb sections by the "/" in the logged name, and pruned of anything
# derivable from what is left: disagreement_steps is agreement_round *
# BARGAIN_SEGMENT, returned_episode_returns_mean is welfare / num_agents, and
# equality already carries what the spread across agents said.
JOINT_BARGAIN_METRICS = {
    # Efficiency, which here is SPEED OF AGREEMENT rather than welfare: above the
    # contract threshold welfare is flat and cannot discriminate between contracts,
    # while every round of disagreement burns a measurable slice of the episode.
    "agreement_rate": "agree/rate",
    "agreement_round": "agree/round",
    # Is the mechanism doing anything, and what did it settle on? in_force_rate
    # sitting near zero is the cold-start failure, not a bug -- see _runner.
    "contract_in_force_rate": "contract/in_force_rate",
    # The FIRST segment specifically. On a depletable commons the pooled rate above
    # can look healthy while every contract arrived too late to price anything: in
    # Harvest the orchard can be stripped inside the first segment, after which a
    # contract on thin-patch eating has nothing left to charge for. The two series
    # separating is the finding, so both are logged.
    "contract_in_force_rate_round0": "contract/in_force_rate_round0",
    # CONTRACT_SPACE=harvest_density only, and identically zero elsewhere so one
    # wandb view covers both. The threshold is the second half of the contract:
    # theta says how much depletion costs, k says which harvesting counts as
    # depletion at all. Both series, because they fail differently -- a high fine on
    # an inert threshold (k -> 0) prices nothing, and a flat tax (k -> 13) prices
    # every harvest including the ones that cost the commons nothing.
    #   density_k_offered   what proposers ASK for, over active non-probe rounds.
    #   density_k_in_force  what actually governed, over contracted segments.
    "density_k_offered": "contract/k_offered",
    "density_k_in_force": "contract/k_in_force",
    "theta_agreed": "contract/theta_agreed",
    "theta_offered": "contract/theta_offered",
    # The scalar actually in force, averaged over the segments that had a contract.
    # Under CONTRACT_KIND=harvest_tax this is the mean tax RATE in force; under
    # clean_wage it is the mean wage. Distinct from theta_agreed, which under
    # episode binding is the one agreed value rather than what got played.
    "theta_in_force": "contract/theta_in_force",
    # Harvest tax only (identically zero under clean_wage).
    #   tax/revenue  reward levied per step, in reward units.
    #   tax/wage     revenue / cells cleaned -- the per-cell wage the tax actually
    #                produces, which is the number directly comparable to a
    #                clean_wage theta. Unlike theta it is not chosen: it floats with
    #                how much harvesting is taxed and how many agents share the pot.
    "tax_revenue": "tax/revenue",
    "tax_wage": "tax/wage",
    "accept_count": "contract/accept_count",
    # Max - min of the asks within a round. Under BARGAIN_PROTOCOL=median this is
    # the mechanism's own diagnostic -- are ideal points separating by role, or has
    # everyone collapsed onto one number? -- and under alternating offers it is the
    # dispersion of asking prices, only one of which is ever read.
    "theta_ask_spread": "contract/ask_spread",
    "transfer_volume": "contract/transfer_volume",
    # The behaviour block is inserted here, per environment, by joint_metric_map:
    # behaviour/<the contracted act>, its spread across agents, and the commons.
    # The two the result is stated in.
    "welfare": "outcome/welfare",
    "equality": "outcome/equality",
    # Run health. A collapsing gameplay policy is invisible in every series above
    # until it has already happened, so this is the early-warning one.
    "policy_entropy": "health/policy_entropy",
    "policy_entropy_min": "health/policy_entropy_min",
    # The vote's exploration floor, annealing to 0. Worth a series because it says
    # how much of the accept rate above is still the floor rather than the policy.
    "vote_eps": "health/vote_eps",
    # Counterfactual credit for the vote. Always logged (0 under
    # BARGAIN_VOTE_ADVANTAGE=gae, where no branch heads exist) rather than
    # conditionally present, so one wandb view covers both modes and a run that
    # silently fell back to gae is visible as a flat zero.
    #   cf/gap      believed lock-minus-continue value, over consequential votes.
    #               Its SIGN is the mechanism: negative means "this offer is worse
    #               for me than bargaining on", which is what a rejection needs.
    #   cf/pivotal_rate  fraction of those votes that actually decided the round.
    #               Under unanimity this is high by construction; if it collapses,
    #               the vote head is being trained on very little.
    "cf_gap": "cf/gap",
    "cf_pivotal_rate": "cf/pivotal_rate",
    # Claims and audits (reporting.py). Zero when REPORT_ENABLE is off.
    #   report/overclaim  mean chosen overclaim per claim, over in-force windows.
    #                     Honest agents sit at 0; watch it against the analytic
    #                     honesty boundary lam* = (1-p)/p.
    #   report/leakage    overclaimed reward actually PAID per episode (unaudited
    #                     lies x theta) -- the enforcement leak, in reward units.
    "report_overclaim": "report/overclaim",
    "report_leakage": "report/leakage",
}


#: Metric sets a joint bargaining run can log. `full` is every series below;
#: `core` is the ten that answer the questions a renegotiation run is read for, and
#: nothing else. Set per environment in the config -- see WANDB_METRIC_SET.
METRIC_SETS = ("full", "core")


def joint_core_metrics(spec, param_dim: int = 1) -> Tuple[str, ...]:
    """The ten series a renegotiated bargaining run is actually read on.

    Chosen to answer five questions and no more, one or two series each:

        did welfare move, and for whom     welfare, equality
        did the mechanism ever bind        in_force_rate, in_force_rate_round0
        at what price                      theta_in_force
        did BEHAVIOUR change               act_per_episode, <act>_share, <act>_gap
        did the commons survive            <commons>
        is the run alive                   policy_entropy

    What this drops that is genuinely load-bearing, and is worth switching back to
    WANDB_METRIC_SET=full to see: `agree/rate` and `agree/round` (speed of
    agreement -- under `segment` binding in_force_rate carries nearly the same
    information, under `episode` it does not), `contract/transfer_volume` (how much
    reward the contract moved, as opposed to whether it bound), `outcome/welfare_null`
    vs `welfare_contracted` (the welfare half of the split whose behaviour half is
    kept here), and the `health/` and `cf/` diagnostics.

    Deliberately NOT dropped for being near-zero: a series that reads zero because
    the mechanism is not working is the series you need most.
    """
    core = [
        "welfare", "equality",
        "contract_in_force_rate", "contract_in_force_rate_round0",
        "theta_in_force",
        "act_per_episode",
    ]
    if param_dim > 1:
        # The contract's second half. theta_in_force says how much depletion costs;
        # this says which harvesting was counted as depletion at all, and the two
        # are not substitutes -- a large fine on a threshold near 0 prices nothing.
        # Only present when there is a threshold to report, so the scalar spaces do
        # not carry a permanently flat chart.
        core.append("density_k_offered")
    if len(spec.behaviour_metrics) > 1:
        # Separates "the contract stopped the harm" from "the contract stopped the
        # activity" -- the numerator alone cannot, and on a depletable commons the
        # second is the likelier failure.
        core.append(f"{spec.act_label}_share")
    core += [
        f"{spec.act_label}_gap",
        f"{spec.commons_metric}_mean",
        "policy_entropy",
    ]
    return tuple(core)


def joint_metric_map(spec, metric_set: str = "full", param_dim: int = 1) -> dict:
    """JOINT_BARGAIN_METRICS with this environment's behaviour block spliced in.

    Everything a bargaining run is judged on -- speed of agreement, where theta
    landed, welfare, equality -- is environment-independent. What is not is the
    answer to "did behaviour actually change", and without that a welfare number is
    unattributable: transfers are zero-sum, so welfare can only move if the
    contracted act does.

    Logged under one name per section (behaviour/act, behaviour/act_spread,
    behaviour/commons) rather than under the raw info keys, so one wandb view reads
    a Clean Up run and a Harvest run side by side even though the underlying series
    are cleaning and depleting harvests.
    """
    out = dict(JOINT_BARGAIN_METRICS)
    # Named after the quantity, not after a generic slot: `behaviour/act` would read
    # the same on every environment while meaning cleaning in one and theft in
    # another, which is exactly the confusion a cross-environment comparison invites.
    # Clean Up's two names are unchanged from before there were other environments,
    # so its wandb views and tests/test_golden_arms.json keep working.
    out[f"{spec.contracted_act}_mean"] = f"behaviour/{spec.act_label}_per_agent"
    # Spread ACROSS agents: whether the contracted act is concentrated in a few
    # agents. That IS the role specialisation the bargaining result is about --
    # cleaners vs harvesters, thieves vs victims -- and pooled means hide it.
    out[f"{spec.contracted_act}_std"] = f"behaviour/{spec.act_label}_spread"
    # The same act summed over the EPISODE and over agents. `_per_agent` above is the
    # raw info field's mean over agents AND steps -- a per-agent-per-step rate, which
    # on Harvest reads as ~0.001 and is nearly unreadable; this is the count you can
    # hold against "how many apples were on the map to begin with".
    out["act_per_episode"] = f"behaviour/{spec.act_label}_per_episode"
    out[f"{spec.commons_metric}_mean"] = f"behaviour/{spec.commons_metric}"
    for extra in spec.behaviour_metrics[1:]:
        # The denominator series, where the environment has one: a contract that cut
        # depleting harvests by suppressing ALL harvesting is a welfare loss dressed
        # up as a success, and only the ratio tells the two apart.
        out[f"{extra}_mean"] = f"behaviour/{extra}"
    if len(spec.behaviour_metrics) > 1:
        # ...and the ratio itself, because that is the quantity the argument is
        # actually about and neither series carries it alone. On Harvest it is the
        # share of harvesting that depletes a patch, on the Coin Game the share of
        # coin collection that is theft. A contract that works drives this DOWN
        # while the denominator holds; a contract that has merely frightened the
        # agents off the commons drives both down together, which looks identical
        # in the numerator series and obviously different here.
        out[f"{spec.act_label}_share"] = f"behaviour/{spec.act_label}_share"
    # Behaviour and welfare under a contract against under none. `<act>_gap` is the
    # series that must open up -- a mechanism whose behaviour is the same either way
    # has priced nothing, however healthy its agreement rate looks -- and
    # welfare_null is the disagreement point every vote is implicitly cast against.
    out[f"{spec.act_label}_null"] = f"behaviour/{spec.act_label}_null"
    out[f"{spec.act_label}_contracted"] = f"behaviour/{spec.act_label}_contracted"
    out[f"{spec.act_label}_gap"] = f"behaviour/{spec.act_label}_gap"
    out["welfare_null"] = "outcome/welfare_null"
    out["welfare_contracted"] = "outcome/welfare_contracted"
    if metric_set == "core":
        keep = joint_core_metrics(spec, param_dim)
        missing = [k for k in keep if k not in out]
        if missing:
            raise KeyError(
                f"core metric set names series this environment does not log: "
                f"{missing}. Available: {sorted(out)}")
        out = {k: out[k] for k in keep}
    return out


# How the vote head's advantage is computed. See bargain.counterfactual_vote_advantage.
VOTE_ADVANTAGE_MODES = ("gae", "counterfactual")


def regime_split(traj_batch, stats, theta, contract, num_agents, spec) -> dict:
    """Behaviour and welfare under the NULL contract vs. under a contract in force.

    The headline question of contract-conditioned training, and invisible in the
    pooled averages: a policy that behaves identically at theta=0 and theta>0
    produces exactly the same mean as one that has learned the distinction. Splitting
    by regime makes it a live training curve rather than a post-hoc grid_eval run.

    `<act>_contracted - <act>_null` is the thing that must open up. `welfare_null` is
    the disagreement point V_i(s, 0) that every acceptance rule is measured against,
    so it doubles as a check that the null baseline is not itself drifting.

    Both halves are means over whichever envs fell in each regime, so the result is
    meaningful whether the split comes from phase 1's P(Theta) (a fixed fraction, by
    construction) or from a negotiation that happened to reject (whatever the arm
    produced). When one side is empty its mean is 0/1 = 0 rather than a NaN -- and
    `contract_null_frac`, returned alongside, is what says which case you are in.
    """
    null_mask = contract.is_null(theta).astype(jnp.float32)     # (NUM_ENVS,)
    n_null = jnp.maximum(null_mask.sum(), 1.0)
    n_contracted = jnp.maximum((1.0 - null_mask).sum(), 1.0)
    act_per_env = jnp.stack([
        traj_batch[i].info[spec.contracted_act].squeeze(-1)
        for i in range(num_agents)
    ]).sum(axis=(0, 1))                                          # (NUM_ENVS,)
    welfare_per_env = stats["returns"].sum(axis=0)               # (NUM_ENVS,)
    out = {}
    for name, per_env in ((spec.act_label, act_per_env),
                          ("welfare", welfare_per_env)):
        out[f"{name}_null"] = (per_env * null_mask).sum() / n_null
        out[f"{name}_contracted"] = (
            (per_env * (1.0 - null_mask)).sum() / n_contracted
        )
    out[f"{spec.act_label}_gap"] = (
        out[f"{spec.act_label}_contracted"] - out[f"{spec.act_label}_null"]
    )
    out["contract_null_frac"] = null_mask.mean()
    return out


def _select(metrics: dict, allowed: Tuple[str, ...]) -> dict:
    """Restrict `metrics` to `allowed`, raising if a name in the allowlist is absent.

    A silent miss would drop the series from wandb with no error anywhere, which is
    exactly how a broken run looks identical to a working one.
    """
    missing = [k for k in allowed if k not in metrics]
    if missing:
        raise KeyError(
            f"metric allowlist names not produced by this phase: {missing}. "
            f"Available: {sorted(metrics)}"
        )
    return {k: metrics[k] for k in allowed}


class MOCATransition(NamedTuple):
    """PPO transition, plus the contract features the policy was conditioned on."""
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    contract: jnp.ndarray
    info: dict


class NegotiationRecord(NamedTuple):
    """One run of the two-step contracting game, per env.

    Carries both the OUTCOME (what was offered, what took force) and everything the
    PPO update needs to credit it -- the actions, their log-probs under the policy
    that took them, and the critic at both steps. Bundled rather than passed around
    loose because the same record is consumed by three callers (the phase-2 update,
    the single-stage update, and the metrics), and a mismatched subset would be a
    silent mis-crediting rather than an error.

    Shapes: theta_* and accepted/prod_prob are (num_envs,); raws is (2, N, E, A) and
    logps/values are (2, N, E), the leading axis being propose-then-agree.
    """
    theta_prop: jnp.ndarray
    theta_eff: jnp.ndarray
    accepted: jnp.ndarray
    prod_prob: jnp.ndarray
    obs_propose: jnp.ndarray
    obs_agree: jnp.ndarray
    raws: jnp.ndarray
    logps: jnp.ndarray
    values: jnp.ndarray


def make_train(config):
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    # Which environment, and therefore which contract space, which behaviour series
    # and which theta range. Resolved before anything else so an unsupported env
    # fails on the config rather than on a missing info key inside a traced rollout.
    spec = envs.spec_for(config["ENV_NAME"])
    envs.check_reward_scale(spec, config.get("ENV_KWARGS", {}))
    # Which wandb series a joint bargaining run logs. Defaults to `full`, so every
    # existing run and every existing wandb view is unchanged; the per-environment
    # config is where `core` gets asked for.
    config.setdefault("WANDB_METRIC_SET", "full")
    if config["WANDB_METRIC_SET"] not in METRIC_SETS:
        raise ValueError(
            f"WANDB_METRIC_SET must be one of {', '.join(METRIC_SETS)}, got "
            f"{config['WANDB_METRIC_SET']!r}")
    STAGE1_METRICS = stage1_metrics(spec)
    STAGE2_METRICS = stage2_metrics(spec)
    STAGE2_SOLVER_METRICS = stage2_solver_metrics(spec)
    STAGE2_NEGOTIATE_METRICS = stage2_negotiate_metrics(spec)
    COMBINED_METRICS = combined_metrics(spec)
    # The contract itself is built further down, but its SHAPE is fixed by the space
    # name, and that is all the metric map needs -- whether there is a threshold to
    # report alongside the fine.
    JOINT_METRICS = joint_metric_map(
        spec, config["WANDB_METRIC_SET"],
        contracts.CONTRACT_SPACES[
            config.get("CONTRACT_SPACE") or spec.contract_space].PARAM_DIM)
    commons_scale = envs.commons_scale(spec, env)

    if config["PARAMETER_SHARING"]:
        raise NotImplementedError(
            "MOCA supports PARAMETER_SHARING=False only: each agent needs its own "
            "proposal/voting policy, and role specialisation needs distinct gameplay "
            "policies. Re-run with PARAMETER_SHARING=False."
        )

    num_agents = env.num_agents
    config["NUM_ACTORS"] = config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    # A contract is agreed once per EPISODE and held fixed for its duration, so one
    # PPO rollout must be exactly one episode. The env auto-resets at
    # inner_t == num_inner_steps, so that means NUM_STEPS == num_inner_steps.
    inner_steps = config["ENV_KWARGS"]["num_inner_steps"]
    if config["NUM_STEPS"] != inner_steps:
        raise ValueError(
            f"MOCA needs one rollout to be exactly one episode so a contract spans "
            f"the episode it was agreed for, but NUM_STEPS={config['NUM_STEPS']} != "
            f"num_inner_steps={inner_steps}. Set them equal."
        )

    # Skip phase 1 entirely and run phase 2 against a saved gameplay policy.
    #
    # The point of this is controlled comparison. A phase-2 decision rule does not
    # touch phase 1, and the solver family does not train in phase 2 at all, so
    # comparing N rules by running N full trainings would retrain an identical
    # phase 1 N times -- and any difference in the phase-1 policies would then
    # confound the comparison it was meant to isolate. Loading one frozen policy
    # makes every arm exact.
    phase1_from = config.get("PHASE1_FROM")
    phase1_only = bool(config.get("PHASE1_ONLY", False))
    if phase1_only and phase1_from:
        raise ValueError(
            "PHASE1_ONLY and PHASE1_FROM are opposites (train phase 1 vs load it); "
            "set exactly one"
        )
    config["PHASE1_ONLY"] = phase1_only

    # An UNSET shell variable expands to nothing, so `PHASE1_FROM="$P1"` arrives as
    # the empty string -- which is falsy, and would silently fall through to a full
    # phase-1 training run that looks superficially fine. Refuse it instead: the
    # difference only shows up as a 90/10 update split buried in the progress lines.
    if phase1_from is not None and not str(phase1_from).strip():
        raise ValueError(
            "PHASE1_FROM was set but is empty -- the shell variable holding the glob "
            "is probably unset (it does not survive between Colab cells). Define it "
            "in the SAME command, or omit PHASE1_FROM to train phase 1 deliberately."
        )

    loaded_phase1 = None
    if phase1_from:
        import glob
        matches = sorted(glob.glob(phase1_from))
        if len(matches) != num_agents:
            raise ValueError(
                f"PHASE1_FROM={phase1_from!r} matched {len(matches)} files but the env "
                f"has {num_agents} agents. Point it at the per-agent GAMEPLAY policies "
                f"(not _contract_/_proposal_/_voting_/_resume_): {matches}"
            )
        loaded_phase1 = [load_params(m) for m in matches]
        print(f"[MOCA] phase 1 loaded from {len(matches)} checkpoints; skipping phase-1 "
              f"training and running phase 2 only", flush=True)

    # Split the update budget 90/10 as in Algorithm 1 -- or give it all to whichever
    # phase is actually running.
    phase1_frac = 0.0 if phase1_from else config.get("PHASE1_FRAC", 0.9)
    if phase1_from:
        config["NUM_UPDATES_PHASE1"] = 0
    elif phase1_only:
        config["NUM_UPDATES_PHASE1"] = config["NUM_UPDATES"]
    else:
        config["NUM_UPDATES_PHASE1"] = max(int(config["NUM_UPDATES"] * phase1_frac), 1)
    config["NUM_UPDATES_PHASE2"] = (
        0 if phase1_only
        else max(config["NUM_UPDATES"] - config["NUM_UPDATES_PHASE1"], 1)
    )

    # Which phase 2 to run. "solver" is what the paper's Cleanup experiments used
    # (see algorithms/MOCA/solver.py); "reinforce" is this repo's discretised
    # proposal/voting game, kept selectable so both can be compared.
    phase2_mode = config.get("PHASE2_MODE", "solver")
    if phase2_mode not in ("solver", "reinforce", "negotiate", "bargain"):
        raise ValueError(
            f"PHASE2_MODE must be 'solver', 'negotiate', 'reinforce' or 'bargain', "
            f"got {phase2_mode!r}"
        )
    config["PHASE2_MODE"] = phase2_mode

    # ---- How the contracting stage is TRAINED, as opposed to which protocol -----
    #   "two_phase" -- MOCA, Algorithm 1.
    #   "combined"  -- single-stage contracting: negotiate at the start of every
    #                  episode, play it, and update BOTH policies. Nothing frozen,
    #                  no P(Theta). The reference's SeparateContractCombinedStage.
    #   "joint"     -- the Rubinstein bargaining extension (Clean Up only).
    training_mode = config.get("TRAINING_MODE", "two_phase")
    if training_mode not in ("two_phase", "combined", "joint"):
        raise ValueError(
            f"TRAINING_MODE must be 'two_phase', 'combined' or 'joint', "
            f"got {training_mode!r}")
    config["TRAINING_MODE"] = training_mode
    if (training_mode == "joint") != (phase2_mode == "bargain"):
        raise ValueError(
            "TRAINING_MODE='joint' and PHASE2_MODE='bargain' currently go together: "
            "joint has no other protocol implemented, and bargain has no two-phase "
            f"variant yet. Got TRAINING_MODE={training_mode!r}, "
            f"PHASE2_MODE={phase2_mode!r}."
        )

    # ---- Single-stage contracting -------------------------------------------
    if training_mode == "combined":
        if phase2_mode != "negotiate":
            raise ValueError(
                f"TRAINING_MODE=combined runs the reference's contracting game every "
                f"episode, which is PHASE2_MODE=negotiate. 'solver' scores contracts "
                f"with a critic that is only meaningful once gameplay is frozen, and "
                f"'reinforce' is a phase-2 bandit over a fixed subgame. Got "
                f"PHASE2_MODE={phase2_mode!r}.")
        if phase1_from or phase1_only:
            raise ValueError(
                "PHASE1_FROM / PHASE1_ONLY are meaningless under "
                "TRAINING_MODE=combined: there is no phase split and nothing is "
                "frozen. Drop them, or use TRAINING_MODE=two_phase.")
        # One loop, so the whole budget is one kind of update. Recorded on the config
        # (rather than left at the 90/10 split computed above) so the sidecar, the
        # progress lines and checkpoint_filename all describe the run that ran.
        config["NUM_UPDATES_PHASE1"] = 0
        config["NUM_UPDATES_PHASE2"] = config["NUM_UPDATES"]
        if float(config.get("NULL_CONTRACT_FRAC", 0.0)) > 0.0:
            print(
                f"[MOCA] TRAINING_MODE=combined ignores NULL_CONTRACT_FRAC "
                f"({config['NULL_CONTRACT_FRAC']}): there is no P(Theta) to reweight. "
                f"Gameplay meets the null contract exactly as often as the "
                f"negotiation rejects, which is the arm's defining limitation and is "
                f"logged as combined/contract_null_frac.", flush=True)
    quorum_b = 0
    if training_mode == "joint":
        if phase1_from or phase1_only:
            raise ValueError(
                "PHASE1_FROM / PHASE1_ONLY are meaningless under TRAINING_MODE=joint: "
                "there is no phase split and no frozen policy. Drop them, or use "
                "TRAINING_MODE=two_phase."
            )
        seg = int(config.get("BARGAIN_SEGMENT", 100))
        inner = int(config["ENV_KWARGS"]["num_inner_steps"])
        if seg <= 0 or inner % seg != 0:
            raise ValueError(
                f"BARGAIN_SEGMENT ({seg}) must be a positive divisor of "
                f"num_inner_steps ({inner}), so the episode splits into whole rounds."
            )
        config["BARGAIN_SEGMENT"] = seg
        config["BARGAIN_ROUNDS"] = inner // seg
        config.setdefault("BARGAIN_PROPOSER", "rotate")
        if config["BARGAIN_PROPOSER"] not in bargain.PROPOSER_MODES:
            raise ValueError(
                f"BARGAIN_PROPOSER must be one of "
                f"{', '.join(bargain.PROPOSER_MODES)}, "
                f"got {config['BARGAIN_PROPOSER']!r}")
        config.setdefault("BARGAIN_FEATURES", "private")
        config.setdefault("BARGAIN_ROTATE_START", "random")
        if config["BARGAIN_ROTATE_START"] not in ("random", "fixed"):
            raise ValueError(
                f"BARGAIN_ROTATE_START must be 'random' or 'fixed', "
                f"got {config['BARGAIN_ROTATE_START']!r}")
        quorum_b = bargain.quorum_size(
            config.setdefault("BARGAIN_QUORUM", "all"), num_agents)
        for key, default in (("BARGAIN_LR", 3e-4), ("BARGAIN_UPDATE_EPOCHS", 4),
                             ("BARGAIN_CLIP_EPS", 0.2), ("BARGAIN_ENT_COEF", 0.01),
                             ("BARGAIN_VF_COEF", 0.5), ("BARGAIN_ACCEPT_BIAS", 1.0),
                             ("BARGAIN_GAE_LAMBDA", 0.95), ("BARGAIN_HIDDEN", 64),
                             ("BARGAIN_VOTE_EPS", 0.05),
                             ("BARGAIN_VOTE_EPS_END", 0.0),
                             ("BARGAIN_PROBE_FRAC", 0.0)):
            config.setdefault(key, default)
        if not 0.0 <= float(config["BARGAIN_VOTE_EPS"]) < 0.5:
            raise ValueError(
                f"BARGAIN_VOTE_EPS must be in [0, 0.5) -- it is a floor on BOTH "
                f"branches of the vote, so 0.5 is a coin flip and above it the floor "
                f"inverts. Got {config['BARGAIN_VOTE_EPS']!r}.")
        if not (0.0 <= float(config["BARGAIN_VOTE_EPS_END"])
                <= float(config["BARGAIN_VOTE_EPS"])):
            raise ValueError(
                f"BARGAIN_VOTE_EPS_END is the floor the anneal ends at, so it must "
                f"lie in [0, BARGAIN_VOTE_EPS={config['BARGAIN_VOTE_EPS']}]. Got "
                f"{config['BARGAIN_VOTE_EPS_END']!r}.")
        if not 0.0 <= float(config["BARGAIN_PROBE_FRAC"]) <= 0.5:
            raise ValueError(
                f"BARGAIN_PROBE_FRAC is the fraction of rounds whose offer is "
                f"replaced by a scripted probe; above 0.5 the run is mostly probing "
                f"rather than bargaining. Got {config['BARGAIN_PROBE_FRAC']!r}.")
        config.setdefault("BARGAIN_PROBE_NULL_FRAC", 0.2)
        if not 0.0 <= float(config["BARGAIN_PROBE_NULL_FRAC"]) <= 1.0:
            raise ValueError(
                f"BARGAIN_PROBE_NULL_FRAC is the fraction OF PROBES that offer the "
                f"null contract, so it must be in [0, 1]. Got "
                f"{config['BARGAIN_PROBE_NULL_FRAC']!r}.")
        # Claims and audits (reporting.py): contract payment on REPORTED cleaning.
        # Off by default -- transfers stay perfectly enforced unless asked.
        config.setdefault("REPORT_ENABLE", False)
        for key, default in (("REPORT_AUDIT_P", 0.25), ("REPORT_FINE_MULT", 2.0),
                             ("REPORT_MAX_OVERCLAIM", 20.0), ("REPORT_LR", 3e-4)):
            config.setdefault(key, default)
        if config["REPORT_ENABLE"]:
            p_audit = float(config["REPORT_AUDIT_P"])
            if not 0.0 < p_audit <= 1.0:
                raise ValueError(
                    f"REPORT_AUDIT_P must be in (0, 1] -- at 0 the cap is the only "
                    f"limit on overclaiming and the run measures nothing. Got "
                    f"{config['REPORT_AUDIT_P']!r}.")
            if float(config["REPORT_FINE_MULT"]) < 0.0:
                raise ValueError(f"REPORT_FINE_MULT must be >= 0, got "
                                 f"{config['REPORT_FINE_MULT']!r}.")
            if float(config["REPORT_MAX_OVERCLAIM"]) <= 0.0:
                raise ValueError(f"REPORT_MAX_OVERCLAIM must be > 0, got "
                                 f"{config['REPORT_MAX_OVERCLAIM']!r}.")
            print(f"[MOCA] reporting on: audit p={p_audit}, fine x"
                  f"{config['REPORT_FINE_MULT']} (honesty needs > "
                  f"{reporting.honesty_threshold(p_audit):.2f}), overclaim cap "
                  f"{config['REPORT_MAX_OVERCLAIM']}", flush=True)
        # How long an accepted contract binds. See bargain.apply_binding.
        config.setdefault("BARGAIN_BINDING", "episode")
        if config["BARGAIN_BINDING"] not in bargain.BINDING_MODES:
            raise ValueError(
                f"BARGAIN_BINDING must be one of "
                f"{', '.join(bargain.BINDING_MODES)}; got "
                f"{config['BARGAIN_BINDING']!r}. 'episode' is the original game -- "
                f"the first offer to carry binds for the rest of the episode. "
                f"'segment' renegotiates every segment with a null fallback, and "
                f"'sticky' renegotiates but leaves the incumbent contract in force "
                f"when a round fails. The renegotiating modes are NOT Rubinstein: "
                f"there is no shrinking pie and no delay cost.")
        # How the VOTE head is credited. The proposal head keeps the GAE advantage
        # either way -- a proposal has no pivotality and no branch structure.
        config.setdefault("BARGAIN_VOTE_ADVANTAGE", "gae")
        if config["BARGAIN_VOTE_ADVANTAGE"] not in VOTE_ADVANTAGE_MODES:
            raise ValueError(
                f"BARGAIN_VOTE_ADVANTAGE must be one of "
                f"{', '.join(VOTE_ADVANTAGE_MODES)}; got "
                f"{config['BARGAIN_VOTE_ADVANTAGE']!r}. 'gae' is the shared round "
                f"advantage (the default, and what every run before 2026-08-12 "
                f"used); 'counterfactual' credits each vote with pivotality x "
                f"(lock value - continue value) and adds two value heads to the "
                f"bargaining network.")
        # Which contracting game each round plays. "alternating" is everything
        # above: one proposer, a vote, a quorum. "median" is one simultaneous
        # move -- every agent names a theta, the median of the N asks binds for
        # the segment -- so the whole accept/reject apparatus (quorum, probes,
        # vote floor, vote credit, binding modes other than per-segment) has
        # nothing to act on. Those settings are REWRITTEN to what actually runs
        # rather than left as recorded lies in the sidecar: a median run is
        # per-segment by construction, and a vote that does not exist cannot
        # carry counterfactual credit or branch value heads.
        config.setdefault("BARGAIN_PROTOCOL", "alternating")
        if config["BARGAIN_PROTOCOL"] not in bargain.PROTOCOL_MODES:
            raise ValueError(
                f"BARGAIN_PROTOCOL must be one of "
                f"{', '.join(bargain.PROTOCOL_MODES)}; got "
                f"{config['BARGAIN_PROTOCOL']!r}. 'alternating' is the "
                f"proposer-and-vote game; 'median' binds the median of "
                f"everyone's simultaneous asks, with no vote.")
        _space = config.get("CONTRACT_SPACE") or spec.contract_space
        if (config["BARGAIN_PROTOCOL"] == "median"
                and contracts.CONTRACT_SPACES[_space].PARAM_DIM > 1):
            # The median mechanism binds the middle of N simultaneous asks. "The
            # middle" of a set of vectors is not defined -- a per-component median is
            # a contract nobody offered, which is exactly the object the mechanism's
            # argument says the median is not.
            raise ValueError(
                f"BARGAIN_PROTOCOL=median needs a one-dimensional contract: it binds "
                f"the median ask, and CONTRACT_SPACE={_space!r} offers "
                f"{contracts.CONTRACT_SPACES[_space].PARAM_DIM} numbers at once. "
                f"Use BARGAIN_PROTOCOL=alternating.")
        if config["BARGAIN_PROTOCOL"] == "median":
            forced = {"BARGAIN_BINDING": "segment",
                      "BARGAIN_VOTE_ADVANTAGE": "gae",
                      "BARGAIN_PROBE_FRAC": 0.0}
            overridden = {k: config[k] for k, v in forced.items()
                          if config.get(k) not in (None, v)}
            config.update(forced)
            if overridden:
                print(f"[MOCA] BARGAIN_PROTOCOL=median has no vote and no "
                      f"multi-round agreement, so these settings are inert and "
                      f"recorded as their forced values: {overridden} -> "
                      f"{ {k: forced[k] for k in overridden} }", flush=True)
        # Recorded so a replay tool can refuse a checkpoint it cannot read, rather
        # than reading it as a mechanism that was never trained.
        config["BARGAIN_FEATURE_VERSION"] = bargain.FEATURE_VERSION
        # Not a free hyperparameter: impatience is already realised as reward lost to
        # disagreement in the environment, so discounting rounds on top would count
        # the same delay cost twice and hand the proposer an advantage it has not
        # earned. See bargain.round_gae.
        if float(config.setdefault("BARGAIN_GAMMA", 1.0)) != 1.0:
            print(f"[MOCA warning] BARGAIN_GAMMA={config['BARGAIN_GAMMA']} != 1.0. "
                  f"Delay is already costly in-environment; an extra discount "
                  f"double-counts it.", flush=True)

    # Which (environment, arm) pairs actually exist. Deliberately AFTER the block
    # above, so it sees the defaults that block fills in rather than only what the
    # config file happened to state.
    envs.check_arm(spec, phase2_mode, training_mode, config)
    if phase2_mode == "negotiate":
        # Reference branch: 2 sampled non-proposers above 3 agents, all of them
        # below. Overridable, but the default is the rule as coded.
        # null in the config means "use the reference's rule for this num_agents".
        if config.get("NEGOTIATE_NU") is None:
            config["NEGOTIATE_NU"] = negotiate.default_nu(num_agents)
        if not 1 <= config["NEGOTIATE_NU"] <= num_agents - 1:
            raise ValueError(
                f"NEGOTIATE_NU must be in [1, {num_agents - 1}], "
                f"got {config['NEGOTIATE_NU']}"
            )
        # Appendix D of the extended paper. Its PPO settings are stated "for all
        # experiments in all domains", but they are applied to the negotiation
        # stage only: phase 1 here is this repo's existing CNN/Adam PPO setup,
        # shared with the IPPO arms, and retuning it to the paper's SGD/MLP
        # configuration would make the phase-1 policy incomparable to them.
        for key, value in (
            ("NEGOTIATE_LR", 1e-4),            # D.1: "a learning rate of 1e-4"
            ("NEGOTIATE_UPDATE_EPOCHS", 30),   # D.1: "30 SGD updates per iteration"
            ("NEGOTIATE_CLIP_EPS", 0.3),       # D.4: "clip parameter of 0.3"
            ("NEGOTIATE_VF_COEF", 1.0),        # D.4: "value function coefficient of 1.0"
        ):
            if config.get(key) is None:
                config[key] = value
    nu_neg = config.get("NEGOTIATE_NU") or negotiate.default_nu(num_agents)
    if phase2_mode == "solver":
        config.setdefault("SOLVER_SAMPLES", 50)
        config.setdefault("SOLVER_DECISION_RULE", "majority")
        if config["SOLVER_SAMPLES"] < 1:
            raise ValueError(f"SOLVER_SAMPLES must be >= 1, got {config['SOLVER_SAMPLES']}")
        if config["SOLVER_DECISION_RULE"] not in solver.DECISION_RULES:
            raise ValueError(
                f"SOLVER_DECISION_RULE must be one of "
                f"{', '.join(solver.DECISION_RULES)}, got "
                f"{config['SOLVER_DECISION_RULE']!r}"
            )

    # Number of non-proposers polled on a proposal. The paper does NOT poll every
    # agent: "we sample nu agents from the space of non-proposing agents, and only
    # use these agent's accept-reject probabilities in determining contract
    # acceptance", with nu=2 reported as strong across all domains.
    nu = int(config.get("VOTER_SAMPLE_NU", 2))
    if not 1 <= nu <= num_agents - 1:
        raise ValueError(
            f"VOTER_SAMPLE_NU must be in [1, num_agents-1] = [1, {num_agents - 1}], got {nu}"
        )
    config["VOTER_SAMPLE_NU"] = nu

    # Algorithm 1 takes one gradient step per EPISODE. A phase-2 update here plays
    # NUM_ENVS episodes in parallel, so folding them into a single step costs a
    # factor of NUM_ENVS in gradient steps -- and Adam displaces each logit by at
    # most CONTRACT_LR per step, capping total movement at
    # NUM_UPDATES_PHASE2 * CONTRACT_LR. Splitting each batch into sequential
    # minibatch updates restores the granularity Algorithm 1 assumes.
    n_contract_mb = int(config.get("CONTRACT_MINIBATCHES", 1))
    if config["NUM_ENVS"] % n_contract_mb != 0:
        raise ValueError(
            f"CONTRACT_MINIBATCHES={n_contract_mb} must divide NUM_ENVS={config['NUM_ENVS']}"
        )
    config["CONTRACT_MINIBATCHES"] = n_contract_mb
    logit_budget = config["NUM_UPDATES_PHASE2"] * n_contract_mb * config["CONTRACT_LR"]
    if phase2_mode == "reinforce" and logit_budget < 3.0:
        print(
            f"[MOCA warning] phase-2 logit budget is only "
            f"{config['NUM_UPDATES_PHASE2']} updates x {n_contract_mb} minibatches x "
            f"CONTRACT_LR={config['CONTRACT_LR']} = {logit_budget:.2f}. Adam moves a "
            f"logit by at most CONTRACT_LR per step, and a categorical needs logit "
            f"gaps of ~3-5 to depart from uniform, so the proposal policy cannot "
            f"converge. Raise CONTRACT_LR, CONTRACT_MINIBATCHES, or the phase-2 share.",
            flush=True,
        )

    # Renamed from NULL_CONTRACT_PROB when the i.i.d. per-env draw became an exact
    # stratified split. Refused rather than aliased: a run launched with the old
    # override would otherwise train at the default null fraction while its command
    # line and wandb config both claim otherwise -- and null mass is the variable
    # under test.
    if "NULL_CONTRACT_PROB" in config:
        raise ValueError(
            "NULL_CONTRACT_PROB was renamed to NULL_CONTRACT_FRAC (it is now an "
            "exact fraction of envs per update, not an i.i.d. draw probability). "
            "Update the override."
        )
    if not 0.0 <= config["NULL_CONTRACT_FRAC"] < 1.0:
        raise ValueError(
            f"NULL_CONTRACT_FRAC must be in [0, 1), got "
            f"{config['NULL_CONTRACT_FRAC']} -- phase 1 must see contracted play."
        )

    # What the bargained scalar means. Validated here rather than at the call site so
    # a typo is a message about CONTRACT_KIND rather than a missing-attribute error
    # a thousand lines later.
    contract_kind = config.setdefault("CONTRACT_KIND", "clean_wage")
    if contract_kind not in contracts.CONTRACT_KINDS:
        raise ValueError(
            f"CONTRACT_KIND must be one of {', '.join(contracts.CONTRACT_KINDS)}; "
            f"got {contract_kind!r}. 'clean_wage' pays theta per waste cell cleaned, "
            f"funded evenly by the others; 'harvest_tax' levies theta as a rate on "
            f"harvest income and shares the pot out by recent cleaning.")
    if contract_kind == "harvest_tax":
        # Every other code path calls compute_transfer(theta, cleaned), which the
        # tax contract cannot answer -- it needs the harvest and the trailing
        # window. Refuse up front rather than at the first phase-1 rollout.
        if config.get("TRAINING_MODE") != "joint" or phase2_mode != "bargain":
            raise ValueError(
                "CONTRACT_KIND=harvest_tax is implemented for the joint bargaining "
                "path only (TRAINING_MODE=joint, PHASE2_MODE=bargain). The phase-1 "
                "and one-shot paths compute transfers from cleaning alone and have "
                "no harvest base or trailing window to tax against.")
        window = int(config.setdefault("TAX_WINDOW", 20))
        if window < 1:
            raise ValueError(f"TAX_WINDOW must be >= 1 steps, got {window}")
        config["TAX_WINDOW"] = window
    # The contract space follows the environment. Overridable, but a mismatch is
    # refused: the space names which env info fields the transfer reads, so pairing
    # the wrong one with an env is either a crash or -- worse, if the fields happen
    # to exist -- a run that redistributes on the wrong quantity and looks fine.
    contract_space = config.setdefault("CONTRACT_SPACE", spec.contract_space)
    allowed_spaces = envs.CONTRACT_SPACES_BY_ENV.get(
        spec.env_name, (spec.contract_space,))
    if contract_space not in allowed_spaces:
        raise ValueError(
            f"CONTRACT_SPACE={contract_space!r} does not belong to "
            f"ENV_NAME={spec.env_name!r}, whose spaces are "
            f"{', '.join(allowed_spaces)}.")
    space_kwargs = {}
    if contract_space == "harvest_density":
        # The threshold's own range, separate from the fine's CONTRACT_LOW/HIGH.
        # Defaults come from the contract class (0..13, i.e. inert through flat tax).
        for cfg_key, arg in (("DENSITY_LOW", "density_low"),
                             ("DENSITY_HIGH", "density_high")):
            if config.get(cfg_key) is not None:
                space_kwargs[arg] = float(config[cfg_key])
    contract = make_contract(
        contract_space,
        num_agents,
        low=config["CONTRACT_LOW"],
        high=config["CONTRACT_HIGH"],
        kind=contract_kind,
        **space_kwargs,
    )
    # The discretised grid only exists for a one-dimensional space, and only
    # PHASE2_MODE=reinforce reads it. Built lazily so a 2-D space is not refused at
    # config time for an arm it is not running.
    contract_grid = (contract.grid(config["NUM_CONTRACT_BINS"])
                     if contract.PARAM_DIM == 1 else None)

    # How many numbers ONE offer is. The bargaining loop is the same game either way;
    # only the shape of the offer differs, so rather than branching all the way down,
    # these four carry the difference:
    #
    #   PDIM            1 for every space the reference defines.
    #   param_low/high  the bounds `unsquash`/`normalise_theta` map onto -- scalars at
    #                   PDIM=1, so those calls are untouched, (P,) vectors above it.
    #   _offer_shape    the shape of a per-env offer: (E,) or (E, P).
    #   _as_offer       a per-env decision, reshaped to select against an offer.
    #
    # At PDIM=1 each is the identity on what the code did before, which is what keeps
    # the scalar arms bit-for-bit unchanged (test_golden_arms).
    PDIM = contract.PARAM_DIM
    param_low = contract.low if PDIM == 1 else contract.param_low
    param_high = contract.high if PDIM == 1 else contract.param_high

    def _offer_shape(n_envs):
        return (n_envs,) if PDIM == 1 else (n_envs, PDIM)

    def _as_offer(cond):
        """(E,) -> (E, 1) when offers carry a parameter axis, else untouched.

        Without it a jnp.where against an (E, P) offer broadcasts the env axis
        against the parameter axis -- not an error when E == P, and silently the
        wrong contract when it is.
        """
        return cond if PDIM == 1 else cond[:, None]
    # Plain Python copy of the grid, purely for building metric NAMES. Formatting a
    # device array with float() fails under tracing, and label text must never depend
    # on a traced value anyway. Read off the grid itself rather than recomputed from
    # low/high: the grid is not a plain linspace when the range excludes weak
    # contracts (index 0 is then the null contract), and labels that disagree with it
    # would mislabel every proposal-probability series.
    # Empty for a multi-parameter space, which has no grid to label -- only
    # PHASE2_MODE=reinforce reads either, and it is refused for such a space.
    contract_grid_labels = ([] if contract_grid is None
                            else [float(x) for x in np.asarray(contract_grid)])

    env = LogWrapper(env, replace_info=False)

    def linear_schedule(count):
        # Denominator must be the number of updates that actually TOUCH the gameplay
        # policy. Two-phase: phase 1 only, since phase 2 freezes it. Joint: the whole
        # run. Using the phase-1 count under joint sent the rate through zero at 90%
        # and NEGATIVE for the rest -- gradient ascent on a loss containing
        # -ENT_COEF * entropy, which drives the policy deterministic and pins entropy
        # at exactly 0. jnp.maximum makes that unreachable however the counts are
        # configured; a negative learning rate is never the intent.
        total = (config["NUM_UPDATES"] if config.get("TRAINING_MODE") == "joint"
                 else config["NUM_UPDATES_PHASE1"])
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / total
        )
        return config["LR"] * jnp.maximum(frac, 0.0)

    def train(rng):
        progress_state = {"times": {}}

        # ------------------------------------------------------------ networks
        network = [
            ContractActorCritic(env.action_space().n, activation=config["ACTIVATION"])
            for _ in range(num_agents)
        ]
        proposal_net = [ProposalPolicy(config["NUM_CONTRACT_BINS"]) for _ in range(num_agents)]
        voting_net = [VotingPolicy(num_agents) for _ in range(num_agents)]

        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *(env.observation_space()[0]).shape))
        init_c = jnp.zeros((1, contract.obs_dim))
        init_onehot = jnp.zeros((1, num_agents))
        init_theta = jnp.zeros((1,))

        network_params = (
            loaded_phase1
            if loaded_phase1 is not None
            else [network[i].init(_rng, init_x, init_c) for i in range(num_agents)]
        )
        proposal_params = [proposal_net[i].init(_rng) for i in range(num_agents)]
        voting_params = [
            voting_net[i].init(_rng, init_onehot, init_theta) for i in range(num_agents)
        ]

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        contract_tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["CONTRACT_LR"], eps=1e-5),
        )

        train_state = [
            TrainState.create(apply_fn=network[i].apply, params=network_params[i], tx=tx)
            for i in range(num_agents)
        ]
        proposal_state = [
            TrainState.create(
                apply_fn=proposal_net[i].apply, params=proposal_params[i], tx=contract_tx
            )
            for i in range(num_agents)
        ]
        voting_state = [
            TrainState.create(
                apply_fn=voting_net[i].apply, params=voting_params[i], tx=contract_tx
            )
            for i in range(num_agents)
        ]

        # Learned negotiation stage. A separate network per agent over the Box
        # action [contract params..., accept_prob], trained by its own PPO -- the
        # reference builds a second RLlib trainer for the negotiation env rather
        # than reusing the gameplay policy.
        negotiate_net = [
            NegotiationActorCritic(2, activation=config["ACTIVATION"])
            for _ in range(num_agents)
        ]
        negotiate_params = [
            negotiate_net[i].init(_rng, init_x, init_c) for i in range(num_agents)
        ]
        negotiate_tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config.get("NEGOTIATE_LR") or 1e-4, eps=1e-5),
        )
        negotiate_state = [
            TrainState.create(
                apply_fn=negotiate_net[i].apply, params=negotiate_params[i], tx=negotiate_tx
            )
            for i in range(num_agents)
        ]

        # Rubinstein bargaining policies. Built unconditionally (cheap, and keeps the
        # PRNG stream identical across modes) but only trained under TRAINING_MODE=joint.
        # The branch value heads exist only where they are used: with
        # BARGAIN_VOTE_ADVANTAGE=gae the module is exactly what it was before they
        # were written, down to the parameter tree.
        cf_vote = config.get("BARGAIN_VOTE_ADVANTAGE", "gae") == "counterfactual"
        # Read once here rather than per-function: the rollout and the metrics both
        # branch on it, and they must never disagree about which game was played.
        binding = config.get("BARGAIN_BINDING", "episode")
        # Median protocol: simultaneous asks, the median binds, no vote. make_train
        # already forced binding="segment" and vote advantage="gae" for it, so
        # everything downstream of the rollout needs no median-specific branches.
        median_protocol = config.get("BARGAIN_PROTOCOL", "alternating") == "median"
        # Whether the bargained scalar is a tax rate rather than a cleaning wage.
        # Python-level, so with clean_wage every branch below compiles to exactly
        # the graph it did before the tax existed and no key is drawn differently.
        tax_kind = config.get("CONTRACT_KIND", "clean_wage") == "harvest_tax"
        bargain_net = [
            BargainingActorCritic(
                hidden=int(config.get("BARGAIN_HIDDEN", 64)),
                activation=config["ACTIVATION"],
                accept_bias=float(config.get("BARGAIN_ACCEPT_BIAS", 1.0)),
                aux_heads=cf_vote,
                param_dim=contract.PARAM_DIM,
            )
            for _ in range(num_agents)
        ]
        # Width follows the contract: the two offer slots hold one column per
        # contract parameter, so a 2-D space widens the state the policy reads.
        init_b = jnp.zeros((1, bargain.feature_dim(num_agents, PDIM)))
        bargain_tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config.get("BARGAIN_LR") or 3e-4, eps=1e-5),
        )
        bargain_state = [
            TrainState.create(
                apply_fn=bargain_net[i].apply,
                params=bargain_net[i].init(_rng, init_b),
                tx=bargain_tx,
            )
            for i in range(num_agents)
        ]

        # Claim policies (reporting.py), built only when reporting is on. The init
        # reuses `_rng` like every network above rather than splitting a fresh key,
        # so the main PRNG stream is identical whether or not reporting exists.
        report_on = bool(config.get("REPORT_ENABLE", False))
        claim_net, claim_state = None, None
        if report_on:
            claim_net = [reporting.ClaimPolicy() for _ in range(num_agents)]
            init_cl = jnp.zeros((1, reporting.FEATURE_DIM))
            claim_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config.get("REPORT_LR") or 3e-4, eps=1e-5),
            )
            claim_state = [
                TrainState.create(
                    apply_fn=claim_net[i].apply,
                    params=claim_net[i].init(_rng, init_cl),
                    tx=claim_tx,
                )
                for i in range(num_agents)
            ]

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # ------------------------------------------------- shared rollout logic
        def rollout(params_list, env_state, last_obs, theta, rng):
            """Play one full episode under contract `theta` (one value per env).

            Returns the trajectory batch plus the final env/obs state. Rewards are
            contract-augmented: R'_i = R_i + transfer_i, with transfers zero-sum
            across agents.
            """
            contract_obs = contract.to_obs(theta)  # (NUM_ENVS, contract_obs_dim)

            def _env_step(carry, unused):
                env_state, last_obs, rng = carry
                rng, _rng = jax.random.split(rng)

                obs_batch = jnp.transpose(last_obs, (1, 0, 2, 3, 4))
                env_act, log_prob, value = {}, [], []
                act_keys = jax.random.split(_rng, num_agents)
                for i in range(num_agents):
                    pi, value_i = network[i].apply(params_list[i], obs_batch[i], contract_obs)
                    action = pi.sample(seed=act_keys[i])
                    log_prob.append(pi.log_prob(action))
                    env_act[env.agents[i]] = action
                    value.append(value_i)
                env_act_list = [v for v in env_act.values()]

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act_list)

                # Contract transfers: zero-sum redistribution on top of base reward.
                # Which env signals the contract reads is the contract's own business
                # (contracts.ScalarContract.SIGNAL_KEYS), so this line is the same on
                # every environment.
                transfers = contract.transfer_from_info(theta, info)
                reward = reward + transfers
                info = dict(info)
                info["contract_transfer"] = transfers
                info["contract_theta"] = jnp.broadcast_to(
                    theta[:, None], transfers.shape
                )

                done_list = [v for v in done.values()]
                transition = []
                for i in range(num_agents):
                    info_i = {
                        k: jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"]), 1), v[:, i])
                        for k, v in info.items()
                    }
                    transition.append(
                        MOCATransition(
                            done_list[i],
                            env_act_list[i],
                            value[i],
                            reward[:, i],
                            log_prob[i],
                            obs_batch[i],
                            contract_obs,
                            info_i,
                        )
                    )
                return (env_state, obsv, rng), transition

            (env_state, last_obs, rng), traj_batch = jax.lax.scan(
                _env_step, (env_state, last_obs, rng), None, config["NUM_STEPS"]
            )
            return traj_batch, env_state, last_obs, rng

        def compute_gae(traj_batch_i, last_val_i):
            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                done, value, reward = transition.done, transition.value, transition.reward
                delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val_i), last_val_i),
                traj_batch_i,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch_i.value

        def gameplay_bootstrap(train_state, last_obs, theta):
            """Critic value at the state the rollout ended in, for GAE's bootstrap."""
            contract_obs = contract.to_obs(theta)
            last_obs_batch = jnp.transpose(last_obs, (1, 0, 2, 3, 4))
            last_val = []
            for i in range(num_agents):
                _, v = network[i].apply(
                    train_state[i].params, last_obs_batch[i], contract_obs)
                last_val.append(v)
            return last_val

        def gameplay_ppo_update(train_state, traj_batch, last_val, rng):
            """One PPO update of the contract-conditioned gameplay policy, per agent.

            Shared by phase 1 and by single-stage contracting, which differ only in
            where their theta came from -- P(Theta) in one, the negotiation in the
            other -- and not at all in how the policy is then trained on it. Kept
            byte-for-byte the phase-1 body it was lifted out of, including the order
            in which keys are split: PRNG draw order is load-bearing here and
            tests/test_golden_arms.py pins the published arms against it.

            Returns (train_state, rng, per-agent metric dicts).
            """
            def _loss_fn(params, traj_batch, gae, targets, net):
                pi, value = net.apply(params, traj_batch.obs, traj_batch.contract)
                log_prob = pi.log_prob(traj_batch.action)
                value_pred_clipped = traj_batch.value + (
                    value - traj_batch.value
                ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                value_losses = jnp.square(value - targets)
                value_losses_clipped = jnp.square(value_pred_clipped - targets)
                value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                ratio = jnp.exp(log_prob - traj_batch.log_prob)
                gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                loss_actor = -jnp.minimum(
                    ratio * gae,
                    jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae,
                ).mean()
                entropy = pi.entropy().mean()
                total = loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                return total, (value_loss, loss_actor, entropy)

            metric = []
            for i in range(num_agents):
                advantages_i, targets_i = compute_gae(traj_batch[i], last_val[i])

                def _update_epoch(update_state, unused, i=i):
                    def _update_minbatch(ts, batch_info):
                        tb, adv, tgt = batch_info
                        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                        loss, grads = grad_fn(ts.params, tb, adv, tgt, network[i])
                        return ts.apply_gradients(grads=grads), loss

                    ts, tb, adv, tgt, rng = update_state
                    rng, _rng = jax.random.split(rng)
                    batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                    permutation = jax.random.permutation(_rng, batch_size)
                    batch = (tb, adv, tgt)
                    batch = jax.tree_util.tree_map(
                        lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                    )
                    shuffled = jax.tree_util.tree_map(
                        lambda x: jnp.take(x, permutation, axis=0), batch
                    )
                    minibatches = jax.tree_util.tree_map(
                        lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])),
                        shuffled,
                    )
                    ts, loss_info = jax.lax.scan(_update_minbatch, ts, minibatches)
                    return (ts, tb, adv, tgt, rng), loss_info

                update_state = (train_state[i], traj_batch[i], advantages_i, targets_i, rng)
                update_state, loss_info = jax.lax.scan(
                    _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
                )
                train_state[i] = update_state[0]
                rng = update_state[-1]

                metric_i = dict(traj_batch[i].info)
                metric_i["loss"] = loss_info[0]
                metric_i["value_loss"] = loss_info[1][0]
                metric_i["actor_loss"] = loss_info[1][1]
                metric_i["entropy"] = loss_info[1][2]
                metric.append(metric_i)

            return train_state, rng, metric

        def gameplay_metrics(traj_batch, metric):
            """Pooled per-agent info + PPO losses, as _mean/_std over agents."""
            metric = jax.tree.map(lambda x: x.mean(), metric)
            keys = list(metric[0].keys())
            stacked = {k: jnp.stack([d[k] for d in metric]) for k in keys}
            out = {}
            for k, v in stacked.items():
                out[f"{k}_mean"] = v.mean()
                out[f"{k}_std"] = v.std()
            stats = episode_stats(traj_batch, num_agents)
            out["welfare"] = stats["welfare"]
            out["equality"] = stats["equality"]
            out["transfer_volume"] = stats["transfer_volume"]
            return out, stats

        # =================================================================
        # PHASE 1 -- learn to PLAY under contracts drawn from P(Theta)
        # =================================================================
        def _update_step_phase1(runner_state, unused):
            train_state, env_state, last_obs, update_step, rng = runner_state

            # Contract for this episode, independent of any proposal policy.
            # Stratified with an exact null block -- see CleanupContract.sample_batch.
            rng, _rng = jax.random.split(rng)
            theta = contract.sample_batch(
                _rng, config["NUM_ENVS"], null_frac=config["NULL_CONTRACT_FRAC"]
            )

            params_list = [ts.params for ts in train_state]
            traj_batch, env_state, last_obs, rng = rollout(
                params_list, env_state, last_obs, theta, rng
            )

            last_val = gameplay_bootstrap(train_state, last_obs, theta)
            train_state, rng, metric = gameplay_ppo_update(
                train_state, traj_batch, last_val, rng)

            update_step = update_step + 1
            jax.debug.callback(checkpoint_callback, train_state, update_step)

            out, stats = gameplay_metrics(traj_batch, metric)
            out["contract_theta_sampled"] = theta.mean()

            out.update(regime_split(traj_batch, stats, theta, contract,
                                    num_agents, spec))

            # Namespaced by stage, as the reference logger does (stage_1/..., stage_2/...):
            # the two phases measure different things, so sharing a key would splice a
            # subgame-learning curve onto a contract-negotiation curve.
            out = {f"stage_1/{k}": v for k, v in _select(out, STAGE1_METRICS).items()}
            out["phase"] = jnp.float32(1.0)
            out["update_step"] = update_step
            out["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            jax.debug.callback(log_callback, out)
            jax.debug.callback(
                progress_callback, update_step,
                out[f"stage_1/{spec.progress_metric}_mean"], 1
            )

            return (train_state, env_state, last_obs, update_step, rng), out

        # =================================================================
        # PHASE 2 -- with gameplay FROZEN, learn WHICH contract to sign
        # =================================================================
        def _update_step_phase2(runner_state, unused):
            (frozen_params, proposal_state, voting_state,
             env_state, last_obs, update_step, rng) = runner_state

            # -- contracting stage: propose, then vote --
            rng, k_prop, k_idx, k_vote, k_sel = jax.random.split(rng, 5)
            proposer = jax.random.randint(k_prop, (config["NUM_ENVS"],), 0, num_agents)
            proposer_onehot = jax.nn.one_hot(proposer, num_agents)

            # theta ~ pi_p(p, 0) for whichever agent p proposes in each env
            all_logits = jnp.stack(
                [proposal_net[i].apply(proposal_state[i].params).logits for i in range(num_agents)]
            )                                        # (N, K)
            sel_logits = all_logits[proposer]        # (NUM_ENVS, K)
            idx = jax.random.categorical(k_idx, sel_logits)      # (NUM_ENVS,)
            theta_prop = contract_grid[idx]
            theta_norm = (theta_prop - contract.low) / (contract.high - contract.low)

            # Sample nu voters uniformly WITHOUT replacement from the non-proposers;
            # only their accept/reject probabilities bind. Polling all N-1 agents
            # instead makes acceptance a product of N-1 roughly-even probabilities --
            # 0.5**6 = 1.6% at N=7, since VotingPolicy initialises near 50/50 -- so
            # almost every episode falls back to the null contract and the proposal
            # gradient is dominated by rollouts in which theta had no effect at all.
            u = jax.random.uniform(k_sel, (num_agents, config["NUM_ENVS"]))
            u = jnp.where(
                jnp.arange(num_agents)[:, None] == proposer[None, :], 2.0, u
            )                                            # proposer is never sampled
            rank = jnp.argsort(jnp.argsort(u, axis=0), axis=0)
            voter_mask = rank < nu                       # (N, NUM_ENVS)

            vote_keys = jax.random.split(k_vote, num_agents)
            votes = []
            accept_all = jnp.ones((config["NUM_ENVS"],), dtype=bool)
            for j in range(num_agents):
                pi_j = voting_net[j].apply(voting_state[j].params, proposer_onehot, theta_norm)
                v_j = pi_j.sample(seed=vote_keys[j])     # 0 = reject, 1 = accept
                votes.append(v_j)
                # Unsampled agents and the proposer do not get a veto.
                accept_all = accept_all & jnp.where(voter_mask[j], v_j == 1, True)
            votes = jnp.stack(votes)                        # (N, NUM_ENVS)

            # Rejected proposals fall back to the null contract (no transfers).
            theta_eff = jnp.where(accept_all, theta_prop, jnp.float32(contract.null))

            # -- play the episode with the FROZEN gameplay policy --
            traj_batch, env_state, last_obs, rng = rollout(
                frozen_params, env_state, last_obs, theta_eff, rng
            )
            # Episode return per agent -- the payoff the contracting policies optimise.
            returns = jnp.stack(
                [traj_batch[i].reward.sum(axis=0) for i in range(num_agents)]
            )                                                # (N, NUM_ENVS)

            # -- REINFORCE on the contracting policies --
            # One decision per episode, so this is an episode-level bandit; a
            # batch-mean baseline per agent keeps the gradient variance manageable.
            baseline = returns.mean(axis=1, keepdims=True)
            adv = returns - baseline                          # (N, NUM_ENVS)

            def proposal_loss(params, i, b):
                pi = proposal_net[i].apply(params)
                logp = pi.log_prob(b["idx"])
                # only envs where agent i actually proposed contribute
                mask = (b["proposer"] == i).astype(jnp.float32)
                n = jnp.maximum(mask.sum(), 1.0)
                return -(logp * b["adv"][:, i] * mask).sum() / n

            def voting_loss(params, j, b):
                pi = voting_net[j].apply(params, b["proposer_onehot"], b["theta_norm"])
                logp = pi.log_prob(b["votes"][:, j])
                # Only envs where j was one of the nu sampled voters: elsewhere its
                # vote was discarded, so it cannot be credited or blamed for it.
                mask = b["voter_mask"][:, j].astype(jnp.float32)
                n = jnp.maximum(mask.sum(), 1.0)
                return -(logp * b["adv"][:, j] * mask).sum() / n

            # Sequential minibatch updates, recovering the per-episode gradient
            # granularity of Algorithm 1 (see CONTRACT_MINIBATCHES in make_train).
            n_mb, mb = config["CONTRACT_MINIBATCHES"], config["NUM_ENVS"] // config["CONTRACT_MINIBATCHES"]
            mb_batch = {
                "idx": idx.reshape(n_mb, mb),
                "proposer": proposer.reshape(n_mb, mb),
                "proposer_onehot": proposer_onehot.reshape(n_mb, mb, num_agents),
                "theta_norm": theta_norm.reshape(n_mb, mb),
                "votes": votes.T.reshape(n_mb, mb, num_agents),
                "voter_mask": voter_mask.T.reshape(n_mb, mb, num_agents),
                "adv": adv.T.reshape(n_mb, mb, num_agents),
            }

            def _contract_minibatch(states, b):
                proposal_state, voting_state = states
                for i in range(num_agents):
                    g = jax.grad(proposal_loss)(proposal_state[i].params, i, b)
                    proposal_state[i] = proposal_state[i].apply_gradients(grads=g)
                    gv = jax.grad(voting_loss)(voting_state[i].params, i, b)
                    voting_state[i] = voting_state[i].apply_gradients(grads=gv)
                return (proposal_state, voting_state), None

            (proposal_state, voting_state), _ = jax.lax.scan(
                _contract_minibatch, (proposal_state, voting_state), mb_batch
            )

            update_step = update_step + 1
            jax.debug.callback(
                contract_checkpoint_callback, proposal_state, voting_state, update_step
            )

            metric = jax.tree.map(lambda x: x.mean(), [dict(traj_batch[i].info) for i in range(num_agents)])
            keys = list(metric[0].keys())
            stacked = {k: jnp.stack([d[k] for d in metric]) for k in keys}
            out = {}
            for k, v in stacked.items():
                out[f"{k}_mean"] = v.mean()
                out[f"{k}_std"] = v.std()
            # The headline MOCA diagnostics: what is being proposed, and is it signed?
            out["contract_theta_proposed"] = theta_prop.mean()
            out["contract_theta_effective"] = theta_eff.mean()
            out["contract_accept_rate"] = accept_all.mean()
            out["contract_returns_mean"] = returns.mean()
            stats = episode_stats(traj_batch, num_agents)
            out["welfare"] = stats["welfare"]
            out["equality"] = stats["equality"]
            out["transfer_volume"] = stats["transfer_volume"]
            # Spread across envs is the REINFORCE signal itself: the advantage is
            # returns minus their per-agent batch mean, so if every env returns the
            # same the contracting policies get an exactly zero gradient.
            out["contract_returns_std"] = returns.std()
            out["contract_adv_absmean"] = jnp.abs(adv).mean()
            # Recomputed POST-update so the logged distribution is the current one.
            post_logits = jnp.stack(
                [proposal_net[i].apply(proposal_state[i].params).logits for i in range(num_agents)]
            )                                                # (N, K)
            per_agent_probs = jax.nn.softmax(post_logits, axis=-1)
            probs = per_agent_probs.mean(axis=0)              # (K,)
            out["contract_proposal_argmax"] = contract_grid[jnp.argmax(probs)]
            # Entropy is the convergence diagnostic for phase 2: it starts at log(K)
            # and must FALL for the proposal policy to have learned anything. A run
            # that ends at log(K) has learned nothing, whatever its argmax says.
            out["contract_proposal_entropy"] = -(
                per_agent_probs * jnp.log(per_agent_probs + 1e-12)
            ).sum(axis=-1).mean()
            out["contract_proposal_entropy_max"] = jnp.log(
                jnp.float32(config["NUM_CONTRACT_BINS"])
            )
            # Share of sampled voters that accepted, independent of whether the
            # proposal cleared unanimity among them.
            out["contract_voter_accept_rate"] = (
                (votes * voter_mask).sum() / jnp.maximum(voter_mask.sum(), 1)
            )
            # Per-bin proposal mass. NOT logged per step -- it is NUM_CONTRACT_BINS
            # series on its own, and contract_proposal_entropy summarises the same
            # thing in one number. _runner.single_run prints the full converged
            # distribution from the saved proposal checkpoints at the end of a run.
            for k in range(config["NUM_CONTRACT_BINS"]):
                out[f"proposal_p_theta{contract_grid_labels[k]:.3f}"] = probs[k]
            out = {f"stage_2/{k}": v for k, v in _select(out, STAGE2_METRICS).items()}
            out["phase"] = jnp.float32(2.0)
            out["update_step"] = update_step
            out["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            jax.debug.callback(log_callback, out)
            jax.debug.callback(
                progress_callback, update_step, out["stage_2/welfare"], 2
            )

            return (frozen_params, proposal_state, voting_state,
                    env_state, last_obs, update_step, rng), out

        # =================================================================
        # PHASE 2 (solver) -- no proposal game: sample contracts, score them
        # with the frozen critic, pick one by the decision rule, play it.
        # This is what the paper's Cleanup experiments actually ran; see
        # algorithms/MOCA/solver.py for the provenance.
        # =================================================================
        def _update_step_phase2_solver(runner_state, unused):
            (frozen_params, env_state, last_obs, update_step, rng) = runner_state

            rng, k_neg = jax.random.split(rng)
            # last_obs is the initial state of the episode about to be played: one
            # rollout is exactly one episode and the env auto-resets at the end, so
            # the observation carried out of the previous rollout is s_0 here.
            obs_batch = jnp.transpose(last_obs, (1, 0, 2, 3, 4))
            theta_eff, solver_info = solver.negotiate(
                k_neg, network, frozen_params, obs_batch, contract,
                config["SOLVER_SAMPLES"], config["SOLVER_DECISION_RULE"],
            )

            traj_batch, env_state, last_obs, rng = rollout(
                frozen_params, env_state, last_obs, theta_eff, rng
            )

            update_step = update_step + 1

            metric = jax.tree.map(
                lambda x: x.mean(), [dict(traj_batch[i].info) for i in range(num_agents)]
            )
            keys = list(metric[0].keys())
            stacked = {k: jnp.stack([d[k] for d in metric]) for k in keys}
            out = {}
            for k, v in stacked.items():
                out[f"{k}_mean"] = v.mean()
                out[f"{k}_std"] = v.std()
            out["contract_theta_effective"] = theta_eff.mean()
            # Spread across envs: each env negotiates its own episode, so this says
            # whether the solver converges on one contract or keeps disagreeing.
            out["contract_theta_std"] = theta_eff.std()
            out.update(solver_info)
            stats = episode_stats(traj_batch, num_agents)
            out["welfare"] = stats["welfare"]
            out["equality"] = stats["equality"]
            out["transfer_volume"] = stats["transfer_volume"]
            out = {f"stage_2/{k}": v for k, v in _select(out, STAGE2_SOLVER_METRICS).items()}
            out["phase"] = jnp.float32(2.0)
            out["update_step"] = update_step
            out["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            jax.debug.callback(log_callback, out)
            jax.debug.callback(
                progress_callback, update_step, out["stage_2/welfare"], 2
            )

            return (frozen_params, env_state, last_obs, update_step, rng), out

        # =================================================================
        # THE CONTRACTING GAME (Algorithm 1's stage 2, and the whole of
        # single-stage contracting): agent 0 proposes, nu sampled agents accept
        # with some probability. See algorithms/MOCA/negotiate.py.
        # =================================================================
        # Split out of the phase-2 body so `combined` runs the SAME protocol rather
        # than a lookalike -- an arm comparison in which the two arms negotiate
        # slightly differently measures the difference between the implementations,
        # not between the algorithms. The key splits happen in the order they always
        # did, which tests/test_golden_arms.py pins.
        def negotiate_contract(negotiate_state, obs_batch, rng):
            """Run the two-step contracting game once per env.

            Returns (record, rng), where `record` carries both what was agreed and
            everything the PPO update needs to credit it.
            """
            low, high = contract.low, contract.high
            n_envs = config["NUM_ENVS"]

            def act(params_list, contract_obs, keys):
                """Every agent's action, log-prob and value at one negotiation step."""
                raws, logps, vals = [], [], []
                for i in range(num_agents):
                    pi, v = negotiate_net[i].apply(
                        params_list[i], obs_batch[i], contract_obs
                    )
                    raw = pi.sample(seed=keys[i])
                    raws.append(raw)
                    logps.append(pi.log_prob(raw))
                    vals.append(v)
                return jnp.stack(raws), jnp.stack(logps), jnp.stack(vals)

            params_list = [ts.params for ts in negotiate_state]
            rng, k0, k1, k_sel, k_acc = jax.random.split(rng, 5)

            # -- step 0: PROPOSE. Only agent 0's contract component is read.
            obs_propose = contract.to_obs(jnp.zeros((n_envs,)), stage=PROPOSE)
            raw0, logp0, val0 = act(params_list, obs_propose, jax.random.split(k0, num_agents))
            theta_prop = negotiate.unsquash(raw0[0, :, 0], low, high)

            # -- step 1: AGREE. nu sampled non-proposers gate the contract.
            obs_agree = contract.to_obs(theta_prop, stage=AGREE)
            raw1, logp1, val1 = act(params_list, obs_agree, jax.random.split(k1, num_agents))
            accept_probs = negotiate.unsquash(raw1[:, :, 1], 0.0, 1.0)   # (N, E)
            voter_mask = negotiate.sample_voters(k_sel, num_agents, nu_neg, n_envs)
            accepted, prod_prob = negotiate.acceptance(k_acc, accept_probs, voter_mask)
            theta_eff = jnp.where(accepted, theta_prop, jnp.float32(contract.null))

            return NegotiationRecord(
                theta_prop=theta_prop, theta_eff=theta_eff, accepted=accepted,
                prod_prob=prod_prob, obs_propose=obs_propose, obs_agree=obs_agree,
                raws=jnp.stack([raw0, raw1]),                       # (2, N, E, A)
                logps=jnp.stack([logp0, logp1]),                    # (2, N, E)
                values=jnp.stack([val0, val1]),                     # (2, N, E)
            ), rng

        def negotiate_ppo_update(negotiate_state, obs_batch, rec, returns):
            """PPO on the two-step negotiation episode, with the episode as reward."""
            # Reward lands only on the agreement step, as in the reference: the
            # proposal step returns zeros and the agreement step returns the
            # accumulated episode reward.
            rewards = jnp.stack([jnp.zeros_like(returns), returns])   # (2, N, E)
            advantages, targets = negotiate.two_step_gae(
                rewards, rec.values, config["GAMMA"], config["GAE_LAMBDA"]
            )

            def ppo_loss(params, i):
                adv_i = advantages[:, i]
                adv_i = (adv_i - adv_i.mean()) / (adv_i.std() + 1e-8)
                loss = 0.0
                for t, cobs in enumerate((rec.obs_propose, rec.obs_agree)):
                    pi, value = negotiate_net[i].apply(params, obs_batch[i], cobs)
                    logp = pi.log_prob(rec.raws[t, i])
                    ratio = jnp.exp(logp - rec.logps[t, i])
                    a = adv_i[t]
                    clip_eps = config["NEGOTIATE_CLIP_EPS"]
                    actor = -jnp.minimum(
                        ratio * a,
                        jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * a,
                    ).mean()
                    v_loss = jnp.square(value - targets[t, i]).mean()
                    # No entropy bonus: Appendix D.4 gives an entropy coefficient of
                    # 0.0, so exploration in the contract space comes from the
                    # Gaussian's own learned scale rather than an added term.
                    loss = loss + actor + config["NEGOTIATE_VF_COEF"] * v_loss
                return loss

            def _epoch(state, unused):
                for i in range(num_agents):
                    g = jax.grad(ppo_loss)(state[i].params, i)
                    state[i] = state[i].apply_gradients(grads=g)
                return state, None

            negotiate_state, _ = jax.lax.scan(
                _epoch, negotiate_state, None, config["NEGOTIATE_UPDATE_EPOCHS"]
            )
            return negotiate_state

        def negotiate_metrics(negotiate_state, obs_batch, rec) -> dict:
            """What was offered, what took force, and whether the proposal is settling."""
            return {
                "contract_theta_proposed": rec.theta_prop.mean(),
                "contract_theta_effective": rec.theta_eff.mean(),
                "contract_accept_rate": rec.accepted.mean(),
                "contract_accept_prob": rec.prod_prob.mean(),
                # Gaussian entropy of the proposer's policy, recomputed post-update:
                # the analogue of the categorical entropy the REINFORCE mode tracks,
                # and the same convergence question -- is the proposal narrowing?
                "negotiate_policy_entropy": negotiate_net[0].apply(
                    negotiate_state[0].params, obs_batch[0], rec.obs_propose
                )[0].entropy().mean(),
            }

        # =================================================================
        # PHASE 2 (negotiate) -- the contracting game against a FROZEN policy.
        # =================================================================
        def _update_step_phase2_negotiate(runner_state, unused):
            (frozen_params, negotiate_state, env_state, last_obs,
             update_step, rng) = runner_state

            obs_batch = jnp.transpose(last_obs, (1, 0, 2, 3, 4))   # (N, E, ...)
            rec, rng = negotiate_contract(negotiate_state, obs_batch, rng)

            # -- play the episode with the FROZEN gameplay policy --
            traj_batch, env_state, last_obs, rng = rollout(
                frozen_params, env_state, last_obs, rec.theta_eff, rng
            )
            returns = jnp.stack(
                [traj_batch[i].reward.sum(axis=0) for i in range(num_agents)]
            )                                                    # (N, E)

            negotiate_state = negotiate_ppo_update(
                negotiate_state, obs_batch, rec, returns)

            update_step = update_step + 1
            jax.debug.callback(
                negotiate_checkpoint_callback, negotiate_state, update_step
            )

            metric = jax.tree.map(
                lambda x: x.mean(), [dict(traj_batch[i].info) for i in range(num_agents)]
            )
            keys = list(metric[0].keys())
            stacked = {k: jnp.stack([d[k] for d in metric]) for k in keys}
            out = {}
            for k, v in stacked.items():
                out[f"{k}_mean"] = v.mean()
                out[f"{k}_std"] = v.std()
            out.update(negotiate_metrics(negotiate_state, obs_batch, rec))
            stats = episode_stats(traj_batch, num_agents)
            out["welfare"] = stats["welfare"]
            out["equality"] = stats["equality"]
            out["transfer_volume"] = stats["transfer_volume"]
            out = {f"stage_2/{k}": v
                   for k, v in _select(out, STAGE2_NEGOTIATE_METRICS).items()}
            out["phase"] = jnp.float32(2.0)
            out["update_step"] = update_step
            out["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            jax.debug.callback(log_callback, out)
            jax.debug.callback(
                progress_callback, update_step, out["stage_2/welfare"], 2
            )

            return (frozen_params, negotiate_state, env_state, last_obs,
                    update_step, rng), out

        # =================================================================
        # SINGLE-STAGE ("vanilla") CONTRACTING -- no phase split, nothing frozen
        # =================================================================
        # The reference's `SeparateContractCombinedStage`: every episode opens with
        # the same two-step contracting game, the episode is then played under
        # whatever was agreed, and BOTH policies take a gradient step on it. Every
        # round is a negotiated round.
        #
        # The comparison this arm exists for is against MOCA, and the difference is
        # entirely in what each policy is trained against:
        #
        #   * gameplay sees only contracts the current proposer likes, so V_i(s, theta)
        #     is learned on-distribution rather than across the contract space;
        #   * the proposer is scored against a gameplay policy that is still changing,
        #     so a contract that looks good now may not be good against the policy
        #     that contract eventually produces;
        #   * the null contract appears only when the negotiation fails, so the
        #     disagreement point every acceptance decision is implicitly measured
        #     against may be barely represented at all.
        #
        # None of those are bugs to fix here. They are what MOCA's two phases buy,
        # and the size of the gap is the result.
        def _update_step_combined(runner_state, unused):
            (train_state, negotiate_state, env_state, last_obs,
             update_step, rng) = runner_state

            obs_batch = jnp.transpose(last_obs, (1, 0, 2, 3, 4))   # (N, E, ...)
            rec, rng = negotiate_contract(negotiate_state, obs_batch, rng)

            # -- play the episode with the CURRENT, still-learning policy --
            params_list = [ts.params for ts in train_state]
            traj_batch, env_state, last_obs, rng = rollout(
                params_list, env_state, last_obs, rec.theta_eff, rng
            )
            returns = jnp.stack(
                [traj_batch[i].reward.sum(axis=0) for i in range(num_agents)]
            )                                                    # (N, E)

            # Both halves learn from the same episode. Gameplay first, so its update
            # consumes keys in the position phase 1 established.
            last_val = gameplay_bootstrap(train_state, last_obs, rec.theta_eff)
            train_state, rng, metric = gameplay_ppo_update(
                train_state, traj_batch, last_val, rng)
            negotiate_state = negotiate_ppo_update(
                negotiate_state, obs_batch, rec, returns)

            update_step = update_step + 1
            jax.debug.callback(checkpoint_callback, train_state, update_step)
            jax.debug.callback(
                negotiate_checkpoint_callback, negotiate_state, update_step
            )

            out, stats = gameplay_metrics(traj_batch, metric)
            out.update(negotiate_metrics(negotiate_state, obs_batch, rec))
            # The null/contracted split is the same diagnostic phase 1 runs, but here
            # the split is produced by REJECTIONS rather than by P(Theta) -- so
            # contract_null_frac is an outcome of the run, not a setting, and a run
            # that drives it to 0 has stopped estimating the disagreement point.
            out.update(regime_split(traj_batch, stats, rec.theta_eff, contract,
                                    num_agents, spec))

            out = {f"combined/{k}": v
                   for k, v in _select(out, COMBINED_METRICS).items()}
            # Phase 0: one loop, neither of the two-phase stages. Kept as a numeric
            # series so a wandb view can tell an arm apart from a phase.
            out["phase"] = jnp.float32(0.0)
            out["update_step"] = update_step
            out["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            jax.debug.callback(log_callback, out)
            jax.debug.callback(
                progress_callback, update_step, out["combined/welfare"], 0
            )

            return (train_state, negotiate_state, env_state, last_obs,
                    update_step, rng), out

        # =================================================================
        # RUBINSTEIN BARGAINING, trained JOINTLY (no MOCA phase split)
        # =================================================================
        # Everything below is a separate code path. It deliberately duplicates a
        # little of the phase-1 PPO body rather than refactoring it: the four arms
        # above are published baselines, and restructuring their shared rollout would
        # change how PRNG keys are consumed, silently moving numbers that have
        # already been collected. tests/test_golden_arms.py pins them against exactly
        # that. Unify once this path has earned its keep.
        def _bargain_ppo_loss(params, traj_batch, gae, targets, net):
            """Phase-1's PPO objective, copied so phase 1 stays untouched."""
            pi, value = net.apply(params, traj_batch.obs, traj_batch.contract)
            log_prob = pi.log_prob(traj_batch.action)
            value_pred_clipped = traj_batch.value + (
                value - traj_batch.value
            ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
            value_loss = 0.5 * jnp.maximum(
                jnp.square(value - targets), jnp.square(value_pred_clipped - targets)
            ).mean()
            ratio = jnp.exp(log_prob - traj_batch.log_prob)
            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
            loss_actor = -jnp.minimum(
                ratio * gae,
                jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae,
            ).mean()
            entropy = pi.entropy().mean()
            total = (loss_actor + config["VF_COEF"] * value_loss
                     - config["ENT_COEF"] * entropy)
            return total, entropy

        def rollout_bargaining(gameplay_params, bargain_params, claim_params,
                               env_state, last_obs, vote_eps, rng):
            """One episode of alternating-offers bargaining interleaved with play.

            Scans over ROUNDS; each round makes one bargaining decision and then
            plays `BARGAIN_SEGMENT` steps under whatever contract is in force. A
            scan rather than a Python loop so only one segment body is compiled.

            Each round is TWO forward passes over the same network: the proposer
            decides with an empty table, then every responder decides having seen the
            offer. `vote_eps` is the annealed exploration floor on the vote.

            Returns the gameplay trajectory reshaped to (NUM_STEPS, ...) -- so the
            existing GAE and loss consume it unchanged -- plus the per-round record
            the bargaining update needs.
            """
            K, x = config["BARGAIN_ROUNDS"], config["BARGAIN_SEGMENT"]
            n_envs = config["NUM_ENVS"]
            feat_mask = bargain.feature_mask(config["BARGAIN_FEATURES"], num_agents,
                                             PDIM)
            # Scales so the bargaining state arrives at roughly unit range. Episode
            # return is bounded by one unit of primary reward per step; the
            # contracted act likewise, since every one of them is at most one event
            # per agent per step. The commons scale is per environment -- the river
            # is counted in grid cells, the orchard in apples -- so it comes from the
            # spec rather than from a grid size that only Clean Up has.
            ret_scale = float(config["NUM_STEPS"]) * float(
                config["ENV_KWARGS"].get(spec.reward_scale_kwarg, 1.0))
            clean_scale = float(config["NUM_STEPS"])
            river_scale = commons_scale

            # Who moves first, drawn once per episode per env. Constant across the
            # rounds of an episode, so alternation is preserved; varying across
            # episodes, so no agent owns the first move. See proposer_for_round.
            rng, k_start = jax.random.split(rng)
            start_offset = (
                jax.random.randint(k_start, (n_envs,), 0, num_agents)
                if config["BARGAIN_ROTATE_START"] == "random" else None
            )

            def _seg_step(carry, unused):
                """One gameplay step under the segment's contract. Mirrors `rollout`."""
                env_state, last_obs, theta, rng, tax_window = carry
                contract_obs = contract.to_obs(theta)
                rng, _rng = jax.random.split(rng)
                obs_batch = jnp.transpose(last_obs, (1, 0, 2, 3, 4))
                env_act, log_prob, value = {}, [], []
                act_keys = jax.random.split(_rng, num_agents)
                for i in range(num_agents):
                    pi, value_i = network[i].apply(
                        gameplay_params[i], obs_batch[i], contract_obs)
                    action = pi.sample(seed=act_keys[i])
                    log_prob.append(pi.log_prob(action))
                    env_act[env.agents[i]] = action
                    value.append(value_i)
                env_act_list = [v for v in env_act.values()]

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, n_envs)
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0))(rng_step, env_state, env_act_list)

                # The contract's redistribution for this step, whatever kind it is.
                # It lands in `reward` before anything downstream sees it, so GAE,
                # the segment returns the bargaining rounds are scored on, and every
                # welfare metric all read the same number.
                if tax_kind:
                    # Roll the trailing window forward FIRST: the payout weighting is
                    # inclusive of this step, so an agent that cleans and harvests on
                    # the same step is credited for both.
                    tax_window = contracts.push_tax_window(
                        tax_window, info["cleaned_by_agent"])
                    transfers = contract.tax_transfer(
                        theta, info["original_rewards"], tax_window)
                    # What was actually levied, for the metrics: the receipts side of
                    # a zero-sum transfer. Zero on steps where nobody has cleaned
                    # recently and no tax is charged at all.
                    tax_pot = jnp.maximum(transfers, 0.0).sum(axis=-1)      # (E,)
                else:
                    transfers = contract.transfer_from_info(theta, info)
                    tax_pot = jnp.zeros((n_envs,), jnp.float32)
                reward = reward + transfers
                info = dict(info)
                info["contract_transfer"] = transfers
                # The FINE, broadcast per agent. Everything downstream that reads
                # this series treats it as a scalar theta, so a 2-D contract records
                # its first component here and its threshold in contract/k_* instead.
                theta_fine = theta if PDIM == 1 else theta[..., 0]
                info["contract_theta"] = jnp.broadcast_to(theta_fine[:, None],
                                                          transfers.shape)

                done_list = [v for v in done.values()]
                transition = []
                for i in range(num_agents):
                    info_i = {
                        k: jax.tree.map(lambda z: z.reshape((config["NUM_ACTORS"]), 1), v[:, i])
                        for k, v in info.items()
                    }
                    transition.append(MOCATransition(
                        done_list[i], env_act_list[i], value[i], reward[:, i],
                        log_prob[i], obs_batch[i], contract_obs, info_i))
                # The act the contract prices, and the state of the commons. Both go
                # into the bargaining state (an agent's own contribution so far; the
                # public tier's aggregates) and into the behaviour metrics.
                # What the contract actually charged for this step. Identical to
                # info[spec.contracted_act] for every scalar space; under a bargained
                # threshold the priced set moves with the contract, so reading a
                # fixed info field would make raising the threshold look inert.
                cleaned = contract.act_from_info(theta, info)        # (E, N)
                clear = info[spec.commons_metric][:, 0]             # (E,)
                return ((env_state, obsv, theta, rng, tax_window),
                        (transition, cleaned, clear, tax_pot))

            # Python-level, not traced: with probes off the round draws exactly the
            # PRNG keys it always did, so existing runs and golden tests replay
            # bit-for-bit.
            probe_frac = float(config["BARGAIN_PROBE_FRAC"])
            probe_null_frac = float(config.get("BARGAIN_PROBE_NULL_FRAC", 0.2))

            def _round(carry, r):
                (env_state, last_obs, rng, agreed, locked, cum_return, cum_clean,
                 last_theta_n, had_offer, n_reject, river,
                 last_votes, last_n_acc, last_rejecters, tax_window) = carry
                # Key budget is decided at trace time from static config, and the
                # first keys are always assigned in the same order, so switching a
                # feature OFF reproduces the exact stream it had before the feature
                # existed.
                n_keys = 4 + (2 if probe_frac > 0.0 else 0) + (2 if report_on else 0)
                keys = jax.random.split(rng, n_keys)
                rng, k_prop, k_theta, k_vote = keys[0], keys[1], keys[2], keys[3]
                nxt = 4
                if probe_frac > 0.0:
                    k_probe, k_probe_theta = keys[4], keys[5]
                    nxt = 6
                if report_on:
                    k_claim, k_audit = keys[nxt], keys[nxt + 1]

                proposer = bargain.proposer_for_round(
                    r, num_agents, n_envs, config["BARGAIN_PROPOSER"],
                    key=k_prop, contributions=cum_clean, start_offset=start_offset,
                    holdouts=last_rejecters)

                # Everything about the round except which phase it is. The two passes
                # must agree on the history or the responders would be voting on a
                # different game from the one the proposer played.
                def feats_at(live_theta_n, live):
                    return bargain.bargaining_features(
                        r, K, proposer, num_agents, last_theta_n, had_offer, n_reject,
                        live_theta_n, live, last_votes, last_n_acc,
                        cum_return / ret_scale, cum_clean / clean_scale,
                        river / river_scale, feat_mask)              # (N, E, F)

                # ---- pass 1: propose, with nothing yet on the table.
                # The empty OFFER and the "is an offer live" flag are different
                # shapes once a contract is more than one number -- the flag stays
                # one column per env however many components the offer has.
                no_offer = jnp.zeros(_offer_shape(n_envs), jnp.float32)
                not_live = jnp.zeros((n_envs,), jnp.float32)
                feats_prop = feats_at(no_offer, not_live)
                raws, lp_t, v_prop = [], [], []
                kt = jax.random.split(k_theta, num_agents)
                for i in range(num_agents):
                    pi_theta, _, v = bargain_net[i].apply(
                        bargain_params[i], feats_prop[i])
                    raw = pi_theta.sample(seed=kt[i])                # (E, P)
                    # A scalar space drops the trailing axis and keeps every shape
                    # below exactly as it was; a multi-parameter one carries it.
                    raws.append(raw[:, 0] if PDIM == 1 else raw)
                    lp_t.append(pi_theta.log_prob(raw))
                    v_prop.append(v)
                raw = jnp.stack(raws)                                # (N, E[, P])

                theta_all = negotiate.unsquash(raw, param_low, param_high)
                mine = bargain.is_proposer_mask(proposer, num_agents)
                # The proposer's offer, picked out of the N asks. With a parameter
                # axis the mask needs a trailing axis of its own, or the sum collapses
                # the wrong one and every env gets a blend of two agents' contracts.
                theta_offer = jnp.sum(
                    jnp.where(_as_offer(mine), theta_all, 0.0), axis=0)   # (E[, P])

                # Scripted probe offers. With probability BARGAIN_PROBE_FRAC the
                # proposer's offer is replaced by a theta drawn uniformly over the
                # whole contract range, and everything downstream -- the vote, the
                # contract if it passes, the history it leaves -- treats it as a real
                # offer. This exists because the vote head only stays calibrated on
                # offers it keeps seeing: once the learned proposers settle into a
                # narrow band, a responder's threshold outside that band stops
                # receiving evidence, goes stale, and softens -- which is exactly the
                # opening the fixesV1 lowballers walked through. Probes are a stream
                # of offers the proposers no longer make, at both ends of the range.
                # The PROPOSAL is not trained on probe rounds (the proposer did not
                # choose the offer); the VOTE is trained normally, because voting on
                # a probe is a genuine decision with genuine consequences.
                if probe_frac > 0.0:
                    u = jax.random.uniform(k_probe, (n_envs,))
                    is_probe = u < probe_frac
                    # Uniform over the whole space, component by component: a probe
                    # is meant to be an offer the learned proposers no longer make,
                    # and holding one component fixed would only probe a slice.
                    probe_theta = jax.random.uniform(
                        k_probe_theta, _offer_shape(n_envs),
                        minval=param_low, maxval=param_high)
                    # BARGAIN_PROBE_NULL_FRAC of the probes offer the NULL contract
                    # itself (nested thresholds on one draw, so no extra key). A
                    # null offer never locks -- see `newly` below -- so these probes
                    # double as forced-null exposure: the segment plays uncontracted
                    # whatever the vote, and gameplay keeps meeting theta=0.
                    is_null_probe = u < probe_frac * probe_null_frac
                    probe_theta = jnp.where(
                        _as_offer(is_null_probe), jnp.float32(contract.null),
                        probe_theta)
                    theta_offer = jnp.where(
                        _as_offer(is_probe), probe_theta, theta_offer)
                else:
                    is_probe = jnp.zeros((n_envs,), bool)

                # ---- pass 2: vote, now that theta_r is on the table. The critic has
                # to see it too: a theta-blind baseline cannot credit a rejection
                # against the size of the offer that was refused.
                feats_vote = feats_at(
                    bargain.normalise_theta(theta_offer, param_low, param_high),
                    jnp.ones((n_envs,), jnp.float32))
                votes, lp_v, v_vote = [], [], []
                kv = jax.random.split(k_vote, num_agents)
                for i in range(num_agents):
                    _, pi_vote, v = bargain_net[i].apply(
                        bargain_params[i], feats_vote[i])
                    vote, lp = bargain.floored_vote(pi_vote, vote_eps, kv[i])
                    votes.append(vote)
                    lp_v.append(lp)
                    v_vote.append(v)
                votes = jnp.stack(votes)

                # Each agent's DECISION row and DECISION-POINT value: the proposer
                # decided before the offer existed, the responders after. Those are
                # the states their respective actions were taken from, so those are
                # what the loss must recompute log-probs from and what GAE must
                # bootstrap through.
                feats = jnp.where(mine[..., None], feats_prop, feats_vote)
                values = jnp.where(mine, jnp.stack(v_prop), jnp.stack(v_vote))

                passed, n_accept = bargain.accepted(
                    votes.astype(bool), proposer, quorum_b, num_agents)
                # An offer of exactly the null contract never LOCKS: accepted or
                # not, the segment plays uncontracted and negotiation reopens next
                # round -- a formal "pass this segment", categorically different
                # from every theta > 0, which binds for all remaining segments.
                # Null offers arise from null probes, and (with CONTRACT_LOW=0)
                # from proposers themselves: the clipped unsquash puts an atom of
                # the Gaussian's mass at exactly the lower bound.
                # An offer of exactly the null contract never takes force: accepted
                # or not, the segment plays uncontracted (or, under `sticky`, under
                # whatever was already in force) -- a formal "pass this segment".
                # How long an offer that DOES carry binds for is the whole of
                # BARGAIN_BINDING; see bargain.apply_binding.
                offer_null = contract.is_null(theta_offer)
                newly, theta_eff, next_agreed, next_locked = bargain.apply_binding(
                    binding, passed, offer_null, theta_offer, agreed, locked,
                    contract.null)

                ((env_state, last_obs, _, rng, tax_window),
                 (traj, cleaned, clear, tax_pot)) = jax.lax.scan(
                    _seg_step, (env_state, last_obs, theta_eff, rng, tax_window),
                    None, x)

                # Claims and audits (reporting.py). After the window has been
                # played, each agent files an overclaim on its cleaning; audited
                # claims are voided and fined. The settlement lands on the window's
                # LAST step like any transfer, so it flows into gameplay returns,
                # the bargaining round rewards (agents negotiating theta feel the
                # enforcement leakage) and every welfare metric without a second
                # reward path. Zero under the null contract by construction.
                if report_on:
                    window_clean = jnp.transpose(cleaned.sum(axis=0))         # (N,E)
                    feats_claim = reporting.claim_features(
                        window_clean, theta_eff, x, contract.low, contract.high)
                    kc = jax.random.split(k_claim, num_agents)
                    raws_c = []
                    for i in range(num_agents):
                        pi_c = claim_net[i].apply(claim_params[i], feats_claim[i])
                        raws_c.append(pi_c.sample(seed=kc[i])[:, 0])
                    raw_claim = jnp.stack(raws_c)                             # (N,E)
                    overclaim = negotiate.unsquash(
                        raw_claim, 0.0, float(config["REPORT_MAX_OVERCLAIM"]))
                    audit = jax.random.uniform(
                        k_audit, (num_agents, n_envs)) < float(config["REPORT_AUDIT_P"])
                    settle, claim_own = reporting.settle_claims(
                        theta_eff, overclaim, audit,
                        float(config["REPORT_FINE_MULT"]), num_agents)
                    traj = [
                        traj[i]._replace(reward=traj[i].reward.at[-1].add(settle[i]))
                        for i in range(num_agents)
                    ]

                seg_return = jnp.stack(
                    [traj[i].reward.sum(axis=0) for i in range(num_agents)])     # (N,E)
                record = {
                    "feats": feats, "raw": raw, "logp_theta": jnp.stack(lp_t),
                    "vote": votes, "logp_vote": jnp.stack(lp_v), "value": values,
                    "seg_return": seg_return,
                    # Per-segment aggregates for the null-vs-contracted split: the
                    # contracted act summed over the segment and across agents, and
                    # the segment's welfare. Kept per ROUND because that is the grain
                    # at which theta varies under renegotiation -- an episode-level
                    # split would average a contracted segment together with an
                    # uncontracted one and report neither.
                    "seg_act": cleaned.sum(axis=(0, 2)),
                    "seg_welfare": seg_return.sum(axis=0),
                    # Harvest-tax revenue actually levied over the segment, for the
                    # metrics. Identically zero under clean_wage.
                    "tax_pot": tax_pot.sum(axis=0),

                    "active": ~agreed, "newly": newly,
                    "is_proposer": mine, "theta_offer": theta_offer,
                    # Every agent's sampled ask, read or not. Only the proposer's
                    # is acted on here, but the dispersion of asks is the series
                    # that says whether ideal points are separating by role -- and
                    # under the median protocol it is the mechanism itself.
                    "theta_all": theta_all,
                    "n_accept": n_accept, "theta_eff": theta_eff,
                    "is_probe": is_probe, "offer_null": offer_null,
                }
                if report_on:
                    # The claim bandit's training record. `claim_own` (not the full
                    # zero-sum settlement) is the reward: the funding share of
                    # OTHERS' claims does not depend on this agent's action.
                    record.update({
                        "claim_feats": feats_claim, "claim_raw": raw_claim,
                        "claim_reward": claim_own, "overclaim": overclaim,
                        "audited": audit,
                        "claim_w": (theta_eff > contract.null).astype(jnp.float32),
                    })
                carry = (env_state, last_obs, rng,
                         next_agreed,
                         next_locked,
                         cum_return + seg_return,
                         cum_clean + jnp.transpose(cleaned.sum(axis=0)),
                         bargain.normalise_theta(theta_offer, param_low, param_high),
                         jnp.ones_like(had_offer),
                         n_reject + (~agreed & ~passed).astype(jnp.int32),
                         clear[-1].astype(jnp.float32),
                         # Only COUNTED votes carry forward: the proposer's own vote
                         # is ignored by the quorum, so recording it would read as a
                         # refusal it never made.
                         (votes.astype(bool) & ~mine).astype(jnp.float32),
                         n_accept.astype(jnp.float32),
                         # Who refused, for BARGAIN_PROPOSER=holdout: counted
                         # rejections only. Consumed next round; zeros mean random
                         # recognition there.
                         (~votes.astype(bool) & ~mine).astype(jnp.float32),
                         tax_window)
                return carry, (record, traj)

            def _round_median(carry, r):
                """One MEDIAN round: every agent asks, the median binds the segment.

                Same carry and record layout as `_round`, so the semi-MDP reward,
                the GAE, the loss masks and the metrics are all shared. The
                vote-shaped fields are structural zeros, and `is_proposer` is all
                True -- every agent's ask is a trained decision -- which is what
                routes the whole round through the proposal head's mask and zeroes
                the vote credit mask without any downstream special-casing.
                """
                (env_state, last_obs, rng, agreed, locked, cum_return, cum_clean,
                 last_theta_n, had_offer, n_reject, river,
                 last_votes, last_n_acc, last_rejecters, tax_window) = carry
                n_keys = 2 + (2 if report_on else 0)
                keys = jax.random.split(rng, n_keys)
                rng, k_theta = keys[0], keys[1]
                if report_on:
                    k_claim, k_audit = keys[2], keys[3]

                feats_prop = bargain.median_round_features(
                    r, K, num_agents, last_theta_n, had_offer,
                    cum_return / ret_scale, cum_clean / clean_scale,
                    river / river_scale, feat_mask)                   # (N, E, F)

                raws, lp_t, v_prop = [], [], []
                kt = jax.random.split(k_theta, num_agents)
                for i in range(num_agents):
                    pi_theta, _, v = bargain_net[i].apply(
                        bargain_params[i], feats_prop[i])
                    raw_i = pi_theta.sample(seed=kt[i])               # (E, 1)
                    raws.append(raw_i[:, 0])
                    lp_t.append(pi_theta.log_prob(raw_i))
                    v_prop.append(v)
                raw = jnp.stack(raws)                                 # (N, E)

                theta_all = negotiate.unsquash(raw, contract.low, contract.high)
                theta_offer = bargain.median_offer(theta_all)         # (E,)
                # A null median is only reachable when CONTRACT_LOW is 0 (the
                # clipped unsquash then puts an atom at exactly 0, and half the
                # asks must sit on it). It plays the segment uncontracted, same
                # as a null offer everywhere else.
                offer_null = contract.is_null(theta_offer)
                theta_eff = theta_offer
                newly = ~offer_null

                ((env_state, last_obs, _, rng, tax_window),
                 (traj, cleaned, clear, tax_pot)) = jax.lax.scan(
                    _seg_step, (env_state, last_obs, theta_eff, rng, tax_window),
                    None, x)

                # Claims and audits, exactly as in `_round`: the enforcement layer
                # is orthogonal to how theta was chosen.
                if report_on:
                    window_clean = jnp.transpose(cleaned.sum(axis=0))         # (N,E)
                    feats_claim = reporting.claim_features(
                        window_clean, theta_eff, x, contract.low, contract.high)
                    kc = jax.random.split(k_claim, num_agents)
                    raws_c = []
                    for i in range(num_agents):
                        pi_c = claim_net[i].apply(claim_params[i], feats_claim[i])
                        raws_c.append(pi_c.sample(seed=kc[i])[:, 0])
                    raw_claim = jnp.stack(raws_c)                             # (N,E)
                    overclaim = negotiate.unsquash(
                        raw_claim, 0.0, float(config["REPORT_MAX_OVERCLAIM"]))
                    audit = jax.random.uniform(
                        k_audit, (num_agents, n_envs)) < float(config["REPORT_AUDIT_P"])
                    settle, claim_own = reporting.settle_claims(
                        theta_eff, overclaim, audit,
                        float(config["REPORT_FINE_MULT"]), num_agents)
                    traj = [
                        traj[i]._replace(reward=traj[i].reward.at[-1].add(settle[i]))
                        for i in range(num_agents)
                    ]

                seg_return = jnp.stack(
                    [traj[i].reward.sum(axis=0) for i in range(num_agents)])   # (N,E)
                record = {
                    "feats": feats_prop, "raw": raw,
                    "logp_theta": jnp.stack(lp_t),
                    "vote": jnp.zeros((num_agents, n_envs), jnp.int32),
                    "logp_vote": jnp.zeros((num_agents, n_envs), jnp.float32),
                    "value": jnp.stack(v_prop),
                    "seg_return": seg_return,
                    # Per-segment aggregates for the null-vs-contracted split: the
                    # contracted act summed over the segment and across agents, and
                    # the segment's welfare. Kept per ROUND because that is the grain
                    # at which theta varies under renegotiation -- an episode-level
                    # split would average a contracted segment together with an
                    # uncontracted one and report neither.
                    "seg_act": cleaned.sum(axis=(0, 2)),
                    "seg_welfare": seg_return.sum(axis=0),
                    # Harvest-tax revenue actually levied over the segment, for the
                    # metrics. Identically zero under clean_wage.
                    "tax_pot": tax_pot.sum(axis=0),

                    "active": ~agreed, "newly": newly,
                    "is_proposer": jnp.ones((num_agents, n_envs), bool),
                    "theta_offer": theta_offer, "theta_all": theta_all,
                    "n_accept": jnp.zeros((n_envs,), jnp.int32),
                    "theta_eff": theta_eff,
                    "is_probe": jnp.zeros((n_envs,), bool),
                    "offer_null": offer_null,
                }
                if report_on:
                    record.update({
                        "claim_feats": feats_claim, "claim_raw": raw_claim,
                        "claim_reward": claim_own, "overclaim": overclaim,
                        "audited": audit,
                        "claim_w": (theta_eff > contract.null).astype(jnp.float32),
                    })
                carry = (env_state, last_obs, rng,
                         agreed,                     # never absorbs: every round asks
                         theta_eff,                  # what the gameplay critic
                                                     # bootstraps against at the end
                         cum_return + seg_return,
                         cum_clean + jnp.transpose(cleaned.sum(axis=0)),
                         bargain.normalise_theta(theta_offer, contract.low,
                                                 contract.high),
                         jnp.ones_like(had_offer),
                         n_reject,                   # rejections do not exist here
                         clear[-1].astype(jnp.float32),
                         jnp.zeros_like(last_votes),
                         jnp.zeros_like(last_n_acc),
                         jnp.zeros_like(last_rejecters),
                         tax_window)
                return carry, (record, traj)

            zeros_e = jnp.zeros((n_envs,), jnp.float32)
            init = (env_state, last_obs, rng,
                    jnp.zeros((n_envs,), bool),
                    jnp.full(_offer_shape(n_envs), contract.null, jnp.float32),
                    jnp.zeros((num_agents, n_envs), jnp.float32),
                    jnp.zeros((num_agents, n_envs), jnp.float32),
                    jnp.zeros(_offer_shape(n_envs), jnp.float32),
                    jnp.zeros((n_envs,), bool),
                    jnp.zeros((n_envs,), jnp.int32), zeros_e,
                    jnp.zeros((num_agents, n_envs), jnp.float32), zeros_e,
                    jnp.zeros((num_agents, n_envs), jnp.float32),
                    # Trailing cleaning, for the harvest tax's payout weighting.
                    # Fresh per EPISODE and carried across segment boundaries: it is
                    # a weighting, not part of the contract, so a cleaner that worked
                    # through the last segment is still owed by the next one. Length
                    # 0 under clean_wage, where it is never read.
                    contracts.new_tax_window(
                        config["TAX_WINDOW"] if tax_kind else 0,
                        num_agents, batch=(n_envs,)))
            carry, (rounds, traj) = jax.lax.scan(
                _round_median if median_protocol else _round, init, jnp.arange(K))
            env_state, last_obs, rng, agreed, locked = carry[0], carry[1], carry[2], carry[3], carry[4]

            # (K, x, ...) -> (NUM_STEPS, ...), so the existing GAE/loss are unchanged.
            traj_batch = [
                jax.tree.map(lambda z: z.reshape((K * x,) + z.shape[2:]), traj[i])
                for i in range(num_agents)
            ]

            # Semi-MDP reward. A rejected round pays only its own null segment; the
            # round where the offer is accepted pays every remaining segment at once,
            # because bargaining ends there and those segments are its consequence.
            #
            # Under the renegotiating modes this degenerates to exactly the right
            # thing with no branch: nothing is ever inactive, so `post` is zero and
            # every round pays its own segment and no more. That lump sum is what
            # dominated the vote's advantage in the first place, so removing it is
            # half of what BARGAIN_BINDING=segment is for.
            active = rounds["active"]                                    # (K, E)
            post = jnp.sum(
                jnp.where(active[:, None, :], 0.0, rounds["seg_return"]), axis=0)  # (N,E)
            rounds["reward"] = rounds["seg_return"] + (
                rounds["newly"][:, None, :] * post[None, :, :])
            last_round = jnp.arange(K)[:, None] == (K - 1)
            # A carried offer ENDS the bargaining episode only when it binds for the
            # rest of it. Under renegotiation the episode runs on regardless, so
            # bootstrapping must not stop at a segment that happened to agree.
            rounds["terminal"] = ((rounds["newly"] | last_round) if binding == "episode"
                                  else jnp.broadcast_to(last_round, active.shape))
            # What the gameplay critic bootstraps its final value against: the locked
            # contract under `episode`, the last segment's contract otherwise (which
            # is what `locked` carries there).
            final_theta = (jnp.where(_as_offer(agreed), locked,
                                     jnp.float32(contract.null))
                           if binding == "episode" else locked)
            return traj_batch, rounds, env_state, last_obs, final_theta, rng

        def _update_step_joint_bargain(runner_state, unused):
            if report_on:
                (train_state, bargain_state, claim_state, env_state, last_obs,
                 update_step, rng) = runner_state
                c_params = [cs.params for cs in claim_state]
            else:
                (train_state, bargain_state, env_state, last_obs,
                 update_step, rng) = runner_state
                claim_state, c_params = None, None
            params_list = [ts.params for ts in train_state]
            b_params = [bs.params for bs in bargain_state]

            # Exploration floor on the vote: full strength early, while a saturated
            # accept/reject habit would still be self-sealing, annealed down to
            # BARGAIN_VOTE_EPS_END -- and kept there, because rejection is only ever
            # maintained by being occasionally sampled. See bargain.vote_eps_at.
            vote_eps = bargain.vote_eps_at(
                config["BARGAIN_VOTE_EPS"], update_step, config["NUM_UPDATES"],
                end=config["BARGAIN_VOTE_EPS_END"])

            (traj_batch, rounds, env_state, last_obs, final_theta,
             rng) = rollout_bargaining(params_list, b_params, c_params,
                                       env_state, last_obs, vote_eps, rng)

            # ---------------------------------------------------- gameplay PPO
            contract_obs = contract.to_obs(final_theta)
            last_obs_batch = jnp.transpose(last_obs, (1, 0, 2, 3, 4))
            last_val = [network[i].apply(params_list[i], last_obs_batch[i],
                                         contract_obs)[1] for i in range(num_agents)]
            entropies = []
            for i in range(num_agents):
                advantages_i, targets_i = compute_gae(traj_batch[i], last_val[i])

                def _epoch(state, unused, i=i):
                    def _minibatch(ts, b):
                        tb, adv, tgt = b
                        grads, ent = jax.grad(_bargain_ppo_loss, has_aux=True)(
                            ts.params, tb, adv, tgt, network[i])
                        return ts.apply_gradients(grads=grads), ent

                    ts, tb, adv, tgt, rng_ = state
                    rng_, _rng = jax.random.split(rng_)
                    batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                    perm = jax.random.permutation(_rng, batch_size)
                    batch = jax.tree_util.tree_map(
                        lambda z: z.reshape((batch_size,) + z.shape[2:]), (tb, adv, tgt))
                    shuffled = jax.tree_util.tree_map(
                        lambda z: jnp.take(z, perm, axis=0), batch)
                    mbs = jax.tree_util.tree_map(
                        lambda z: jnp.reshape(
                            z, [config["NUM_MINIBATCHES"], -1] + list(z.shape[1:])),
                        shuffled)
                    ts, ent = jax.lax.scan(_minibatch, ts, mbs)
                    return (ts, tb, adv, tgt, rng_), ent

                state = (train_state[i], traj_batch[i], advantages_i, targets_i, rng)
                state, ent = jax.lax.scan(_epoch, state, None, config["UPDATE_EPOCHS"])
                train_state[i] = state[0]
                rng = state[-1]
                entropies.append(ent.mean())

            # -------------------------------------------------- bargaining PPO
            adv_b, targ_b = bargain.round_gae(
                rounds["reward"], rounds["value"], rounds["active"][:, None, :],
                rounds["terminal"][:, None, :],
                config["BARGAIN_GAMMA"], config["BARGAIN_GAE_LAMBDA"])

            # Realised remaining return per round, the regression target for both
            # branch heads, and who was actually pivotal. Only built in the mode
            # that uses them, so the `gae` path is untouched arithmetic-for-
            # arithmetic.
            rtg = None
            if cf_vote:
                rtg = bargain.return_to_go(rounds["reward"],
                                           rounds["active"][:, None, :])
                rounds["pivotal"] = bargain.pivotal_mask(
                    rounds["n_accept"], rounds["vote"], rounds["is_proposer"],
                    quorum_b)

            def bargain_loss(params, i):
                # rounds["feats"] holds each agent's DECISION-POINT state: the
                # proposal pass for whoever proposed, the vote pass for everyone
                # else. Recomputing log-probs from it is what keeps the PPO ratio
                # exactly on-policy now that the two decisions are taken from
                # different states.
                if cf_vote:
                    pi_theta, pi_vote, value, lock_v, cont_v = bargain_net[i].apply(
                        params, rounds["feats"][:, i], return_aux=True)
                else:
                    pi_theta, pi_vote, value = bargain_net[i].apply(
                        params, rounds["feats"][:, i])              # (K, E, ...)
                act = rounds["active"].astype(jnp.float32)          # (K, E)
                # Over ACTIVE rounds only -- the rest of the block is structural
                # zeros. Every term below is masked by `act`, so the garbage this
                # leaves outside the mask never reaches a gradient.
                a = bargain.masked_standardise(adv_b[:, i], act)
                mine = rounds["is_proposer"][:, i].astype(jnp.float32)
                # A probe round replaced the proposer's offer with a scripted one,
                # so its proposal was never acted on and must not be trained on --
                # crediting it with the probe's consequences would teach the
                # proposal head from offers it did not make. Votes on a probe were
                # real decisions and train as usual; what the VOTE excludes is in
                # `vote_credit_mask`.
                not_probe = 1.0 - rounds["is_probe"].astype(jnp.float32)
                w_prop = act * mine * not_probe
                w_vote = bargain.vote_credit_mask(act, mine, rounds["offer_null"])
                eps = config["BARGAIN_CLIP_EPS"]

                def clipped(logp, old_logp, w, adv):
                    ratio = jnp.exp(logp - old_logp)
                    obj = jnp.minimum(
                        ratio * adv, jnp.clip(ratio, 1.0 - eps, 1.0 + eps) * adv)
                    return -(obj * w).sum() / (w.sum() + 1e-8)

                # An agent proposes OR votes in a given round, never both, so the two
                # heads are trained on disjoint masks. Rounds after agreement carry no
                # decision and are excluded from all three terms.
                loss = clipped(pi_theta.log_prob(rounds["raw"][:, i][..., None]),
                               rounds["logp_theta"][:, i], w_prop, a)

                # The vote's advantage, and the mask it is averaged over. Under
                # `counterfactual` both narrow: only pivotal votes carry signal, and
                # what they carry is their own branch difference rather than the
                # round's shared outcome.
                vote_adv, w_vote_pg = a, w_vote
                if cf_vote:
                    pivotal = rounds["pivotal"][:, i].astype(jnp.float32)
                    w_vote_pg = bargain.vote_credit_mask(
                        act, mine, rounds["offer_null"], pivotal)
                    # Scale-only, NOT standardised: the counterfactual advantage is
                    # already measured against its own baseline (the other branch),
                    # so its batch mean is signal, not artefact. Centering it
                    # stripped the level and left only the slope -- the cf-vote
                    # run's harvesters believed "reject" at every theta while
                    # their acceptance level sat untrained at ~0.9. See
                    # bargain.masked_scale.
                    vote_adv = bargain.masked_scale(
                        bargain.counterfactual_vote_advantage(
                            rounds["vote"][:, i], lock_v, cont_v, pivotal),
                        w_vote_pg)
                # The stored vote log-prob is under the eps-FLOORED distribution the
                # vote was drawn from, while this one is under the policy itself, so
                # the ratio is a proper importance weight against the behaviour
                # distribution. Evaluating the numerator through the floor instead
                # would zero the gradient of exactly the saturated agents the floor
                # exists to rescue.
                loss = loss + clipped(pi_vote.log_prob(rounds["vote"][:, i]),
                                      rounds["logp_vote"][:, i], w_vote_pg, vote_adv)
                v_loss = (jnp.square(value - targ_b[:, i]) * act).sum() / (act.sum() + 1e-8)
                if cf_vote:
                    # Each branch head regresses on the rounds where that branch was
                    # REALISED: lock on the rounds that locked, continue on the rest
                    # (including null offers, which are pure continuation samples).
                    # Both are restricted to responder rows -- a proposer's decision
                    # state has no offer on the table, so a branch value there would
                    # be fitted to a state the counterfactual never asks about.
                    responder = act * (1.0 - mine)
                    newly = rounds["newly"].astype(jnp.float32)
                    for head, w_head in ((lock_v, responder * newly),
                                         (cont_v, responder * (1.0 - newly))):
                        v_loss = v_loss + (
                            jnp.square(head - rtg[:, i]) * w_head
                        ).sum() / (w_head.sum() + 1e-8)
                # Entropy stays on the WIDER mask: a non-pivotal vote earns no
                # policy gradient, but it is still a vote the agent will cast again,
                # and letting it saturate for want of regularisation is how the
                # always-accept equilibrium formed in the first place.
                entropy = (pi_vote.entropy() * w_vote).sum() / (w_vote.sum() + 1e-8)
                return (loss + config["BARGAIN_VF_COEF"] * v_loss
                        - config["BARGAIN_ENT_COEF"] * entropy)

            def _bargain_epoch(state, unused):
                for i in range(num_agents):
                    g = jax.grad(bargain_loss)(state[i].params, i)
                    state[i] = state[i].apply_gradients(grads=g)
                return state, None

            bargain_state, _ = jax.lax.scan(
                _bargain_epoch, bargain_state, None, config["BARGAIN_UPDATE_EPOCHS"])

            # ------------------------------------------------- claim bandit
            # Exogenous audits make the claim a contextual bandit: the settlement
            # lands immediately and nothing carries over, so plain REINFORCE
            # against a batch-standardised baseline is the whole update. Windows
            # with no contract in force are masked -- their settlement is
            # identically zero whatever the claim, so they carry only noise.
            if report_on:
                def claim_loss(params, i):
                    pi_c = claim_net[i].apply(params, rounds["claim_feats"][:, i])
                    logp = pi_c.log_prob(rounds["claim_raw"][:, i][..., None])
                    w = rounds["claim_w"]                                # (K, E)
                    adv = bargain.masked_standardise(
                        rounds["claim_reward"][:, i], w)
                    return -(logp * adv * w).sum() / (w.sum() + 1e-8)

                for i in range(num_agents):
                    g = jax.grad(claim_loss)(claim_state[i].params, i)
                    claim_state[i] = claim_state[i].apply_gradients(grads=g)

            update_step = update_step + 1
            jax.debug.callback(checkpoint_callback, train_state, update_step)
            jax.debug.callback(bargain_checkpoint_callback, bargain_state, update_step)
            if report_on:
                jax.debug.callback(claim_checkpoint_callback, claim_state, update_step)

            # ------------------------------------------------------- metrics
            metric = jax.tree.map(
                lambda z: z.mean(), [dict(traj_batch[i].info) for i in range(num_agents)])
            stacked = {k: jnp.stack([d[k] for d in metric]) for k in metric[0]}
            out = {}
            for k, v in stacked.items():
                out[f"{k}_mean"] = v.mean()
                out[f"{k}_std"] = v.std()
            stats = episode_stats(traj_batch, num_agents)
            out["welfare"] = stats["welfare"]
            out["equality"] = stats["equality"]
            out["transfer_volume"] = stats["transfer_volume"]

            agreed_any = rounds["newly"].any(axis=0)                   # (E,)
            K = config["BARGAIN_ROUNDS"]
            # FIRST round in which an offer carried, K if none ever did. Under
            # `episode` that is the round of agreement, and its mean is directly
            # "how many rounds of bargaining were burned"; under renegotiation it is
            # how long the agents took to get a contract in force at all, and the
            # series that carries the outcome from then on is in_force_rate.
            round_idx = jnp.arange(K)[:, None]
            agree_round = jnp.where(
                agreed_any, jnp.argmax(rounds["newly"].astype(jnp.int32), axis=0), K)
            # Under `episode` an episode agrees at most once, so the fraction of
            # ROUNDS that carried and the fraction of EPISODES that agreed are the
            # same question asked twice; under renegotiation they are not, and the
            # per-round rate is the one that means anything.
            out["agreement_rate"] = (agreed_any.mean() if binding == "episode"
                                     else rounds["newly"].mean())
            out["agreement_round"] = agree_round.mean()
            out["disagreement_steps"] = agree_round.mean() * config["BARGAIN_SEGMENT"]
            # The FINE decides whether a contract is in force; a threshold on its
            # own moves nothing. With a 2-D space theta_eff is (K, E, P), so the
            # comparison has to name component 0 rather than reduce over both.
            theta_eff_f = (rounds["theta_eff"] if PDIM == 1
                           else rounds["theta_eff"][..., 0])
            theta_offer_f = (rounds["theta_offer"] if PDIM == 1
                             else rounds["theta_offer"][..., 0])
            in_force = theta_eff_f > contract.null
            out["contract_in_force_rate"] = in_force.mean()
            # Round 0 alone. Not derivable from the pooled rate, and on a commons
            # that can be spent inside one segment it is the one that decides whether
            # the mechanism ever had anything to price.
            out["contract_in_force_rate_round0"] = in_force[0].mean()
            # The bargained threshold. Zero on every one-dimensional space, so the
            # series exists in both cases and a run that silently fell back to the
            # scalar contract reads as a flat zero rather than as a missing chart.
            if PDIM > 1:
                k_offer = rounds["theta_offer"][..., 1]
                k_eff = rounds["theta_eff"][..., 1]
                w_offer = rounds["active"] * (
                    1.0 - rounds["is_probe"].astype(jnp.float32))
                out["density_k_offered"] = ((k_offer * w_offer).sum()
                                            / jnp.maximum(w_offer.sum(), 1.0))
                out["density_k_in_force"] = ((k_eff * in_force).sum()
                                             / jnp.maximum(in_force.sum(), 1.0))
            else:
                out["density_k_offered"] = jnp.float32(0.0)
                out["density_k_in_force"] = jnp.float32(0.0)
            if binding == "episode":
                # theta actually agreed, averaged over the envs that agreed at all.
                agreed_theta = jnp.sum(
                    jnp.where(rounds["newly"], theta_offer_f, 0.0), axis=0)
                out["theta_agreed"] = (jnp.sum(agreed_theta) /
                                       jnp.maximum(agreed_any.sum(), 1.0))
            else:
                # There is no single agreed theta -- there are up to K of them, one
                # per segment -- so this becomes the theta actually PLAYED UNDER,
                # averaged over the segments that had a contract at all.
                out["theta_agreed"] = (
                    (theta_eff_f * in_force).sum()
                    / jnp.maximum(in_force.sum(), 1.0))
            out["theta_in_force"] = ((theta_eff_f * in_force).sum()
                                     / jnp.maximum(in_force.sum(), 1.0))
            # Harvest tax. Revenue is levied per step, so it is reported per step;
            # the realised wage divides it by the cells that earned it, which is the
            # figure to hold against a clean_wage theta. Both are structurally zero
            # under clean_wage, where no pot exists.
            steps = float(config["BARGAIN_ROUNDS"] * config["BARGAIN_SEGMENT"])
            out["tax_revenue"] = rounds["tax_pot"].sum() / (steps * config["NUM_ENVS"])
            # The contracted act per episode: seg_act is already summed over the
            # segment's steps and over agents, so summing the rounds gives one
            # episode's total and the mean is over envs.
            out["act_per_episode"] = rounds["seg_act"].sum(axis=0).mean()
            cells_per_step = stacked[spec.contracted_act].mean() * num_agents
            out["tax_wage"] = out["tax_revenue"] / jnp.maximum(cells_per_step, 1e-6)
            # Masked by `active`: agents still emit an offer in rounds after
            # agreement, but it is never read, so averaging it in would report
            # untrained noise as the policy's asking price. Probe rounds are masked
            # too -- their offer is scripted, and this series is meant to show what
            # the POLICY is asking.
            n_active = jnp.maximum(rounds["active"].sum(), 1.0)
            w_own_offer = rounds["active"] * (
                1.0 - rounds["is_probe"].astype(jnp.float32))
            out["theta_offered"] = ((theta_offer_f * w_own_offer).sum()
                                    / jnp.maximum(w_own_offer.sum(), 1.0))
            out["accept_count"] = (
                rounds["n_accept"] * rounds["active"]).sum() / n_active
            # Dispersion of the asks within a round (max - min across agents,
            # averaged over active rounds). Under the median protocol this is the
            # series that says whether ideal points are separating by role --
            # cleaners asking high, harvesters low -- or whether everyone has
            # collapsed onto one number; under alternating offers it is the same
            # question about the asks only one of which is ever read.
            theta_all_f = (rounds["theta_all"] if PDIM == 1
                           else rounds["theta_all"][..., 0])
            ask_spread = (theta_all_f.max(axis=1)
                          - theta_all_f.min(axis=1))                   # (K, E)
            out["theta_ask_spread"] = (
                ask_spread * rounds["active"]).sum() / n_active
            # ---- did the contract change anything, and was it worth signing? ----
            # Split by SEGMENT on whether a contract was in force. Under
            # renegotiation both regimes occur throughout a single run, so the
            # comparison phase 1 gets from P(Theta) is available here too -- live,
            # rather than only from an offline theta sweep against saved weights.
            #
            # It is correlational, not controlled: which segments run uncontracted is
            # decided by the negotiation itself, so a role that rejects when the
            # commons is already spent will bias the null side. BARGAIN_PROBE_FRAC
            # with BARGAIN_PROBE_NULL_FRAC forces null segments independently of the
            # policies, and is what moves this toward a causal reading.
            contracted = in_force.astype(jnp.float32)             # (K, E)
            uncontracted = 1.0 - contracted
            n_con = jnp.maximum(contracted.sum(), 1.0)
            n_unc = jnp.maximum(uncontracted.sum(), 1.0)
            for name, per_round in ((spec.act_label, rounds["seg_act"]),
                                    ("welfare", rounds["seg_welfare"])):
                out[f"{name}_null"] = (per_round * uncontracted).sum() / n_unc
                out[f"{name}_contracted"] = (per_round * contracted).sum() / n_con
            # Per SEGMENT, so it is not comparable to phase 1's per-episode figures;
            # multiply by BARGAIN_ROUNDS for that. Both sides read 0 when a run never
            # leaves one regime, which in_force_rate is what disambiguates.
            out[f"{spec.act_label}_gap"] = (
                out[f"{spec.act_label}_contracted"] - out[f"{spec.act_label}_null"])
            if len(spec.behaviour_metrics) > 1:
                # The contracted act as a share of the denominator the environment
                # supplies -- depleting eats per apple eaten, stolen coins per coin
                # taken. Separates "the contract stopped the harm" from "the contract
                # stopped the activity", which the numerator alone cannot.
                out[f"{spec.act_label}_share"] = (
                    out[f"{spec.contracted_act}_mean"]
                    / jnp.maximum(out[f"{spec.behaviour_metrics[1]}_mean"], 1e-8))
            # Gameplay policy entropy, averaged over agents. Nothing else in this set
            # detects a collapsing policy: welfare and cleaning stay plausible right
            # up until the policies go deterministic, and then everything drops to
            # zero in a single update with no warning in any other series. Watch this
            # one -- if it trends toward 0 the run is dying, whatever else says.
            out["policy_entropy"] = jnp.stack(entropies).mean()
            out["policy_entropy_min"] = jnp.stack(entropies).min()
            out["vote_eps"] = vote_eps
            # Counterfactual diagnostics, read off the params that GENERATED the
            # rollout rather than the just-updated ones, so they describe the
            # beliefs the logged votes were actually cast under.
            out["cf_gap"] = jnp.float32(0.0)
            out["cf_pivotal_rate"] = jnp.float32(0.0)
            if cf_vote:
                w_vote_all = bargain.vote_credit_mask(
                    rounds["active"][:, None, :], rounds["is_proposer"],
                    rounds["offer_null"][:, None, :])                  # (K, N, E)
                gaps = []
                for i in range(num_agents):
                    _, _, _, lock_v, cont_v = bargain_net[i].apply(
                        b_params[i], rounds["feats"][:, i], return_aux=True)
                    gaps.append(lock_v - cont_v)
                total_w = jnp.maximum(w_vote_all.sum(), 1.0)
                out["cf_gap"] = (jnp.stack(gaps, axis=1) * w_vote_all).sum() / total_w
                out["cf_pivotal_rate"] = (
                    rounds["pivotal"].astype(jnp.float32) * w_vote_all
                ).sum() / total_w
            # Reporting series, zero when the mechanism is off (same convention as
            # the cf/ pair: always present, so one wandb view covers both modes).
            out["report_overclaim"] = jnp.float32(0.0)
            out["report_leakage"] = jnp.float32(0.0)
            if report_on:
                w_claim = rounds["claim_w"]                              # (K, E)
                n_claims = jnp.maximum((w_claim.sum() * num_agents), 1.0)
                out["report_overclaim"] = (
                    rounds["overclaim"] * w_claim[:, None, :]).sum() / n_claims
                # Reward that left honest pockets: overclaims PAID (unaudited),
                # per episode. The mechanism's leak rate in reward units.
                paid = (rounds["overclaim"]
                        * (1.0 - rounds["audited"].astype(jnp.float32))
                        * rounds["theta_eff"][:, None, :])
                out["report_leakage"] = paid.sum() / config["NUM_ENVS"]

            out = {f"joint/{JOINT_METRICS[k]}": v
                   for k, v in _select(out, tuple(JOINT_METRICS)).items()}
            # Un-namespaced duplicate so welfare is findable at the top level of a
            # wandb run rather than only inside the joint/ section.
            out["welfare"] = out["joint/outcome/welfare"]
            out["phase"] = jnp.float32(3.0)
            out["update_step"] = update_step
            out["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            jax.debug.callback(log_callback, out)
            jax.debug.callback(
                progress_callback, update_step, out["joint/outcome/welfare"], 3)

            if report_on:
                return (train_state, bargain_state, claim_state, env_state,
                        last_obs, update_step, rng), out
            return (train_state, bargain_state, env_state, last_obs,
                    update_step, rng), out

        # ----------------------------------------------------------- callbacks
        def log_callback(metric):
            wandb.log({k: float(v) for k, v in metric.items()})

        def checkpoint_callback(train_state, update_step):
            update_step = int(update_step)
            every = config.get("CHECKPOINT_EVERY", 20)
            if every <= 0 or update_step % every != 0:
                return
            filename = checkpoint_filename(config, latest=True)
            for i in range(num_agents):
                save_params(train_state[i], f"./checkpoints/moca/{filename}_{i}.pkl")
                save_train_state(
                    train_state[i], update_step, f"./checkpoints/moca/{filename}_resume_{i}.pkl"
                )
            print(f"[checkpoint] MOCA phase-1 gameplay policy at update {update_step}")

        def contract_checkpoint_callback(proposal_state, voting_state, update_step):
            update_step = int(update_step)
            every = config.get("CHECKPOINT_EVERY", 20)
            if every <= 0 or update_step % every != 0:
                return
            filename = checkpoint_filename(config, latest=True)
            for i in range(num_agents):
                save_params(proposal_state[i], f"./checkpoints/moca/{filename}_proposal_{i}.pkl")
                save_params(voting_state[i], f"./checkpoints/moca/{filename}_voting_{i}.pkl")
            print(f"[checkpoint] MOCA phase-2 contracting policies at update {update_step}")

        def negotiate_checkpoint_callback(negotiate_state, update_step):
            update_step = int(update_step)
            every = config.get("CHECKPOINT_EVERY", 20)
            if every <= 0 or update_step % every != 0:
                return
            filename = checkpoint_filename(config, latest=True)
            for i in range(num_agents):
                # "_contract_", not "_negotiate_": the run stem already ends in the
                # PHASE2_MODE token, so a "_negotiate_" role suffix would make
                # ..._negotiate_0.pkl (gameplay) and ..._negotiate_negotiate_0.pkl
                # (contracting) indistinguishable by substring, which is how the
                # viewer separates the two sets.
                save_params(negotiate_state[i], f"./checkpoints/moca/{filename}_contract_{i}.pkl")
            print(f"[checkpoint] MOCA negotiation policies at update {update_step}")

        def bargain_checkpoint_callback(bargain_state, update_step):
            update_step = int(update_step)
            every = config.get("CHECKPOINT_EVERY", 20)
            if every <= 0 or update_step % every != 0:
                return
            filename = checkpoint_filename(config, latest=True)
            for i in range(num_agents):
                # "_contract_", the same role suffix the negotiation stage uses, so
                # the viewer's gameplay/contracting split keeps working unchanged.
                save_params(bargain_state[i],
                            f"./checkpoints/moca/{filename}_contract_{i}.pkl")
            print(f"[checkpoint] bargaining policies at update {update_step}")

        def claim_checkpoint_callback(claim_state, update_step):
            update_step = int(update_step)
            every = config.get("CHECKPOINT_EVERY", 20)
            if every <= 0 or update_step % every != 0:
                return
            filename = checkpoint_filename(config, latest=True)
            for i in range(num_agents):
                save_params(claim_state[i],
                            f"./checkpoints/moca/{filename}_claim_{i}.pkl")
            print(f"[checkpoint] claim policies at update {update_step}")

        def progress_callback(update_step, mean_val, phase):
            update_step = int(update_step)
            now = time.time()
            progress_state["times"][update_step] = now
            first = progress_state.setdefault("first_update", update_step)
            every = config.get("PROGRESS_EVERY", 1)
            if every <= 0 or update_step % every != 0:
                return
            total = config["NUM_UPDATES_PHASE1"] + config["NUM_UPDATES_PHASE2"]
            if update_step <= first:
                print(f"[progress] phase {int(phase)} update {update_step}/{total} "
                      f"(JIT compiling -- first update is slow)", flush=True)
                return
            t1 = progress_state["times"].get(first)
            if t1 is None:
                print(f"[progress] phase {int(phase)} update {update_step}/{total}", flush=True)
                return
            rate = (now - t1) / (update_step - first)
            eta = rate * (total - update_step) / 60
            print(
                f"[progress] phase {int(phase)} update {update_step}/{total} "
                f"({100*update_step/total:.1f}%) ~{rate:.1f}s/update, ETA ~{eta:.1f} min, "
                f"metric={float(mean_val):.3f}",
                flush=True,
            )

        # --------------------------------------------------------------- run
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, 0, _rng)
        if config["TRAINING_MODE"] == "joint":
            # One loop, nothing frozen: gameplay and bargaining learn together for
            # the whole budget. Returns early -- there is no phase 1 or phase 2 here,
            # so the two-phase bookkeeping below does not apply.
            runner_state = ((train_state, bargain_state, claim_state, env_state,
                             obsv, jnp.array(0), _rng) if report_on else
                            (train_state, bargain_state, env_state, obsv,
                             jnp.array(0), _rng))
            runner_state, metric_j = jax.lax.scan(
                _update_step_joint_bargain, runner_state, None, config["NUM_UPDATES"]
            )
            result = {
                "runner_state": (runner_state[0],),
                "bargain_state": runner_state[1],
                "contract_grid": contract_grid,
                "metrics_joint": metric_j,
            }
            if report_on:
                result["claim_state"] = runner_state[2]
            return result

        if config["TRAINING_MODE"] == "combined":
            # Single-stage contracting: one loop over the whole budget, negotiating
            # afresh each episode. Returns early for the same reason `joint` does --
            # there is no phase split, so the two-phase bookkeeping below is not just
            # unnecessary but would misreport what ran.
            runner_state = (train_state, negotiate_state, env_state, obsv, 0, _rng)
            runner_state, metric_c = jax.lax.scan(
                _update_step_combined, runner_state, None, config["NUM_UPDATES"]
            )
            return {
                "runner_state": (runner_state[0],),
                "negotiate_state": runner_state[1],
                "contract_grid": contract_grid,
                "metrics_combined": metric_c,
            }

        if config["NUM_UPDATES_PHASE1"] == 0:
            # PHASE1_FROM: the policy is already trained, so there is nothing to
            # scan over. Skipped in Python rather than run as a zero-length scan so
            # metrics_phase1 is absent rather than empty.
            metric1 = None
        else:
            runner_state, metric1 = jax.lax.scan(
                _update_step_phase1, runner_state, None, config["NUM_UPDATES_PHASE1"]
            )

        # FREEZE the gameplay policy: phase 2 only ever reads these params.
        train_state, env_state, last_obs, update_step, rng = runner_state
        frozen_params = [ts.params for ts in train_state]

        out = {
            "runner_state": (train_state,),
            "contract_grid": contract_grid,
        }
        if metric1 is not None:
            out["metrics_phase1"] = metric1

        if config["NUM_UPDATES_PHASE2"] == 0:
            # PHASE1_ONLY: stop here. _runner saves the gameplay policies and prints
            # the glob to hand to PHASE1_FROM.
            return out

        if phase2_mode == "solver":
            # No contracting policies to carry: the solver reads the frozen critic
            # and picks a contract, so phase 2 learns nothing and stores nothing.
            runner_state2 = (frozen_params, env_state, last_obs, update_step, rng)
            runner_state2, metric2 = jax.lax.scan(
                _update_step_phase2_solver, runner_state2, None,
                config["NUM_UPDATES_PHASE2"],
            )
        elif phase2_mode == "negotiate":
            runner_state2 = (frozen_params, negotiate_state, env_state, last_obs,
                             update_step, rng)
            runner_state2, metric2 = jax.lax.scan(
                _update_step_phase2_negotiate, runner_state2, None,
                config["NUM_UPDATES_PHASE2"],
            )
            out["negotiate_state"] = runner_state2[1]
        else:
            runner_state2 = (frozen_params, proposal_state, voting_state,
                             env_state, last_obs, update_step, rng)
            runner_state2, metric2 = jax.lax.scan(
                _update_step_phase2, runner_state2, None, config["NUM_UPDATES_PHASE2"]
            )
            out["proposal_state"] = runner_state2[1]
            out["voting_state"] = runner_state2[2]

        out["metrics_phase2"] = metric2
        return out

    return train
