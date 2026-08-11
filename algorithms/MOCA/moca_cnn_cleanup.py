"""MOCA (Mutually Optimal Contract Algorithm) on Clean Up.

Implements Algorithm 1 of Christoffersen et al., "Formal Contracts Mitigate Social
Dilemmas in Multi-Agent RL" (arXiv:2208.10469 / AAMAS 2023):

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

Gameplay rewards are R'_i = R_i + theta_i(s,a) with sum_i theta_i = 0, so contracts
redistribute welfare without creating it (see contracts.py).

Only PARAMETER_SHARING=False is supported: MOCA gives every agent its own proposal
and voting policy, and the specialisation this environment is studied for (some
agents cleaning, others harvesting) requires distinct gameplay policies too.
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
from algorithms.MOCA import bargain, negotiate, solver
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
STAGE1_METRICS = (
    # Learning curve and the emergent division of labour it produces.
    "returned_episode_returns_mean",   # per-agent episode return
    "returned_episode_returns_std",    # spread ACROSS agents = specialisation
    "shaped_rewards_mean",             # read by progress_callback
    "cleaned_by_agent_mean",           # public-good provision
    "cleaned_by_agent_std",            # is cleaning concentrated in a few agents?
    "waste_cleared_mean",              # river state: cells NOT dirt, higher = cleaner
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
    # Does the policy CONDITION on theta? cleaned_gap must open up; if it stays at
    # ~0 the contract is inert and every phase-2 comparison downstream is moot.
    "cleaned_null",
    "cleaned_contracted",
    "cleaned_gap",
    # V_i(s, 0), the disagreement point phase 2 measures acceptance against.
    "welfare_null",
    "welfare_contracted",
)

STAGE2_METRICS = (
    # What is offered, what ends up in force, and whether it is signed at all.
    "contract_theta_proposed",
    "contract_theta_effective",
    "contract_accept_rate",
    # Convergence diagnostic. Starts at log(NUM_CONTRACT_BINS) and MUST fall: a run
    # that ends at the maximum has learned nothing, whatever its argmax says.
    "contract_proposal_entropy",
    # Did the contract change behaviour? Transfers are zero-sum, so welfare can
    # only move if cleaning does.
    "cleaned_by_agent_mean",
    "waste_cleared_mean",              # river state: cells NOT dirt, higher = cleaner
    "transfer_volume",
    # Headline outcomes.
    "welfare",
    "equality",
)

# Solver phase 2 has no proposal or acceptance policy to track, so its diagnostics
# are about what the sampling search settled on instead.
STAGE2_SOLVER_METRICS = (
    "contract_theta_effective",
    "contract_theta_std",       # do the per-env negotiations agree?
    "solver_null_rate",         # how often nothing beat the null contract
    "solver_accept_count",      # agents preferring the chosen contract to null
    "solver_predicted_welfare",  # critic's estimate; compare against `welfare`
    "cleaned_by_agent_mean",
    "waste_cleared_mean",              # river state: cells NOT dirt, higher = cleaner
    "welfare",                  # realised
    "equality",
)

# Learned negotiation stage: a proposal game again, so the diagnostics are what
# agent 0 offers and whether the polled agents sign it.
STAGE2_NEGOTIATE_METRICS = (
    "contract_theta_proposed",
    "contract_theta_effective",
    "contract_accept_rate",      # realised signings
    "contract_accept_prob",      # the product of the polled agents' probabilities
    "negotiate_policy_entropy",  # falling entropy = the proposal is converging
    "cleaned_by_agent_mean",
    "waste_cleared_mean",              # river state: cells NOT dirt, higher = cleaner
    "welfare",
    "equality",
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
    "theta_agreed": "contract/theta_agreed",
    "theta_offered": "contract/theta_offered",
    "accept_count": "contract/accept_count",
    "transfer_volume": "contract/transfer_volume",
    # Did behaviour actually change? Transfers are zero-sum, so welfare can only
    # move if cleaning does.
    "cleaned_by_agent_mean": "behaviour/cleaned_per_agent",
    "waste_cleared_mean": "behaviour/waste_cleared",
    # The two the result is stated in.
    "welfare": "outcome/welfare",
    "equality": "outcome/equality",
}


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


def make_train(config):
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])

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

    # ---- Rubinstein bargaining, trained jointly -----------------------------
    # A separate training schedule, not a phase-2 protocol: gameplay and bargaining
    # learn together and nothing is ever frozen. It exists to get the GAME right
    # before layering MOCA's two-phase construction on top of it.
    training_mode = config.get("TRAINING_MODE", "two_phase")
    if training_mode not in ("two_phase", "joint"):
        raise ValueError(
            f"TRAINING_MODE must be 'two_phase' or 'joint', got {training_mode!r}")
    config["TRAINING_MODE"] = training_mode
    if (training_mode == "joint") != (phase2_mode == "bargain"):
        raise ValueError(
            "TRAINING_MODE='joint' and PHASE2_MODE='bargain' currently go together: "
            "joint has no other protocol implemented, and bargain has no two-phase "
            f"variant yet. Got TRAINING_MODE={training_mode!r}, "
            f"PHASE2_MODE={phase2_mode!r}."
        )
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
                             ("BARGAIN_GAE_LAMBDA", 0.95), ("BARGAIN_HIDDEN", 64)):
            config.setdefault(key, default)
        # Not a free hyperparameter: impatience is already realised as reward lost to
        # disagreement in the environment, so discounting rounds on top would count
        # the same delay cost twice and hand the proposer an advantage it has not
        # earned. See bargain.round_gae.
        if float(config.setdefault("BARGAIN_GAMMA", 1.0)) != 1.0:
            print(f"[MOCA warning] BARGAIN_GAMMA={config['BARGAIN_GAMMA']} != 1.0. "
                  f"Delay is already costly in-environment; an extra discount "
                  f"double-counts it.", flush=True)
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

    contract = make_contract(
        config.get("CONTRACT_SPACE", "cleanup"),
        num_agents,
        low=config["CONTRACT_LOW"],
        high=config["CONTRACT_HIGH"],
    )
    contract_grid = contract.grid(config["NUM_CONTRACT_BINS"])
    # Plain Python copy of the grid, purely for building metric NAMES. Formatting a
    # device array with float() fails under tracing, and label text must never depend
    # on a traced value anyway. Read off the grid itself rather than recomputed from
    # low/high: the grid is not a plain linspace when the range excludes weak
    # contracts (index 0 is then the null contract), and labels that disagree with it
    # would mislabel every proposal-probability series.
    contract_grid_labels = [float(x) for x in np.asarray(contract_grid)]

    env = LogWrapper(env, replace_info=False)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES_PHASE1"]
        )
        return config["LR"] * frac

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
        bargain_net = [
            BargainingActorCritic(
                hidden=int(config.get("BARGAIN_HIDDEN", 64)),
                activation=config["ACTIVATION"],
                accept_bias=float(config.get("BARGAIN_ACCEPT_BIAS", 1.0)),
            )
            for _ in range(num_agents)
        ]
        init_b = jnp.zeros((1, bargain.feature_dim(num_agents)))
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
                transfers = contract.compute_transfer(theta, info["cleaned_by_agent"])
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

            # bootstrap value
            contract_obs = contract.to_obs(theta)
            last_obs_batch = jnp.transpose(last_obs, (1, 0, 2, 3, 4))
            last_val = []
            for i in range(num_agents):
                _, v = network[i].apply(train_state[i].params, last_obs_batch[i], contract_obs)
                last_val.append(v)

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

            update_step = update_step + 1
            jax.debug.callback(checkpoint_callback, train_state, update_step)

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
            out["contract_theta_sampled"] = theta.mean()

            # Is the policy actually CONDITIONING on theta? The headline question of
            # phase 1, and invisible in the pooled averages above: a policy that
            # cleans identically at theta=0 and theta>0 produces exactly the same
            # cleaned_by_agent_mean as one that has learned the distinction.
            #
            # Splitting by regime makes it a live training curve rather than a
            # post-hoc grid_eval run. cleaned_contracted - cleaned_null is the thing
            # that must open up; welfare_null is the disagreement point V_i(s, 0)
            # that phase 2's acceptance rules will be measured against, so it doubles
            # as a check that the null baseline is not itself drifting.
            null_mask = contract.is_null(theta).astype(jnp.float32)     # (NUM_ENVS,)
            n_null = jnp.maximum(null_mask.sum(), 1.0)
            n_contracted = jnp.maximum((1.0 - null_mask).sum(), 1.0)
            cleaned_per_env = jnp.stack([
                traj_batch[i].info["cleaned_by_agent"].squeeze(-1)
                for i in range(num_agents)
            ]).sum(axis=(0, 1))                                          # (NUM_ENVS,)
            welfare_per_env = stats["returns"].sum(axis=0)               # (NUM_ENVS,)
            for name, per_env in (("cleaned", cleaned_per_env),
                                  ("welfare", welfare_per_env)):
                out[f"{name}_null"] = (per_env * null_mask).sum() / n_null
                out[f"{name}_contracted"] = (
                    (per_env * (1.0 - null_mask)).sum() / n_contracted
                )
            out["cleaned_gap"] = out["cleaned_contracted"] - out["cleaned_null"]
            out["contract_null_frac"] = null_mask.mean()

            # Namespaced by stage, as the reference logger does (stage_1/..., stage_2/...):
            # the two phases measure different things, so sharing a key would splice a
            # subgame-learning curve onto a contract-negotiation curve.
            out = {f"stage_1/{k}": v for k, v in _select(out, STAGE1_METRICS).items()}
            out["phase"] = jnp.float32(1.0)
            out["update_step"] = update_step
            out["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            jax.debug.callback(log_callback, out)
            jax.debug.callback(
                progress_callback, update_step, out["stage_1/shaped_rewards_mean"], 1
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
        # PHASE 2 (negotiate) -- Algorithm 1's contracting game: agent 0
        # proposes, nu sampled agents accept with some probability, then the
        # frozen policy plays the episode. See algorithms/MOCA/negotiate.py.
        # =================================================================
        def _update_step_phase2_negotiate(runner_state, unused):
            (frozen_params, negotiate_state, env_state, last_obs,
             update_step, rng) = runner_state

            low, high = contract.low, contract.high
            obs_batch = jnp.transpose(last_obs, (1, 0, 2, 3, 4))   # (N, E, ...)
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

            # -- play the episode with the FROZEN gameplay policy --
            traj_batch, env_state, last_obs, rng = rollout(
                frozen_params, env_state, last_obs, theta_eff, rng
            )
            returns = jnp.stack(
                [traj_batch[i].reward.sum(axis=0) for i in range(num_agents)]
            )                                                    # (N, E)

            # Reward lands only on the agreement step, as in the reference: the
            # proposal step returns zeros and the agreement step returns the
            # accumulated episode reward.
            rewards = jnp.stack([jnp.zeros_like(returns), returns])   # (2, N, E)
            values = jnp.stack([val0, val1])                          # (2, N, E)
            advantages, targets = negotiate.two_step_gae(
                rewards, values, config["GAMMA"], config["GAE_LAMBDA"]
            )
            raws = jnp.stack([raw0, raw1])                            # (2, N, E, A)
            logps = jnp.stack([logp0, logp1])                         # (2, N, E)

            # -- PPO on the two-step negotiation episode --
            def ppo_loss(params, i):
                adv_i = advantages[:, i]
                adv_i = (adv_i - adv_i.mean()) / (adv_i.std() + 1e-8)
                loss = 0.0
                for t, cobs in enumerate((obs_propose, obs_agree)):
                    pi, value = negotiate_net[i].apply(params, obs_batch[i], cobs)
                    logp = pi.log_prob(raws[t, i])
                    ratio = jnp.exp(logp - logps[t, i])
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
            out["contract_theta_proposed"] = theta_prop.mean()
            out["contract_theta_effective"] = theta_eff.mean()
            out["contract_accept_rate"] = accepted.mean()
            out["contract_accept_prob"] = prod_prob.mean()
            # Gaussian entropy of the proposer's policy, recomputed post-update:
            # the analogue of the categorical entropy the REINFORCE mode tracks,
            # and the same convergence question -- is the proposal narrowing?
            post_pi, _ = negotiate_net[0].apply(
                negotiate_state[0].params, obs_batch[0], obs_propose
            )
            out["negotiate_policy_entropy"] = post_pi.entropy().mean()
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
            return (loss_actor + config["VF_COEF"] * value_loss
                    - config["ENT_COEF"] * entropy)

        def rollout_bargaining(gameplay_params, bargain_params, env_state, last_obs, rng):
            """One episode of alternating-offers bargaining interleaved with play.

            Scans over ROUNDS; each round makes one bargaining decision and then
            plays `BARGAIN_SEGMENT` steps under whatever contract is in force. A
            scan rather than a Python loop so only one segment body is compiled.

            Returns the gameplay trajectory reshaped to (NUM_STEPS, ...) -- so the
            existing GAE and loss consume it unchanged -- plus the per-round record
            the bargaining update needs.
            """
            K, x = config["BARGAIN_ROUNDS"], config["BARGAIN_SEGMENT"]
            n_envs = config["NUM_ENVS"]
            feat_mask = bargain.feature_mask(config["BARGAIN_FEATURES"], num_agents)
            # Scales so the state arrives at roughly unit range. Episode return is
            # bounded by one apple per step; cleaning likewise.
            ret_scale = float(config["NUM_STEPS"]) * float(
                config["ENV_KWARGS"].get("apple_reward", 1.0))
            clean_scale = float(config["NUM_STEPS"])
            river_scale = float(env.GRID_SIZE_ROW * env.GRID_SIZE_COL)

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
                env_state, last_obs, theta, rng = carry
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

                transfers = contract.compute_transfer(theta, info["cleaned_by_agent"])
                reward = reward + transfers
                info = dict(info)
                info["contract_transfer"] = transfers
                info["contract_theta"] = jnp.broadcast_to(theta[:, None], transfers.shape)

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
                cleaned = info["cleaned_by_agent"]                  # (E, N)
                clear = info["waste_cleared"][:, 0]                 # (E,)
                return (env_state, obsv, theta, rng), (transition, cleaned, clear)

            def _round(carry, r):
                (env_state, last_obs, rng, agreed, locked, cum_return, cum_clean,
                 last_theta_n, had_offer, n_reject, river) = carry
                rng, k_prop, k_theta, k_vote = jax.random.split(rng, 4)

                proposer = bargain.proposer_for_round(
                    r, num_agents, n_envs, config["BARGAIN_PROPOSER"],
                    key=k_prop, contributions=cum_clean, start_offset=start_offset)
                feats = bargain.bargaining_features(
                    r, K, proposer, num_agents, last_theta_n, had_offer, n_reject,
                    cum_return / ret_scale, cum_clean / clean_scale,
                    river / river_scale, feat_mask)                  # (N, E, F)

                raws, lp_t, votes, lp_v, vals = [], [], [], [], []
                kt = jax.random.split(k_theta, num_agents)
                kv = jax.random.split(k_vote, num_agents)
                for i in range(num_agents):
                    pi_theta, pi_vote, v = bargain_net[i].apply(bargain_params[i], feats[i])
                    raw = pi_theta.sample(seed=kt[i])                # (E, 1)
                    vote = pi_vote.sample(seed=kv[i])                # (E,)
                    raws.append(raw[:, 0])
                    lp_t.append(pi_theta.log_prob(raw))
                    votes.append(vote)
                    lp_v.append(pi_vote.log_prob(vote))
                    vals.append(v)
                raw = jnp.stack(raws)                                # (N, E)
                votes = jnp.stack(votes)
                values = jnp.stack(vals)

                theta_all = negotiate.unsquash(raw, contract.low, contract.high)
                mine = bargain.is_proposer_mask(proposer, num_agents)
                theta_offer = jnp.sum(jnp.where(mine, theta_all, 0.0), axis=0)   # (E,)
                passed, n_accept = bargain.accepted(
                    votes.astype(bool), proposer, quorum_b, num_agents)
                newly = passed & ~agreed
                theta_eff = jnp.where(
                    agreed, locked,
                    jnp.where(newly, theta_offer, jnp.float32(contract.null)))

                (env_state, last_obs, _, rng), (traj, cleaned, clear) = jax.lax.scan(
                    _seg_step, (env_state, last_obs, theta_eff, rng), None, x)

                seg_return = jnp.stack(
                    [traj[i].reward.sum(axis=0) for i in range(num_agents)])     # (N,E)
                record = {
                    "feats": feats, "raw": raw, "logp_theta": jnp.stack(lp_t),
                    "vote": votes, "logp_vote": jnp.stack(lp_v), "value": values,
                    "seg_return": seg_return,
                    "active": ~agreed, "newly": newly,
                    "is_proposer": mine, "theta_offer": theta_offer,
                    "n_accept": n_accept, "theta_eff": theta_eff,
                }
                carry = (env_state, last_obs, rng,
                         agreed | newly,
                         jnp.where(newly, theta_offer, locked),
                         cum_return + seg_return,
                         cum_clean + jnp.transpose(cleaned.sum(axis=0)),
                         2.0 * (theta_offer - contract.low)
                         / (contract.high - contract.low) - 1.0,
                         jnp.ones_like(had_offer),
                         n_reject + (~agreed & ~passed).astype(jnp.int32),
                         clear[-1].astype(jnp.float32))
                return carry, (record, traj)

            zeros_e = jnp.zeros((n_envs,), jnp.float32)
            init = (env_state, last_obs, rng,
                    jnp.zeros((n_envs,), bool), jnp.full((n_envs,), contract.null),
                    jnp.zeros((num_agents, n_envs), jnp.float32),
                    jnp.zeros((num_agents, n_envs), jnp.float32),
                    zeros_e, jnp.zeros((n_envs,), bool),
                    jnp.zeros((n_envs,), jnp.int32), zeros_e)
            carry, (rounds, traj) = jax.lax.scan(_round, init, jnp.arange(K))
            env_state, last_obs, rng, agreed, locked = carry[0], carry[1], carry[2], carry[3], carry[4]

            # (K, x, ...) -> (NUM_STEPS, ...), so the existing GAE/loss are unchanged.
            traj_batch = [
                jax.tree.map(lambda z: z.reshape((K * x,) + z.shape[2:]), traj[i])
                for i in range(num_agents)
            ]

            # Semi-MDP reward. A rejected round pays only its own null segment; the
            # round where the offer is accepted pays every remaining segment at once,
            # because bargaining ends there and those segments are its consequence.
            active = rounds["active"]                                    # (K, E)
            post = jnp.sum(
                jnp.where(active[:, None, :], 0.0, rounds["seg_return"]), axis=0)  # (N,E)
            rounds["reward"] = rounds["seg_return"] + (
                rounds["newly"][:, None, :] * post[None, :, :])
            last_round = jnp.arange(K)[:, None] == (K - 1)
            rounds["terminal"] = rounds["newly"] | last_round
            final_theta = jnp.where(agreed, locked, jnp.float32(contract.null))
            return traj_batch, rounds, env_state, last_obs, final_theta, rng

        def _update_step_joint_bargain(runner_state, unused):
            (train_state, bargain_state, env_state, last_obs, update_step, rng) = runner_state
            params_list = [ts.params for ts in train_state]
            b_params = [bs.params for bs in bargain_state]

            (traj_batch, rounds, env_state, last_obs, final_theta,
             rng) = rollout_bargaining(params_list, b_params, env_state, last_obs, rng)

            # ---------------------------------------------------- gameplay PPO
            contract_obs = contract.to_obs(final_theta)
            last_obs_batch = jnp.transpose(last_obs, (1, 0, 2, 3, 4))
            last_val = [network[i].apply(params_list[i], last_obs_batch[i],
                                         contract_obs)[1] for i in range(num_agents)]
            for i in range(num_agents):
                advantages_i, targets_i = compute_gae(traj_batch[i], last_val[i])

                def _epoch(state, unused, i=i):
                    def _minibatch(ts, b):
                        tb, adv, tgt = b
                        grads = jax.grad(_bargain_ppo_loss)(
                            ts.params, tb, adv, tgt, network[i])
                        return ts.apply_gradients(grads=grads), None

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
                    ts, _ = jax.lax.scan(_minibatch, ts, mbs)
                    return (ts, tb, adv, tgt, rng_), None

                state = (train_state[i], traj_batch[i], advantages_i, targets_i, rng)
                state, _ = jax.lax.scan(_epoch, state, None, config["UPDATE_EPOCHS"])
                train_state[i] = state[0]
                rng = state[-1]

            # -------------------------------------------------- bargaining PPO
            adv_b, targ_b = bargain.round_gae(
                rounds["reward"], rounds["value"], rounds["active"][:, None, :],
                rounds["terminal"][:, None, :],
                config["BARGAIN_GAMMA"], config["BARGAIN_GAE_LAMBDA"])

            def bargain_loss(params, i):
                pi_theta, pi_vote, value = bargain_net[i].apply(
                    params, rounds["feats"][:, i])                  # (K, E, ...)
                a = adv_b[:, i]
                a = (a - a.mean()) / (a.std() + 1e-8)
                act = rounds["active"].astype(jnp.float32)          # (K, E)
                mine = rounds["is_proposer"][:, i].astype(jnp.float32)
                w_prop, w_vote = act * mine, act * (1.0 - mine)
                eps = config["BARGAIN_CLIP_EPS"]

                def clipped(logp, old_logp, w):
                    ratio = jnp.exp(logp - old_logp)
                    obj = jnp.minimum(
                        ratio * a, jnp.clip(ratio, 1.0 - eps, 1.0 + eps) * a)
                    return -(obj * w).sum() / (w.sum() + 1e-8)

                # An agent proposes OR votes in a given round, never both, so the two
                # heads are trained on disjoint masks. Rounds after agreement carry no
                # decision and are excluded from all three terms.
                loss = clipped(pi_theta.log_prob(rounds["raw"][:, i][..., None]),
                               rounds["logp_theta"][:, i], w_prop)
                loss = loss + clipped(pi_vote.log_prob(rounds["vote"][:, i]),
                                      rounds["logp_vote"][:, i], w_vote)
                v_loss = (jnp.square(value - targ_b[:, i]) * act).sum() / (act.sum() + 1e-8)
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

            update_step = update_step + 1
            jax.debug.callback(checkpoint_callback, train_state, update_step)
            jax.debug.callback(bargain_checkpoint_callback, bargain_state, update_step)

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
            # Round of agreement, K if the episode never agreed -- so the mean is
            # directly "how many rounds of bargaining were burned".
            round_idx = jnp.arange(K)[:, None]
            agree_round = jnp.where(
                agreed_any, jnp.sum(jnp.where(rounds["newly"], round_idx, 0), axis=0), K)
            out["agreement_rate"] = agreed_any.mean()
            out["agreement_round"] = agree_round.mean()
            out["disagreement_steps"] = agree_round.mean() * config["BARGAIN_SEGMENT"]
            out["contract_in_force_rate"] = (
                rounds["theta_eff"] > contract.null).mean()
            # theta actually agreed, averaged over the envs that agreed at all.
            agreed_theta = jnp.sum(jnp.where(rounds["newly"], rounds["theta_offer"], 0.0),
                                   axis=0)
            out["theta_agreed"] = (jnp.sum(agreed_theta) /
                                   jnp.maximum(agreed_any.sum(), 1.0))
            # Masked by `active`: agents still emit an offer in rounds after
            # agreement, but it is never read, so averaging it in would report
            # untrained noise as the policy's asking price.
            n_active = jnp.maximum(rounds["active"].sum(), 1.0)
            out["theta_offered"] = (
                rounds["theta_offer"] * rounds["active"]).sum() / n_active
            out["accept_count"] = (
                rounds["n_accept"] * rounds["active"]).sum() / n_active
            out = {f"joint/{JOINT_BARGAIN_METRICS[k]}": v
                   for k, v in _select(out, tuple(JOINT_BARGAIN_METRICS)).items()}
            out["phase"] = jnp.float32(3.0)
            out["update_step"] = update_step
            out["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            jax.debug.callback(log_callback, out)
            jax.debug.callback(
                progress_callback, update_step, out["joint/outcome/welfare"], 3)

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
            runner_state = (train_state, bargain_state, env_state, obsv,
                            jnp.array(0), _rng)
            runner_state, metric_j = jax.lax.scan(
                _update_step_joint_bargain, runner_state, None, config["NUM_UPDATES"]
            )
            return {
                "runner_state": (runner_state[0],),
                "bargain_state": runner_state[1],
                "contract_grid": contract_grid,
                "metrics_joint": metric_j,
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


SINGLE_RUN_KWARGS = {"wandb_name": "moca_cnn_cleanup"}
TUNE_KWARGS = {"sweep_name": "cleanup"}
