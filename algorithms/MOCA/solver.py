"""Sampling-based negotiation solver -- the authors' phase 2 for Clean Up.

This is `NegotiationSolver` from the reference implementation
(github.com/Algorithmic-Alignment-Lab/contracts,
environments/two_stage_train.py). It is what actually produced the paper's Cleanup
numbers: `experiment_configs/cleanup-contracting.json` sets `"solver": true`, and
`utils/ray_config_utils.py` then disables the learned negotiation stage outright::

    if params_dict['joint'] or params_dict['separate'] or params_dict['combined'] \
            or params_dict['solver']:
        params_dict['negotiate'] = False

So for Cleanup there is no proposal game and no proposer at all. Instead, at every
episode reset the solver

  1. samples `contract_samples` contracts uniformly from the CONTINUOUS contract
     space, and prepends the null contract;
  2. scores each with the FROZEN CRITIC, V_i(s_0, theta) -- one forward pass per
     (agent, contract), no rollouts;
  3. picks one by a decision rule, and plays the episode under it.

Two decision rules, quoting the reference docstring:

    1) max: find the contract that maximizes sum of agent values
    2) majority: find the contract that maximizes sum of agent values subject to
       majority of agents accepting the contract.
    Note - an agent accepts a contract if V(s,c) >= V(s,0) i.e., the agent gets a
    higher value with that contract compared to a null contract

`majority` is the reference default and the rule used for the reported results.

Note what this does NOT contain, since it bears directly on how the results should
be read: there is no single proposer extracting surplus, and acceptance is by
majority rather than unanimity. The "proposer holds every other agent to its
disagreement payoff" argument (the paper's Proposition 4.5) describes the LEARNED
negotiation stage, which Cleanup never ran.
"""
import jax
import jax.numpy as jnp


def sample_contract_candidates(key, contract, num_samples: int, num_envs: int):
    """Candidate contracts for one negotiation, with the null contract first.

    The reference samples from `gym.spaces.Box(low, high)` -- CONTINUOUS, not a
    grid. Each env negotiates independently, as each is a separate episode reset,
    so candidates are drawn per env.

    Returns:
        (num_samples + 1, num_envs) float32. Row 0 is the null contract, which
        `select_contract` relies on as both the disagreement point and the
        guaranteed-feasible fallback.
    """
    sampled = jax.random.uniform(
        key, (num_samples, num_envs), minval=contract.low, maxval=contract.high
    )
    null = jnp.full((1, num_envs), contract.low, dtype=jnp.float32)
    return jnp.concatenate([null, sampled.astype(jnp.float32)], axis=0)


def contract_values(networks, params, obs_batch, contract, thetas):
    """V_i(s_0, theta) for every agent and every candidate contract.

    This is the frozen critic evaluated at the episode's initial state, which is
    the whole reason the solver is cheap: no rollout is needed to score a contract.

    Args:
        networks: per-agent ContractActorCritic modules.
        params: per-agent frozen parameters.
        obs_batch: (N, num_envs, ...) initial observations, agent-major.
        contract: the contract space (for `to_obs`).
        thetas: (K, num_envs) candidate contracts.

    Returns:
        (K, N, num_envs) float32 values.
    """
    num_agents = len(networks)

    def values_for(theta_row):                       # (num_envs,)
        contract_obs = contract.to_obs(theta_row)    # (num_envs, obs_dim)
        return jnp.stack([
            networks[i].apply(params[i], obs_batch[i], contract_obs)[1]
            for i in range(num_agents)
        ])                                           # (N, num_envs)

    # lax.map scans rather than vmapping over K, keeping peak memory at one
    # contract's worth of CNN activations instead of K.
    return jax.lax.map(values_for, thetas)


BARGAINING_RULES = ("nash", "kalai_smorodinsky", "egalitarian")
DECISION_RULES = ("majority", "max") + BARGAINING_RULES


def _bargaining_select(values, rule: str):
    """Axiomatic bargaining solutions over the sampled contracts.

    These differ from the reference's rules in two ways, both deliberate:

    * Participation is UNANIMOUS. A bargaining solution is defined on the set of
      outcomes that dominate the disagreement point for EVERY agent, so a contract
      a single agent dislikes is infeasible -- where `majority` would let it pass.
    * Feasibility uses a weak inequality (gain >= 0), the axiomatic convention,
      against the reference's strict `>` for counting acceptances. The null contract
      therefore always sits in the feasible set with a gain of exactly 0, so the
      mechanism can decline.

    The disagreement point is V_i(s_0, 0), the same object the acceptance rule and
    the solver already use.
    """
    disagreement = values[0]                          # (N, num_envs)
    gains = values - disagreement[None, :, :]         # (K, N, num_envs)
    feasible = (gains >= 0.0).all(axis=1)             # (K, num_envs), unanimity
    feasible = feasible.at[0].set(True)

    if rule == "egalitarian":
        score = gains.min(axis=1)                     # maximin on absolute gain
    elif rule == "nash":
        # Sum of logs, not a raw product: with several agents and gains of order
        # 100 the product overflows, and log is monotone so the argmax is identical.
        # Infeasible entries are masked out below, so clipping at ~0 only affects
        # the null contract, which correctly scores worst.
        score = jnp.log(jnp.maximum(gains, 1e-12)).sum(axis=1)
    elif rule == "kalai_smorodinsky":
        # Each agent's best attainable gain over the FEASIBLE set, so the ideal
        # point respects participation rather than being read off contracts nobody
        # would sign.
        masked = jnp.where(feasible[:, None, :], gains, -jnp.inf)
        ideal = jnp.maximum(masked.max(axis=0), 1e-12)          # (N, num_envs)
        score = (gains / ideal[None, :, :]).min(axis=1)
    else:
        raise ValueError(f"unknown bargaining rule {rule!r}")

    return jnp.argmax(jnp.where(feasible, score, -jnp.inf), axis=0).astype(jnp.int32)


def select_contract(values, rule: str = "majority"):
    """Index of the chosen contract per env, given (K, N, num_envs) critic values.

    Row 0 of `values` must be the null contract: it supplies each agent's
    disagreement value V_i(s_0, 0) and is always left feasible, so the solver can
    fall back to "no contract" when nothing else commands a majority. The reference
    builds this in by seeding the accepted list with the null contract before
    filtering (`accepted_vals, accepted_params = [default_vals], [all_params[0]]`).

    Returns:
        (num_envs,) int32 index into the leading axis of `values`.
    """
    if rule not in DECISION_RULES:
        raise ValueError(
            f"unknown decision rule {rule!r} (available: {', '.join(DECISION_RULES)})"
        )
    if rule in BARGAINING_RULES:
        return _bargaining_select(values, rule)

    welfare = values.sum(axis=1)                     # (K, num_envs)
    if rule == "max":
        return jnp.argmax(welfare, axis=0).astype(jnp.int32)

    num_agents = values.shape[1]
    disagreement = values[0]                         # (N, num_envs)
    # An agent accepts iff the contract beats its null-contract value.
    accepts = (values > disagreement[None, :, :]).sum(axis=1)   # (K, num_envs)
    # Reference condition is `accepted >= rejected`, i.e. accepts >= N - accepts.
    eligible = 2 * accepts >= num_agents
    eligible = eligible.at[0].set(True)              # null contract always feasible
    masked = jnp.where(eligible, welfare, -jnp.inf)
    return jnp.argmax(masked, axis=0).astype(jnp.int32)


def negotiate(key, networks, params, obs_batch, contract, num_samples, rule="majority"):
    """One full solver negotiation: sample, score, choose.

    Returns:
        theta: (num_envs,) chosen contract per env.
        info: diagnostics -- how often the solver settled for the null contract,
            and how many agents would accept the chosen contract.
    """
    num_envs = obs_batch.shape[1]
    thetas = sample_contract_candidates(key, contract, num_samples, num_envs)
    values = contract_values(networks, params, obs_batch, contract, thetas)
    best = select_contract(values, rule)
    theta = jnp.take_along_axis(thetas, best[None, :], axis=0)[0]

    disagreement = values[0]
    chosen_values = jnp.take_along_axis(values, best[None, None, :], axis=0)[0]  # (N, E)
    info = {
        "solver_null_rate": (best == 0).mean(),
        "solver_accept_count": (chosen_values > disagreement).sum(axis=0).mean(),
        "solver_value_gain": (chosen_values - disagreement).sum(axis=0).mean(),
        # Predicted welfare at the chosen contract. Logged so it can be compared
        # against the welfare the episode ACTUALLY produces: every rule here selects
        # on critic estimates, so if the two diverge the whole solver family is
        # choosing on bad information and the comparison between rules is moot.
        "solver_predicted_welfare": chosen_values.sum(axis=0).mean(),
    }
    return theta, info
