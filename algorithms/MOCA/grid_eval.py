"""Tabulate V_i(s_0, theta) for a frozen MOCA phase-1 policy.

Phase 2 -- in every mode -- is a choice over a single scalar made against a FROZEN
gameplay policy. Everything it decides is therefore a deterministic function of one
table:

    V_i(theta) = agent i's expected episode return with contract theta in force

This script measures that table directly instead of inferring it from a learned
proposal, which separates three questions that are otherwise indistinguishable from
a training run alone:

  1. Does the policy respond to theta at all? Read cleaned/step against theta. Flat
     means contracts cannot change behaviour, and the null contract is then the
     correct answer rather than a failure -- a proposer who is not a cleaner should
     offer 0 and free-ride, because paying buys it nothing.
  2. What SHOULD phase 2 converge to? argmax over theta of the proposer's own
     return, subject to the others accepting. Comparing that to what phase 2 did
     converge to distinguishes "the equilibrium is unattractive" from "phase 2 never
     converged" -- the ambiguity that made the first broken run so hard to read.
  3. How unfair is the outcome? With the whole table in hand the disagreement point
     V_i(0), the individually-rational set, and the utilitarian/egalitarian optima
     are all directly computable, so "cleaners get too little" becomes a measured
     gap rather than an impression.

Cost is roughly one phase-2 update for the whole sweep, and unlike learned phase 2
every episode is informative: theta is in force in all of them.

Usage:
    python algorithms/MOCA/grid_eval.py \
        --checkpoint 'runs/moca_baseline/clean_up_seed42_..._negotiate*.pkl' \
        --num-agents 5 --contract-high 0.2 \
        --env-kwarg dirt_spawn_cells=1 --env-kwarg dirtSpawnProbability=0.5
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp
import numpy as np

import socialjax
from socialjax.wrappers.baselines import LogWrapper
from algorithms.utils import load_params
from algorithms.MOCA.contracts import CleanupContract
from algorithms.MOCA.networks import ContractActorCritic


def rollout_at_theta(env, network, params, contract, theta, num_envs, num_steps, key):
    """One batch of full episodes with `theta` in force. Returns per-agent metrics.

    Rewards are contract-augmented exactly as in training (R'_i = R_i + transfer_i),
    and base and transfer components are kept apart -- the split between what an
    agent earns and what the contract moves is the whole fairness question.
    """
    num_agents = env.num_agents
    key, k_reset = jax.random.split(key)
    obsv, env_state = jax.vmap(env.reset)(jax.random.split(k_reset, num_envs))
    contract_obs = contract.to_obs(jnp.full((num_envs,), theta))

    def step(carry, _):
        env_state, last_obs, rng = carry
        rng, k_act, k_step = jax.random.split(rng, 3)
        obs_batch = jnp.transpose(last_obs, (1, 0, 2, 3, 4))      # (N, E, ...)
        act_keys = jax.random.split(k_act, num_agents)
        actions = []
        for i in range(num_agents):
            pi, _ = network.apply(params[i], obs_batch[i], contract_obs)
            actions.append(pi.sample(seed=act_keys[i]))
        obsv, env_state, reward, done, info = jax.vmap(env.step)(
            jax.random.split(k_step, num_envs), env_state, actions
        )
        cleaned = info["cleaned_by_agent"]                         # (E, N)
        transfers = contract.compute_transfer(jnp.full((num_envs,), theta), cleaned)
        return (env_state, obsv, rng), (reward, transfers, cleaned)

    (_, _, _), (reward, transfers, cleaned) = jax.lax.scan(
        step, (env_state, obsv, key), None, num_steps
    )                                                              # each (T, E, N)
    base = reward.sum(axis=0)                                      # (E, N)
    moved = transfers.sum(axis=0)
    return {
        "base_return": base.mean(axis=0),                          # (N,)
        "transfer": moved.mean(axis=0),
        "return": (base + moved).mean(axis=0),
        "cleaned_per_step": cleaned.mean(axis=(0, 1)),
        "transfer_volume": jnp.maximum(transfers, 0.0).sum(axis=(0, 2)).mean(),
    }


def evaluate_grid(env, params, contract, thetas, num_envs, num_steps, seed):
    network = ContractActorCritic(env.action_space().n, activation="relu")
    rows = []
    key = jax.random.PRNGKey(seed)
    for t in thetas:
        key, k = jax.random.split(key)
        # Same key per theta would be better paired, but the env reset already
        # dominates the variance and a shared key would correlate the estimates.
        out = rollout_at_theta(env, network, params, contract, float(t),
                               num_envs, num_steps, k)
        rows.append(jax.tree.map(np.asarray, out))
        print(f"  theta={t:.4f} done", flush=True)
    return rows


def _gini_equality(v):
    diffs = np.abs(v[:, None] - v[None, :]).sum()
    return 1.0 - diffs / (2.0 * len(v) * np.abs(v).sum() + 1e-8)


def report(thetas, rows, num_agents):
    ret = np.stack([r["return"] for r in rows])                    # (K, N)
    base = np.stack([r["base_return"] for r in rows])
    clean = np.stack([r["cleaned_per_step"] for r in rows])
    welfare = ret.sum(axis=1)

    print("\n=== cleaning per step, per agent (the theta-responsiveness test) ===")
    hdr = "  theta  " + "".join(f"  ag{i:<6d}" for i in range(num_agents)) + "   total"
    print(hdr)
    for k, t in enumerate(thetas):
        print(f"  {t:.3f}  " + "".join(f"  {c:7.3f}" for c in clean[k])
              + f"  {clean[k].sum():7.3f}")

    print("\n=== episode return, per agent (base + transfers) ===")
    print(hdr.replace("   total", "  welfare  equality"))
    for k, t in enumerate(thetas):
        print(f"  {t:.3f}  " + "".join(f"  {v:7.2f}" for v in ret[k])
              + f"  {welfare[k]:7.2f}  {_gini_equality(ret[k]):7.3f}")

    print("\n=== base return only, before transfers ===")
    for k, t in enumerate(thetas):
        print(f"  {t:.3f}  " + "".join(f"  {v:7.2f}" for v in base[k]))

    # --- the questions the table exists to answer ---
    d = ret[0]                                                     # V_i(0)
    print("\n=== diagnosis ===")
    span = clean.sum(axis=1)
    print(f"  total cleaning at theta=0 : {span[0]:.3f} cells/step")
    print(f"  total cleaning at theta=max: {span[-1]:.3f} cells/step")
    if span.max() - span.min() < 0.05 * max(span.max(), 1e-6):
        print("  -> FLAT in theta: the policy does not gate cleaning on payment, so no")
        print("     contract can change behaviour and theta=0 is the correct proposal.")
    else:
        print("  -> cleaning RESPONDS to theta: contracts can change behaviour.")

    print(f"\n  proposer (agent 0) return by theta:")
    for k, t in enumerate(thetas):
        mark = "  <-- best for agent 0" if k == int(np.argmax(ret[:, 0])) else ""
        print(f"    theta={t:.3f}  V_0={ret[k, 0]:8.2f}{mark}")

    ir = np.array([bool(np.all(ret[k] >= d - 1e-6)) for k in range(len(thetas))])
    print(f"\n  individually rational (every agent >= its theta=0 return): "
          f"{[f'{t:.3f}' for t, ok in zip(thetas, ir) if ok]}")
    if ir.any():
        best_prop = int(np.argmax(np.where(ir, ret[:, 0], -np.inf)))
        print(f"  MOCA-style outcome (agent 0's best IR contract): theta="
              f"{thetas[best_prop]:.3f}, welfare={welfare[best_prop]:.2f}, "
              f"equality={_gini_equality(ret[best_prop]):.3f}")
    print(f"  utilitarian optimum: theta={thetas[int(np.argmax(welfare))]:.3f}, "
          f"welfare={welfare.max():.2f}")
    gains = ret - d
    egal = int(np.argmax(gains.min(axis=1)))
    print(f"  egalitarian (maximin gain over no contract): theta={thetas[egal]:.3f}, "
          f"welfare={welfare[egal]:.2f}, equality={_gini_equality(ret[egal]):.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True,
                   help="glob matching the per-agent GAMEPLAY policies")
    p.add_argument("--env", default="clean_up")
    p.add_argument("--num-agents", type=int, required=True)
    p.add_argument("--contract-low", type=float, default=0.0)
    p.add_argument("--contract-high", type=float, required=True,
                   help="must match the run's CONTRACT_HIGH; theta is normalised by it")
    p.add_argument("--points", type=int, default=6, help="theta values to evaluate")
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--num-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env-kwarg", action="append", default=[], metavar="KEY=VALUE",
                   help="env kwargs; set these to the ECOLOGY THE POLICY WAS TRAINED "
                        "ON, otherwise the policy is measured out of distribution")
    p.add_argument("--save", default=None, metavar="PATH",
                   help="write the table to a .npz so protocols.py can search over it "
                        "without re-running the sweep")
    args = p.parse_args()

    from viz.interactive_viewer import _gameplay_checkpoints, _parse_env_kwarg_value

    paths = _gameplay_checkpoints(args.checkpoint)
    if len(paths) != args.num_agents:
        raise SystemExit(f"expected {args.num_agents} gameplay checkpoints, got "
                         f"{len(paths)}: {paths}")
    print(f"Loading {len(paths)} gameplay policies")
    params = [load_params(p_) for p_ in paths]

    env_kwargs = {"num_agents": args.num_agents, "shared_rewards": False,
                  "cnn": True, "jit": True, "apple_reward": 1.0,
                  "num_inner_steps": args.num_steps}
    for kv in args.env_kwarg:
        key, _, raw = kv.partition("=")
        if not _:
            raise SystemExit(f"--env-kwarg expects KEY=VALUE, got {kv!r}")
        env_kwargs[key] = _parse_env_kwarg_value(raw)
    print(f"Env kwargs: {env_kwargs}")

    env = LogWrapper(socialjax.make(args.env, **env_kwargs), replace_info=False)
    contract = CleanupContract(args.num_agents, args.contract_low, args.contract_high)
    thetas = np.linspace(args.contract_low, args.contract_high, args.points)

    print(f"Evaluating {len(thetas)} contracts x {args.num_envs} envs x "
          f"{args.num_steps} steps")
    rows = evaluate_grid(env, params, contract, thetas, args.num_envs,
                         args.num_steps, args.seed)
    report(thetas, rows, args.num_agents)

    if args.save:
        np.savez(
            args.save,
            thetas=thetas,
            returns=np.stack([r["return"] for r in rows]),
            base_returns=np.stack([r["base_return"] for r in rows]),
            cleaned=np.stack([r["cleaned_per_step"] for r in rows]),
        )
        print(f"\nTable written to {args.save}")


if __name__ == "__main__":
    main()
