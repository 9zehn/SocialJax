"""Compare the phase-2 arms: which contract each mechanism picks, and what it buys.

WHAT THIS CAN AND CANNOT READ FROM THE CHECKPOINTS
--------------------------------------------------
Phase 2 never updates the gameplay policy, so the per-agent gameplay .pkl files
saved by all six arms are byte-identical copies of the shared phase-1 policy. The
only file content unique to an arm is the negotiation arm's `_contract_` policies.
Every arm's OUTCOME therefore has to be re-derived here rather than loaded:

  * solver arms  -- the decision rule is a deterministic function of the frozen
                    critic, so re-running it reproduces the arm exactly;
  * negotiate    -- theta comes from evaluating the saved contracting policies at
                    s_0 (agent 0 proposes; the mean of its Gaussian, not a sample,
                    since we want the policy's choice rather than its noise);
  * null         -- theta = 0, the disagreement point everything is measured from.

Each arm's contract is then REPLAYED for real episodes, so every number below is a
realised outcome rather than a critic estimate. The gap between the two is itself
reported, because every solver rule *selects* on critic estimates -- if those do not
predict outcomes, the ranking between rules is measuring noise.

WHAT IT ASKS
------------
  1. Does the mechanism do anything?      theta, transfer volume, cleaning
  2. Is it efficient?                     welfare, and the gap to the best theta
  3. Is it fair?                          equality, cleaner:harvester, worst-off
  4. Would anyone refuse?                 individual rationality vs the null contract
  5. Does the commons survive?            river stock, mean and final
  6. Is the choice stable?                spread of theta across episodes
  7. Is the critic trustworthy?           predicted vs realised welfare

Usage:
    python algorithms/MOCA/evaluate.py \
        --phase1 'checkpoints/moca/..._negotiate_?.pkl' \
        --negotiate-contract 'checkpoints/moca/..._negotiate_contract_?.pkl' \
        --num-agents 7 --contract-high 1.0
"""
import argparse
import glob as globlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp
import numpy as np

import socialjax
from socialjax.wrappers.baselines import LogWrapper
from algorithms.utils import load_params
from algorithms.MOCA import negotiate as neg
from algorithms.MOCA import solver
from algorithms.MOCA.contracts import PROPOSE, CleanupContract
from algorithms.MOCA.grid_eval import rollout_at_theta
from algorithms.MOCA.networks import ContractActorCritic, NegotiationActorCritic

SOLVER_RULES = ("majority", "max", "nash", "kalai_smorodinsky", "egalitarian")


def _load_glob(pattern, n, what):
    paths = sorted(globlib.glob(pattern))
    if len(paths) != n:
        raise SystemExit(f"{what}: expected {n} files, {pattern!r} matched "
                         f"{len(paths)}: {paths}")
    return [load_params(p) for p in paths]


def initial_obs(env, num_envs, seed):
    """(N, num_envs, ...) observations at s_0 -- the state every arm negotiates from."""
    obsv, _ = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(seed), num_envs))
    return jnp.transpose(obsv, (1, 0, 2, 3, 4))


def solver_thetas(rule, nets, params, obs0, contract, num_samples, key):
    """Re-run a solver arm: sample, score with the frozen critic, apply the rule."""
    cands = solver.sample_contract_candidates(key, contract, num_samples, obs0.shape[1])
    values = solver.contract_values(nets, params, obs0, contract, cands)
    best = solver.select_contract(values, rule)
    theta = jnp.take_along_axis(cands, best[None, :], axis=0)[0]
    chosen = jnp.take_along_axis(values, best[None, None, :], axis=0)[0]   # (N, E)
    return theta, {"predicted_welfare": float(chosen.sum(axis=0).mean()),
                   "null_rate": float((best == 0).mean())}


def negotiate_thetas(cparams, obs0, contract):
    """Agent 0's proposal per env, from the saved contracting policies.

    The MEAN of the Gaussian, not a draw: we want the contract the policy would
    settle on, not one realisation of its exploration noise.
    """
    net = NegotiationActorCritic(2, activation="relu")
    n_envs = obs0.shape[1]
    pi0, _ = net.apply(cparams[0], obs0[0],
                       contract.to_obs(jnp.zeros((n_envs,)), stage=PROPOSE))
    return neg.unsquash(pi0.mean()[:, 0], contract.low, contract.high)


def negotiate_sign_prob(cparams, obs0, contract, theta, nu):
    """P(the offer is signed), and each non-proposer's accept probability.

    Agent 0 proposes; nu of the other N-1 are drawn uniformly WITHOUT replacement and
    the product of their accept probabilities is the signing probability. Averaging
    that product over every possible nu-subset gives the exact expectation rather than
    a sampled estimate -- with 6 non-proposers there are only C(6,nu) of them.

    This is where nu bites: acceptance is a PRODUCT, so polling more agents multiplies
    in more sub-unit terms. Raising nu from 2 to all-6 does not make agents pickier,
    it just makes agreement combinatorially rarer.
    """
    import itertools
    net = NegotiationActorCritic(2, activation="relu")
    from algorithms.MOCA.contracts import AGREE
    agree_obs = contract.to_obs(theta, stage=AGREE)
    probs = []
    for j in range(1, len(cparams)):
        pi, _ = net.apply(cparams[j], obs0[j], agree_obs)
        probs.append(np.asarray(neg.unsquash(pi.mean()[:, 1], 0.0, 1.0)))
    probs = np.stack(probs)                                    # (N-1, num_envs)
    subsets = list(itertools.combinations(range(probs.shape[0]), nu))
    p_sign = np.mean([np.prod(probs[list(s)], axis=0) for s in subsets], axis=0)
    return p_sign, probs


def _blend(signed, null, p):
    """Expected outcome when the offer is signed with probability p, else null."""
    out = {}
    for k in ("return", "cleaned_per_step", "theta", "river_final", "transfer_volume"):
        out[k] = p * np.asarray(signed[k]) + (1.0 - p) * np.asarray(null[k])
    out["theta_std"] = signed["theta_std"]
    return out


def equality(v):
    diffs = np.abs(v[:, None] - v[None, :]).sum()
    return 1.0 - diffs / (2.0 * len(v) * np.abs(v).sum() + 1e-8)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase1", required=True, help="glob for the shared gameplay policies")
    p.add_argument("--negotiate-contract", default=None,
                   help="glob for the negotiate arm's _contract_ policies (optional)")
    p.add_argument("--num-agents", type=int, required=True)
    p.add_argument("--contract-low", type=float, default=0.0)
    p.add_argument("--contract-high", type=float, required=True,
                   help="MUST match the run's CONTRACT_HIGH: theta is normalised by it")
    p.add_argument("--env", default="clean_up")
    p.add_argument("--num-envs", type=int, default=32, help="episodes per arm")
    p.add_argument("--num-steps", type=int, default=1000)
    p.add_argument("--solver-samples", type=int, default=50)
    p.add_argument("--negotiate-nu", type=int, default=None,
                   help="non-proposers polled on the offer. Default follows the "
                        "reference rule (2 above 3 agents); pass num_agents-1 for "
                        "the poll-everyone comparison")
    p.add_argument("--grid-points", type=int, default=11,
                   help="theta grid for the efficiency/fairness benchmarks")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env-kwarg", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--save", default=None, help="write results to a .npz")
    args = p.parse_args()

    from viz.interactive_viewer import _parse_env_kwarg_value

    n = args.num_agents
    params = _load_glob(args.phase1, n, "phase-1 gameplay policies")
    cparams = (_load_glob(args.negotiate_contract, n, "negotiate contracting policies")
               if args.negotiate_contract else None)

    env_kwargs = {"num_agents": n, "shared_rewards": False, "cnn": True, "jit": True,
                  "apple_reward": 1.0, "num_inner_steps": args.num_steps}
    for kv in args.env_kwarg:
        k, _, v = kv.partition("=")
        env_kwargs[k] = _parse_env_kwarg_value(v)
    env = LogWrapper(socialjax.make(args.env, **env_kwargs), replace_info=False)
    contract = CleanupContract(n, args.contract_low, args.contract_high)
    nets = [ContractActorCritic(env.action_space().n, activation="relu") for _ in range(n)]
    net = nets[0]

    obs0 = initial_obs(env, args.num_envs, args.seed)
    key = jax.random.PRNGKey(args.seed)

    # ---- what contract does each arm choose? --------------------------------
    arms, extra = {}, {}
    arms["null"] = jnp.zeros((args.num_envs,))
    for rule in SOLVER_RULES:
        key, k = jax.random.split(key)
        t, info = solver_thetas(rule, nets, params, obs0, contract, args.solver_samples, k)
        arms[f"solver:{rule}"] = t
        extra[f"solver:{rule}"] = info
    sign = None
    if cparams is not None:
        t_neg = negotiate_thetas(cparams, obs0, contract)
        arms["negotiate"] = t_neg
        nu = args.negotiate_nu or neg.default_nu(n)
        if not 1 <= nu <= n - 1:
            raise SystemExit(f"--negotiate-nu must be in [1, {n-1}], got {nu}")
        p_sign, acc = negotiate_sign_prob(cparams, obs0, contract, t_neg, nu)
        sign = {"nu": nu, "p": p_sign, "per_agent": acc}

    # ---- replay each contract for real episodes -----------------------------
    print(f"Replaying {len(arms)} arms x {args.num_envs} episodes x {args.num_steps} steps")
    results = {}
    for name, theta in arms.items():
        key, k = jax.random.split(key)
        results[name] = jax.tree.map(np.asarray, rollout_at_theta(
            env, net, params, contract, theta, args.num_envs, args.num_steps, k))
        results[name]["theta_std"] = float(np.std(np.asarray(theta)))
        print(f"  {name} done", flush=True)

    # ---- benchmarks: the best theta on each criterion -----------------------
    grid = np.linspace(args.contract_low, args.contract_high, args.grid_points)
    print(f"\nSweeping {len(grid)} contracts for benchmarks")
    bench = []
    for t in grid:
        key, k = jax.random.split(key)
        bench.append(jax.tree.map(np.asarray, rollout_at_theta(
            env, net, params, contract, float(t), args.num_envs, args.num_steps, k)))
    bret = np.stack([b["return"] for b in bench])                   # (K, N)
    d = bret[0]
    bwelf = bret.sum(axis=1)
    gains = bret - d
    feas = (gains >= 0).all(axis=1); feas[0] = True
    best = {
        "utilitarian": float(grid[int(np.argmax(bwelf))]),
        "nash": float(grid[int(np.argmax(np.where(feas, np.log(np.maximum(gains, 1e-12)).sum(1), -np.inf)))]),
        "egalitarian": float(grid[int(np.argmax(np.where(feas, gains.min(1), -np.inf)))]),
        "max_equality": float(grid[int(np.argmax([equality(r) for r in bret]))]),
    }
    # Best welfare seen ANYWHERE in this evaluation -- the grid sweep alone is not an
    # upper bound, since arms pick theta off-grid and each rollout uses its own key.
    best_welfare = float(bwelf.max())

    # ---- roles, from who actually cleans ------------------------------------
    grid_clean = np.stack([b["cleaned_per_step"] for b in bench]).mean(axis=0)
    is_cleaner = grid_clean > grid_clean.mean()
    print(f"\nRoles (mean cleaning/step over the sweep): "
          f"cleaners={list(np.where(is_cleaner)[0])} {np.round(grid_clean[is_cleaner],3)}, "
          f"harvesters={list(np.where(~is_cleaner)[0])}")

    # ---- report --------------------------------------------------------------
    # Expected outcome of the negotiation arm: the offer only takes force if signed,
    # and on rejection the episode runs under the null contract.
    if sign is not None:
        results["negotiate(E)"] = _blend(results["negotiate"], results["null"],
                                         float(sign["p"].mean()))
        arms["negotiate(E)"] = arms["negotiate"]

    best_welfare = max(best_welfare,
                       max(float(results[a]["return"].sum()) for a in arms))
    null_ret = results["null"]["return"]
    hdr = (f"\n{'arm':<22}{'theta':>7}{'+-':>6}{'welfare':>9}{'%best':>7}"
           f"{'equal':>7}{'clnr':>8}{'harv':>8}{'ratio':>7}{'worst':>8}{'IR!':>5}"
           f"{'clean':>7}{'river':>7}")
    print(hdr)
    print("-" * (len(hdr) - 1))
    for name in arms:
        r = results[name]
        v = r["return"]
        cl_v = v[is_cleaner].mean() if is_cleaner.any() else np.nan
        hv_v = v[~is_cleaner].mean() if (~is_cleaner).any() else np.nan
        ir = int((v < null_ret - 1e-6).sum())        # agents worse off than no contract
        print(f"{name:<22}{float(r['theta']):>7.3f}{r['theta_std']:>6.2f}"
              f"{v.sum():>9.1f}{100*v.sum()/max(best_welfare,1e-9):>7.1f}"
              f"{equality(v):>7.3f}{cl_v:>8.1f}{hv_v:>8.1f}{cl_v/hv_v if hv_v else np.nan:>7.2f}"
              f"{v.min():>8.1f}{ir:>5d}"
              f"{r['cleaned_per_step'].sum():>7.2f}{float(r['river_final']):>7.1f}")

    if sign is not None:
        import itertools as _it
        pa = sign["per_agent"]
        allnu = n - 1
        p_all = np.mean([np.prod(pa[list(s_)], axis=0)
                         for s_ in _it.combinations(range(pa.shape[0]), allnu)], axis=0)
        print(f"\nnegotiation acceptance (nu={sign['nu']} of {allnu} non-proposers polled):")
        for j, pr in enumerate(pa, start=1):
            print(f"    agent {j} accepts with p={pr.mean():.3f}")
        print(f"  P(signed) at nu={sign['nu']:<2}          : {sign['p'].mean():.3f}")
        print(f"  P(signed) at nu={allnu} (poll everyone): {p_all.mean():.3f}")
        print("  -> 'negotiate' is the outcome IF signed; 'negotiate(E)' weights by P(signed)")

    print("\nbenchmarks (theta maximising each criterion on the sweep):")
    for k_, v_ in best.items():
        print(f"  {k_:<14} theta={v_:.3f}")
    print(f"  best welfare on the sweep: {best_welfare:.1f}")

    print("\ncritic validity -- solver rules SELECT on these estimates:")
    print("  (only meaningful if --num-steps matches the episode length phase 1 was\n   trained at: the critic predicts a FULL episode's return)")
    for name in arms:
        if name in extra:
            pred = extra[name]["predicted_welfare"]
            real = float(results[name]["return"].sum())
            print(f"  {name:<22} predicted {pred:>9.1f}  realised {real:>9.1f}  "
                  f"error {100*(pred-real)/max(abs(real),1e-9):>+7.1f}%"
                  f"   null-rate {extra[name]['null_rate']:.2f}")

    print("\nper-agent return (realised, base + transfers):")
    print("  arm                   " + "".join(f"  ag{i:<5d}" for i in range(n)))
    for name in arms:
        print(f"  {name:<22}" + "".join(f"  {x:7.1f}" for x in results[name]["return"]))
    print("  roles                 " + "".join(f"  {('cleaner' if is_cleaner[i] else 'harvest'):>7}"
                                               for i in range(n)))

    if args.save:
        np.savez(args.save,
                 arms=np.array(list(arms.keys())),
                 theta=np.array([float(results[a]["theta"]) for a in arms]),
                 returns=np.stack([results[a]["return"] for a in arms]),
                 cleaned=np.stack([results[a]["cleaned_per_step"] for a in arms]),
                 is_cleaner=is_cleaner, grid=grid, grid_returns=bret)
        print(f"\nwritten to {args.save}")


if __name__ == "__main__":
    main()
