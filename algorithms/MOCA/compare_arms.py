"""Compare contracting arms on identical episodes: bargaining vs one-shot vs null.

Every arm is replayed over the SAME number of full episodes with the same env, so
welfare, equality and cleaning are directly comparable. Each arm's contract is
produced by its own mechanism:

  bargain    the Rubinstein stage runs during the episode (see evaluate_bargain)
  negotiate  the paper's one-shot ultimatum: agent 0 proposes at s_0, nu sampled
             non-proposers gate it by the PRODUCT of their accept probabilities,
             and rejection means the null contract for the whole episode. Sampled,
             not taken at the Gaussian's mean, so the reported spread is the
             mechanism's real variability rather than a point estimate.
  null       theta = 0 throughout: the disagreement point every acceptance rule is
             measured against, and the only honest zero for "what did contracting
             buy".

Usage:
    python algorithms/MOCA/compare_arms.py \
        --run bargain=runs/.../..._bargain_seg100_joint_'[0-9]'.pkl \
        --run base42=runs/.../..._negotiate_nu2_'[0-9]'.pkl \
        --null --episodes 20
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
from algorithms.MOCA import bargain as bg
from algorithms.MOCA import negotiate as neg
from algorithms.MOCA.contracts import AGREE, PROPOSE, CleanupContract
from algorithms.MOCA.evaluate_bargain import rollout as bargain_rollout
from algorithms.MOCA.networks import (BargainingActorCritic, ContractActorCritic,
                                      NegotiationActorCritic)


def play(env, gp, contract, theta, num_envs, num_steps, seed):
    """Roll `num_envs` episodes under a per-env contract held for the whole episode."""
    n = env.num_agents
    net = ContractActorCritic(env.action_space().n, activation="relu")
    key = jax.random.PRNGKey(seed)
    key, kr = jax.random.split(key)
    obsv, st = jax.vmap(env.reset)(jax.random.split(kr, num_envs))
    cobs = contract.to_obs(theta)

    def step(carry, _):
        st, ob, rng = carry
        rng, ka, ks = jax.random.split(rng, 3)
        b = jnp.transpose(ob, (1, 0, 2, 3, 4))
        keys = jax.random.split(ka, n)
        acts = [net.apply(gp[i], b[i], cobs)[0].sample(seed=keys[i]) for i in range(n)]
        ob2, st2, rew, done, info = jax.vmap(env.step)(
            jax.random.split(ks, num_envs), st, acts)
        tr = contract.compute_transfer(theta, info["cleaned_by_agent"])
        return (st2, ob2, rng), (jnp.transpose(info["cleaned_by_agent"]),
                                 jnp.transpose(rew), jnp.transpose(tr),
                                 info["waste_cleared"][:, 0])

    _, (cl, rew, tr, riv) = jax.lax.scan(step, (st, obsv, key), None, num_steps)
    return jax.tree.map(np.asarray, (cl, rew, tr, riv))


def one_shot_theta(env, cpaths, contract, num_envs, seed, nu=None):
    """Replay the paper's one-shot stage: propose at s_0, nu voters gate by product."""
    n = env.num_agents
    nu = nu or neg.default_nu(n)
    nets = [NegotiationActorCritic(2, activation="relu") for _ in range(n)]
    params = [load_params(p) for p in cpaths]
    key = jax.random.PRNGKey(seed)
    key, kr = jax.random.split(key)
    obsv, _ = jax.vmap(env.reset)(jax.random.split(kr, num_envs))
    ob = jnp.transpose(obsv, (1, 0, 2, 3, 4))

    key, k0, k1, ks, ka = jax.random.split(key, 5)
    prop_obs = contract.to_obs(jnp.zeros((num_envs,)), stage=PROPOSE)
    raw0 = jnp.stack([nets[i].apply(params[i], ob[i], prop_obs)[0]
                      .sample(seed=jax.random.split(k0, n)[i]) for i in range(n)])
    theta_prop = neg.unsquash(raw0[0, :, 0], contract.low, contract.high)

    agree_obs = contract.to_obs(theta_prop, stage=AGREE)
    raw1 = jnp.stack([nets[i].apply(params[i], ob[i], agree_obs)[0]
                      .sample(seed=jax.random.split(k1, n)[i]) for i in range(n)])
    accept_probs = neg.unsquash(raw1[:, :, 1], 0.0, 1.0)
    mask = neg.sample_voters(ks, n, nu, num_envs)
    accepted, prod = neg.acceptance(ka, accept_probs, mask)
    theta_eff = jnp.where(accepted, theta_prop, jnp.float32(contract.null))
    return (np.asarray(theta_eff), np.asarray(theta_prop), np.asarray(accepted),
            np.asarray(prod), nu)


def gini_equality(v):
    d = np.abs(v[:, None, :] - v[None, :, :]).sum((0, 1))
    return 1.0 - d / (2.0 * v.shape[0] * np.abs(v).sum(0) + 1e-8)


def summarise(cl, rew, tr, riv, num_steps):
    """Per-episode outcome arrays from raw (T, N, E) / (T, E) rollout records."""
    ret = (rew + tr).sum(0)                                   # (N, E)
    return {
        "return": ret,
        "welfare": ret.sum(0),
        "equality": gini_equality(ret),
        "cleaned": cl.sum((0, 1)) / num_steps,                # cells/step, all agents
        "cleaned_agent": cl.sum(0) / num_steps,               # (N, E)
        "river": riv.mean(0),
        "transfer_volume": np.maximum(tr, 0).sum((0, 1)),
        "base": rew.sum(0),
    }


def roles(cleaned_agent):
    """Cleaners = above-average cleaning. Crude, but the split is unambiguous here."""
    m = cleaned_agent.mean(1)
    return m > m.mean()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", default=[], metavar="LABEL=GLOB",
                   help="repeatable; mode is detected from the checkpoint name")
    p.add_argument("--null", action="store_true",
                   help="add a theta=0 arm using the FIRST run's gameplay policies")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--num-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--contract-low", type=float, default=0.2)
    p.add_argument("--contract-high", type=float, default=1.0)
    args = p.parse_args()

    from viz.interactive_viewer import (_gameplay_checkpoints, detect_moca,
                                        infer_bargain_config)

    arms, first_gp = {}, None
    for spec in args.run:
        label, _, glob_ = spec.partition("=")
        moca = detect_moca(glob_)
        gp = [load_params(q) for q in _gameplay_checkpoints(glob_)]
        n = len(gp)
        env = LogWrapper(socialjax.make(
            "clean_up", num_agents=n, shared_rewards=False, cnn=True, jit=True,
            apple_reward=1.0, num_inner_steps=args.num_steps), replace_info=False)
        contract = CleanupContract(n, args.contract_low, args.contract_high)
        if first_gp is None:
            first_gp = (gp, env, contract, n)

        mode = moca["mode"] if moca else "none"
        print(f"[{label}] mode={mode}  agents={n}", flush=True)
        if mode == "bargain":
            cfg = infer_bargain_config(moca["stem"])
            cfg.setdefault("hidden", 64)
            cfg.setdefault("accept_bias", 1.0)
            bp = [load_params(q) for q in moca["contract_paths"]]
            rec, K = bargain_rollout(env, gp, bp, contract, cfg, args.episodes,
                                     args.num_steps, args.seed)
            ret = (rec["base"] + rec["transfer"]).sum(0)
            cl_agent = rec["cleaned"].sum(0) / args.num_steps
            agreed = rec["newly"].any(0)
            r_idx = np.arange(K)[:, None]
            s = {"return": ret, "welfare": ret.sum(0),
                 "equality": gini_equality(ret),
                 "cleaned": rec["cleaned"].sum((0, 1)) / args.num_steps,
                 "cleaned_agent": cl_agent, "river": rec["river"].mean(0),
                 "transfer_volume": np.maximum(rec["transfer"], 0).sum((0, 1)),
                 "base": rec["base"].sum(0),
                 "theta": (rec["newly"] * rec["offer"]).sum(0),
                 "agreed": agreed,
                 "agree_round": np.where(agreed, (rec["newly"] * r_idx).sum(0), K),
                 "uncontracted_steps": (rec["theta_eff"] <= contract.null + 1e-9
                                        ).sum(0) * cfg["segment"]}
            print(f"          segment={cfg['segment']} quorum={cfg['quorum']} "
                  f"proposer={cfg['proposer']}")
        elif mode == "negotiate":
            th, prop, acc, prod, nu = one_shot_theta(
                env, moca["contract_paths"], contract, args.episodes, args.seed)
            s = summarise(*play(env, gp, contract, jnp.asarray(th), args.episodes,
                                args.num_steps, args.seed), args.num_steps)
            s.update({"theta": th, "theta_proposed": prop, "agreed": acc,
                      "accept_prob": prod,
                      "uncontracted_steps": (~acc) * args.num_steps})
            print(f"          nu={nu}  proposer=agent 0 (fixed)")
        else:
            raise SystemExit(f"[{label}] unsupported mode {mode!r}")
        arms[label] = s

    if args.null:
        gp, env, contract, n = first_gp
        th = jnp.full((args.episodes,), contract.null)
        arms["null"] = summarise(*play(env, gp, contract, th, args.episodes,
                                       args.num_steps, args.seed), args.num_steps)
        arms["null"]["theta"] = np.zeros(args.episodes)
        print("[null] theta=0 throughout (uses the first run's gameplay policies)")

    # ------------------------------------------------------------------ report
    def blk(t):
        print(f"\n{t}\n" + "-" * 78)

    blk(f"outcomes over {args.episodes} episodes x {args.num_steps} steps "
        f"(mean +- sd)")
    hdr = (f"  {'arm':<12}{'welfare':>16}{'equality':>14}{'clean/step':>14}"
           f"{'river':>9}{'theta':>9}")
    print(hdr)
    for label, s in arms.items():
        print(f"  {label:<12}{s['welfare'].mean():>9.1f} +-{s['welfare'].std():>5.0f}"
              f"{s['equality'].mean():>9.3f} +-{s['equality'].std():>4.3f}"
              f"{s['cleaned'].mean():>9.3f} +-{s['cleaned'].std():>4.3f}"
              f"{s['river'].mean():>9.1f}{s['theta'].mean():>9.3f}")

    blk("distribution: who ends up with what")
    print(f"  {'arm':<12}{'cleaners':>10}{'harvesters':>12}{'ratio':>8}"
          f"{'worst-off':>11}{'transfers':>11}{'n_cleaners':>12}")
    for label, s in arms.items():
        is_cl = roles(s["cleaned_agent"])
        r = s["return"].mean(1)
        c = r[is_cl].mean() if is_cl.any() else np.nan
        h = r[~is_cl].mean() if (~is_cl).any() else np.nan
        print(f"  {label:<12}{c:>10.1f}{h:>12.1f}{c / h:>8.3f}{r.min():>11.1f}"
              f"{s['transfer_volume'].mean():>11.1f}{int(is_cl.sum()):>12}")

    blk("mechanism")
    print(f"  {'arm':<12}{'agreed':>9}{'theta':>18}{'steps uncontracted':>22}")
    for label, s in arms.items():
        if "agreed" not in s:
            print(f"  {label:<12}{'--':>9}{'0 (null arm)':>18}"
                  f"{args.num_steps:>22}")
            continue
        t = s["theta"][s["agreed"]] if s["agreed"].any() else np.array([np.nan])
        extra = ""
        if "agree_round" in s:
            extra = f"   (round {s['agree_round'].mean():.2f})"
        print(f"  {label:<12}{s['agreed'].mean():>9.2f}"
              f"{t.mean():>11.3f} +-{t.std():>4.3f}"
              f"{s['uncontracted_steps'].mean():>22.0f}{extra}")

    blk("per episode: welfare / equality / clean-per-step")
    labels = list(arms)
    print("  ep  " + "".join(f"{l:>24}" for l in labels))
    for e in range(args.episodes):
        cells = "".join(
            f"{arms[l]['welfare'][e]:>10.0f}{arms[l]['equality'][e]:>7.3f}"
            f"{arms[l]['cleaned'][e]:>7.2f}" for l in labels)
        print(f"  {e:>3} " + cells)
    print("  " + "-" * (4 + 24 * len(labels)))
    print("  mean" + "".join(
        f"{arms[l]['welfare'].mean():>10.0f}{arms[l]['equality'].mean():>7.3f}"
        f"{arms[l]['cleaned'].mean():>7.2f}" for l in labels))


if __name__ == "__main__":
    main()
