"""Statistics for a Rubinstein bargaining run: what was negotiated, and what it bought.

No rendering. Replays whole episodes in parallel and reports the things the panel
can only show one episode at a time.

The questions it answers, in the order the tables come out:

  1. What did the episodes produce?      welfare, equality, cleaning, river
  2. Did the contract change behaviour?  the SAME policies, split by whether a
                                         contract was in force that segment. This is
                                         the comparison the mechanism lives or dies
                                         on, and it is within-run: no separate
                                         baseline to confound it.
  3. How did bargaining go?              agreement rate and round, theta agreed,
                                         steps burned on disagreement
  4. Do proposers concede over rounds?   offers by round index -- the Rubinstein
                                         signature. Offers should soften as the
                                         remaining episode shrinks.
  5. Who wanted what?                    per agent: role, what it proposed, how it
                                         voted, whether its offers carried
  6. Every episode                       one line each, so outliers are visible
                                         rather than averaged away

Usage:
    python algorithms/MOCA/evaluate_bargain.py \
        --checkpoint 'runs/rubensteinV1/run1_step300/..._joint_latest_[0-9].pkl' \
        --episodes 20 --num-steps 1000
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
from algorithms.utils import contract_range, load_params
from algorithms.MOCA import bargain as bg
from algorithms.MOCA import negotiate as neg
from algorithms.MOCA.contracts import CleanupContract
from algorithms.MOCA.networks import BargainingActorCritic, ContractActorCritic


def rollout(env, gp, bp, contract, cfg, num_envs, num_steps, seed):
    """`num_envs` full episodes in parallel. Returns per-round and per-segment records.

    Mirrors the training rollout exactly -- same feature scales, same proposer rule,
    same quorum -- because a policy evaluated on a differently-scaled state is not
    the policy that was trained.
    """
    n = env.num_agents
    net = ContractActorCritic(env.action_space().n, activation="relu")
    bnet = BargainingActorCritic(hidden=cfg["hidden"], activation="relu",
                                 accept_bias=cfg["accept_bias"])
    mask = bg.feature_mask(cfg["features"], n)
    quorum = bg.quorum_size(cfg["quorum"], n)
    x, K = cfg["segment"], num_steps // cfg["segment"]
    inner = int(getattr(env, "num_inner_steps", num_steps))
    ret_s = float(inner) * float(getattr(env, "apple_reward", 1.0))
    cl_s = float(inner)
    riv_s = float(env.GRID_SIZE_ROW * env.GRID_SIZE_COL)

    key = jax.random.PRNGKey(seed)
    key, k_reset, k_start = jax.random.split(key, 3)
    obsv, st = jax.vmap(env.reset)(jax.random.split(k_reset, num_envs))
    offset = (jax.random.randint(k_start, (num_envs,), 0, n)
              if cfg["rotate_start"] == "random" else None)

    def seg_step(carry, _):
        st, ob, theta, rng = carry
        cobs = contract.to_obs(theta)
        rng, ka, ks = jax.random.split(rng, 3)
        b = jnp.transpose(ob, (1, 0, 2, 3, 4))
        keys = jax.random.split(ka, n)
        acts = [net.apply(gp[i], b[i], cobs)[0].sample(seed=keys[i]) for i in range(n)]
        ob2, st2, rew, done, info = jax.vmap(env.step)(
            jax.random.split(ks, num_envs), st, acts)
        tr = contract.compute_transfer(theta, info["cleaned_by_agent"])
        return (st2, ob2, theta, rng), (jnp.transpose(info["cleaned_by_agent"]),
                                        jnp.transpose(rew), jnp.transpose(tr),
                                        info["waste_cleared"][:, 0])

    def round_(carry, r):
        (st, ob, rng, agreed, locked, cum_r, cum_c, last_tn, had, nrej, riv) = carry
        rng, kp, kt, kv = jax.random.split(rng, 4)
        prop = bg.proposer_for_round(r, n, num_envs, cfg["proposer"], key=kp,
                                     contributions=cum_c, start_offset=offset)
        feats = bg.bargaining_features(r, K, prop, n, last_tn, had, nrej,
                                       cum_r / ret_s, cum_c / cl_s, riv / riv_s, mask)
        kts, kvs = jax.random.split(kt, n), jax.random.split(kv, n)
        raw, votes = [], []
        for i in range(n):
            pt, pv, _ = bnet.apply(bp[i], feats[i])
            raw.append(pt.sample(seed=kts[i])[:, 0])
            votes.append(pv.sample(seed=kvs[i]))
        raw, votes = jnp.stack(raw), jnp.stack(votes)
        theta_all = neg.unsquash(raw, contract.low, contract.high)
        mine = bg.is_proposer_mask(prop, n)
        offer = jnp.sum(jnp.where(mine, theta_all, 0.0), axis=0)
        passed, n_acc = bg.accepted(votes.astype(bool), prop, quorum, n)
        newly = passed & ~agreed
        theta_eff = jnp.where(agreed, locked,
                              jnp.where(newly, offer, jnp.float32(contract.null)))

        (st, ob, _, rng), (cl, rew, tr, riv_t) = jax.lax.scan(
            seg_step, (st, ob, theta_eff, rng), None, x)
        seg_cl, seg_base, seg_tr = cl.sum(0), rew.sum(0), tr.sum(0)      # (N, E)

        rec = {"proposer": prop, "offer": offer, "votes": votes, "accepted": passed,
               "newly": newly, "active": ~agreed, "n_accept": n_acc,
               "theta_eff": theta_eff, "cleaned": seg_cl, "base": seg_base,
               "transfer": seg_tr, "river": riv_t.mean(0)}
        carry = (st, ob, rng, agreed | newly, jnp.where(newly, offer, locked),
                 cum_r + seg_base + seg_tr, cum_c + seg_cl,
                 2.0 * (offer - contract.low) / (contract.high - contract.low) - 1.0,
                 jnp.ones_like(had), nrej + (~agreed & ~passed).astype(jnp.int32),
                 riv_t[-1].astype(jnp.float32))
        return carry, rec

    z_e = jnp.zeros((num_envs,), jnp.float32)
    init = (st, obsv, key, jnp.zeros((num_envs,), bool),
            jnp.full((num_envs,), contract.null),
            jnp.zeros((n, num_envs), jnp.float32), jnp.zeros((n, num_envs), jnp.float32),
            z_e, jnp.zeros((num_envs,), bool), jnp.zeros((num_envs,), jnp.int32), z_e)
    _, rec = jax.lax.scan(round_, init, jnp.arange(K))
    return jax.tree.map(np.asarray, rec), K


def gini_equality(v, axis=0):
    d = np.abs(np.expand_dims(v, axis) - np.expand_dims(v, axis + 1)).sum((axis, axis + 1))
    return 1.0 - d / (2.0 * v.shape[axis] * np.abs(v).sum(axis) + 1e-8)


def report(rec, K, cfg, contract, n, num_steps):
    x = cfg["segment"]
    ret = (rec["base"] + rec["transfer"]).sum(0)               # (N, E) episode return
    welfare = ret.sum(0)                                        # (E,)
    equality = gini_equality(ret, axis=0)
    contracted = rec["theta_eff"] > contract.null + 1e-9        # (K, E)
    cl_per_step = rec["cleaned"].sum(1) / x                     # (K, E) all agents
    agreed_any = rec["newly"].any(0)                            # (E,)
    r_idx = np.arange(K)[:, None]
    agree_round = np.where(agreed_any, (rec["newly"] * r_idx).sum(0), K)
    theta_ag = (rec["newly"] * rec["offer"]).sum(0)

    def blk(t):
        print(f"\n\033[1m{t}\033[0m" if sys.stdout.isatty() else f"\n{t}")

    print(f"protocol: segment={x}  rounds={K}  proposer={cfg['proposer']}"
          f"(start {cfg['rotate_start']})  quorum={cfg['quorum']}"
          f"={bg.quorum_size(cfg['quorum'], n)}/{n-1}  features={cfg['features']}")
    print(f"contract space: {{{contract.null:g}}} u "
          f"[{contract.low:g}, {contract.high:g}]")
    print(f"{ret.shape[1]} episodes x {num_steps} steps")

    blk("outcomes (mean +- sd over episodes)")
    for name, v in (("welfare", welfare), ("equality", equality),
                    ("cleaning /step", cl_per_step.mean(0)),
                    ("river clear", rec["river"].mean(0)),
                    ("transfer volume", np.maximum(rec["transfer"], 0).sum((0, 1)))):
        print(f"  {name:<18}{v.mean():10.3f} +- {v.std():.3f}")

    blk("contract vs no contract  (same policies, split by segment)")
    # Read with care. Uncontracted segments are always the EARLY ones -- bargaining
    # only runs until it succeeds -- so this is confounded with episode phase. Apples
    # need a clean river and time to grow, so welfare/step is heavily biased against
    # the no-contract row. clean/step is the safer comparison: cleaning is capped by
    # the dirt spawn rate rather than by accumulated stock.
    print("  (no-contract segments are always the episode's first ones, so "
          "welfare/step is confounded with time; compare clean/step)")
    print(f"  {'':<16}{'segments':>10}{'clean/step':>12}{'welfare/step':>14}{'river':>9}")
    for label, m in (("no contract", ~contracted), ("under contract", contracted)):
        if not m.any():
            print(f"  {label:<16}{0:>10}{'--':>12}{'--':>14}{'--':>9}")
            continue
        w = (rec["base"] + rec["transfer"]).sum(1)[m] / x        # welfare per step
        print(f"  {label:<16}{int(m.sum()):>10}{cl_per_step[m].mean():>12.3f}"
              f"{w.mean():>14.3f}{rec['river'][m].mean():>9.1f}")

    blk("negotiation")
    print(f"  agreement rate    {agreed_any.mean():.3f}  "
          f"({int(agreed_any.sum())}/{len(agreed_any)} episodes)")
    print(f"  agreement round   {agree_round.mean():.2f} +- {agree_round.std():.2f}"
          f"   (K={K} means never agreed)")
    hist = np.bincount(agree_round, minlength=K + 1)
    print("  round histogram   " + "  ".join(
        f"R{i}:{c}" for i, c in enumerate(hist) if c) )
    if agreed_any.any():
        t = theta_ag[agreed_any]
        print(f"  theta agreed      {t.mean():.4f} +- {t.std():.4f}   "
              f"[{t.min():.3f}, {t.max():.3f}]")
    print(f"  steps uncontracted {(~contracted).sum(0).mean() * x:.0f} of {num_steps}")

    blk("offers by round  (do proposers concede as the episode shrinks?)")
    print(f"  {'round':>6}{'offers':>8}{'mean theta':>12}{'accepted':>10}{'accept rate':>13}")
    for r in range(K):
        act = rec["active"][r]
        if not act.any():
            continue
        print(f"  {r:>6}{int(act.sum()):>8}{rec['offer'][r][act].mean():>12.4f}"
              f"{int(rec['newly'][r].sum()):>10}"
              f"{rec['newly'][r][act].mean():>13.3f}")

    blk("per agent")
    cl_agent = rec["cleaned"].sum(0).mean(1) / num_steps         # (N,) cells/step
    is_cleaner = cl_agent > cl_agent.mean()
    print(f"  {'agent':<7}{'clean/step':>11}{'return':>9}{'role':>10}"
          f"{'proposed':>10}{'mean theta':>12}{'carried':>9}{'votes yes':>11}")
    for i in range(n):
        mine = rec["proposer"] == i                              # (K, E)
        prop_act = mine & rec["active"]
        voted = rec["active"] & ~mine
        yes = (rec["votes"][:, i] == 1) & voted
        print(f"  A{i:<6}{cl_agent[i]:>11.3f}{ret[i].mean():>9.1f}"
              f"{'cleaner' if is_cleaner[i] else 'harvester':>10}"
              f"{int(prop_act.sum()):>10}"
              f"{(rec['offer'][prop_act].mean() if prop_act.any() else np.nan):>12.4f}"
              f"{int((rec['newly'] & mine).sum()):>9}"
              f"{(yes.sum() / max(voted.sum(), 1)):>11.3f}")
    if is_cleaner.any() and (~is_cleaner).any():
        c, h = ret[is_cleaner].mean(), ret[~is_cleaner].mean()
        print(f"\n  cleaner:harvester return ratio  {c / h:.3f}   "
              f"(cleaners {c:.1f}, harvesters {h:.1f})")
        # The direct test of whether rotating proposal rights does any work: if a
        # cleaner's turn produces a systematically higher offer than a harvester's,
        # then who holds the move is shifting the split.
        for label, m in (("cleaners", is_cleaner), ("harvesters", ~is_cleaner)):
            sel = np.isin(rec["proposer"], np.where(m)[0]) & rec["active"]
            if sel.any():
                print(f"  mean theta offered by {label:<11}{rec['offer'][sel].mean():.4f}"
                      f"   (carried {int((rec['newly'] & sel).sum())})")

    blk("per episode")
    print(f"  {'ep':>3}{'agree':>7}{'theta':>8}{'by':>5}{'welfare':>10}"
          f"{'equality':>10}{'clean/step':>12}")
    for e in range(ret.shape[1]):
        if agreed_any[e]:
            r = int(agree_round[e])
            who = f"A{int(rec['proposer'][r, e])}"
            ag, th = f"R{r}", f"{theta_ag[e]:.3f}"
        else:
            who, ag, th = "-", "none", "-"
        print(f"  {e:>3}{ag:>7}{th:>8}{who:>5}{welfare[e]:>10.1f}"
              f"{equality[e]:>10.3f}{cl_per_step[:, e].mean():>12.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True,
                   help="glob matching the per-agent GAMEPLAY policies")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--num-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--contract-low", type=float, default=None)
    p.add_argument("--contract-high", type=float, default=None,
                   help="theta bounds. Read from the run's .run.yaml sidecar when it "
                        "has one; these flags override it. The bounds are not in the "
                        "filename, so for runs predating the sidecar there is no "
                        "record at all and a mismatch silently rescales theta -- "
                        "supply them explicitly for those.")
    p.add_argument("--bargain-segment", type=int, default=None,
                   help="override the _seg<N> read from the checkpoint name")
    p.add_argument("--env-kwarg", action="append", default=[], metavar="KEY=VALUE")
    args = p.parse_args()

    from viz.interactive_viewer import (_gameplay_checkpoints, _parse_env_kwarg_value,
                                        detect_moca, infer_bargain_config)

    moca = detect_moca(args.checkpoint)
    if moca is None or moca["mode"] != "bargain":
        raise SystemExit(
            f"not a bargaining run (detected: {moca['mode'] if moca else 'none'}). "
            f"Point --checkpoint at the gameplay policies of a PHASE2_MODE=bargain run."
        )
    gp = [load_params(q) for q in _gameplay_checkpoints(args.checkpoint)]
    bp = [load_params(q) for q in moca["contract_paths"]]
    n = len(gp)
    if len(bp) != n:
        raise SystemExit(f"{n} gameplay but {len(bp)} bargaining policies")

    cfg = infer_bargain_config(moca["stem"])
    cfg.setdefault("hidden", 64)
    cfg.setdefault("accept_bias", 1.0)
    if args.bargain_segment:
        cfg["segment"] = args.bargain_segment
    if args.num_steps % cfg["segment"]:
        raise SystemExit(f"--num-steps {args.num_steps} must be a multiple of the "
                         f"segment length {cfg['segment']}")

    env_kwargs = {"num_agents": n, "shared_rewards": False, "cnn": True, "jit": True,
                  "apple_reward": 1.0, "num_inner_steps": args.num_steps}
    for kv in args.env_kwarg:
        k, _, raw = kv.partition("=")
        env_kwargs[k] = _parse_env_kwarg_value(raw)
    env = LogWrapper(socialjax.make("clean_up", **env_kwargs), replace_info=False)
    lo, hi, source = contract_range(args.checkpoint, args.contract_low,
                                    args.contract_high)
    contract = CleanupContract(n, lo, hi)

    print(f"run: {moca['stem']}")
    print(f"contract: theta in [{lo:g}, {hi:g}] (from {source})")
    if source == "fallback":
        print("  [warning] no .run.yaml sidecar and no --contract-low/--contract-high: "
              "this is a GUESS. If the run was not trained on this range, every theta "
              "below is rescaled and the numbers are wrong.")
    rec, K = rollout(env, gp, bp, contract, cfg, args.episodes, args.num_steps, args.seed)
    report(rec, K, cfg, contract, n, args.num_steps)


if __name__ == "__main__":
    main()
