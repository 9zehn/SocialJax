"""Sweep theta through a trained vote head: where are the reservation values?

evaluate_bargain.py can only bin votes by the offers the proposers actually made,
and converged proposers cluster (the seed-42 eval had one offer each in the middle
bins). This asks the vote head directly: for every theta on a grid, and every
possible proposer, what is each responder's accept probability -- no environment,
no rollout, no sampling noise.

At round 0 the sweep is EXACT, not an approximation: the round-0 bargaining state
is fully determined by (theta, proposer). Every history feature -- standing offer,
rejections, last votes, own return, own cleaning -- is zero at the first decision
of an episode, and the public tier (river stock) is masked out under the default
BARGAIN_FEATURES=private. Since ~3/4 of agreements land in round 0, this is also
the state distribution that matters most.

For later rounds (--round > 0) the history must be invented, so the tool fills it
with a plausible default (all earlier rounds failed, standing offer as given) and
says so; read those numbers as indicative, not exact.

Usage:
    python algorithms/MOCA/probe_votes.py \
        --checkpoint 'runs/<run>/clean_up_..._bargain_seg100_joint_[0-9].pkl'

    # later-round state, custom grid, per-proposer detail
    python algorithms/MOCA/probe_votes.py --checkpoint '...' \
        --round 2 --last-theta 0.4 --grid 13 --by-proposer

    # ...plus evaluate_bargain's full episode report underneath, so the swept
    # thresholds can be read against the offers the proposers actually made
    python algorithms/MOCA/probe_votes.py --checkpoint '...' --episodes 40
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax.numpy as jnp
import numpy as np

from algorithms.utils import contract_range, load_params
from algorithms.MOCA import bargain as bg
from algorithms.MOCA.networks import BargainingActorCritic


def pass_probability(p_accept, quorum):
    """P(#accepts >= quorum) for independent Bernoulli responders (exact DP)."""
    dist = np.zeros(len(p_accept) + 1)
    dist[0] = 1.0
    for p in p_accept:
        new = np.zeros_like(dist)
        new[0] = dist[0] * (1.0 - p)
        new[1:] = dist[1:] * (1.0 - p) + dist[:-1] * p
        dist = new
    return dist[quorum:].sum()


def crossings(thetas, probs, level=0.5):
    """theta values where p(accept) crosses `level`, linearly interpolated."""
    out = []
    for a, b, pa, pb in zip(thetas[:-1], thetas[1:], probs[:-1], probs[1:]):
        if (pa - level) * (pb - level) < 0:
            out.append(a + (level - pa) * (b - a) / (pb - pa))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True,
                   help="glob matching the run's GAMEPLAY policies (the bargaining "
                        "policies are found beside them, as in evaluate_bargain)")
    p.add_argument("--grid", type=int, default=25, help="theta grid points")
    p.add_argument("--round", type=int, default=0,
                   help="round index to probe (0 = exact; later rounds need an "
                        "invented history)")
    p.add_argument("--last-theta", type=float, default=None,
                   help="standing rejected offer for --round > 0 (default: range "
                        "midpoint)")
    p.add_argument("--num-steps", type=int, default=1000,
                   help="episode length, to compute the number of rounds K")
    p.add_argument("--contract-low", type=float, default=None)
    p.add_argument("--contract-high", type=float, default=None,
                   help="theta bounds; read from the .run.yaml sidecar when present")
    p.add_argument("--by-proposer", action="store_true",
                   help="also print each (proposer, responder) pair separately")
    p.add_argument("--binding", choices=bg.BINDING_MODES, default=None,
                   help="how long a carried offer bound; read from the .run.yaml "
                        "sidecar when there is one. Only affects --episodes (the "
                        "sweep itself is per-round either way), but a renegotiated "
                        "run replayed as 'episode' reports its first contracted "
                        "segment as an episode-long agreement.")
    p.add_argument("--episodes", type=int, default=0,
                   help="also roll out this many episodes and print "
                        "evaluate_bargain's full report underneath the sweep. The "
                        "two answer different questions and are worth reading "
                        "together: the sweep says what the policy WOULD do at every "
                        "theta, the rollout says what actually happened at the "
                        "thetas the proposers really offered. 0 (default) skips it "
                        "and keeps this tool rollout-free.")
    p.add_argument("--env-kwarg", action="append", default=[], metavar="KEY=VALUE")
    args = p.parse_args()

    from viz.interactive_viewer import detect_moca, infer_bargain_config

    moca = detect_moca(args.checkpoint)
    if moca is None or moca["mode"] != "bargain":
        raise SystemExit(
            f"not a bargaining run (detected: {moca['mode'] if moca else 'none'})")
    bp = [load_params(q) for q in moca["contract_paths"]]
    n = len(bp)
    cfg = infer_bargain_config(moca["stem"], checkpoint=args.checkpoint)
    if cfg.get("protocol") == "median":
        raise SystemExit(
            "this is a BARGAIN_PROTOCOL=median run: there is no vote to probe -- "
            "the vote head exists in the network but was never trained, so its "
            "curves would describe an initialisation, not a mechanism. The "
            "analogue is the asks table in evaluate_bargain.py (per-agent mean "
            "ask, held-median rate, ask spread).")
    if args.binding:
        cfg["binding"], cfg["binding_source"] = args.binding, "flag"
    try:
        bg.check_params_compatible(bp[0], n, cfg.get("feature_version"),
                                   hidden=cfg["hidden"], label=moca["stem"])
    except ValueError as e:
        raise SystemExit(f"[incompatible checkpoint] {e}")

    lo, hi, source = contract_range(args.checkpoint, args.contract_low,
                                    args.contract_high)
    if source == "fallback":
        print("[warning] no .run.yaml sidecar and no --contract-low/--contract-high: "
              "the range is a GUESS, and a wrong range shifts every threshold below.")
    K = args.num_steps // cfg["segment"]
    if not 0 <= args.round < K:
        raise SystemExit(f"--round must be in [0, {K - 1}] for {K} rounds")
    quorum = bg.quorum_size(cfg["quorum"], n)
    # Read from the weights, so both a gae-trained and a counterfactual-trained
    # checkpoint sweep with no flag from the user.
    has_aux = bg.params_have_aux_heads(bp[0])
    bnet = BargainingActorCritic(hidden=cfg["hidden"], activation="relu",
                                 accept_bias=cfg["accept_bias"], aux_heads=has_aux)
    mask = bg.feature_mask(cfg["features"], n)

    # One feature row per (proposer, theta) combination; e = p * G + g.
    G = args.grid
    thetas = np.linspace(lo, hi, G)
    E = n * G
    theta_col = jnp.asarray(np.tile(thetas, n), jnp.float32)              # (E,)
    proposer = jnp.asarray(np.repeat(np.arange(n), G), jnp.int32)         # (E,)

    r = args.round
    zeros_e = jnp.zeros((E,), jnp.float32)
    if r == 0:
        last_tn, had, nrej = zeros_e, jnp.zeros((E,), bool), jnp.zeros((E,), jnp.int32)
        note = "round 0 (exact: round-0 history is all zeros)"
    else:
        last_theta = args.last_theta if args.last_theta is not None else (lo + hi) / 2
        last_tn = jnp.full((E,), bg.normalise_theta(last_theta, lo, hi), jnp.float32)
        had = jnp.ones((E,), bool)
        nrej = jnp.full((E,), r, jnp.int32)
        note = (f"round {r} (INVENTED history: {r} failed rounds, standing offer "
                f"{last_theta:g}, no vote record, zero own-payoff features)")
    feats = bg.bargaining_features(
        r, K, proposer, n, last_tn, had, nrej,
        bg.normalise_theta(theta_col, lo, hi), jnp.ones((E,), jnp.float32),
        jnp.zeros((n, E), jnp.float32), zeros_e,
        jnp.zeros((n, E), jnp.float32), jnp.zeros((n, E), jnp.float32),
        zeros_e, mask)                                                    # (N, E, F)

    p_acc = np.zeros((n, n, G))                     # [responder, proposer, theta]
    gap = np.zeros((n, n, G)) if has_aux else None  # lock value - continue value
    for i in range(n):
        if has_aux:
            _, pi_vote, _, lock_v, cont_v = bnet.apply(bp[i], feats[i],
                                                       return_aux=True)
            gap[i] = np.asarray(lock_v - cont_v).reshape(n, G)
        else:
            _, pi_vote, _ = bnet.apply(bp[i], feats[i])
        p_acc[i] = np.asarray(pi_vote.probs[..., 1]).reshape(n, G)

    print(f"run: {moca['stem']}")
    print(f"contract: theta in [{lo:g}, {hi:g}] (from {source})")
    print(f"features={cfg['features']}  quorum={cfg['quorum']}={quorum}/{n - 1}  "
          f"K={K} rounds (segment={cfg['segment']}, {args.num_steps} steps)")
    print(f"vote-pass sweep at {note}\n")

    resp = ~np.eye(n, dtype=bool)                   # [responder, proposer]
    print("p(accept) by offered theta -- per responder, averaged over proposers;")
    print("P(pass) is the exact quorum probability, averaged over proposers")
    print("  theta " + "".join(f"{f'A{i}':>8}" for i in range(n)) + f"{'P(pass)':>9}")
    for g, th in enumerate(thetas):
        cells = "".join(
            f"{p_acc[i, resp[i], g].mean():>8.3f}" for i in range(n))
        pp = np.mean([
            pass_probability(p_acc[resp[:, pr], pr, g], quorum) for pr in range(n)])
        print(f"  {th:5.2f} {cells}{pp:>9.3f}")

    print("\nper responder: range of p(accept) over theta, and 0.5 crossings")
    for i in range(n):
        curve = p_acc[i, resp[i], :].mean(axis=0)
        cross = crossings(thetas, curve)
        cross_s = ", ".join(f"{c:.2f}" for c in cross) if cross else "none"
        print(f"  A{i}: p(accept) {curve.min():.3f}..{curve.max():.3f}  "
              f"slope {'+' if curve[-1] >= curve[0] else '-'}"
              f"{abs(curve[-1] - curve[0]):.3f}  crosses 0.5 at: {cross_s}")

    if has_aux:
        # What the agent BELIEVES the vote is worth, next to what it does about it.
        # The gap is (value if this offer locks now) - (value of one more null
        # segment and reopening), so its sign is the whole reservation-value
        # question: negative means "I am better off bargaining on". Printed beside
        # p(accept) because the two together separate wrong beliefs from right
        # beliefs acted on wrongly -- an agent that knows a lowball hurts it and
        # accepts anyway has a policy problem, not a value problem.
        print("\nbelieved lock-continue value gap (negative = better to keep "
              "bargaining), averaged over proposers")
        print("  theta " + "".join(f"{f'A{i}':>8}" for i in range(n)))
        for g, th in enumerate(thetas):
            cells = "".join(f"{gap[i, resp[i], g].mean():>8.2f}" for i in range(n))
            print(f"  {th:5.2f} {cells}")
        print("\n  agreement between belief and vote, per responder")
        for i in range(n):
            curve = p_acc[i, resp[i], :].mean(axis=0)
            gcurve = gap[i, resp[i], :].mean(axis=0)
            wants = gcurve > 0
            # Where belief and action disagree: it thinks continuing is better yet
            # still accepts (or the reverse). This is the quantity the
            # counterfactual advantage is meant to drive to zero.
            mismatch = np.mean(wants != (curve > 0.5))
            zero = crossings(thetas, gcurve, level=0.0)
            zero_s = ", ".join(f"{c:.2f}" for c in zero) if zero else "none"
            print(f"  A{i}: gap {gcurve.min():+.2f}..{gcurve.max():+.2f}  "
                  f"indifferent at: {zero_s}   "
                  f"belief/action mismatch {mismatch:.0%} of the grid")

    if args.by_proposer:
        for pr in range(n):
            print(f"\np(accept) when A{pr} proposes")
            print("  theta " + "".join(
                f"{f'A{i}':>8}" for i in range(n) if i != pr) + f"{'P(pass)':>9}")
            for g, th in enumerate(thetas):
                cells = "".join(
                    f"{p_acc[i, pr, g]:>8.3f}" for i in range(n) if i != pr)
                pp = pass_probability(p_acc[resp[:, pr], pr, g], quorum)
                print(f"  {th:5.2f} {cells}{pp:>9.3f}")

    if args.episodes:
        episode_report(args, moca, cfg, contract=(lo, hi), n=n, bp=bp)


def episode_report(args, moca, cfg, contract, n, bp):
    """evaluate_bargain's rollout and report, underneath the sweep.

    The sweep above is what the policy WOULD do at every theta; this is what it
    actually did at the thetas the proposers really offered, and what that bought.
    Neither substitutes for the other -- a threshold at theta=0.6 means nothing if
    every realised offer is 2.4, and a good welfare number means nothing if the vote
    turns out to be unconditional.

    Delegates rather than reimplements, so the two tools cannot drift apart on the
    feature scales or the protocol.
    """
    import socialjax
    from socialjax.wrappers.baselines import LogWrapper
    from algorithms.MOCA.contracts import CleanupContract
    from algorithms.MOCA.evaluate_bargain import report, rollout
    from viz.interactive_viewer import _gameplay_checkpoints, _parse_env_kwarg_value

    lo, hi = contract
    if args.num_steps % cfg["segment"]:
        raise SystemExit(f"--num-steps {args.num_steps} must be a multiple of the "
                         f"segment length {cfg['segment']}")
    gp = [load_params(q) for q in _gameplay_checkpoints(args.checkpoint)]
    if len(gp) != n:
        raise SystemExit(f"{len(gp)} gameplay but {n} bargaining policies")
    # A claims-and-audits run replays under its own enforcement regime, which needs
    # parameters this tool does not take. Refuse rather than quietly report a
    # perfectly-enforced version of a run that was never perfectly enforced.
    if any(Path(q.replace("_contract_", "_claim_")).exists()
           for q in moca["contract_paths"]):
        raise SystemExit(
            "this run has _claim_ checkpoints (claims and audits). Run "
            "evaluate_bargain.py for the episode statistics -- it takes the audit "
            "probability and fine that the replay needs.")

    env_kwargs = {"num_agents": n, "shared_rewards": False, "cnn": True, "jit": True,
                  "apple_reward": 1.0, "num_inner_steps": args.num_steps}
    for kv in args.env_kwarg:
        k, _, raw = kv.partition("=")
        env_kwargs[k] = _parse_env_kwarg_value(raw)
    env = LogWrapper(socialjax.make("clean_up", **env_kwargs), replace_info=False)
    contract_obj = CleanupContract(n, lo, hi)

    print(f"\n{'=' * 72}")
    print(f"rollout: {args.episodes} episodes x {args.num_steps} steps")
    rec, K = rollout(env, gp, bp, contract_obj, cfg, args.episodes, args.num_steps,
                     seed=0)
    report(rec, K, cfg, contract_obj, n, args.num_steps)


if __name__ == "__main__":
    main()
