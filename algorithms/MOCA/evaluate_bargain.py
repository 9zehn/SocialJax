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
  5. Is the vote a REPLY to the offer?   accept rate and p(accept) binned by the
                                         theta on the table, pooled and per agent.
                                         A reservation value shows up here as a
                                         slope; a flat row means the agents are not
                                         conditioning on the offer at all.
  6. Who wanted what?                    per agent: role, what it proposed, how it
                                         voted, whether its offers carried
  7. Every episode                       one line each, so outliers are visible
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
from algorithms.utils import contract_range, load_params, load_run_config
from algorithms.MOCA import bargain as bg
from algorithms.MOCA import negotiate as neg
from algorithms.MOCA import reporting
from algorithms.MOCA.contracts import CleanupContract
from algorithms.MOCA.networks import BargainingActorCritic, ContractActorCritic


def rollout(env, gp, bp, contract, cfg, num_envs, num_steps, seed, cp=None):
    """`num_envs` full episodes in parallel. Returns per-round and per-segment records.

    Mirrors the training rollout exactly -- same feature scales, same proposer rule,
    same quorum -- because a policy evaluated on a differently-scaled state is not
    the policy that was trained. When `cp` (claim policies) is given, the claims-
    and-audits layer replays too, with the run's audit probability and fine, since
    a reporting run evaluated with perfect enforcement is not the run that trained.
    """
    n = env.num_agents
    report = cp is not None
    cnet = reporting.ClaimPolicy() if report else None
    net = ContractActorCritic(env.action_space().n, activation="relu")
    # Whether the run trained the counterfactual branch heads is read off the
    # weights, not off a flag: flax needs the module structure to match the params,
    # and the params are the one source that cannot be out of date.
    bnet = BargainingActorCritic(hidden=cfg["hidden"], activation="relu",
                                 accept_bias=cfg["accept_bias"],
                                 aux_heads=bg.params_have_aux_heads(bp[0]))
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
        (st, ob, rng, agreed, locked, cum_r, cum_c, last_tn, had, nrej, riv,
         last_votes, last_nacc, last_rej) = carry
        if report:
            rng, kp, kt, kv, kc, ka = jax.random.split(rng, 6)
        else:
            rng, kp, kt, kv = jax.random.split(rng, 4)
        prop = bg.proposer_for_round(r, n, num_envs, cfg["proposer"], key=kp,
                                     contributions=cum_c, start_offset=offset,
                                     holdouts=last_rej)

        def feats_at(live_tn, live):
            return bg.bargaining_features(r, K, prop, n, last_tn, had, nrej,
                                          live_tn, live, last_votes, last_nacc,
                                          cum_r / ret_s, cum_c / cl_s, riv / riv_s,
                                          mask)

        # Two passes, exactly as in training: propose first, then vote on what was
        # proposed. Collapsing them back into one would replay a policy that had
        # never been trained -- the votes would be reading an empty table.
        zeros_e = jnp.zeros((num_envs,), jnp.float32)
        feats_prop = feats_at(zeros_e, zeros_e)
        kts, kvs = jax.random.split(kt, n), jax.random.split(kv, n)
        raw = []
        for i in range(n):
            pt, _, _ = bnet.apply(bp[i], feats_prop[i])
            raw.append(pt.sample(seed=kts[i])[:, 0])
        raw = jnp.stack(raw)
        theta_all = neg.unsquash(raw, contract.low, contract.high)
        mine = bg.is_proposer_mask(prop, n)
        offer = jnp.sum(jnp.where(mine, theta_all, 0.0), axis=0)

        feats_vote = feats_at(bg.normalise_theta(offer, contract.low, contract.high),
                              jnp.ones((num_envs,), jnp.float32))
        votes, p_acc = [], []
        for i in range(n):
            _, pv, _ = bnet.apply(bp[i], feats_vote[i])
            # eps=0: the exploration floor is a training device, so what is measured
            # here is the policy itself. p_accept is kept as well -- it is the
            # offer-conditioned quantity, and reading it off the distribution rather
            # than the sampled votes gives a far less noisy picture per bin.
            v, _ = bg.floored_vote(pv, 0.0, kvs[i])
            votes.append(v)
            p_acc.append(pv.probs[..., 1])
        votes, p_acc = jnp.stack(votes), jnp.stack(p_acc)
        passed, n_acc = bg.accepted(votes.astype(bool), prop, quorum, n)
        # As in training: an offer of exactly the null contract never takes force --
        # accepted or not, the fallback applies and negotiation reopens. How long an
        # offer that DOES carry governs is BARGAIN_BINDING, read from the run's
        # sidecar; a run without one predates the setting and is `episode`.
        newly, theta_eff, next_agreed, next_locked = bg.apply_binding(
            cfg.get("binding", "episode"), passed, contract.is_null(offer), offer,
            agreed, locked, contract.null)

        (st, ob, _, rng), (cl, rew, tr, riv_t) = jax.lax.scan(
            seg_step, (st, ob, theta_eff, rng), None, x)
        seg_cl, seg_base, seg_tr = cl.sum(0), rew.sum(0), tr.sum(0)      # (N, E)

        # Claims and audits, as in training: file after the window, audit against
        # ground truth, settle zero-sum. Identically zero under the null contract.
        overclaim = jnp.zeros((n, num_envs), jnp.float32)
        audited = jnp.zeros((n, num_envs), bool)
        settle = jnp.zeros((n, num_envs), jnp.float32)
        if report:
            feats_claim = reporting.claim_features(
                seg_cl, theta_eff, x, contract.low, contract.high)
            kcs = jax.random.split(kc, n)
            raw_c = jnp.stack([
                cnet.apply(cp[i], feats_claim[i]).sample(seed=kcs[i])[:, 0]
                for i in range(n)
            ])
            overclaim = neg.unsquash(raw_c, 0.0, cfg["report_max_overclaim"])
            audited = jax.random.uniform(ka, (n, num_envs)) < cfg["report_audit_p"]
            settle, _ = reporting.settle_claims(
                theta_eff, overclaim, audited, cfg["report_fine_mult"], n)

        rec = {"proposer": prop, "offer": offer, "votes": votes, "accepted": passed,
               "newly": newly, "active": ~agreed, "n_accept": n_acc,
               "p_accept": p_acc, "theta_eff": theta_eff, "cleaned": seg_cl,
               "base": seg_base, "transfer": seg_tr, "river": riv_t.mean(0),
               "overclaim": overclaim, "audited": audited, "report_tr": settle}
        carry = (st, ob, rng, next_agreed, next_locked,
                 cum_r + seg_base + seg_tr + settle, cum_c + seg_cl,
                 bg.normalise_theta(offer, contract.low, contract.high),
                 jnp.ones_like(had), nrej + (~agreed & ~passed).astype(jnp.int32),
                 riv_t[-1].astype(jnp.float32),
                 (votes.astype(bool) & ~mine).astype(jnp.float32),
                 n_acc.astype(jnp.float32),
                 (~votes.astype(bool) & ~mine).astype(jnp.float32))
        return carry, rec

    z_e = jnp.zeros((num_envs,), jnp.float32)
    init = (st, obsv, key, jnp.zeros((num_envs,), bool),
            jnp.full((num_envs,), contract.null),
            jnp.zeros((n, num_envs), jnp.float32), jnp.zeros((n, num_envs), jnp.float32),
            z_e, jnp.zeros((num_envs,), bool), jnp.zeros((num_envs,), jnp.int32), z_e,
            jnp.zeros((n, num_envs), jnp.float32), z_e,
            jnp.zeros((n, num_envs), jnp.float32))
    _, rec = jax.lax.scan(round_, init, jnp.arange(K))
    return jax.tree.map(np.asarray, rec), K


def voting_vs_offer(rec, contract, n, blk, nbins=6):
    """Accept rate as a function of the theta on the table.

    The question this answers is whether the vote is a REPLY to the offer at all.
    Under the one-pass round the vote head never saw theta_r, so the only voting
    strategies it could express were "accept whoever proposed" and "always accept" --
    which is precisely what both early runs converged to. A flat row here now means
    the agents chose not to condition on the offer; a downward slope means a
    reservation value, which is the strategy the whole mechanism rests on.

    Two measures per bin, because they fail differently: the sampled vote rate is
    what actually happened, and the mean accept PROBABILITY is the same quantity
    with the sampling noise taken out (offers cluster, so some bins are thin).
    """
    act = rec["active"]                                            # (K, E)
    if not act.any():
        return
    is_prop = rec["proposer"][:, None, :] == np.arange(n)[None, :, None]   # (K,N,E)
    responder = act[:, None, :] & ~is_prop
    edges = np.linspace(contract.low, contract.high, nbins + 1)
    idx = np.clip(np.digitize(rec["offer"], edges) - 1, 0, nbins - 1)      # (K, E)

    blk("voting vs the offer  (does the vote condition on theta at all?)")
    print(f"  {'theta bin':>16}{'offers':>8}{'mean theta':>12}"
          f"{'accept rate':>13}{'p(accept)':>11}{'passed':>8}")
    rows = []
    for b in range(nbins):
        sel = act & (idx == b)
        if not sel.any():
            continue
        m = responder & sel[:, None, :]
        rate = rec["votes"][m].mean()
        prob = rec["p_accept"][m].mean()
        rows.append((rec["offer"][sel].mean(), rate, prob))
        print(f"  [{edges[b]:>6.3f},{edges[b+1]:>6.3f}){int(sel.sum()):>8}"
              f"{rec['offer'][sel].mean():>12.4f}{rate:>13.3f}{prob:>11.3f}"
              f"{rec['newly'][sel].mean():>8.3f}")
    if len(rows) >= 2:
        probs = np.array([r[2] for r in rows])
        thetas = np.array([r[0] for r in rows])
        spread = probs.max() - probs.min()
        corr = (np.corrcoef(thetas, probs)[0, 1] if probs.std() > 1e-9 else 0.0)
        print(f"\n  spread in p(accept) across bins {spread:.3f}"
              f"   correlation with theta {corr:+.3f}")
        if spread < 0.02:
            print("  -> voting is effectively CONSTANT in theta: whatever is being "
                  "learned, it is not a reservation value.")

    # Per agent, because a threshold that only one agent holds is invisible in the
    # pooled row above -- and one holdout is exactly the seed-42 veto-dictator shape.
    print("\n  p(accept) by agent and theta bin  ('--' = never a responder there)")
    header = "".join(f"{edges[b]:>9.2f}" for b in range(nbins))
    print(f"  {'agent':<7}{header}")
    for i in range(n):
        cells = []
        for b in range(nbins):
            m = responder[:, i, :] & act & (idx == b)
            cells.append(f"{rec['p_accept'][:, i, :][m].mean():>9.3f}"
                         if m.any() else f"{'--':>9}")
        print(f"  A{i:<6}" + "".join(cells))


def renegotiation_block(rec, K, contract, n, blk, agreed_any, agree_round, theta_ag,
                        contracted):
    """The `negotiation` block's analogue when there is no single agreement.

    Under BARGAIN_BINDING=segment/sticky an episode does not agree once; it holds up
    to K separate negotiations. So the questions change: how much of the episode ends
    up governed, what theta it is governed AT, and -- the one that decides how the
    result may be described -- whether that theta is a bargained consensus or just an
    average over whoever held the pen.
    """
    carried = rec["newly"]                                       # (K, E) new deal
    blk("renegotiation  (one bargain per segment)")
    print(f"  segments contracted  {contracted.mean():.3f}  "
          f"({int(contracted.sum())}/{contracted.size})")
    print(f"  offers that carried  {carried.mean():.3f} per round")
    print(f"  first contracted     R{agree_round.mean():.2f} +- "
          f"{agree_round.std():.2f}   (K={K} means never)")
    if contracted.any():
        t = rec["theta_eff"][contracted]
        print(f"  theta in force       {t.mean():.4f} +- {t.std():.4f}   "
              f"[{t.min():.3f}, {t.max():.3f}]")
    # Churn: how often the contract in force actually changes between segments. Near
    # zero means renegotiation is nominal -- the first deal is re-ratified every
    # segment and the protocol has collapsed back onto `episode` in all but name.
    if K > 1:
        churn = np.abs(np.diff(rec["theta_eff"], axis=0)) > 1e-9
        print(f"  contract changes     {churn.mean():.3f} per segment boundary")
        half = K // 2
        early = rec["theta_eff"][:half][contracted[:half]]
        late = rec["theta_eff"][half:][contracted[half:]]
        if early.size and late.size:
            # Escalation or concession over the episode: the proposer's leverage
            # should fall as the remaining episode shrinks, which is the one piece
            # of Rubinstein intuition that survives per-segment renegotiation.
            print(f"  theta drift          {early.mean():.4f} (first half) -> "
                  f"{late.mean():.4f} (second half)")

    # THE check on how this result may be described. If each proposer carries its own
    # very different theta, an even average is turn-taking -- every agent gets a turn
    # as dictator -- not bargaining. Convergence across proposers is the evidence
    # that the responders, not the rotation, are setting the number.
    per_prop = []
    for i in range(n):
        m = carried & (rec["proposer"] == i)
        per_prop.append(rec["offer"][m].mean() if m.any() else np.nan)
    per_prop = np.array(per_prop)
    if np.isfinite(per_prop).sum() >= 2:
        lo, hi = np.nanmin(per_prop), np.nanmax(per_prop)
        print(f"  theta carried by proposer  " + "  ".join(
            f"A{i}:{v:.2f}" if np.isfinite(v) else f"A{i}:--"
            for i, v in enumerate(per_prop)))
        print(f"    spread {hi - lo:.3f} across proposers -- a wide spread means the "
              f"agreed theta is\n    whoever's turn it was, not what was bargained")


def gini_equality(v, axis=0):
    d = np.abs(np.expand_dims(v, axis) - np.expand_dims(v, axis + 1)).sum((axis, axis + 1))
    return 1.0 - d / (2.0 * v.shape[axis] * np.abs(v).sum(axis) + 1e-8)


def report(rec, K, cfg, contract, n, num_steps):
    x = cfg["segment"]
    # Episode return includes the claim settlements when the run has them --
    # rec["report_tr"] is identically zero otherwise.
    ret = (rec["base"] + rec["transfer"] + rec["report_tr"]).sum(0)    # (N, E)
    welfare = ret.sum(0)                                        # (E,)
    equality = gini_equality(ret, axis=0)
    contracted = rec["theta_eff"] > contract.null + 1e-9        # (K, E)
    cl_per_step = rec["cleaned"].sum(1) / x                     # (K, E) all agents
    agreed_any = rec["newly"].any(0)                            # (E,)
    binding = cfg.get("binding", "episode")
    # Under `episode` at most one round can carry, so "the round it agreed" and
    # "the theta it agreed on" are single well-defined numbers. Under renegotiation
    # there are up to K of each, so the first carried segment and the theta actually
    # PLAYED UNDER are the analogues -- summing over carried rounds, as the episode
    # formulas do, would add several agreements together into a number that is not
    # any theta anyone ever offered.
    agree_round = np.where(agreed_any, np.argmax(rec["newly"], axis=0), K)
    if binding == "episode":
        theta_ag = (rec["newly"] * rec["offer"]).sum(0)                    # (E,)
    else:
        theta_ag = ((rec["theta_eff"] * contracted).sum(0)
                    / np.maximum(contracted.sum(0), 1))

    def blk(t):
        print(f"\n\033[1m{t}\033[0m" if sys.stdout.isatty() else f"\n{t}")

    print(f"protocol: segment={x}  rounds={K}  binding={binding}  "
          f"proposer={cfg['proposer']}"
          f"(start {cfg['rotate_start']})  quorum={cfg['quorum']}"
          f"={bg.quorum_size(cfg['quorum'], n)}/{n-1}  features={cfg['features']}")
    if binding != "episode":
        print("  renegotiated every segment: a carried offer governs ITS OWN segment"
              + (", a failed round keeps the incumbent" if binding == "sticky"
                 else ", a failed round plays uncontracted"))
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

    if binding == "episode":
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
    else:
        renegotiation_block(rec, K, contract, n, blk, agreed_any, agree_round,
                            theta_ag, contracted)
    print(f"  steps uncontracted {(~contracted).sum(0).mean() * x:.0f} of {num_steps}")

    if cfg.get("report"):
        p_a, lam = cfg["report_audit_p"], cfg["report_fine_mult"]
        lam_star = reporting.honesty_threshold(p_a)
        side = "lying has NEGATIVE EV" if lam > lam_star else "lying has POSITIVE EV"
        blk("claims and audits  (payment on reported cleaning)")
        print(f"  audit p={p_a:g}  fine x{lam:g}  honesty needs fine > "
              f"{lam_star:.2f}  ->  {side}")
        w = contracted.astype(np.float32)                        # (K, E) in force
        n_claims = np.maximum(w.sum(), 1.0)
        oc = rec["overclaim"]                                    # (K, N, E)
        paid = oc * ~rec["audited"] * rec["theta_eff"][:, None, :]
        fined = (oc * rec["audited"] * rec["theta_eff"][:, None, :]
                 * lam)
        print(f"  mean overclaim/claim  {(oc * w[:, None, :]).sum() / (n_claims * n):.3f}"
              f"   (cap {cfg['report_max_overclaim']:g})")
        print(f"  leakage /episode      {paid.sum() / oc.shape[-1]:.1f}   "
              f"fines /episode {fined.sum() / oc.shape[-1]:.1f}")
        print(f"  {'agent':<7}{'overclaim':>11}{'true clean/win':>15}{'caught rate':>13}")
        for i in range(n):
            oc_i = (oc[:, i] * w).sum() / n_claims
            cl_i = (rec["cleaned"][:, i] * w).sum() / n_claims
            lied = (oc[:, i] > 0.5) & (w > 0)
            caught = (lied & rec["audited"][:, i]).sum() / max(lied.sum(), 1)
            print(f"  A{i:<6}{oc_i:>11.3f}{cl_i:>15.2f}{caught:>13.3f}")

    blk("offers by round  (do proposers concede as the episode shrinks?)")
    print(f"  {'round':>6}{'offers':>8}{'mean theta':>12}{'accepted':>10}{'accept rate':>13}")
    for r in range(K):
        act = rec["active"][r]
        if not act.any():
            continue
        print(f"  {r:>6}{int(act.sum()):>8}{rec['offer'][r][act].mean():>12.4f}"
              f"{int(rec['newly'][r].sum()):>10}"
              f"{rec['newly'][r][act].mean():>13.3f}")

    voting_vs_offer(rec, contract, n, blk)

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
    if binding == "episode":
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
    else:
        # "agreed at R3 by A5" has no meaning when every segment is its own bargain,
        # so the columns become the episode's contracting HISTORY: how many of its
        # segments were governed, at what mean theta, and how many distinct deals it
        # took to get there.
        print(f"  {'ep':>3}{'segs':>7}{'theta':>8}{'deals':>7}{'welfare':>10}"
              f"{'equality':>10}{'clean/step':>12}")
        for e in range(ret.shape[1]):
            n_seg = int(contracted[:, e].sum())
            th = f"{theta_ag[e]:.3f}" if n_seg else "-"
            print(f"  {e:>3}{f'{n_seg}/{K}':>7}{th:>8}"
                  f"{int(rec['newly'][:, e].sum()):>7}{welfare[e]:>10.1f}"
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
    p.add_argument("--report-audit-p", type=float, default=None)
    p.add_argument("--report-fine-mult", type=float, default=None)
    p.add_argument("--report-max-overclaim", type=float, default=None,
                   help="claims-and-audits parameters, needed only for runs with "
                        "_claim_ checkpoints. Read from the .run.yaml sidecar when "
                        "it has one; these flags override it. A mismatch replays a "
                        "different enforcement regime than the one trained.")
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

    cfg = infer_bargain_config(moca["stem"], checkpoint=args.checkpoint)
    try:
        bg.check_params_compatible(bp[0], n, cfg.get("feature_version"),
                                   hidden=cfg["hidden"], label=moca["stem"])
    except ValueError as e:
        raise SystemExit(f"[incompatible checkpoint] {e}")
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

    # Claims-and-audits runs leave `_claim_` checkpoints beside the others. When
    # they exist the enforcement layer replays too -- silently evaluating such a
    # run under perfect enforcement would report a mechanism it never trained.
    claim_paths = [q.replace("_contract_", "_claim_") for q in moca["contract_paths"]]
    cp = None
    if all(Path(q).exists() for q in claim_paths):
        run_cfg = load_run_config(args.checkpoint) or {}
        report_params = {}
        for key, flag in (("report_audit_p", args.report_audit_p),
                          ("report_fine_mult", args.report_fine_mult),
                          ("report_max_overclaim", args.report_max_overclaim)):
            val = flag if flag is not None else run_cfg.get(key.upper())
            if val is None:
                raise SystemExit(
                    f"this run has _claim_ checkpoints but {key.upper()} is not in "
                    f"its .run.yaml sidecar and --{key.replace('_', '-')} was not "
                    f"given. Refusing to guess an enforcement regime.")
            report_params[key] = float(val)
        cfg.update(report_params, report=True)
        cp = [load_params(q) for q in claim_paths]
        print(f"claims: audit p={cfg['report_audit_p']:g}, fine x"
              f"{cfg['report_fine_mult']:g}, overclaim cap "
              f"{cfg['report_max_overclaim']:g}")

    rec, K = rollout(env, gp, bp, contract, cfg, args.episodes, args.num_steps,
                     args.seed, cp=cp)
    report(rec, K, cfg, contract, n, args.num_steps)


if __name__ == "__main__":
    main()
