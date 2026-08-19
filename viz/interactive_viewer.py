"""Local rollout viewer for SocialJax.

Runs a policy (random, or a trained checkpoint saved by algorithms/utils/io_utils.py)
against any registered environment, renders each step, and either exports a GIF
or opens a local matplotlib scrubber (with play/pause, speed, and loop controls)
to step through the episode.

Rollout itself is cheap (single episode, no gradients) even on a CPU-only laptop.
Rendering each frame is comparatively expensive (the per-tile drawing is plain
Python/numpy, not jitted) so it's parallelized across CPU cores via --render-workers.

Examples:
    # No checkpoint yet -> random policy, just to check the env/renderer
    python viz/interactive_viewer.py --env clean_up --steps 200

    # Trained IPPO checkpoint, export a GIF instead of opening a window
    python viz/interactive_viewer.py --env clean_up \\
        --checkpoint checkpoints/clean_up_seed30_reward_individual.pkl \\
        --gif viz/out/clean_up_rollout.gif --no-interactive

    # PARAMETER_SHARING=False run (one .pkl per agent): pass a glob, quoted so
    # the shell doesn't expand it; files are sorted and assigned to agents 0..N-1
    python viz/interactive_viewer.py --env clean_up \\
        --checkpoint 'checkpoints/individual/clean_up_seed0_reward_individual_*.pkl'

    # Contracting run on any of the three environments -- --env is optional, since
    # the run's .run.yaml sidecar records which one it was trained on. Renegotiated
    # bargaining replays segment by segment, with the round log in the panel:
    python viz/interactive_viewer.py \\
        --checkpoint 'runs/.../harvest_common_open_seed42_reward_individual_agents7_bargain_seg100_segment_joint_[0-9].pkl'
"""
import argparse
import functools
import math
import os
import re
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

import socialjax
from algorithms.utils.io_utils import load_params, load_run_config



def rollout(env, params, num_steps, seed, contract=None, theta=None, spec=None):
    """Step the env and collect raw states (fast, sequential — each step depends on the last).

    Rendering is deferred to render_states() so it can be parallelized separately.

    Returns (states, extras) with three per-step (N,) series:
        act        the CONTRACTED ACT, i.e. the quantity the contract prices --
                   cells cleaned on Clean Up, thin-patch apples eaten on Harvest.
                   Which info field that is comes from `spec` (algorithms/MOCA/envs),
                   so the viewer never has to know one environment's field names.
        reward     the environment's own per-agent reward, BEFORE transfers.
        transfers  the zero-sum contract transfer (all zeros unless a MOCA `contract`
                   and `theta` are supplied, in which case the gameplay policy is the
                   contract-conditioned network).
    """
    use_contract = contract is not None and theta is not None
    act_key = spec.contracted_act if spec is not None else "cleaned_by_agent"
    rng = jax.random.PRNGKey(seed)
    rng, rng_reset = jax.random.split(rng)
    obs, state = env.reset(rng_reset)

    states = [state]
    transfers_per_step = [np.zeros(env.num_agents)]
    act_per_step = [np.zeros(env.num_agents, dtype=np.float32)]
    reward_per_step = [np.zeros(env.num_agents, dtype=np.float32)]

    network = None
    contract_vec = None
    if params is not None:
        if use_contract:
            from algorithms.MOCA.networks import ContractActorCritic
            network = ContractActorCritic(action_dim=env.action_space().n, activation="relu")
            # Same feature vector the policy was trained on: [theta_normalised, stage].
            contract_vec = contract.to_obs(jnp.float32(theta))[None, ...]
        else:
            from algorithms.utils.networks import ActorCritic
            network = ActorCritic(action_dim=env.action_space().n, activation="relu")
        if isinstance(params, list) and len(params) != env.num_agents:
            raise SystemExit(
                f"got {len(params)} per-agent checkpoints but env has {env.num_agents} agents"
            )

    def _apply(p, x):
        return network.apply(p, x, contract_vec) if use_contract else network.apply(p, x)

    for _ in range(num_steps):
        rng, rng_act, rng_step = jax.random.split(rng, 3)

        if network is None:
            action_keys = jax.random.split(rng_act, env.num_agents)
            actions = [
                int(jax.random.randint(k, (), 0, env.action_space().n))
                for k in action_keys
            ]
        elif isinstance(params, list):
            # PARAMETER_SHARING=False: one set of weights per agent
            actions = []
            act_keys = jax.random.split(rng_act, env.num_agents)
            for i, a in enumerate(env.agents):
                pi, _ = _apply(params[i], obs[a][None, ...])
                actions.append(int(pi.sample(seed=act_keys[i]).squeeze()))
        else:
            obs_batch = jnp.stack([obs[a] for a in env.agents])
            pi, _ = _apply(params, obs_batch)
            sampled = pi.sample(seed=rng_act)
            actions = [int(sampled[i]) for i in range(env.num_agents)]

        prev_state = state
        obs, state, reward, done, info = env.step(rng_step, state, actions)
        states.append(state)

        act = (np.atleast_1d(np.array(info[act_key], dtype=np.float32))
               if act_key in info else np.zeros(env.num_agents, dtype=np.float32))
        act_per_step.append(act)
        reward_per_step.append(
            np.atleast_1d(np.asarray(reward, dtype=np.float32)).reshape(-1))

        if use_contract:
            # Read straight out of the info dict, so WHICH signal is redistributed is
            # the contract's business and not the viewer's -- the same call the
            # training loop makes, and the reason a Harvest contract charges for
            # thin-patch eating here without the viewer knowing that it does.
            tr = np.array(contract.transfer_from_info(jnp.float32(theta), info))
            transfers_per_step.append(tr)
        else:
            transfers_per_step.append(np.zeros(env.num_agents))

        if bool(done["__all__"]):
            break

    return states, {"transfers": transfers_per_step, "act": act_per_step,
                    "reward": reward_per_step}



def infer_bargain_config(stem, checkpoint=None):
    """Recover the bargaining settings: the .run.yaml sidecar first, the stem second.

    checkpoint_filename encodes segment length always and the other three knobs when
    off-default, so the whole protocol is recoverable from the filename -- which
    matters because replaying under the wrong quorum or proposer rule silently
    reproduces a DIFFERENT mechanism rather than failing. The sidecar carries the
    same settings plus the ones no filename could hold (network width, accept bias,
    feature version), so it wins wherever both exist.

    `checkpoint` is any path or glob naming the run's files; omit it to fall back to
    the filename alone, as runs predating the sidecar must.
    """
    seg = re.search(r"_seg(\d+)", stem)
    # Anchored on a following "_" or end-of-stem so "_segment" cannot be read out of
    # "_seg100", which precedes it in every bargaining name.
    binding = re.search(r"_(episode|segment|sticky)(?=_|$)", stem)
    quorum = re.search(r"_q(majority|\d+)", stem)
    proposer = ("contribution" if "_contribution" in stem
                else "holdout" if "_holdout" in stem
                else "random" if "_random" in stem else "rotate")
    features = ("protocol" if "_protocol" in stem
                else "public" if "_public" in stem else "private")
    cfg = {
        "segment": int(seg.group(1)) if seg else 100,
        "quorum": (quorum.group(1) if quorum else "all"),
        "proposer": proposer,
        "features": features,
        # Simultaneous-move protocols are tagged in the stem (checkpoint_filename
        # writes the protocol INSTEAD of the proposer/quorum tags), so a median
        # run is recoverable from its filename alone. Replaying one as
        # alternating would ask the vote head questions it was never trained on.
        "protocol": "median" if "_median" in stem else "alternating",
        "rotate_start": "fixed" if "_fixedstart" in stem else "random",
        # Not in any filename, so a run trained off these defaults is unreplayable
        # without its sidecar -- the shape check in bargain.check_params_compatible
        # is what turns that into an error rather than wrong numbers.
        "hidden": 64,
        "accept_bias": 1.0,
        "feature_version": None,
        # How long a carried offer binds. checkpoint_filename marks this for EVERY
        # binding rather than only off-default ones, so a stem that carries the token
        # settles it on its own and a stem that does not is a run predating the
        # setting, for which "episode" is the only correct reading. Worth recovering
        # rather than defaulting: replaying a renegotiated run as an absorbing one
        # reports a first carried segment as an episode-long agreement, and every
        # downstream number follows it while looking entirely normal.
        "binding": (binding.group(1) if binding else "episode"),
        # What the bargained scalar MEANS: a wage per cell cleaned, or a tax rate on
        # harvest income. Same fallback logic, and a worse failure if it is wrong --
        # the two pay different agents from different bases, so a mismatch is a
        # complete fiction reported in ordinary-looking units.
        "contract_kind": "clean_wage",
        "tax_window": 20,
    }
    side = load_run_config(checkpoint) if checkpoint else None
    for key, recorded in (("segment", "BARGAIN_SEGMENT"), ("quorum", "BARGAIN_QUORUM"),
                          ("proposer", "BARGAIN_PROPOSER"),
                          ("features", "BARGAIN_FEATURES"),
                          ("rotate_start", "BARGAIN_ROTATE_START"),
                          ("hidden", "BARGAIN_HIDDEN"),
                          ("accept_bias", "BARGAIN_ACCEPT_BIAS"),
                          ("binding", "BARGAIN_BINDING"),
                          ("protocol", "BARGAIN_PROTOCOL"),
                          ("contract_kind", "CONTRACT_KIND"),
                          ("tax_window", "TAX_WINDOW"),
                          ("feature_version", "BARGAIN_FEATURE_VERSION")):
        if side is not None and side.get(recorded) is not None:
            cfg[key] = side[recorded]
    cfg["segment"] = int(cfg["segment"])
    cfg["hidden"] = int(cfg["hidden"])
    cfg["accept_bias"] = float(cfg["accept_bias"])
    # Whether the binding mode is known or assumed. Unlike every other setting here,
    # getting this one wrong does not fail or look odd -- a renegotiated run replays
    # as an episode-long agreement struck in its first contracted segment, and every
    # number that follows is internally consistent and wrong. So the tools say which
    # it was, the way contract_range does.
    cfg["binding_source"] = (
        "sidecar" if side is not None and side.get("BARGAIN_BINDING") is not None
        else "stem" if binding else "fallback")
    cfg["tax_window"] = int(cfg["tax_window"])
    cfg["kind_source"] = ("sidecar" if side is not None
                          and side.get("CONTRACT_KIND") is not None else "fallback")
    return cfg


def rollout_bargaining(env, gameplay_params, bargain_params, num_steps, seed,
                       contract, cfg, spec, fixed_theta=None):
    """Replay a Rubinstein bargaining run: negotiate, play a segment, repeat.

    Parallel to `rollout` rather than folded into it, so every pre-bargaining
    checkpoint keeps its exact replay path.

    Environment-independent by way of `spec` (an algorithms/MOCA/envs.EnvSpec):
    everything the bargaining state reads off the world -- the act the contract
    prices, the state of the commons, and the scales that put both at unit range --
    comes from there rather than from Clean Up's field names, because the bargaining
    policy was TRAINED on those exact features and a substitute would be a different
    input vector fed to the same weights.

    Returns (states, extras) with the same keys `rollout` produces, plus
    extras["rounds"]: one record per round in which a decision was actually made --
    what was offered, by whom, how each agent voted, and whether it carried. That
    is what the panel draws.

    `fixed_theta` overrides the negotiation entirely (replay a counterfactual
    contract); the round records are then empty, since nothing was negotiated.
    """
    from algorithms.MOCA import bargain as bg
    from algorithms.MOCA import contracts as bg_contracts
    from algorithms.MOCA import envs as moca_envs
    from algorithms.MOCA import negotiate as neg
    from algorithms.MOCA.networks import BargainingActorCritic, ContractActorCritic

    n = env.num_agents
    net = ContractActorCritic(action_dim=env.action_space().n, activation="relu")
    # aux_heads comes from the weights themselves: a counterfactual-trained run has
    # extra params, and flax needs the module to match what it is handed.
    bnet = BargainingActorCritic(hidden=cfg.get("hidden", 64), activation="relu",
                                 accept_bias=cfg.get("accept_bias", 1.0),
                                 aux_heads=bg.params_have_aux_heads(bargain_params[0]))
    mask = bg.feature_mask(cfg["features"], n)
    quorum = bg.quorum_size(cfg["quorum"], n)
    seg = int(cfg["segment"])

    # Scales must match training exactly or the policy reads a different state.
    # These are moca_cnn.py's, term for term: an episode's worth of unit reward, an
    # episode's worth of the contracted act (at most one per agent per step), and the
    # commons at its maximum -- which is grid cells on Clean Up and apple spawn
    # points on Harvest, hence read off the spec rather than off a grid size.
    act_key, commons_key = spec.contracted_act, spec.commons_metric
    inner = int(getattr(env, "num_inner_steps", num_steps))
    ret_scale = float(inner) * float(getattr(env, spec.reward_scale_kwarg, 1.0))
    act_scale = float(inner)
    commons_scale = float(moca_envs.commons_scale(spec, env))

    rng = jax.random.PRNGKey(seed)
    rng, k_reset, k_start = jax.random.split(rng, 3)
    obs, state = env.reset(k_reset)
    offset = (jax.random.randint(k_start, (1,), 0, n)
              if cfg["rotate_start"] == "random" else None)

    states, transfers, act_hist, rounds = [state], [np.zeros(n)], [np.zeros(n)], []
    rewards_hist = [np.zeros(n, np.float32)]
    cum_ret, cum_act = np.zeros(n, np.float32), np.zeros(n, np.float32)
    agreed, locked = False, float(contract.null)
    last_tn, had_offer, n_reject, commons = 0.0, False, 0, 0.0
    last_votes, last_n_accept = np.zeros(n, np.float32), 0.0
    last_rejecters = np.zeros(n, np.float32)
    step = 0
    num_rounds = max(1, int(np.ceil(num_steps / seg)))

    binding = cfg.get("binding", "episode")
    tax_kind = cfg.get("contract_kind", "clean_wage") == "harvest_tax"
    # Trailing cleaning for the harvest tax's payout weighting: one episode's worth,
    # carried across segment boundaries, never read under clean_wage.
    tax_win = bg_contracts.new_tax_window(
        cfg.get("tax_window", 20) if tax_kind else 0, n)
    for r in range(num_rounds):
        if fixed_theta is not None:
            theta = float(fixed_theta)
        elif agreed:
            # Only `episode` is absorbing. Under renegotiation `agreed` never
            # becomes True, so every segment bargains again.
            theta = locked
        elif cfg.get("protocol") == "median":
            # One simultaneous pass: every agent asks, the median binds for this
            # segment. No proposer, no vote -- the round record marks that with
            # proposer=-1 and carries the asks instead.
            rng, k_theta = jax.random.split(rng)
            feats = bg.median_round_features(
                r, num_rounds, n, jnp.array([last_tn], jnp.float32),
                jnp.array([had_offer]),
                jnp.asarray(cum_ret)[:, None] / ret_scale,
                jnp.asarray(cum_act)[:, None] / act_scale,
                jnp.array([commons], jnp.float32) / commons_scale, mask)
            kt = jax.random.split(k_theta, n)
            asks = []
            for i in range(n):
                pi_theta, _, _ = bnet.apply(bargain_params[i], feats[i])
                raw = pi_theta.sample(seed=kt[i])[0, 0]
                asks.append(float(neg.unsquash(raw, contract.low, contract.high)))
            theta_offer = float(np.median(asks))
            # A null median (only reachable when the range floor is 0) plays the
            # segment uncontracted, like a null offer everywhere else.
            took_force = theta_offer > contract.null + 1e-6
            theta = theta_offer if took_force else float(contract.null)
            locked = theta
            rounds.append({
                "round": r, "step": step, "proposer": -1, "theta": theta_offer,
                "votes": [], "accepted": bool(took_force), "n_accept": 0,
                "quorum": 0, "p_accept": [], "asks": asks,
                "in_force": float(theta),
            })
            last_tn = float(bg.normalise_theta(theta_offer, contract.low,
                                               contract.high))
            had_offer = True
        else:
            rng, k_prop, k_theta, k_vote = jax.random.split(rng, 4)
            proposer = int(bg.proposer_for_round(
                r, n, 1, cfg["proposer"], key=k_prop,
                contributions=jnp.asarray(cum_act)[:, None], start_offset=offset,
                holdouts=jnp.asarray(last_rejecters)[:, None])[0])

            def feats_at(live_tn, live):
                return bg.bargaining_features(
                    r, num_rounds, jnp.array([proposer]), n,
                    jnp.array([last_tn], jnp.float32), jnp.array([had_offer]),
                    jnp.array([n_reject], jnp.int32),
                    jnp.array([live_tn], jnp.float32), jnp.array([live], jnp.float32),
                    jnp.asarray(last_votes)[:, None], jnp.array([last_n_accept]),
                    jnp.asarray(cum_ret)[:, None] / ret_scale,
                    jnp.asarray(cum_act)[:, None] / act_scale,
                    jnp.array([commons], jnp.float32) / commons_scale, mask)

            # Two passes, as in training: the offer is made, and only then voted on.
            # One pass would hand the vote head an empty offer slot it never saw
            # during training, so the replay would not be this policy at all.
            kt = jax.random.split(k_theta, n)
            kv = jax.random.split(k_vote, n)
            feats_prop = feats_at(0.0, 0.0)
            raw = []
            for i in range(n):
                pi_theta, _, _ = bnet.apply(bargain_params[i], feats_prop[i])
                raw.append(float(pi_theta.sample(seed=kt[i])[0, 0]))
            theta_offer = float(neg.unsquash(
                jnp.float32(raw[proposer]), contract.low, contract.high))

            feats_vote = feats_at(
                float(bg.normalise_theta(theta_offer, contract.low, contract.high)),
                1.0)
            votes, p_accept = [], []
            for i in range(n):
                _, pi_vote, _ = bnet.apply(bargain_params[i], feats_vote[i])
                # eps=0: the exploration floor is a training device, so a replay
                # shows the policy rather than the floor.
                vote, _ = bg.floored_vote(pi_vote, 0.0, kv[i])
                votes.append(int(vote[0]))
                p_accept.append(float(pi_vote.probs[0, 1]))
            n_accept = sum(v for i, v in enumerate(votes) if i != proposer)
            passed = n_accept >= quorum

            rounds.append({
                "round": r, "step": step, "proposer": proposer,
                "theta": theta_offer, "votes": list(votes), "accepted": bool(passed),
                "n_accept": int(n_accept), "quorum": quorum,
                "p_accept": p_accept,
            })
            last_tn = float(bg.normalise_theta(theta_offer, contract.low, contract.high))
            had_offer = True
            # Only counted votes carry forward -- the proposer casts none.
            last_votes = np.array([0.0 if i == proposer else float(v)
                                   for i, v in enumerate(votes)], np.float32)
            last_n_accept = float(n_accept)
            last_rejecters = np.array([0.0 if i == proposer else 1.0 - float(v)
                                       for i, v in enumerate(votes)], np.float32)
            # A null offer never takes force in any mode: accepting it buys one
            # segment under the fallback and negotiation reopens next round.
            took_force = passed and theta_offer > contract.null + 1e-6
            if not passed:
                n_reject += 1
            if binding == "episode":
                if took_force:
                    agreed, locked = True, theta_offer
                theta = locked if agreed else float(contract.null)
            else:
                # Renegotiated: this segment plays under the new offer if it
                # carried, and otherwise under the fallback -- null under
                # `segment`, the incumbent under `sticky`. `locked` carries the
                # incumbent forward and `agreed` stays False, so the next round
                # bargains again.
                fallback = locked if binding == "sticky" else float(contract.null)
                theta = theta_offer if took_force else fallback
                locked = theta
            rounds[-1]["in_force"] = float(theta)

        contract_vec = contract.to_obs(jnp.float32(theta))[None, ...]
        for _ in range(min(seg, num_steps - step)):
            rng, k_act, k_step = jax.random.split(rng, 3)
            act_keys = jax.random.split(k_act, n)
            actions = []
            for i, a in enumerate(env.agents):
                pi, _ = net.apply(gameplay_params[i], obs[a][None, ...], contract_vec)
                actions.append(int(pi.sample(seed=act_keys[i]).squeeze()))
            obs, state, reward, done, info = env.step(k_step, state, actions)
            states.append(state)
            step += 1

            act = np.atleast_1d(np.array(info[act_key], np.float32))
            rew = np.atleast_1d(np.asarray(reward, np.float32)).reshape(-1)
            # The transfer the run was trained under. A harvest tax reads the
            # harvest and a trailing cleaning window instead of this step's cleaning
            # alone; the window is per episode and survives segment boundaries.
            # Every other kind reads its own signals out of the info dict, so which
            # act is priced stays the contract's business.
            if tax_kind:
                tax_win = bg_contracts.push_tax_window(tax_win, jnp.asarray(act))
                harvest = np.atleast_1d(
                    np.array(info["original_rewards"], np.float32))
                tr = np.array(contract.tax_transfer(
                    jnp.float32(theta), jnp.asarray(harvest), tax_win))
            else:
                tr = np.array(contract.transfer_from_info(jnp.float32(theta), info))
            act_hist.append(act)
            rewards_hist.append(rew)
            transfers.append(tr)
            cum_act = cum_act + act
            cum_ret = cum_ret + rew + tr
            commons = float(np.array(info[commons_key]).reshape(-1)[0])
            if bool(done["__all__"]):
                return states, {"transfers": transfers, "act": act_hist,
                                "reward": rewards_hist, "rounds": rounds}
        if step >= num_steps:
            break

    return states, {"transfers": transfers, "act": act_hist,
                    "reward": rewards_hist, "rounds": rounds}


def _agent_snapshot(state):
    snapshot = {}
    if hasattr(state, "agent_locs"):
        snapshot["locs"] = np.array(state.agent_locs)
    if hasattr(state, "agent_invs"):
        snapshot["invs"] = np.array(state.agent_invs)
    return snapshot


# Module-level worker globals for ProcessPoolExecutor (must be top-level to be
# picklable/importable by spawned worker processes).
_WORKER_ENV = None


def _init_render_worker(env_name, env_kwargs):
    global _WORKER_ENV
    _WORKER_ENV = socialjax.make(env_name, **env_kwargs)


def _render_worker(state):
    return np.array(_WORKER_ENV.render(state))


def recording_meta(env, env_name=None, contract_info=None):
    """Everything render_recording() needs to rasterise without the env or JAX.

    `n_items` comes off the env rather than from Clean Up's Items enum: every env
    numbers its agents as `len(Items) + i`, and the palette has to reserve exactly
    that many item codes. Harvest has 6 items to Clean Up's 10, so borrowing Clean
    Up's count paints Harvest's agent cells in river/dirt colours.
    """
    from viz.recording import BACKGROUNDS, RECORDING_VERSION, _BACKGROUND

    return {
        "version": RECORDING_VERSION,
        "n_items": int(np.asarray(env._agents).reshape(-1)[0]),
        "background": list(BACKGROUNDS.get(env_name, _BACKGROUND)),
        "act_header": getattr(env, "act_header", "Clean"),
        "theta_unit": getattr(env, "theta_unit", "cell"),
        "padding": int(env.PADDING),
        "player_colours": [list(map(int, c)) for c in env.PLAYER_COLOURS],
        "num_agents": int(env.num_agents),
        "grid_rows": int(env.GRID_SIZE_ROW),
        "grid_cols": int(env.GRID_SIZE_COL),
        "shared_rewards": bool(getattr(env, "shared_rewards", True)),
        "pay_mode": getattr(env, "pay_mode", "off"),
        "pay_scheme": getattr(env, "pay_scheme", "instant"),
        "split_recipients": bool(getattr(env, "split_recipients", False)),
        "share_fraction": float(getattr(env, "share_fraction", 0.5)),
        "pay_amount": float(getattr(env, "pay_amount", 1.0)),
        "pay_clean_window": int(getattr(env, "pay_clean_window", 50)),
        "apple_reward": float(getattr(env, "apple_reward", env.num_agents)),
        "contract_info": contract_info,
    }


class _ReplayEnv:
    """Stand-in for the env when replaying a recording.

    A recording is rendered and panelled without importing the environment at all, but
    the panel code reads a few attributes off `env`; this exposes exactly those
    from the recorded metadata so replay and live rendering share one code path.
    """

    def __init__(self, meta):
        self.num_agents = meta["num_agents"]
        self.PLAYER_COLOURS = [tuple(c) for c in meta["player_colours"]]
        self.GRID_SIZE_ROW = meta["grid_rows"]
        self.GRID_SIZE_COL = meta["grid_cols"]
        self.PADDING = meta["padding"]
        self.shared_rewards = meta["shared_rewards"]
        self.pay_mode = meta["pay_mode"]
        self.pay_scheme = meta["pay_scheme"]
        self.split_recipients = meta["split_recipients"]
        self.share_fraction = meta["share_fraction"]
        self.pay_amount = meta["pay_amount"]
        self.pay_clean_window = meta["pay_clean_window"]
        self.apple_reward = meta["apple_reward"]
        # The panel's environment vocabulary, so a replay is labelled the way the
        # live rollout was rather than in Clean Up's words by default.
        self.act_header = meta.get("act_header", "Clean")
        self.theta_unit = meta.get("theta_unit", "cell")


def render_states(env, env_name, env_kwargs, states, workers=None):
    """Render a list of states to RGB frames, parallelized across processes."""
    if workers is None:
        workers = os.cpu_count() or 1

    if workers <= 1 or len(states) < 8:
        return [np.array(env.render(s)) for s in states]

    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_render_worker, initargs=(env_name, env_kwargs)
    ) as ex:
        return list(ex.map(_render_worker, states))



# ---------------------------------------------------------------------------
# Per-agent info panel (drawn to the right of the grid).
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=16)
def _load_font(size):
    """A scalable TrueType font at `size`, falling back gracefully.

    Tries matplotlib's bundled DejaVu Sans (matplotlib is already a viewer
    dependency, so it's always importable) then a couple of common OS paths, then
    Pillow's built-in default. Cached so per-frame rendering doesn't reload the
    font file. Returns an ImageFont.
    """
    from PIL import ImageFont

    candidates = []
    try:
        from matplotlib import font_manager

        candidates.append(font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans")))
    except Exception:
        pass
    candidates += [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1 scalable default
    except TypeError:
        return ImageFont.load_default()


def _panel_supported(states, env):
    """The panel needs per-agent colours and at least one state to read."""
    return len(states) > 0 and getattr(env, "PLAYER_COLOURS", None) is not None


def collect_panel_data(states, env, transfers=None, act=None, rewards=None):
    """Per-step, per-agent stats for the info panel, aligned with `states`.

    Returns a list (one entry per state) of dicts:
        balance:   (N,) cumulative net reward. Clean Up carries this in State
                   (`agent_balance`, the spendable balance the pay mechanism moves);
                   every other environment has no such field, so it is accumulated
                   from the per-step `rewards` the rollout collected. Environment
                   reward only in both cases -- contract transfers are the next
                   column, and adding them here would double-count them.
        act:       (N,) running count of the CONTRACTED ACT -- cells cleared on
                   Clean Up, thin-patch apples eaten on Harvest. Taken from the
                   env's own per-step info when `act` is supplied. The Clean Up-only
                   fallback (last_clean_t changing) can only count cleaning STEPS,
                   and the beam covers 4 tiles, so it under-reports whenever an agent
                   clears more than one cell in a single action.
        share:     (N,) bool, share-mode toggle currently ON (tithe scheme only;
                   all-False otherwise).
        transfer:  (N,) CUMULATIVE contract transfer received (negative = net
                   funder). Only meaningful under MOCA; zeros otherwise.

    Note the contract case needs `transfers` passed in: contract transfers are
    computed by the viewer during the rollout, not stored in env State the way the
    pay mechanism's balance is.
    """
    n = env.num_agents
    act_counts = np.zeros(n, dtype=int)
    cum_transfer = np.zeros(n)
    cum_reward = np.zeros(n, dtype=np.float32)
    prev_lct = None
    out = []
    for idx, s in enumerate(states):
        if act is not None:
            if idx < len(act):
                act_counts = act_counts + np.asarray(act[idx]).reshape(-1).astype(int)
        else:
            lct = np.array(s.last_clean_t) if hasattr(s, "last_clean_t") else None
            if lct is not None and prev_lct is not None:
                act_counts = act_counts + (lct != prev_lct).astype(int)
            prev_lct = lct
        if rewards is not None and idx < len(rewards):
            cum_reward = cum_reward + np.asarray(rewards[idx], np.float32).reshape(-1)
        balance = (np.array(s.agent_balance) if hasattr(s, "agent_balance")
                   else cum_reward)
        if hasattr(s, "share_expiry_t"):
            share = np.array(s.share_expiry_t) > int(s.inner_t)
        else:
            share = np.zeros(n, dtype=bool)
        if transfers is not None and idx < len(transfers):
            cum_transfer = cum_transfer + np.asarray(transfers[idx]).reshape(-1)
        out.append({"balance": np.asarray(balance).reshape(-1).copy(),
                    "act": act_counts.copy(),
                    "share": np.asarray(share).reshape(-1),
                    "transfer": cum_transfer.copy()})
    return out


# Panel palette (dark, so the colored agent swatches pop and it reads next to the grid).
_PANEL_BG = (26, 27, 38)
_PANEL_FG = (228, 230, 240)
_PANEL_MUTED = (150, 154, 170)
_PANEL_ROW = (37, 39, 55)
_PANEL_ON = (80, 220, 130)
_PANEL_OFF = (222, 70, 74)


# Short display wording per environment: what one unit of theta is priced PER (the
# panel caption) and the column header for the act the contract prices. Both are
# derivable from the EnvSpec -- see the fallback in `act_labels` -- so a new
# environment renders sensibly without an entry here. The table exists only because
# act_label ("thin_patch_eats") is wider than the column, which collides with the
# Reward column at around eight characters.
_ACT_DISPLAY = {
    "clean_up": ("cell", "Clean"),
    "harvest_common_open": ("thin-patch eat", "Thin eat"),
    "coin_game": ("stolen coin", "Steal"),
    "coin_game_n": ("stolen coin", "Steal"),
}


def act_labels(spec):
    """(unit theta is priced per, panel column header) for a contracting env."""
    if spec is None:
        return ("unit", "Act")
    named = _ACT_DISPLAY.get(spec.env_name)
    if named is not None:
        return named
    words = spec.act_label.replace("_", " ")
    return (words, words.split()[0][:7].title())


def _scheme_caption(env, contract_info=None):
    """One-line description of the active reward + redistribution mechanism, shown in
    the panel header. Names the reward mode explicitly because the env DEFAULTS to
    shared/common reward (every agent gets each apple), which silently turns an
    individual-reward checkpoint's replay into a common-reward economy -- and names
    the mechanism so a MOCA contract, a tithe toggle and the old instant one-off
    payment can't be confused for one another.
    """
    reward = "shared-reward" if getattr(env, "shared_rewards", True) else "individual"
    if contract_info is not None:
        theta = contract_info["theta"]
        src = contract_info.get("source", "")
        # The unit is the environment's, not Clean Up's: the same number means a wage
        # per cell cleaned on one env and a fine per thin-patch apple on another.
        # Three significant figures, matching the bargaining log's rows -- under
        # renegotiation this theta is a mean over segments, and printing it to five
        # would be precision the number does not have. Where it came from is its own
        # " · " part so the caption can wrap there when the units are long.
        unit = getattr(env, "theta_unit", "cell")
        origin = f" · {src.strip('()')}" if src else ""
        return f"{reward} · contract θ={theta:.3g}/{unit}{origin}"
    if not hasattr(env, "pay_mode"):
        return reward          # no redistribution mechanism on this environment
    mode = env.pay_mode
    if mode == "off":
        return f"{reward} · no payments"
    scheme = getattr(env, "pay_scheme", "?")
    tag = "" if mode == "on" else " [placebo]"
    if scheme == "tithe":
        who = "→all cleaners" if getattr(env, "split_recipients", False) else "→latest cleaner"
        return f"{reward} · tithe {getattr(env, 'share_fraction', 0.5):g} {who}{tag}"
    return f"{reward} · instant pays {getattr(env, 'pay_amount', 1.0):g}{tag}"


def _wrap_caption(draw, text, font, max_width):
    """Split `text` on its " · " separators into lines that fit `max_width`.

    The caption names the reward mode and the mechanism, and the mechanism's units
    are the environment's -- "per thin-patch eat" is half again as wide as "per
    cell". Truncating would drop the units, which is the part that says what the
    number means, so it wraps instead.
    """
    parts = text.split(" · ")
    lines, current = [], ""
    for part in parts:
        trial = f"{current} · {part}" if current else part
        if current and draw.textlength(trial, font=font) > max_width:
            lines.append(current)
            current = part
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def _render_bargain_log(draw, x0, y0, x1, y1, rounds, colors, width, n):
    """The bargaining log: one row per round that actually had a decision.

    Reads left to right as the round played out -- who proposed, what they asked
    for, how each agent voted, and whether it carried. One vote dot per agent, in
    agent order so the column lines up with the table above: green accept, red
    reject, and hollow for the proposer, which never votes on its own offer.
    """
    from PIL import ImageDraw  # noqa: F401  (draw is already an ImageDraw)

    head_f = _load_font(max(11, int(width * 0.04)))
    cell_f = _load_font(max(12, int(width * 0.044)))
    pad = max(14, int(width * 0.045))

    draw.line([x0 - 4, y0, x1 + 4, y0], fill=_PANEL_ROW, width=2)
    y = y0 + int(head_f.size * 0.8)
    draw.text((x0, y), "Bargaining", font=_load_font(max(13, int(width * 0.052))),
              fill=_PANEL_FG)
    if not rounds:
        draw.text((x1, y + 2), "no offer yet", font=head_f, fill=_PANEL_MUTED,
                  anchor="ra")
        return
    # Under renegotiation there is no single settlement to report, so the header
    # becomes how much of the episode so far is governed, and at what.
    if any("in_force" in r for r in rounds) and len(
            {r.get("in_force") for r in rounds}) > 1:
        live = [r["in_force"] for r in rounds if r.get("in_force", 0.0) > 1e-6]
        status = (f"{len(live)}/{len(rounds)} segs at θ~"
                  f"{sum(live) / len(live):.3f}" if live
                  else f"{len(rounds)} segs uncontracted")
        ok = bool(live)
    else:
        settled = next((r for r in rounds if r["accepted"]), None)
        status = (f"agreed R{settled['round']} at θ={settled['theta']:.3f}"
                  if settled else f"{len(rounds)} rejected")
        ok = settled is not None
    draw.text((x1, y + 2), status, font=head_f,
              fill=_PANEL_ON if ok else _PANEL_OFF, anchor="ra")

    y += int(head_f.size * 2.0)
    avail = y1 - y
    # Floored at the text height. Dividing the space by the round count alone sends
    # the row height to a couple of pixels at short segment lengths (seg25 over 1000
    # steps is 40 rounds), which does not fit more rows -- it draws them on top of
    # each other. Past the floor the log keeps the MOST RECENT rounds instead, since
    # under renegotiation the live bargain is the one worth reading; the earlier ones
    # are counted in a line above.
    row_h = max(int(cell_f.size * 1.35),
                min(int(cell_f.size * 1.8),
                    max(1, int(avail / max(len(rounds), 1)))))
    dot = max(6, int(row_h * 0.42))
    # Vote dots sit in a fixed-width strip on the right so rows stay aligned.
    strip_w = (dot + 3) * n
    x_dots = x1 - strip_w - int(width * 0.09)

    n_fit = max(1, int(avail // row_h))
    if len(rounds) > n_fit:
        # One row goes to the "N earlier" marker, so the rest can show.
        dropped = len(rounds) - (n_fit - 1)
        draw.text((x0, y), f"+{dropped} earlier round"
                  + ("s" if dropped != 1 else ""),
                  font=head_f, fill=_PANEL_MUTED)
        y += row_h
        rounds = rounds[dropped:]

    for k, rec in enumerate(rounds):
        ry = y + row_h * k
        if ry + row_h > y1:
            draw.text((x0, ry), f"+{len(rounds) - k} more", font=head_f,
                      fill=_PANEL_MUTED)
            break
        cy = ry + row_h / 2
        ok = rec["accepted"]
        draw.rounded_rectangle([x0 - 4, ry + 1, x1 + 4, ry + row_h - 2],
                               radius=5, fill=_PANEL_ROW)
        p = rec["proposer"]
        draw.text((x0 + 2, cy), f"R{rec['round']}", font=head_f,
                  fill=_PANEL_MUTED, anchor="lm")
        sx = x0 + int(width * 0.075)
        if p < 0:
            # Median round: nobody proposed and nobody voted, so the row shows
            # the mechanism instead -- the label, the spread of asks it took the
            # middle of, and the theta that middle turned out to be.
            draw.text((sx, cy), "median", font=head_f, fill=_PANEL_FG, anchor="lm")
            asks = rec.get("asks") or ()
            if asks:
                draw.text((x_dots + strip_w, cy),
                          f"asks {min(asks):.2f}–{max(asks):.2f}",
                          font=head_f, fill=_PANEL_MUTED, anchor="rm")
        else:
            pcol = (tuple(int(c) for c in colors[p]) if p < len(colors)
                    else (200, 200, 200))
            # Proposer swatch + id, so "who asked" is readable without counting dots.
            draw.rounded_rectangle([sx, cy - dot / 2, sx + dot, cy + dot / 2],
                                   radius=3, fill=pcol, outline=(15, 16, 24))
            draw.text((sx + dot + 4, cy), f"A{p}", font=head_f, fill=_PANEL_FG,
                      anchor="lm")
        draw.text((x_dots - int(width * 0.02), cy), f"θ={rec['theta']:.3f}",
                  font=cell_f, fill=_PANEL_FG, anchor="rm")

        if p >= 0:
            for i in range(n):
                dx = x_dots + i * (dot + 3)
                box = [dx, cy - dot / 2, dx + dot, cy + dot / 2]
                if i == p:
                    draw.ellipse(box, outline=_PANEL_MUTED, width=1)  # proposer: no vote
                else:
                    draw.ellipse(box, fill=_PANEL_ON if rec["votes"][i] else _PANEL_OFF)
        draw.text((x1, cy), "✓" if ok else "✗", font=cell_f,
                  fill=_PANEL_ON if ok else _PANEL_OFF, anchor="rm")


def render_info_panel(height, datum, colors, env, step, width, contract_info=None,
                      bargain_rounds=None, bargain_slots=0):
    """Render one info-panel frame (RGB, height x width) for a single step.

    First column is each agent's color (a swatch + a color bar down the row edge)
    so every stat reads back to the matching agent in the grid.
    Columns: color/label | Reward (balance) | the contracted act's running count,
    headed with that environment's name for it | and then either Share (ON/OFF) for
    the tithe toggle, or Transfer (cumulative net contract transfer, + receiver /
    - funder) under a MOCA contract.
    """
    from PIL import ImageDraw

    show_contract = contract_info is not None
    show_share = (not show_contract
                  and getattr(env, "pay_scheme", None) == "tithe"
                  and getattr(env, "pay_mode", "off") != "off")
    n = env.num_agents

    img = Image.new("RGB", (width, height), _PANEL_BG)
    draw = ImageDraw.Draw(img)

    pad = max(14, int(width * 0.045))
    title_f = _load_font(max(15, int(width * 0.062)))
    head_f = _load_font(max(11, int(width * 0.04)))
    cell_f = _load_font(max(13, int(width * 0.048)))

    # Header: title + step, the active-mechanism caption, then a totals line.
    draw.text((pad, pad), "Agents", font=title_f, fill=_PANEL_FG)
    draw.text((width - pad, pad + 2), f"step {step}", font=head_f, fill=_PANEL_MUTED, anchor="ra")
    y = pad + int(title_f.size * 1.5)
    caption = _wrap_caption(draw, _scheme_caption(env, contract_info), head_f,
                            width - 2 * pad)
    line_h = int(head_f.size * 1.35)
    for i, line in enumerate(caption):
        draw.text((pad, y + i * line_h), line, font=head_f, fill=_PANEL_MUTED)
    y += (len(caption) - 1) * line_h
    act_head = getattr(env, "act_header", "Clean")
    total_reward = float(np.sum(datum["balance"]))
    total_act = int(np.sum(datum["act"]))
    y += int(head_f.size * 1.5)
    totals = (f"total reward {total_reward:.1f}    "
              f"total {act_head.lower()} {total_act}")
    if show_contract:
        # Under a zero-sum contract the transfers must cancel; showing the moved
        # volume instead makes "is the contract doing anything" readable at a glance.
        moved = float(np.sum(np.maximum(datum["transfer"], 0.0)))
        totals += f"    moved {moved:.2f}"
    draw.text((pad, y), totals, font=head_f, fill=_PANEL_MUTED)

    # Column x anchors (right-aligned value columns). The rightmost column (Share's
    # dot + ON/OFF, or the signed Transfer figure) needs the widest slot; leave a
    # clear gap between it and the act count so they never run together.
    x_last = width - pad                                             # rightmost
    has_last = show_share or show_contract
    x_act = (x_last - int(width * 0.26)) if has_last else (width - pad)
    x_reward = x_act - int(width * 0.20)

    # Column header row.
    y_head = y + int(head_f.size * 1.9)
    draw.text((x_reward, y_head), "Reward", font=head_f, fill=_PANEL_MUTED, anchor="ra")
    draw.text((x_act, y_head), act_head, font=head_f, fill=_PANEL_MUTED, anchor="ra")
    if show_share:
        draw.text((x_last, y_head), "Share", font=head_f, fill=_PANEL_MUTED, anchor="ra")
    elif show_contract:
        draw.text((x_last, y_head), "Transfer", font=head_f, fill=_PANEL_MUTED, anchor="ra")
    x_share = x_last

    # Rows. Under bargaining the agent table gives up the lower part of the panel to
    # the round log. Sized from the TOTAL number of rounds rather than the ones
    # revealed so far, so the table does not jump as the scrubber moves.
    top = y_head + int(head_f.size * 1.5)
    bottom = height - pad
    if bargain_rounds is not None:
        need = 0.20 + 0.05 * max(bargain_slots, 1)
        bottom = top + int((height - pad - top) * (1.0 - min(need, 0.55)))
    avail = bottom - top
    row_h = avail / max(n, 1)
    swatch = min(int(row_h * 0.5), int(width * 0.09))

    for i in range(n):
        ry = top + row_h * i
        cy = ry + row_h / 2
        color = tuple(int(c) for c in colors[i]) if i < len(colors) else (200, 200, 200)

        # Row background + left color bar (the "color first column").
        draw.rounded_rectangle([pad - 4, ry + 2, width - pad + 4, ry + row_h - 2],
                               radius=6, fill=_PANEL_ROW)
        draw.rounded_rectangle([pad - 4, ry + 2, pad + 2, ry + row_h - 2], radius=3, fill=color)

        # Color swatch + agent label.
        sx = pad + int(width * 0.02)
        draw.rounded_rectangle([sx, cy - swatch / 2, sx + swatch, cy + swatch / 2],
                               radius=4, fill=color, outline=(15, 16, 24))
        draw.text((sx + swatch + int(width * 0.03), cy), f"A{i}", font=cell_f,
                  fill=_PANEL_FG, anchor="lm")

        # Reward + contracted-act values.
        draw.text((x_reward, cy), f"{float(datum['balance'][i]):.1f}", font=cell_f,
                  fill=_PANEL_FG, anchor="rm")
        draw.text((x_act, cy), f"{int(datum['act'][i])}", font=cell_f,
                  fill=_PANEL_FG, anchor="rm")

        # Share toggle indicator: filled green square = ON, filled red square = OFF,
        # so an inactive pledge is as obvious as an active one.
        if show_share:
            on = bool(datum["share"][i])
            fill = _PANEL_ON if on else _PANEL_OFF
            s = max(10, int(row_h * 0.32))
            bx = x_share - s
            draw.rounded_rectangle([bx, cy - s / 2, bx + s, cy + s / 2],
                                   radius=4, fill=fill, outline=(15, 16, 24))
            draw.text((bx - int(width * 0.015), cy), "ON" if on else "OFF",
                      font=head_f, fill=fill, anchor="rm")

        # Cumulative contract transfer: green = net receiver, red = net funder.
        # Which way is "good" depends on the space -- Clean Up SUBSIDISES a benefit,
        # so receivers are the cleaners; Harvest FINES a harm, so receivers are the
        # agents who stayed out of thin patches. Either way this column is what says
        # whether the contract actually moved anything.
        elif show_contract:
            t = float(datum["transfer"][i])
            if t > 1e-9:
                fill, txt = _PANEL_ON, f"+{t:.2f}"
            elif t < -1e-9:
                fill, txt = _PANEL_OFF, f"{t:.2f}"
            else:
                fill, txt = _PANEL_MUTED, "0.00"
            draw.text((x_share, cy), txt, font=cell_f, fill=fill, anchor="rm")

    if bargain_rounds is not None:
        _render_bargain_log(draw, pad, bottom + int(pad * 0.6), width - pad,
                            height - pad, bargain_rounds, colors, width, n)

    return np.array(img)


def add_info_panels(frames, panel_data, colors, env, width=None, contract_info=None,
                    bargain_rounds=None):
    """Composite a per-step info panel onto the right of each grid frame.

    Returns new (wider) frames of uniform size so both the GIF export and the
    interactive scrubber show the panel. Panel width defaults to ~62% of the grid
    height, clamped to a sensible minimum.
    """
    h = frames[0].shape[0]
    if width is None:
        width = max(360, int(h * 0.75))
    out = []
    for i, frame in enumerate(frames):
        # Reveal rounds as the scrubber reaches them, so the log reads as history
        # accumulating rather than spoiling the outcome from frame 0.
        so_far = (None if bargain_rounds is None
                  else [r for r in bargain_rounds if r["step"] <= i])
        panel = render_info_panel(h, panel_data[i], colors, env, i, width,
                                  contract_info=contract_info,
                                  bargain_rounds=so_far,
                                  bargain_slots=len(bargain_rounds or ()))
        out.append(np.hstack([frame, panel]))
    return out


def save_gif(frames, path, duration=200):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(
        path, format="GIF", save_all=True, append_images=imgs[1:],
        duration=duration, loop=0, optimize=False,
    )


def interactive_view(frames, traces, interval_ms=150):
    import matplotlib
    import matplotlib.backend_bases
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button

    # The native toolbar's Home/Back/Forward/Pan/Zoom don't do anything useful for a
    # fixed-size rendered frame, but on some backends (e.g. macosx) the toolbar is a
    # fixed-arity native widget that can't be filtered down to just "Save" without
    # crashing -- so drop it entirely and provide our own Save button below instead.
    plt.rcParams["toolbar"] = "None"

    state = {"playing": False, "loop": True, "interval": interval_ms}

    # Frames may now be wide (grid + info panel), so pick a figure aspect that
    # matches the frame instead of a fixed portrait size.
    fh, fw = frames[0].shape[:2]
    fig_w = 9.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * fh / fw + 1.4))
    plt.subplots_adjust(bottom=0.22)
    im = ax.imshow(frames[0])
    ax.axis("off")
    ax.set_title(_frame_title(0, traces), fontsize=9)

    ax_slider = plt.axes([0.15, 0.14, 0.7, 0.03])
    slider = Slider(ax_slider, "Step", 0, len(frames) - 1, valinit=0, valstep=1)

    def show_frame(t):
        im.set_data(frames[t])
        ax.set_title(_frame_title(t, traces), fontsize=9)
        fig.canvas.draw_idle()

    def on_slider_changed(val):
        show_frame(int(slider.val))

    slider.on_changed(on_slider_changed)

    # ONE persistent, repeating timer started exactly once. On the macOS backend
    # each call to timer.start() spawns a fresh native CFRunLoop timer without
    # cancelling the previous one, so repeatedly starting/restarting a timer (the
    # previous single-shot approach) stacks N timers -- which is what caused the
    # "wait, then jump several frames" bursts and made pause (a single stop()) fail.
    #
    # Instead the timer fires at a fixed fast rate and we advance frames using a
    # wall-clock accumulator. Speed changes only mutate state["interval"]; play/pause
    # only flips state["playing"]. We never call start()/stop() again.
    CLOCK_MS = 30
    last_advance = {"t": time.monotonic()}

    def tick(_=None):
        if not state["playing"]:
            return
        now = time.monotonic()
        if (now - last_advance["t"]) * 1000.0 < state["interval"]:
            return
        last_advance["t"] = now
        t = int(slider.val) + 1
        if t > len(frames) - 1:
            if state["loop"]:
                t = 0
            else:
                state["playing"] = False
                return
        slider.set_val(t)  # triggers on_slider_changed -> show_frame

    timer = fig.canvas.new_timer(interval=CLOCK_MS)
    timer.add_callback(tick)
    timer.start()

    def start_playing(_event=None):
        state["playing"] = True
        last_advance["t"] = time.monotonic()

    def stop_playing(_event=None):
        state["playing"] = False

    def step_frame(delta):
        # Pause and nudge exactly one frame; clamp at the ends (don't wrap, so
        # stepping is predictable for frame-by-frame inspection).
        state["playing"] = False
        t = max(0, min(len(frames) - 1, int(slider.val) + delta))
        slider.set_val(t)  # triggers on_slider_changed -> show_frame

    def set_speed(factor):
        # lower bound = CLOCK_MS: the frame clock can't advance faster than it ticks,
        # so don't let the label promise an fps we can't actually deliver.
        state["interval"] = max(CLOCK_MS, min(2000, int(state["interval"] * factor)))
        speed_label.set_text(f"{1000 / state['interval']:.1f} fps")
        fig.canvas.draw_idle()

    def toggle_loop(_event):
        state["loop"] = not state["loop"]
        loop_button.ax.set_facecolor("honeydew" if state["loop"] else "0.85")
        fig.canvas.draw_idle()

    def save_frame(_event):
        try:
            out_dir = Path("viz/out").resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            t = int(slider.val)
            path = out_dir / f"frame_{t:05d}.png"
            Image.fromarray(frames[t]).save(path)
            print(f"Saved frame {t} to {path}", flush=True)
            ax.set_title(f"{_frame_title(t, traces)}  [saved -> {path.name}]", fontsize=9)
            fig.canvas.draw_idle()
        except Exception as exc:  # surface errors instead of letting the GUI swallow them
            print(f"Save failed: {exc}", flush=True)

    button_specs = [
        ("◀|", lambda e: step_frame(-1)),      # step one frame back
        ("|▶", lambda e: step_frame(1)),       # step one frame forward
        ("▶", start_playing),    # play
        ("||", stop_playing),         # pause
        ("◀◀", lambda e: set_speed(1.5)),      # slower
        ("▶▶", lambda e: set_speed(1 / 1.5)),  # faster
        ("↻", toggle_loop),      # loop
        ("Save", save_frame),
    ]
    n = len(button_specs)
    width, gap = 0.105, 0.012
    total = n * width + (n - 1) * gap
    x0 = (1 - total) / 2
    buttons = []
    for i, (label, callback) in enumerate(button_specs):
        ax_btn = plt.axes([x0 + i * (width + gap), 0.05, width, 0.05])
        btn = Button(ax_btn, label)
        btn.on_clicked(callback)
        buttons.append(btn)
    loop_button = buttons[6]
    loop_button.ax.set_facecolor("honeydew")

    speed_label = fig.text(0.5, 0.115, f"{1000 / state['interval']:.1f} fps", fontsize=8, ha="center")

    plt.show()


def _frame_title(t, traces):
    # Per-agent detail now lives in the side panel, so the title stays minimal.
    return f"step {t}"


def _parse_env_kwarg_value(raw):
    """Parse an --env-kwarg value: bool/int/float where unambiguous, else string."""
    low = raw.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass
    return raw


# Suffixes of the auxiliary policies a MOCA run saves next to its gameplay policies.
# A plain `..._reward_individual*.pkl` glob matches these too, so they must be
# filtered out before the remaining files are zipped onto agents 0..N-1.
_AUX_CKPT_MARKERS = (
    "_proposal_", "_voting_",   # PHASE2_MODE=reinforce
    "_contract_",               # PHASE2_MODE=negotiate
    "_negotiate_negotiate_",    # legacy name for the same, before it was disambiguated
    "_resume",
)


def _gameplay_checkpoints(arg):
    """Files from `arg` that are GAMEPLAY policies (one per agent), sorted.

    MOCA writes three checkpoints per agent -- gameplay, proposal and voting -- into
    one directory, all sharing the run stem, so the obvious glob returns 3N files.
    Loading those as if they were N per-agent gameplay policies is the failure this
    guards against.
    """
    import glob

    matches = sorted(glob.glob(arg))
    return [m for m in matches if not any(t in os.path.basename(m) for t in _AUX_CKPT_MARKERS)]


def _load_checkpoint(arg):
    """Load --checkpoint: single .pkl -> shared params; glob with N matches -> per-agent list."""
    import glob

    if not sorted(glob.glob(arg)):
        raise SystemExit(f"no checkpoint file matches {arg!r}")
    matches = _gameplay_checkpoints(arg)
    if not matches:
        raise SystemExit(
            f"{arg!r} matched only auxiliary checkpoints "
            f"({'/'.join(_AUX_CKPT_MARKERS)}) and no gameplay policies"
        )
    if len(matches) == 1:
        return load_params(matches[0])
    print(f"Loading {len(matches)} per-agent checkpoints (sorted -> agent 0..{len(matches) - 1})")
    return [load_params(m) for m in matches]


# ---------------------------------------------------------------------------
# MOCA (formal contracting) support.
# ---------------------------------------------------------------------------

def detect_moca(checkpoint_arg):
    """Detect a MOCA run and locate its contracting policies, whichever phase 2 ran.

    The three PHASE2_MODEs leave different things on disk, so detection is by which
    sibling checkpoints exist next to the gameplay policies:

        reinforce  `_proposal_<i>.pkl` + `_voting_<i>.pkl` -- a categorical over a
                   contract grid, readable straight off the weights.
        negotiate  `_contract_<i>.pkl` -- one Box policy per agent emitting
                   [theta, accept_prob]. Continuous, and a function of the initial
                   observation, so recovering theta needs a forward pass.
        solver     nothing at all: phase 2 learns no policy, it searches the frozen
                   critics at reset. Recognised from the `_solver` stem token.

    Returns None for non-MOCA runs, else a dict with "mode", "gameplay", and the
    mode's policy paths.
    """
    import glob

    gameplay = _gameplay_checkpoints(checkpoint_arg)
    if not gameplay:
        return None
    # Contracting policies sit next to the gameplay ones, sharing the run stem.
    stem = re.sub(r"_\d+\.pkl$", "", gameplay[0])

    proposals = sorted(glob.glob(f"{stem}_proposal_*.pkl"))
    if proposals:
        return {
            "mode": "reinforce",
            "proposal_paths": proposals,
            "voting_paths": sorted(glob.glob(f"{stem}_voting_*.pkl")),
            "gameplay": gameplay,
        }

    base = os.path.basename(stem)

    # Rubinstein bargaining writes its policies under the SAME "_contract_" role
    # suffix as the one-shot negotiation stage, so the stem token has to be checked
    # first -- otherwise a bargaining run is read as a negotiate run and its weights
    # are loaded into the wrong network. Checked before the glob for that reason.
    if "_bargain" in base:
        contracts = sorted(glob.glob(f"{stem}_contract_*.pkl"))
        if contracts:
            return {"mode": "bargain", "contract_paths": contracts,
                    "gameplay": gameplay, "stem": base}
        return {"mode": "phase1", "gameplay": gameplay}

    contracts = sorted(glob.glob(f"{stem}_contract_*.pkl"))
    if not contracts:  # runs written before the role suffix was disambiguated
        contracts = sorted(glob.glob(f"{stem}_negotiate_*.pkl"))
    if contracts:
        return {"mode": "negotiate", "contract_paths": contracts, "gameplay": gameplay}

    # Solver runs save no contracting policy, so the stem is the only evidence.
    if "_solver" in base:
        return {"mode": "solver", "gameplay": gameplay}
    # PHASE1_ONLY: checkpoint_filename always marks a PHASE2_MODE, so the stem still
    # carries the token even though phase 2 never ran and no contracting policy was
    # ever written. The gameplay policy is contract-conditioned regardless, so it has
    # to be replayed through the contract path -- its weights will not even load into
    # the plain ActorCritic. There is no learned theta to recover, so the caller must
    # supply one (--contract-theta 0 for the null contract).
    if "_negotiate" in base or "_reinforce" in base:
        return {"mode": "phase1", "gameplay": gameplay}
    return None


def load_contract_policies(moca, low, high):
    """Read the learned proposal distribution of a PHASE2_MODE=reinforce run."""
    logits = np.stack([
        np.asarray(load_params(p)["params"]["proposal_logits"]) for p in moca["proposal_paths"]
    ])                                                     # (N, K)
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = probs / probs.sum(axis=-1, keepdims=True)
    k = logits.shape[-1]
    theta_grid = np.linspace(low, high, k)
    modal_theta = float(theta_grid[int(probs.mean(axis=0).argmax())])
    return {"theta_grid": theta_grid, "probs": probs, "modal_theta": modal_theta}


def _initial_obs_batch(env, seed):
    """(N, ...) observations at s_0 -- the state both phase-2 modes negotiate from."""
    obs, _ = env.reset(jax.random.PRNGKey(seed))
    return jnp.stack([obs[a] for a in env.agents])


def solve_negotiate_theta(moca, contract, env, seed):
    """Agent 0's proposal, and what the others would sign, from a negotiate run.

    Unlike the reinforce mode's logits, the proposal is a Gaussian conditioned on the
    initial observation, so it has to be evaluated rather than read. The MEAN is used
    rather than a sample: it is the policy's actual choice, not one draw from its
    exploration noise.
    """
    from algorithms.MOCA import negotiate as neg
    from algorithms.MOCA.contracts import AGREE, PROPOSE
    from algorithms.MOCA.networks import NegotiationActorCritic

    params = [load_params(p) for p in moca["contract_paths"]]
    net = NegotiationActorCritic(2, activation="relu")
    obs = _initial_obs_batch(env, seed)
    n = len(params)

    # Agent 0 always proposes -- the paper's single-proposer assumption.
    pi0, _ = net.apply(params[0], obs[0][None, ...],
                       contract.to_obs(jnp.zeros((1,)), stage=PROPOSE))
    theta = float(neg.unsquash(pi0.mean()[0, 0], contract.low, contract.high))

    # What every other agent would offer as its accept probability for that theta.
    agree_obs = contract.to_obs(jnp.full((1,), theta), stage=AGREE)
    accept = []
    for i in range(1, n):
        pi, _ = net.apply(params[i], obs[i][None, ...], agree_obs)
        accept.append(float(neg.unsquash(pi.mean()[0, 1], 0.0, 1.0)))
    return {"theta": theta, "accept_probs": np.array(accept), "nu": neg.default_nu(n)}


def solve_solver_theta(params, contract, env, seed, num_samples, rule):
    """Re-run the sampling solver from the frozen critics, as phase 2 did each episode."""
    from algorithms.MOCA import solver as moca_solver
    from algorithms.MOCA.networks import ContractActorCritic

    nets = [ContractActorCritic(action_dim=env.action_space().n, activation="relu")
            for _ in range(env.num_agents)]
    obs = _initial_obs_batch(env, seed)[:, None, ...]      # (N, 1, ...) one "env"
    theta, info = moca_solver.negotiate(
        jax.random.PRNGKey(seed), nets, params, obs, contract, num_samples, rule
    )
    return {"theta": float(theta[0]),
            "null": bool(np.asarray(info["solver_null_rate"]) > 0.5)}


def _infer_env_kwargs_from_checkpoint(checkpoint_arg, env_name="clean_up", spec=None):
    """Best-effort recovery of the env config a checkpoint was TRAINED with, by parsing
    the filename the training loop wrote (algorithms/utils/io_utils.checkpoint_filename).

    That naming scheme OMITS any value left at its default, so an absent token means
    "the env default": no `_pay_*` -> pay_mode off; `_pay_on`/`_pay_noop` without
    `_tithe` -> instant scheme; no `_f`/`_d`/`_win` -> default fraction/duration/window.
    We only trust this when the name is clearly from that scheme (it carries a
    `reward_individual|common` or `_agents` token); for arbitrary/renamed files we
    return {} and let the env defaults + the mismatch warning handle it. Reliability
    caveat: this reads the filename, not the weights, so a mislabeled file misleads it
    -- hence it's overridable by explicit --env-kwarg and printed for inspection.

    `env_name` gates the kwargs that only one environment HAS. The pay mechanism is
    Clean Up's alone, so emitting `pay_mode` for a Harvest run is not a wrong value
    but an unknown keyword argument, and the env constructor raises. `spec` names the
    reward-scale kwarg for the same reason -- it is `apple_reward` on two of the three
    environments and `coin_reward` on the other.

    Returns a dict of env kwargs (possibly empty).
    """
    import glob
    import re

    has_pay_mechanism = env_name == "clean_up"

    matches = sorted(glob.glob(checkpoint_arg))
    name = os.path.basename(matches[0] if matches else checkpoint_arg).lower()

    recognized = ("reward_individual" in name or "reward_common" in name
                  or re.search(r"_agents\d+", name) is not None)
    if not recognized:
        return {}

    kw = {}
    # Reward mode: individual -> shared_rewards=False (differs from the env default!),
    # common -> True. This is the token that most often needs fixing.
    if "reward_individual" in name:
        kw["shared_rewards"] = False
    elif "reward_common" in name:
        kw["shared_rewards"] = True

    # Pay mode: token present for on/noop, absent means off.
    if has_pay_mechanism:
        if "pay_noop" in name:
            kw["pay_mode"] = "noop"
        elif "pay_on" in name:
            kw["pay_mode"] = "on"
        else:
            kw["pay_mode"] = "off"

    # Scheme + its off-default knobs only matter when pay is active.
    if kw.get("pay_mode") in ("on", "noop"):
        kw["pay_scheme"] = "tithe" if "_tithe" in name else "instant"
        if kw["pay_scheme"] == "tithe":
            m = re.search(r"_f([0-9]*\.?[0-9]+)", name)
            if m:
                kw["share_fraction"] = float(m.group(1))
            m = re.search(r"_d(\d+)", name)
            if m:
                kw["share_duration"] = int(m.group(1))
            if "_split" in name:
                kw["split_recipients"] = True
        m = re.search(r"_win(\d+)", name)
        if m:
            kw["pay_clean_window"] = int(m.group(1))

    m = re.search(r"_agents(\d+)", name)
    if m:
        kw["num_agents"] = int(m.group(1))

    # MOCA runs are detected from the sibling contracting checkpoints rather than the
    # filename, which carries no contract tokens. They train with a UNIT apple
    # (apple_reward=1.0, the scale the contracting literature calibrates theta to)
    # and no pay action, neither of which is recoverable from the name -- replaying
    # them at the env's default num_agents-valued apple would silently rescale the
    # whole economy relative to the contract.
    if detect_moca(checkpoint_arg) is not None:
        kw[spec.reward_scale_kwarg if spec is not None else "apple_reward"] = 1.0
        if has_pay_mechanism:
            kw["pay_mode"] = "off"
    return kw


def _report_env_config(env, env_name, checkpoint_arg):
    """Print the reward/pay config the env was actually built with, and warn loudly if
    it contradicts what the checkpoint filename says it was trained with.

    Motivation: the env DEFAULTS to shared/common reward, so a viewer command that
    forgets `--env-kwarg shared_rewards=False` silently replays an individual-reward
    checkpoint inside a common-reward economy (every apple credits all agents) -- which
    looks like a payment bug but is just a mode mismatch. Filenames written by the
    training loop encode reward_{individual,common} and pay_{off,on,noop}[_scheme].
    """
    shared = getattr(env, "shared_rewards", True)
    pay_mode = getattr(env, "pay_mode", "off")
    pay_scheme = getattr(env, "pay_scheme", "?")
    split = getattr(env, "split_recipients", False)
    # The pay mechanism exists on Clean Up only, so on the other environments it is
    # absent rather than off, and reporting "pay_mode=off" there would read as a
    # setting somebody chose.
    pay = ""
    if hasattr(env, "pay_mode"):
        pay = f"pay_mode={pay_mode}, pay_scheme={pay_scheme}"
        if pay_mode != "off" and pay_scheme == "tithe":
            pay += (f", recipients={'split-all' if split else 'latest'}, "
                    f"clean_window={getattr(env, 'pay_clean_window', '?')}")
        pay += ", "
    scale = ("coin_reward" if env_name == "coin_game" else "apple_reward")
    print(
        f"Env config: {env_name}, "
        f"reward={'shared/common' if shared else 'individual'}, {pay}"
        f"{scale}={getattr(env, scale, '?')}, num_agents={env.num_agents}"
    )
    if not checkpoint_arg:
        return
    # checkpoint_arg may be a glob (e.g. ..._reward_individual*.pkl) that truncates
    # before the pay-scheme tokens, so resolve it to a real matched filename first.
    import glob

    matches = sorted(glob.glob(checkpoint_arg))
    name = os.path.basename(matches[0] if matches else checkpoint_arg).lower()
    warnings = []
    if "reward_individual" in name and shared:
        warnings.append("checkpoint says reward_individual but env is shared/common reward "
                        "-> add `--env-kwarg shared_rewards=False`")
    if "reward_common" in name and not shared:
        warnings.append("checkpoint says reward_common but env is individual reward "
                        "-> add `--env-kwarg shared_rewards=True`")
    if "_tithe" in name and pay_scheme != "tithe":
        warnings.append("checkpoint trained with the tithe scheme but env pay_scheme="
                        f"{pay_scheme!r} -> add `--env-kwarg pay_scheme=tithe`")
    if "pay_on" in name and pay_mode != "on":
        warnings.append(f"checkpoint trained with pay_mode=on but env pay_mode={pay_mode!r} "
                        "-> add `--env-kwarg pay_mode=on`")
    if "_split" in name and not split:
        warnings.append("checkpoint trained with split recipients but env uses the single "
                        "most-recent cleaner -> add `--env-kwarg split_recipients=True`")
    for w in warnings:
        print(f"  ⚠️  WARNING: {w}")


def _warn_if_fixed_agent_count(env, env_name, requested_num_agents):
    """Agents spawn onto 'P' tiles baked into each env's map, so you can't have more
    agents than spawn tiles. Asking for too many otherwise fails deep inside the env
    with a cryptic JAX shape-concatenation error, so catch it early with a hint.

    (This only guards the too-many case, which is unambiguous. coin_game additionally
    uses *all* its spawns, so it's effectively 2-player only; asking for exactly 2 is
    fine, and asking for 3+ is caught here.)
    """
    if requested_num_agents is None:
        return
    spawns = getattr(env, "SPAWNS_PLAYERS", None)
    if spawns is None:
        return
    n_spawn = int(spawns.shape[0])
    if requested_num_agents > n_spawn:
        raise SystemExit(
            f"'{env_name}' has only {n_spawn} player spawn point(s) in its map, "
            f"but --num_agents {requested_num_agents} was given. Re-run with "
            f"--num_agents <= {n_spawn} (or omit --num_agents to use the env default)."
        )


def check_action_space(params, env, env_name):
    """Refuse a checkpoint whose action head does not match the env's action space.

    The failure this replaces is a flax ScopeParamShapeError naming "/Dense_1" and a
    pair of kernel shapes, several frames from the config that caused it. The usual
    cause on Harvest is the zap beam: `enable_zap` now defaults to False, so a policy
    trained upstream (8 actions, beam included) meets a 7-action environment.
    """
    one = params[0] if isinstance(params, list) else params
    if one is None:
        return
    try:
        # The actor head is the last Dense before the Categorical; its output width
        # IS the action count the policy was trained on.
        trained = int(one["params"]["Dense_1"]["kernel"].shape[-1])
    except (KeyError, TypeError, IndexError):
        return          # not a shape we can read; let flax report it as before
    have = int(env.action_space().n)
    if trained == have:
        return
    hint = ""
    if env_name == "harvest_common_open" and trained == have + 1:
        hint = ("\n  This is the zap beam: it is off by default now, so an upstream "
                "checkpoint\n  has one action more than the environment offers. Replay "
                "it with\n  `--env-kwarg enable_zap=True`.")
    raise SystemExit(
        f"[incompatible checkpoint] this policy was trained with {trained} actions, "
        f"but {env_name!r} as configured offers {have}.{hint}")


def resolve_env_name(arg_env, checkpoint):
    """Which environment to build, reconciling --env with the run's own record.

    A contracting checkpoint is only replayable on the environment it was trained on
    -- the observation shapes differ, so a mismatch usually dies inside flax with a
    kernel-shape error naming neither environment. The sidecar knows which one it
    was, so it both supplies the default and turns the mismatch into a sentence.
    """
    recorded = (load_run_config(checkpoint) or {}).get("ENV_NAME") if checkpoint else None
    if arg_env is None:
        if recorded is None:
            raise SystemExit(
                "--env is required: this run has no .run.yaml sidecar to read the "
                "environment from (pass e.g. --env harvest_common_open)")
        print(f"Env from the run's sidecar: {recorded}")
        return recorded
    if recorded is not None and recorded != arg_env:
        raise SystemExit(
            f"--env {arg_env!r} but the run's .run.yaml sidecar says this checkpoint "
            f"was trained on {recorded!r}. Replaying it on another environment is not "
            f"the same policy; drop --env to use the recorded one.")
    return arg_env


def spec_for_env(env_name):
    """The contracting EnvSpec for `env_name`, or None if contracting has no spec.

    None is not an error here: the viewer also replays IPPO/pay-mechanism runs on
    environments contracting was never implemented for. It only becomes an error once
    a contract is involved, which is where the message can say what is missing.
    """
    from algorithms.MOCA import envs as moca_envs

    try:
        return moca_envs.spec_for(env_name)
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default=None,
                        help="e.g. clean_up, harvest_common_open, coin_game, coop_mining. "
                             "Optional when the checkpoint has a .run.yaml sidecar, which "
                             "records the environment the run was trained on")
    parser.add_argument("--checkpoint", default=None,
                        help="path to a .pkl saved by save_params(), or a quoted glob matching one "
                             "per-agent .pkl each (PARAMETER_SHARING=False runs); omit for a random policy")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_agents", type=int, default=None,
                        help="override agent count; capped by the env's spawn tiles (coin_game is 2-player only)")
    parser.add_argument("--gif", default=None, help="save the rollout as a GIF to this path")
    parser.add_argument("--no-interactive", action="store_true", help="skip opening the matplotlib scrubber")
    parser.add_argument("--no-jit", action="store_true", help="disable JAX jit (slow; only useful for debugging env internals)")
    parser.add_argument("--render-workers", type=int, default=None, help="parallel processes for rendering (default: all CPU cores)")
    parser.add_argument("--env-kwarg", action="append", default=[], metavar="KEY=VALUE",
                        help="extra env kwarg, repeatable (e.g. --env-kwarg pay_mode=on); "
                             "values parsed as int/float/bool when possible")
    parser.add_argument("--no-panel", action="store_true",
                        help="don't draw the per-agent info panel to the right of the grid")
    parser.add_argument("--no-autoconfig", action="store_true",
                        help="don't infer env kwargs (reward mode / pay scheme / num_agents) from the "
                             "checkpoint filename; use env defaults + explicit --env-kwarg only")
    parser.add_argument("--contract-theta", type=float, default=None,
                        help="MOCA only: replay under this contract value instead of the one the "
                             "learned proposal policy favours (0 = null contract, no transfers)")
    parser.add_argument("--contract-low", type=float, default=None,
                        help="MOCA only: lower bound of the NON-NULL contract range the run was "
                             "trained on (not encoded in the filename; must match CONTRACT_LOW). "
                             "Read from the run's .run.yaml sidecar when it has one; this "
                             "overrides it. The null contract theta=0 is always available and is "
                             "unaffected by these bounds. Without a sidecar this falls back to "
                             "moca_base.yaml's 0.2; pass 0.0 for runs trained before the range "
                             "excluded weak contracts")
    parser.add_argument("--bargain-segment", type=int, default=None,
                        help="PHASE2_MODE=bargain: steps per bargaining round. Read "
                             "from the checkpoint name (_seg<N>) by default; pass this "
                             "to replay the same policies at a different round length")
    parser.add_argument("--solver-samples", type=int, default=50,
                        help="PHASE2_MODE=solver: contracts sampled and scored by the "
                             "frozen critics at reset (training default: 50)")
    parser.add_argument("--solver-rule", default="majority", choices=("majority", "max"),
                        help="PHASE2_MODE=solver: decision rule (training default: majority)")
    parser.add_argument("--contract-high", type=float, default=None,
                        help="MOCA only: upper bound of the contract space (must match the run's "
                             "CONTRACT_HIGH; the bounds are not encoded in the filename, and a "
                             "mismatch silently rescales theta). Read from the run's .run.yaml "
                             "sidecar when it has one; this overrides it. Without a sidecar this "
                             "falls back to moca_base.yaml's 1.0; pass 0.2 for runs on the "
                             "paper's original range")
    parser.add_argument("--record", default=None, metavar="PATH",
                        help="save the rollout to PATH (.npz) so it can be reopened later "
                             "with --replay, without re-running the simulation")
    parser.add_argument("--replay", default=None, metavar="PATH",
                        help="load a recording saved by --record and render it; needs no "
                             "env, policies or checkpoint")
    parser.add_argument("--slow-render", action="store_true",
                        help="use the environment's own per-tile renderer instead of the fast "
                             "vectorised one. ~20x slower; its only visible difference is that "
                             "it also tints each agent's field-of-view window")
    args = parser.parse_args()

    # ---- replay path: no env, no policies, no simulation ------------------
    if args.replay:
        from viz.recording import load_recording, render_recording

        rec = load_recording(args.replay)
        meta = rec["meta"]
        env = _ReplayEnv(meta)
        contract_info = meta.get("contract_info")
        print(f"Replaying {args.replay}: {len(rec['grids'])} steps, "
              f"{meta['num_agents']} agents")
        t0 = time.time()
        frames = render_recording(rec["grids"], rec["agent_locs"], meta)
        print(f"Rendered {len(frames)} frames in {time.time() - t0:.1f}s")
        if not args.no_panel and rec["panel_data"] is not None:
            frames = add_info_panels(frames, rec["panel_data"],
                                     env.PLAYER_COLOURS, env, contract_info=contract_info)
            print("Added per-agent info panel.")
        if args.gif:
            save_gif(frames, args.gif)
            print(f"Saved GIF to {args.gif}")
        if not args.no_interactive:
            interactive_view(frames, [{} for _ in frames])
        return

    user_env_kwargs = {}
    for kv in args.env_kwarg:
        key, _, raw = kv.partition("=")
        if not _:
            raise SystemExit(f"--env-kwarg expects KEY=VALUE, got {kv!r}")
        user_env_kwargs[key] = _parse_env_kwarg_value(raw)

    env_name = resolve_env_name(args.env, args.checkpoint)
    spec = spec_for_env(env_name)

    # Recover the training config from the checkpoint name, then let explicit
    # --env-kwarg / --num_agents win over it (inference is a convenience, not a lock).
    inferred = {}
    if args.checkpoint and not args.no_autoconfig:
        inferred = _infer_env_kwargs_from_checkpoint(args.checkpoint, env_name, spec)
        if inferred:
            overridden = sorted(k for k, v in user_env_kwargs.items()
                                if k in inferred and v != inferred[k])
            print(f"Auto-config from checkpoint name: {inferred}"
                  + (f"  (your --env-kwarg overrides: {overridden})" if overridden else ""))
    env_kwargs = {**inferred, **user_env_kwargs}
    if args.no_jit:
        env_kwargs["jit"] = False
    if args.num_agents is not None:
        env_kwargs["num_agents"] = args.num_agents
    env = socialjax.make(env_name, **env_kwargs)

    # The panel's vocabulary for this environment, carried on the env because that is
    # where every panel function already reads its configuration from (and what
    # _ReplayEnv reconstructs from a recording).
    env.theta_unit, env.act_header = act_labels(spec)

    _warn_if_fixed_agent_count(env, env_name, args.num_agents)

    params = _load_checkpoint(args.checkpoint) if args.checkpoint else None
    check_action_space(params, env, env_name)

    _report_env_config(env, env_name, args.checkpoint)

    # MOCA: pick the contract to replay under. The learned proposal distribution is
    # the run's actual result, so its mode is the default -- what the agents ended up
    # wanting to sign -- with --contract-theta available to replay a counterfactual
    # (notably 0, the null contract, to see what the same policies do unsubsidised).
    contract, theta, contract_info = None, None, None
    bargain_cfg, bargain_params, bargain_rounds = None, None, None
    moca = detect_moca(args.checkpoint) if args.checkpoint else None
    if moca is not None:
        from algorithms.MOCA.contracts import contract_for_params, make_contract
        from algorithms.utils import contract_range

        if spec is None:
            raise SystemExit(
                f"this is a contracting checkpoint, but {env_name!r} has no contract "
                f"space (algorithms/MOCA/envs.py lists the ones that do)")

        # The bounds decide what every theta below MEANS -- replaying at the wrong
        # ones rescales it through both the contract observation the policy reads
        # and the unsquash of the proposal it emits, so the viewer shows a
        # mechanism that was never trained. Prefer the run's own record.
        #
        # The last-resort fallback is the range that environment's config actually
        # ships: moca_base raised Clean Up's above the paper's (0.2, 1.0 rather than
        # 0, 0.2) after the negotiated theta pinned to the ceiling, while Harvest and
        # the Coin Game still run the space their contract class declares.
        c_low, c_high, c_source = contract_range(
            args.checkpoint, args.contract_low, args.contract_high,
            fallback=((0.2, 1.0) if env_name == "clean_up" else spec.contract_range))

        # Matched to the checkpoint's own encoding so pre-fix policies (2 contract
        # features, no is_null flag) stay replayable; without a checkpoint there are
        # no weights to read, so fall back to the current space.
        one = params[0] if isinstance(params, list) else params
        contract = (contract_for_params(one, env.num_agents, c_low, c_high,
                                        space=spec.contract_space)
                    if one is not None
                    else make_contract(spec.contract_space, env.num_agents,
                                       c_low, c_high))
        mode = moca["mode"]
        print(f"MOCA run detected (PHASE2_MODE={mode})")
        print(f"  contract space: {spec.contract_space} -- theta in "
              f"[{c_low:g}, {c_high:g}] per {env.theta_unit} (from {c_source})"
              + (" (continuous)" if mode != "reinforce" else ""))
        if c_source == "fallback":
            print("    [warning] the run has no .run.yaml sidecar, so this range is a "
                  "guess.\n    If it is wrong, every theta shown is rescaled -- pass "
                  "--contract-low/--contract-high.")

        learned = None
        if mode == "reinforce":
            info = load_contract_policies(moca, c_low, c_high)
            learned = info["modal_theta"]
            print(f"  {len(moca['proposal_paths'])} proposal + "
                  f"{len(moca['voting_paths'])} voting policies, "
                  f"{len(info['theta_grid'])} bins")
            print("  learned proposal distribution (mean over agents):")
            for th, p in zip(info["theta_grid"], info["probs"].mean(axis=0)):
                print(f"    theta={th:.3f}  p={p:.3f}  {'#' * int(round(p * 40))}")
        elif mode == "negotiate":
            info = solve_negotiate_theta(moca, contract, env, args.seed)
            learned = info["theta"]
            print(f"  {len(moca['contract_paths'])} negotiation policies "
                  f"(each emits [theta, accept_prob]; agent 0 proposes, "
                  f"nu={info['nu']} of the rest are polled)")
            print(f"  agent 0 proposes theta={info['theta']:.4f}")
            probs = info["accept_probs"]
            print("  accept probability at that theta, per non-proposer:")
            for i, p in enumerate(probs, start=1):
                print(f"    agent {i}: {p:.3f}  {'#' * int(round(p * 40))}")
            # Signing needs nu of them to agree jointly, so the typical pair product
            # is the number that decides whether the contract ever takes force.
            if len(probs):
                print(f"  mean accept prob {probs.mean():.3f} -> a nu={info['nu']} draw "
                      f"signs with probability ~{probs.mean() ** info['nu']:.3f}")
        elif mode == "phase1":
            print("  no contracting policy on disk -- PHASE1_ONLY run, so there is no "
                  "learned theta to recover")
            if args.contract_theta is None:
                raise SystemExit(
                    "this is a PHASE1_ONLY checkpoint: phase 2 never ran, so no contract "
                    "was ever negotiated. Pass --contract-theta explicitly to choose what "
                    "to replay under (--contract-theta 0 is the null contract)."
                )
        elif mode == "solver":
            if params is None:
                raise SystemExit("solver runs need --checkpoint to score contracts")
            info = solve_solver_theta(params, contract, env, args.seed,
                                      args.solver_samples, args.solver_rule)
            learned = info["theta"]
            print(f"  solver ({args.solver_rule} rule, {args.solver_samples} samples) "
                  f"chose theta={info['theta']:.4f}"
                  + ("  [the NULL contract -- nothing beat it]" if info["null"] else ""))

        if mode == "bargain":
            from algorithms.MOCA import bargain as bg

            bargain_cfg = infer_bargain_config(moca["stem"],
                                               checkpoint=args.checkpoint)
            if args.bargain_segment is not None:
                bargain_cfg["segment"] = args.bargain_segment
            bargain_params = [load_params(p) for p in moca["contract_paths"]]
            try:
                bg.check_params_compatible(
                    bargain_params[0], len(bargain_params),
                    bargain_cfg.get("feature_version"),
                    hidden=bargain_cfg["hidden"], label=moca["stem"])
            except ValueError as e:
                raise SystemExit(f"[incompatible checkpoint] {e}")
            if bargain_cfg["contract_kind"] == "harvest_tax" and env_name != "clean_up":
                # A tax on harvest income, shared out by CLEANING, has no counterpart
                # where nobody cleans. check_arm refuses the combination at training
                # time, so this can only be a mis-attributed sidecar -- and replaying
                # it would levy a fiction in ordinary-looking units.
                raise SystemExit(
                    f"the sidecar records CONTRACT_KIND=harvest_tax, which is a Clean "
                    f"Up mechanism, but this run is on {env_name!r}")
            if bargain_cfg["contract_kind"] == "harvest_tax":
                # Same scalar space and the same observation encoding -- only the
                # transfer differs, so the contract object is swapped rather than
                # rebuilt. Replaying this run through the clean-wage transfer would
                # pay a cleaning wage in a run that levied a tax.
                from algorithms.MOCA.contracts import HarvestTaxContract
                contract = HarvestTaxContract(env.num_agents, c_low, c_high)
                print(f"  contract kind: HARVEST TAX "
                      f"({bargain_cfg['kind_source']}) -- theta is a rate on "
                      f"harvest income, shared out by cleaning over "
                      f"{bargain_cfg['tax_window']} steps")
            if bargain_cfg.get("protocol") == "median":
                print(f"  {len(bargain_params)} bargaining policies; "
                      f"protocol=median (simultaneous asks, the median binds), "
                      f"segment={bargain_cfg['segment']}, "
                      f"features={bargain_cfg['features']}")
            else:
                print(f"  {len(bargain_params)} bargaining policies; "
                      f"segment={bargain_cfg['segment']}, "
                      f"proposer={bargain_cfg['proposer']} "
                      f"(start {bargain_cfg['rotate_start']}), "
                      f"quorum={bargain_cfg['quorum']}, "
                      f"features={bargain_cfg['features']}")
            print("  theta is renegotiated during the episode, so it is not fixed "
                  "up front -- see the bargaining log in the panel")

        if args.contract_theta is not None:
            theta, source = float(args.contract_theta), "(manual)"
        else:
            theta, source = learned, "(learned)"
        contract_info = {"theta": theta, "source": source}
        if mode != "bargain" or args.contract_theta is not None:
            print(f"  replaying under theta={theta:g} {source}")

    if bargain_cfg is not None:
        # Parallel replay path: the one-shot arms keep `rollout` untouched.
        states, extras = rollout_bargaining(
            env, params, bargain_params, args.steps, args.seed, contract,
            bargain_cfg, spec, fixed_theta=args.contract_theta)
        bargain_rounds = extras["rounds"]
        settled = next((r for r in bargain_rounds if r["accepted"]), None)
        if args.contract_theta is not None:
            print(f"  --contract-theta given: bargaining bypassed, replaying at "
                  f"theta={args.contract_theta:g} throughout")
        elif bargain_cfg.get("binding", "episode") != "episode":
            live = [r["in_force"] for r in bargain_rounds
                    if r.get("in_force", 0.0) > contract.null + 1e-6]
            mean_theta = sum(live) / len(live) if live else contract.null
            print(f"  renegotiated every segment: {len(live)}/"
                  f"{len(bargain_rounds)} segments contracted, mean theta in force "
                  f"{mean_theta:.4f}")
            # The panel's per-round log is the honest record here; a single headline
            # theta is an average over segments that were each bargained separately.
            contract_info = {"theta": mean_theta, "source": "(mean over segments)"}
        elif settled:
            print(f"  agreed in round {settled['round']} at theta="
                  f"{settled['theta']:.4f} (proposed by agent {settled['proposer']}, "
                  f"{settled['n_accept']}/{settled['quorum']} needed)")
            contract_info = {"theta": settled["theta"], "source": "(bargained)"}
        else:
            print(f"  no agreement in {len(bargain_rounds)} rounds -- played "
                  f"uncontracted")
            contract_info = {"theta": contract.null, "source": "(no agreement)"}
    else:
        states, extras = rollout(
            env, params, args.steps, args.seed, contract=contract, theta=theta,
            spec=spec,
        )
    print(f"Rolled out {len(states) - 1} steps ({'trained checkpoint' if params else 'random policy'}).")

    traces = [_agent_snapshot(s) for s in states]
    t0 = time.time()
    if args.slow_render:
        frames = render_states(env, env_name, env_kwargs, states, workers=args.render_workers)
    else:
        # Vectorised path: one palette lookup + block upscale per frame instead of a
        # Python loop over ~1900 padded tiles (each of which forced a device sync).
        from viz.recording import render_recording

        meta = recording_meta(env, env_name, contract_info)
        frames = render_recording(
            np.stack([np.array(s.grid) for s in states]),
            np.stack([np.array(s.agent_locs) for s in states]),
            meta,
        )
    print(f"Rendered {len(frames)} frames in {time.time() - t0:.1f}s.")

    colors = getattr(env, "PLAYER_COLOURS", None)


    panel_data = None
    if _panel_supported(states, env):
        panel_data = collect_panel_data(states, env, transfers=extras["transfers"],
                                        act=extras["act"], rewards=extras["reward"])

    # Record BEFORE the panel is composited: a recording stores world state, not
    # pixels, so it can be re-rendered later at any size or with a different panel.
    if args.record:
        from viz.recording import save_recording

        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        save_recording(
            args.record,
            np.stack([np.array(s.grid) for s in states]),
            np.stack([np.array(s.agent_locs) for s in states]),
            panel_data,
            recording_meta(env, env_name, contract_info),
        )
        size_kb = Path(args.record).stat().st_size / 1024
        print(f"Recorded {len(states)} steps to {args.record} ({size_kb:.0f} KB) "
              f"-- reopen with --replay {args.record}")

    if not args.no_panel and panel_data is not None:
        frames = add_info_panels(frames, panel_data, colors, env,
                                 contract_info=contract_info,
                                 bargain_rounds=bargain_rounds)
        print("Added per-agent info panel.")

    if args.gif:
        save_gif(frames, args.gif)
        print(f"Saved GIF to {args.gif}")

    if not args.no_interactive:
        interactive_view(frames, traces)


if __name__ == "__main__":
    main()
