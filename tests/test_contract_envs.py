"""Formal contracting on Harvest and the Coin Game, plus single-stage contracting.

Runnable two ways:
    python tests/test_contract_envs.py            # plain runner, no pytest needed
    python -m pytest tests/test_contract_envs.py  # if pytest is installed

tests/test_moca.py covers the Clean Up contract and MOCA's two phases. This covers
what expanding beyond Clean Up added:

  * the two new contract spaces -- Harvest's, transcribed from the authors, and the
    Coin Game's, built to their rules -- against the same zero-sum invariant, plus
    the direction of the transfer, which is opposite to Clean Up's in both;
  * the env signals those spaces condition on, which have to be attributable to an
    individual agent and have to actually fire;
  * TRAINING_MODE=combined, the contracting arm without MOCA's phase split;
  * that every (environment, arm) pair trains end to end, since a contract space is
    only worth anything if the loop can be run with it.

Run it on its own, with OMP_NUM_THREADS=1. On macOS this suite occasionally aborts
mid-run in `recursive_mutex` -- a JAX/XLA problem with repeated compilation in one
process, documented in CLAUDE.md and unrelated to any test here. Re-run; it passes.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("WANDB_MODE", "disabled")

import jax
import jax.numpy as jnp
import numpy as np

import socialjax
from algorithms.MOCA import envs
from algorithms.MOCA.contracts import (
    CleanupContract, CoinGameContract, HarvestContract, make_contract,
)


def _wandb_off():
    import wandb
    wandb.init(mode="disabled")


# ----------------------------------------------------------- Harvest contract

def test_harvest_transfer_is_zero_sum():
    c = HarvestContract(7, 0.0, 10.0)
    rng = np.random.default_rng(0)
    for _ in range(100):
        eaten = jnp.array(rng.integers(0, 2, 7).astype(np.float32))
        theta = float(rng.uniform(0, 10))
        t = c.compute_transfer(jnp.float32(theta), eaten)
        assert abs(float(jnp.sum(t))) < 1e-4, "contracts must never create/destroy welfare"


def test_harvest_depleter_pays_and_the_others_are_paid():
    """The reference's direction: the agent that ate in a thin patch TRANSFERS theta
    to the others, `rews[i] -= transfers[i]` with the rest split evenly. Opposite in
    sign to Clean Up's, where the contracted act is rewarded rather than charged."""
    c = HarvestContract(7, 0.0, 10.0)
    t = np.array(c.compute_transfer(jnp.float32(10.0),
                                    jnp.array([1., 0, 0, 0, 0, 0, 0])))
    assert abs(t[0] + 10.0) < 1e-4, "the depleting eater pays theta"
    assert np.allclose(t[1:], 10.0 / 6, atol=1e-4), "the other N-1 share it evenly"


def test_harvest_charges_a_flat_theta_not_a_count():
    """The reference sets `transfers[key] = params[key][0]` -- theta, whatever the
    count was -- so the charge must not scale with the signal's magnitude."""
    c = HarvestContract(4, 0.0, 10.0)
    once = c.compute_transfer(jnp.float32(5.0), jnp.array([1., 0, 0, 0]))
    thrice = c.compute_transfer(jnp.float32(5.0), jnp.array([3., 0, 0, 0]))
    assert np.allclose(np.array(once), np.array(thrice))


def test_harvest_null_contract_moves_nothing():
    c = HarvestContract(7, 0.0, 10.0)
    t = c.compute_transfer(jnp.float32(c.null), jnp.array([1., 1, 0, 0, 0, 0, 0]))
    assert np.allclose(np.array(t), 0.0)


def test_harvest_range_is_the_papers():
    assert HarvestContract.DEFAULT_RANGE == (0.0, 10.0)


# --------------------------------------------------------- Coin Game contract

def test_coin_transfer_is_zero_sum():
    c = CoinGameContract(2, 0.0, 2.0)
    rng = np.random.default_rng(1)
    for _ in range(100):
        by = jnp.array(rng.integers(0, 3, 2).astype(np.float32))
        frm = jnp.array(by[::-1])          # 2 agents: each theft is the other's loss
        theta = float(rng.uniform(0, 2))
        t = c.compute_transfer(jnp.float32(theta), by, frm)
        assert abs(float(jnp.sum(t))) < 1e-4


def test_coin_thief_pays_the_victim():
    c = CoinGameContract(2, 0.0, 2.0)
    t = np.array(c.compute_transfer(jnp.float32(2.0),
                                    jnp.array([3., 0.]), jnp.array([0., 3.])))
    assert abs(t[0] + 6.0) < 1e-5, "the taker pays theta per stolen coin"
    assert abs(t[1] - 6.0) < 1e-5, "the owner of the coins receives it"


def test_coin_theta_one_makes_stealing_break_even():
    """The range's meaning, and the reason it is [0, 2] rather than tuned: at theta=1
    a stolen coin nets the taker nothing, and at theta=2 the owner is made whole."""
    c = CoinGameContract(2, 0.0, 2.0)
    payoff, penalty = 1.0, -2.0
    for theta, taker_net, victim_net in ((1.0, 0.0, -1.0), (2.0, -1.0, 0.0)):
        t = np.array(c.compute_transfer(jnp.float32(theta),
                                        jnp.array([1., 0.]), jnp.array([0., 1.])))
        assert abs((payoff + t[0]) - taker_net) < 1e-5
        assert abs((penalty + t[1]) - victim_net) < 1e-5


def test_coin_mutual_theft_cancels():
    c = CoinGameContract(2, 0.0, 2.0)
    t = c.compute_transfer(jnp.float32(2.0), jnp.array([2., 2.]), jnp.array([2., 2.]))
    assert np.allclose(np.array(t), 0.0)


# ------------------------------------------------ spaces share one scalar space

def test_every_space_shares_the_observation_encoding():
    """The arms are only comparable across environments if theta means the same
    thing to the policy in each -- normalised onto [-1, 1] with the same null flag,
    whatever the raw range happens to be (0.2 on Clean Up, 10 on Harvest).

    The floor is taken strictly above the null contract here, because at low == 0 the
    floor IS the null contract: the space is {0} u [low, high], and a contract equal
    to the null moves nothing whatever the range says.
    """
    for name, n, hi in (("cleanup", 7, 0.2), ("harvest", 7, 10.0),
                        ("coin_game", 2, 2.0)):
        lo = 0.25 * hi
        c = make_contract(name, n, low=lo, high=hi)
        assert c.obs_dim == 3
        obs = np.array(c.to_obs(jnp.array([c.null, lo, 0.5 * (lo + hi), hi])))
        assert obs.shape == (4, 3)
        assert np.allclose(obs[0], [0.0, 1.0, 0.0]), "null: flag +1, theta pinned to 0"
        assert np.allclose(obs[1], [-1.0, -1.0, 0.0]), "range floor -> -1"
        assert np.allclose(obs[2], [0.0, -1.0, 0.0]), "midpoint -> 0, but flag -1"
        assert np.allclose(obs[3], [1.0, -1.0, 0.0]), "range ceiling -> +1"


def test_a_zero_floor_makes_the_floor_the_null_contract():
    """The paper's ranges all start at 0, so their floor and their null contract are
    the same point. Worth pinning: every acceptance rule compares against V_i(s, 0),
    and a floor that quietly moved reward would corrupt it."""
    for name, n, hi in (("cleanup", 7, 0.2), ("harvest", 7, 10.0),
                        ("coin_game", 2, 2.0)):
        c = make_contract(name, n, low=0.0, high=hi)
        assert bool(c.is_null(jnp.float32(0.0)))
        assert np.allclose(np.array(c.to_obs(jnp.array([0.0])))[0], [0.0, 1.0, 0.0])


def test_contract_space_must_match_the_environment():
    from algorithms.MOCA.moca_cnn import make_train
    cfg = _tiny_cfg("harvest_common_open")
    cfg["CONTRACT_SPACE"] = "cleanup"
    try:
        make_train(cfg)
    except ValueError as e:
        assert "does not belong to" in str(e)
    else:
        raise AssertionError("a mismatched contract space must be refused")


def test_contract_reads_its_signals_from_info():
    """transfer_from_info is what keeps the training loop environment-agnostic, so a
    space paired with an env that does not emit its signals must say so."""
    c = HarvestContract(3, 0.0, 10.0)
    good = c.transfer_from_info(jnp.float32(10.0),
                                {"low_density_eaten": jnp.array([1., 0, 0])})
    assert abs(float(jnp.sum(good))) < 1e-4
    try:
        c.transfer_from_info(jnp.float32(10.0),
                             {"cleaned_by_agent": jnp.array([1., 0, 0])})
    except KeyError as e:
        assert "low_density_eaten" in str(e)
    else:
        raise AssertionError("a missing contract signal must be an error")


# ------------------------------------------------------------- env signals

def test_harvest_emits_per_agent_low_density_signal():
    """The contract conditions on it, so it has to be attributable to an individual
    and it has to be present in the INDIVIDUAL-reward branch, which is the only one
    a social dilemma exists in."""
    env = socialjax.make("harvest_common_open", num_agents=5, num_inner_steps=50,
                         shared_rewards=False, apple_reward=1.0)
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)
    _, _, _, _, info = env.step_env(key, state, [int(0)] * 5)
    for k in ("low_density_eaten", "eaten_apples", "local_apple_density",
              "apple_stock", "original_rewards", "shaped_rewards"):
        assert k in info, f"harvest info is missing {k}"
        assert np.array(info[k]).shape == (5,), f"{k} must be per-agent"
    d = np.array(info["low_density_eaten"])
    assert np.all((d == 0) | (d == 1)), "the predicate is an indicator"
    assert np.all(np.array(info["low_density_eaten"])
                  <= np.array(info["eaten_apples"])), "charged eats are a subset"


def test_harvest_low_density_predicate_fires_and_is_bounded_by_the_threshold():
    """A random policy must produce BOTH kinds of eating, or the contract would be
    conditioning on something that never happens (or always does)."""
    env = socialjax.make("harvest_common_open", num_agents=7, num_inner_steps=400,
                         shared_rewards=False, apple_reward=1.0)
    key = jax.random.PRNGKey(1)
    _, state = env.reset(key)
    step = jax.jit(env.step)
    ate = charged = 0.0
    for _ in range(400):
        key, ka, ks = jax.random.split(key, 3)
        acts = [jax.random.randint(k, (), 0, env.num_actions)
                for k in jax.random.split(ka, 7)]
        _, state, _, _, info = step(ks, state, acts)
        e = np.array(info["eaten_apples"])
        c = np.array(info["low_density_eaten"])
        dens = np.array(info["local_apple_density"])
        # The predicate must agree with the density it is derived from.
        assert np.all(c[c > 0] * dens[c > 0] < env.low_density_threshold)
        assert not np.any((e == 0) & (c > 0)), "cannot be charged without eating"
        ate += e.sum()
        charged += c.sum()
    assert ate > 0, "no apples eaten at all -- the test is not exercising anything"
    assert 0 < charged < ate, (
        f"the contract must price SOME but not ALL harvesting; got {charged} of {ate}")


def test_coin_game_emits_both_sides_of_a_theft():
    env = socialjax.make("coin_game", num_agents=2, num_inner_steps=50,
                         shared_rewards=False, coin_reward=1.0)
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)
    _, _, _, _, info = env.step_env(key, state, [int(6)] * 2)
    for k in ("stolen_by_agent", "stolen_from_agent", "coins_taken"):
        assert k in info, f"coin_game info is missing {k}"
        assert np.array(info[k]).shape == (2,), f"{k} must be per-agent"


def test_coin_theft_sides_always_balance():
    """The zero-sum property of the contract rests on this: every stolen coin appears
    once as somebody's theft and once as somebody's loss."""
    env = socialjax.make("coin_game", num_agents=2, num_inner_steps=300,
                         shared_rewards=False, coin_reward=1.0)
    key = jax.random.PRNGKey(3)
    _, state = env.reset(key)
    step = jax.jit(env.step)
    total = 0.0
    for _ in range(300):
        key, ka, ks = jax.random.split(key, 3)
        acts = [jax.random.randint(k, (), 0, env.num_actions)
                for k in jax.random.split(ka, 2)]
        _, state, _, _, info = step(ks, state, acts)
        by = np.array(info["stolen_by_agent"])
        frm = np.array(info["stolen_from_agent"])
        taken = np.array(info["coins_taken"])
        assert abs(by.sum() - frm.sum()) < 1e-6
        assert np.all(by <= taken), "a stolen coin is also a taken coin"
        total += by.sum()
    assert total > 0, "no theft occurred -- the test is not exercising anything"


def test_reward_scale_defaults_to_num_agents_in_both_new_envs():
    """Contract ranges are quoted against a unit reward, and both envs default their
    scale to num_agents. The knobs that fix that must exist and must work, or every
    theta is silently N times too weak."""
    h = socialjax.make("harvest_common_open", num_agents=7, shared_rewards=False)
    assert h.apple_reward == 7.0
    h1 = socialjax.make("harvest_common_open", num_agents=7, shared_rewards=False,
                        apple_reward=1.0)
    assert h1.apple_reward == 1.0
    c = socialjax.make("coin_game", num_agents=2, shared_rewards=False)
    assert c.coin_reward == 2.0
    c1 = socialjax.make("coin_game", num_agents=2, shared_rewards=False,
                        coin_reward=1.0)
    assert c1.coin_reward == 1.0


# --------------------------------------------------------------- env specs

def test_every_registered_env_has_a_contract_and_matching_signals():
    """An EnvSpec that names signals its environment does not emit is a run that
    fails thousands of traced lines later, so check the pairing directly."""
    for name, spec in envs.ENV_SPECS.items():
        kwargs = dict(num_agents=spec.num_agents, num_inner_steps=20,
                      shared_rewards=False, cnn=True, jit=True)
        kwargs[spec.reward_scale_kwarg] = 1.0
        env = socialjax.make(name, **kwargs)
        key = jax.random.PRNGKey(0)
        _, state = env.reset(key)
        _, _, _, _, info = env.step_env(key, state, [0] * spec.num_agents)
        contract = make_contract(spec.contract_space, spec.num_agents,
                                 *spec.contract_range)
        for k in contract.SIGNAL_KEYS:
            assert k in info, f"{name}: contract needs info[{k!r}]"
        for k in (spec.behaviour_metrics
                  + (spec.commons_metric, spec.progress_metric)):
            assert k in info, f"{name}: EnvSpec names info[{k!r}], which is absent"
        # The transfer must run on the env's real info and stay zero-sum on it.
        t = contract.transfer_from_info(jnp.float32(spec.contract_range[1]), info)
        assert abs(float(jnp.sum(t))) < 1e-4


def test_unregistered_env_is_refused_with_a_list():
    try:
        envs.spec_for("territory_open")
    except ValueError as e:
        assert "clean_up" in str(e) and "coin_game" in str(e)
    else:
        raise AssertionError("an env with no contract space must be refused")


def test_alternating_bargaining_runs_everywhere():
    """The renegotiation arm is not Clean Up-only. Only the machinery underneath it
    that is genuinely about cleaning is."""
    for env_name in envs.ENV_SPECS:
        envs.check_arm(envs.spec_for(env_name), "bargain", "joint",
                       {"BARGAIN_PROTOCOL": "alternating"})


def test_cleanup_only_bargaining_features_are_refused_elsewhere():
    """Each of these reads something only Clean Up has. Refused at config time rather
    than as a KeyError inside a traced rollout, or -- worse for the median case -- as
    a mechanism that runs happily while meaning nothing."""
    cases = (
        ({"BARGAIN_PROTOCOL": "median"}, "median"),
        ({"BARGAIN_PROTOCOL": "alternating", "REPORT_ENABLE": True}, "Clean Up only"),
        ({"BARGAIN_PROTOCOL": "alternating", "CONTRACT_KIND": "harvest_tax"},
         "Clean Up setting"),
    )
    for env_name in ("harvest_common_open", "coin_game"):
        spec = envs.spec_for(env_name)
        for cfg, expected in cases:
            try:
                envs.check_arm(spec, "bargain", "joint", cfg)
            except ValueError as e:
                assert expected in str(e), (env_name, cfg, str(e))
            else:
                raise AssertionError(f"{cfg} must be refused on {env_name}")
    # ... and all three stay available on Clean Up.
    for cfg, _ in cases:
        envs.check_arm(envs.spec_for("clean_up"), "bargain", "joint", cfg)


def test_commons_scale_is_read_off_the_environment():
    """The bargaining state divides the commons by this to reach unit range. A
    hardcoded Clean Up grid size would feed a Harvest policy a feature an order of
    magnitude out of range, which trains without complaint."""
    for env_name, spec in envs.ENV_SPECS.items():
        kwargs = dict(num_agents=spec.num_agents, num_inner_steps=20,
                      shared_rewards=False, cnn=True, jit=True)
        kwargs[spec.reward_scale_kwarg] = 1.0
        env = socialjax.make(env_name, **kwargs)
        s = envs.commons_scale(spec, env)
        assert s > 0, env_name
        key = jax.random.PRNGKey(0)
        _, state = env.reset(key)
        _, _, _, _, info = env.step_env(key, state, [0] * spec.num_agents)
        commons = float(np.array(info[spec.commons_metric]).max())
        assert commons <= s * 1.5, (
            f"{env_name}: commons {commons} against scale {s} -- the normalised "
            f"feature would leave [0, 1]")


# ------------------------------------------------- single-stage contracting

def _tiny_cfg(env_name, **over):
    spec = envs.spec_for(env_name)
    lo, hi = spec.contract_range
    cfg = {
        "LR": 5e-4, "NUM_ENVS": 2, "NUM_STEPS": 8, "TOTAL_TIMESTEPS": 8 * 2 * 4,
        "UPDATE_EPOCHS": 1, "NUM_MINIBATCHES": 1, "GAMMA": 0.99, "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2, "ENT_COEF": 0.01, "VF_COEF": 0.5, "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "relu", "ANNEAL_LR": False, "PARAMETER_SHARING": False,
        "SEED": 0, "NUM_CONTRACT_BINS": 5, "PHASE1_FRAC": 0.5,
        "NULL_CONTRACT_FRAC": 0.5, "VOTER_SAMPLE_NU": 1, "CONTRACT_MINIBATCHES": 1,
        "CONTRACT_LR": 0.01, "NEGOTIATE_UPDATE_EPOCHS": 1, "CHECKPOINT_EVERY": 0,
        "PROGRESS_EVERY": 0, "TRAINING_MODE": "two_phase", "PHASE2_MODE": "negotiate",
        "ENV_NAME": env_name, "CONTRACT_SPACE": spec.contract_space,
        "CONTRACT_LOW": lo, "CONTRACT_HIGH": hi,
        "ENV_KWARGS": {"num_agents": min(spec.num_agents, 3), "num_inner_steps": 8,
                       "shared_rewards": False, "cnn": True, "jit": True,
                       spec.reward_scale_kwarg: 1.0},
    }
    cfg.update(over)
    return cfg


def test_combined_spends_the_whole_budget_on_one_loop():
    """No phase split: unlike Algorithm 1's 9/10 - 1/10, every update negotiates."""
    from algorithms.MOCA.moca_cnn import make_train
    cfg = _tiny_cfg("clean_up", TRAINING_MODE="combined",
                    TOTAL_TIMESTEPS=8 * 2 * 20)
    make_train(cfg)
    assert cfg["NUM_UPDATES"] == 20
    assert cfg["NUM_UPDATES_PHASE1"] == 0
    assert cfg["NUM_UPDATES_PHASE2"] == cfg["NUM_UPDATES"]


def test_combined_requires_the_negotiate_protocol():
    """solver scores contracts with a critic that only means something once gameplay
    is frozen, and reinforce is a bandit over a fixed subgame. Neither is single-stage
    contracting, and running one under that name would mislabel the arm."""
    from algorithms.MOCA.moca_cnn import make_train
    for mode in ("solver", "reinforce"):
        try:
            make_train(_tiny_cfg("clean_up", TRAINING_MODE="combined",
                                 PHASE2_MODE=mode))
        except ValueError as e:
            assert "negotiate" in str(e)
        else:
            raise AssertionError(f"combined + {mode} must be refused")


def test_combined_rejects_phase1_loading():
    from algorithms.MOCA.moca_cnn import make_train
    for key in ("PHASE1_ONLY", "PHASE1_FROM"):
        val = True if key == "PHASE1_ONLY" else "./nowhere_[0-9].pkl"
        try:
            make_train(_tiny_cfg("clean_up", TRAINING_MODE="combined", **{key: val}))
        except ValueError as e:
            assert "combined" in str(e) or "matched 0 files" in str(e)
        else:
            raise AssertionError(f"{key} must be refused under combined")


def test_combined_and_two_phase_get_different_checkpoint_names():
    """The two arms differ only in TRAINING_MODE, so without it in the name the
    second run silently overwrites the first -- and that IS the comparison."""
    from algorithms.utils import checkpoint_filename
    base = {"ENV_NAME": "harvest_common_open", "SEED": 42, "REWARD": "individual",
            "PHASE2_MODE": "negotiate", "NEGOTIATE_NU": 2,
            "ENV_KWARGS": {"num_agents": 7}}
    two = checkpoint_filename({**base, "TRAINING_MODE": "two_phase"})
    comb = checkpoint_filename({**base, "TRAINING_MODE": "combined"})
    assert two != comb
    assert comb.endswith("_combined")
    # and two_phase keeps the name every existing MOCA run already has
    assert two == checkpoint_filename(base)


# Each end-to-end training below runs in its OWN process. Not fastidiousness: on
# macOS a second jax.jit(make_train(...)) compilation in one process aborts in
# recursive_mutex, so an in-process loop over environments takes the whole suite
# down at the second one and reports nothing about any of them. See CLAUDE.md.

def _run_child(env_name, mode):
    """Train one (environment, arm) in a subprocess; assert on what it reported."""
    import json
    import subprocess
    proc = subprocess.run(
        [sys.executable, "-u", str(Path(__file__).resolve()), "--child",
         env_name, mode],
        capture_output=True, text=True,
        env={**os.environ, "OMP_NUM_THREADS": "1", "WANDB_MODE": "disabled",
             "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
    )
    line = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    if not line:
        raise AssertionError(
            f"{env_name}/{mode} produced no result.\n"
            f"stdout tail:\n{proc.stdout[-2000:]}\nstderr tail:\n{proc.stderr[-2000:]}")
    return json.loads(line[-1][len("RESULT "):])


def _bargain_cfg(env_name, binding):
    """The renegotiation arm at test scale: 8-step episodes of two 4-step segments,
    so both a first-round agreement and a second round are reachable."""
    return _tiny_cfg(
        env_name, TRAINING_MODE="joint", PHASE2_MODE="bargain",
        BARGAIN_PROTOCOL="alternating", BARGAIN_BINDING=binding,
        BARGAIN_SEGMENT=4, BARGAIN_PROPOSER="random", BARGAIN_QUORUM="all",
        BARGAIN_FEATURES="private", BARGAIN_VOTE_ADVANTAGE="counterfactual",
        BARGAIN_PROBE_FRAC=0.1, BARGAIN_UPDATE_EPOCHS=1,
    )


def _child_main(env_name, mode):
    import json
    from algorithms.MOCA.moca_cnn import make_train
    _wandb_off()
    if mode == "combined":
        cfg = _tiny_cfg(env_name, TRAINING_MODE="combined")
    elif mode.startswith("bargain:"):
        rest = mode.split(":", 1)[1]
        density = rest.endswith(":density")
        cfg = _bargain_cfg(env_name, rest.replace(":density", ""))
        if density:
            cfg["CONTRACT_SPACE"] = "harvest_density"
    else:
        cfg = _tiny_cfg(env_name, PHASE2_MODE=mode, SOLVER_SAMPLES=3)
    out = jax.jit(make_train(cfg))(jax.random.PRNGKey(0))
    payload = {
        "keys": sorted(out),
        "num_updates": int(cfg["NUM_UPDATES"]),
        "series": {k: sorted(v) for k, v in out.items() if k.startswith("metrics")},
        "lengths": {k: int(len(np.array(next(iter(v.values())))))
                    for k, v in out.items() if k.startswith("metrics")},
    }
    print("RESULT " + json.dumps(payload))


def test_combined_trains_end_to_end_on_every_environment():
    for env_name in envs.ENV_SPECS:
        r = _run_child(env_name, "combined")
        assert "metrics_combined" in r["keys"], env_name
        assert "negotiate_state" in r["keys"], "the negotiation policies are a product"
        m = r["series"]["metrics_combined"]
        spec = envs.spec_for(env_name)
        # The arm's defining diagnostic: null exposure is an OUTCOME here, not a
        # setting, so it has to be logged.
        assert "combined/contract_null_frac" in m, env_name
        assert "combined/contract_theta_effective" in m, env_name
        assert f"combined/{spec.act_label}_gap" in m, env_name
        n = r["lengths"]["metrics_combined"]
        assert n == r["num_updates"], f"{env_name}: {n} != {r['num_updates']}"


def test_moca_trains_end_to_end_on_every_environment():
    for env_name in envs.ENV_SPECS:
        spec = envs.spec_for(env_name)
        for phase2 in ("negotiate", "solver"):
            r = _run_child(env_name, phase2)
            assert "metrics_phase1" in r["keys"], f"{env_name}/{phase2}"
            assert "metrics_phase2" in r["keys"], f"{env_name}/{phase2}"
            m1 = r["series"]["metrics_phase1"]
            # The per-env behaviour series must be there under their own names --
            # a missing one would mean the run cannot say whether behaviour moved.
            assert f"stage_1/{spec.contracted_act}_mean" in m1, env_name
            assert f"stage_1/{spec.act_label}_gap" in m1, env_name
            assert f"stage_1/{spec.commons_metric}_mean" in m1, env_name


def test_renegotiation_trains_end_to_end_at_both_bindings_everywhere():
    """The two bindings are the renegotiation experiment: 'a carried offer binds the
    rest of the episode' against 'renegotiated every segment'. Both have to run on
    every environment, and both have to report the series the comparison is read from
    -- speed of agreement, what took force, and whether behaviour moved."""
    for env_name in envs.ENV_SPECS:
        spec = envs.spec_for(env_name)
        for binding in ("segment", "episode"):
            r = _run_child(env_name, f"bargain:{binding}")
            assert "metrics_joint" in r["keys"], f"{env_name}/{binding}"
            assert "bargain_state" in r["keys"], "the bargaining policies are a product"
            m = r["series"]["metrics_joint"]
            for k in ("joint/agree/rate", "joint/agree/round",
                      "joint/contract/in_force_rate", "joint/contract/theta_in_force",
                      f"joint/behaviour/{spec.act_label}_per_agent",
                      f"joint/behaviour/{spec.act_label}_spread",
                      f"joint/behaviour/{spec.commons_metric}",
                      "joint/outcome/welfare", "joint/outcome/equality",
                      "joint/cf/gap"):
                assert k in m, f"{env_name}/{binding} is missing {k}"
            # The per-env denominator series, where the environment has one.
            for extra in spec.behaviour_metrics[1:]:
                assert f"joint/behaviour/{extra}" in m, f"{env_name}: {extra}"


def test_both_bindings_get_different_checkpoint_names():
    """episode and segment are different games run at the same seed and segment
    length. Without the binding in the name the second overwrites the first -- and
    the pair IS the experiment."""
    from algorithms.utils import checkpoint_filename
    base = {"ENV_NAME": "harvest_common_open", "SEED": 42, "REWARD": "individual",
            "PHASE2_MODE": "bargain", "TRAINING_MODE": "joint",
            "BARGAIN_SEGMENT": 100, "BARGAIN_PROTOCOL": "alternating",
            "BARGAIN_PROPOSER": "random", "BARGAIN_QUORUM": "all",
            "ENV_KWARGS": {"num_agents": 7}}
    seg = checkpoint_filename({**base, "BARGAIN_BINDING": "segment"})
    epi = checkpoint_filename({**base, "BARGAIN_BINDING": "episode"})
    assert seg != epi
    assert seg.endswith("_joint") and "_segment" in seg
    assert "_episode" in epi
    # ... and no two registered environments collide with each other either.
    names = {checkpoint_filename({**base, "ENV_NAME": e, "BARGAIN_BINDING": "segment"})
             for e in envs.ENV_SPECS}
    assert len(names) == len(envs.ENV_SPECS)


def test_contracts_move_reward_but_never_create_it_in_a_real_rollout():
    """End to end, on each environment: summed reward under a contract must equal
    summed reward without one, step by step, or the mechanism is a subsidy."""
    from algorithms.MOCA.contracts import make_contract
    # Long enough that the rarest contracted act happens at least once under a
    # random policy: Harvest's is "ate an apple in an already-thin patch", which a
    # random walker does a handful of times per few hundred steps.
    STEPS = 400
    for env_name, spec in envs.ENV_SPECS.items():
        kwargs = dict(num_agents=spec.num_agents, num_inner_steps=STEPS,
                      shared_rewards=False, cnn=True, jit=True)
        kwargs[spec.reward_scale_kwarg] = 1.0
        env = socialjax.make(env_name, **kwargs)
        contract = make_contract(spec.contract_space, spec.num_agents,
                                 *spec.contract_range)
        theta = jnp.float32(spec.contract_range[1])
        key = jax.random.PRNGKey(1)
        _, state = env.reset(key)
        step = jax.jit(env.step)
        moved = 0.0
        for _ in range(STEPS):
            key, ka, ks = jax.random.split(key, 3)
            acts = [jax.random.randint(k, (), 0, env.num_actions)
                    for k in jax.random.split(ka, spec.num_agents)]
            _, state, reward, _, info = step(ks, state, acts)
            t = contract.transfer_from_info(theta, info)
            assert abs(float(jnp.sum(t))) < 1e-3, env_name
            base = float(jnp.sum(jnp.asarray(reward)))
            after = float(jnp.sum(jnp.asarray(reward) + t))
            assert abs(base - after) < 1e-3, f"{env_name}: welfare changed"
            moved += float(jnp.sum(jnp.abs(t))) / 2
        assert moved > 0, f"{env_name}: the contract never moved anything at theta=max"


# ---------------------------------------------------------------- viewer

def _fake_bargain_run(tmp, env_name, n=3, segment=20):
    """A renegotiated bargaining checkpoint set for `env_name`, sidecar included.

    Random weights: what is under test is the replay PLUMBING -- which info field is
    read, which contract space is built, which scales the bargaining features use --
    not what a trained policy would do with it.
    """
    import optax
    from flax.training.train_state import TrainState

    from algorithms.MOCA import bargain
    from algorithms.MOCA.networks import BargainingActorCritic, ContractActorCritic
    from algorithms.utils.io_utils import (checkpoint_filename, save_params,
                                           save_run_config)

    spec = envs.spec_for(env_name)
    n = min(n, spec.num_agents)
    low, high = spec.contract_range
    config = {
        "ENV_NAME": env_name, "SEED": 42, "REWARD": "individual",
        "CONTRACT_SPACE": spec.contract_space,
        "CONTRACT_LOW": low, "CONTRACT_HIGH": high,
        "PHASE2_MODE": "bargain", "TRAINING_MODE": "joint",
        "BARGAIN_SEGMENT": segment, "BARGAIN_BINDING": "segment",
        "BARGAIN_PROTOCOL": "alternating", "BARGAIN_PROPOSER": "rotate",
        "BARGAIN_QUORUM": "all", "BARGAIN_FEATURES": "private",
        "BARGAIN_ROTATE_START": "random", "BARGAIN_HIDDEN": 64,
        "BARGAIN_ACCEPT_BIAS": 1.0,
        "ENV_KWARGS": {"num_agents": n, "num_inner_steps": 4 * segment, "cnn": True,
                       "jit": True, spec.reward_scale_kwarg: 1.0,
                       "shared_rewards": False},
    }
    stem = str(Path(tmp) / checkpoint_filename(config))

    env = socialjax.make(env_name, **config["ENV_KWARGS"])
    obs, _ = env.reset(jax.random.PRNGKey(0))
    contract = make_contract(spec.contract_space, n, low, high)
    play = ContractActorCritic(action_dim=env.action_space().n, activation="relu")
    bnet = BargainingActorCritic(hidden=64, activation="relu", accept_bias=1.0)
    feats = bargain.bargaining_features(
        0, 4, jnp.array([0]), n, jnp.zeros((1,)), jnp.zeros((1,)),
        jnp.zeros((1,), jnp.int32), jnp.zeros((1,)), jnp.zeros((1,)),
        jnp.zeros((n, 1)), jnp.zeros((1,)), jnp.zeros((n, 1)), jnp.zeros((n, 1)),
        jnp.zeros((1,)), bargain.feature_mask("private", n))
    tx = optax.adam(1e-3)
    for i in range(n):
        p = play.init(jax.random.PRNGKey(i), jnp.zeros((1,) + obs[env.agents[0]].shape),
                      jnp.zeros((1, contract.obs_dim)))
        save_params(TrainState.create(apply_fn=play.apply, params=p, tx=tx),
                    f"{stem}_{i}.pkl")
        bp = bnet.init(jax.random.PRNGKey(100 + i), feats[i])
        save_params(TrainState.create(apply_fn=bnet.apply, params=bp, tx=tx),
                    f"{stem}_contract_{i}.pkl")
    save_run_config(config, stem)
    return f"{stem}_[0-9].pkl", env, config


def test_viewer_replays_a_renegotiated_run_on_every_environment():
    """The viewer's bargaining replay must not be Clean Up-only.

    It reads the contracted act, the commons stock and the commons scale out of the
    environment on every step, and those are `cleaned_by_agent` / `waste_cleared` /
    a grid area on exactly one of the three. On the others, Clean Up's names are a
    KeyError -- which is the good case; the one to fear is a replay that runs and
    prices the wrong act.
    """
    import tempfile

    from viz.interactive_viewer import (detect_moca, infer_bargain_config,
                                        rollout_bargaining, spec_for_env)
    from algorithms.utils.io_utils import load_params

    for env_name in envs.ENV_SPECS:
        with tempfile.TemporaryDirectory() as tmp:
            pattern, env, config = _fake_bargain_run(tmp, env_name)
            spec = spec_for_env(env_name)
            n = config["ENV_KWARGS"]["num_agents"]
            low, high = spec.contract_range
            contract = make_contract(spec.contract_space, n, low, high)
            moca = detect_moca(pattern)
            assert moca is not None and moca["mode"] == "bargain", (env_name, moca)
            cfg = infer_bargain_config(moca["stem"], checkpoint=pattern)
            assert cfg["binding"] == "segment", f"{env_name}: {cfg['binding']}"
            assert cfg["binding_source"] == "sidecar", env_name

            play = [load_params(p) for p in sorted(
                p for p in __import__("glob").glob(pattern))]
            bp = [load_params(p) for p in moca["contract_paths"]]
            steps = 3 * config["BARGAIN_SEGMENT"]
            # theta at the ceiling, so the transfer cannot be zero by construction.
            states, extras = rollout_bargaining(
                env, play, bp, steps, 0, contract, cfg, spec, fixed_theta=high)

            assert len(states) == steps + 1, (env_name, len(states))
            for key in ("transfers", "act", "reward", "rounds"):
                assert key in extras, (env_name, key)
            tr = np.stack(extras["transfers"])
            act = np.stack(extras["act"])
            assert np.abs(tr.sum(axis=-1)).max() < 1e-3, \
                f"{env_name}: transfers are not zero-sum"
            # The act is the one the contract prices, so the money moved is theta
            # times it -- the check that the replay redistributes on the right series
            # rather than merely on some series.
            assert np.isclose(np.maximum(tr, 0.0).sum(), high * act.sum(), rtol=1e-4), \
                (env_name, np.maximum(tr, 0.0).sum(), high * act.sum())


def test_viewer_panel_labels_name_the_environments_own_act():
    """A panel that says "Clean" on Harvest is a wrong label on a real number: the
    column is thin-patch eating, which is a harm the contract FINES rather than a
    public good it subsidises."""
    from viz.interactive_viewer import act_labels

    # Distinct per ACT rather than per environment: two environments that price the
    # same act (the 2- and N-player coin games) SHOULD read identically, and only a
    # label shared across different acts would be the confusion worth failing on.
    by_act = {}
    for env_name, spec in envs.ENV_SPECS.items():
        unit, header = act_labels(spec)
        assert unit and header, env_name
        assert len(header) <= 8, f"{env_name}: {header!r} collides with the Reward column"
        by_act.setdefault(spec.act_label, set()).add((unit, header))
    for act, labels in by_act.items():
        assert len(labels) == 1, f"{act!r} is labelled inconsistently: {labels}"
    flat = [next(iter(v)) for v in by_act.values()]
    assert len(set(flat)) == len(flat), f"different acts share a label: {flat}"
    assert act_labels(None) == ("unit", "Act"), "envs without a spec still need a label"


def test_evaluator_words_every_environments_act_and_its_direction():
    """evaluate_bargain's tables are read as a scoreboard, so the wording has to say
    which way is good -- and it is opposite on Clean Up to the other two.

    Clean Up SUBSIDISES cleaning, so a working contract moves the act UP and the
    heavy actors are net receivers. Harvest and the Coin Game FINE a harm, so it
    moves DOWN and the heavy actors are net payers. Labelling a Harvest run in Clean
    Up's words would invert the reading of a table that otherwise looks fine.
    """
    from algorithms.MOCA.evaluate_bargain import ACT_WORDS, act_words

    by_act = {}
    for env_name, spec in envs.ENV_SPECS.items():
        w = act_words(spec)
        for key in ("act", "rate", "unit", "commons", "roles", "roles_plural",
                    "act_is_harm"):
            assert key in w, f"{env_name}: act_words is missing {key!r}"
        # The direction is not asserted against a hardcoded table -- it is DERIVED
        # from the contract's own transfer, so the wording is checked against the
        # mechanism rather than against a second copy of the same belief.
        contract = make_contract(spec.contract_space, spec.num_agents,
                                 *spec.contract_range)
        # Agent 0 performs the act once; whether that leaves it up or down on the
        # transfer is the whole question. Built through the info dict rather than
        # compute_transfer so a space reading two signals (the Coin Game reads the
        # thief AND the victim) is fed a coherent step: agent 0 took from agent 1.
        acted = np.zeros(spec.num_agents, np.float32)
        acted[0] = 1.0
        info = {spec.contracted_act: jnp.asarray(acted)}
        for key in contract.SIGNAL_KEYS[1:]:
            victim = np.zeros(spec.num_agents, np.float32)
            victim[1] = 1.0
            info[key] = jnp.asarray(victim)
        t = np.asarray(contract.transfer_from_info(
            jnp.float32(spec.contract_range[1]), info))
        assert (t[0] < 0) == w["act_is_harm"], (
            f"{env_name}: act_is_harm={w['act_is_harm']} but the sole actor's "
            f"transfer is {t[0]:+.3f}")
        assert len(w["roles"]) == 2 and len(w["roles_plural"]) == 2, env_name
        by_act.setdefault(spec.act_label, set()).add(w["roles"])
    # Same act -> same role names; different acts -> different ones.
    for act, roles in by_act.items():
        assert len(roles) == 1, f"{act!r} has inconsistent role labels: {roles}"
    flat = [next(iter(v)) for v in by_act.values()]
    assert len(set(flat)) == len(flat), f"different acts share role labels: {flat}"

    # An environment with no entry in the table still gets usable wording -- the
    # table is for phrasing, not for correctness, so adding an env must not be
    # gated on remembering to update it.
    fallback = act_words(envs.ENV_SPECS["clean_up"]._replace(env_name="nowhere"))
    assert fallback["act"] and len(fallback["roles"]) == 2
    assert set(ACT_WORDS) <= set(envs.ENV_SPECS), (
        f"ACT_WORDS names environments that no longer exist: "
        f"{set(ACT_WORDS) - set(envs.ENV_SPECS)}")


# ------------------------------------------- harvest with a bargained threshold

def test_regrow_ring_density_is_the_regrowth_rules_own_count():
    """The contract that bargains a density threshold prices the neighbourhood
    REGROWTH reads, which is not the 21-cell mask the published contract uses.

    Getting this wrong is the whole experiment: if the threshold were compared
    against a count the commons does not respond to, no value of it could protect
    regrowth and a null result would say nothing about the mechanism.
    """
    env = socialjax.make("harvest_common_open", num_agents=3, num_inner_steps=20,
                         cnn=True, jit=True, apple_reward=1.0)
    R, ROW, COL = env.LOCAL_APPLE_R, env.GRID_SIZE_ROW, env.GRID_SIZE_COL
    mask = np.array(env.REGROW_RING_MASK)
    assert env.REGROW_RING_CELLS == 12, env.REGROW_RING_CELLS
    assert mask.sum() == 12 and mask[R, R] == 0, "the centre is not in its own ring"

    # Every offset count_apple sums, transcribed. The env's own guards test the
    # wrong axis on one of them, so agreement is asserted off the boundary.
    offsets = ((-1, 0), (1, 0), (0, -1), (0, 1), (-2, 0), (2, 0), (0, -2), (0, 2),
               (-1, -1), (-1, 1), (1, -1), (1, 1))
    rng = np.random.default_rng(0)
    grid = (rng.random((ROW, COL)) < 0.35).astype(np.float32)
    padded = np.pad(grid, R)
    for r in range(2, ROW - 2):
        for c in range(2, COL - 2):
            via_mask = float((padded[r:r + 2 * R + 1, c:c + 2 * R + 1] * mask).sum())
            direct = float(sum(grid[r + dr, c + dc] for dr, dc in offsets))
            assert via_mask == direct, (r, c, via_mask, direct)

    # ...and it actually reaches the info dict, per agent, every step.
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)
    _, _, _, _, info = env.step_env(key, state, [0, 0, 0])
    ring = np.array(info["regrow_ring_density"]).reshape(-1)
    assert ring.shape == (3,), ring.shape
    assert ((ring >= 0) & (ring <= 12)).all(), ring


def test_density_contract_charges_only_below_the_bargained_threshold():
    """k is what decides which harvests are depletion, so the whole range has to
    mean something: inert at the bottom, a flat tax at the top, and the regrowth
    cliff reachable in between."""
    from algorithms.MOCA.contracts import HarvestDensityContract

    n = 7
    c = make_contract("harvest_density", n, 0.0, 10.0)
    assert c.PARAM_DIM == 2 and c.obs_dim == 4
    assert isinstance(c, HarvestDensityContract)
    assert c.density_low == 0.0 and c.density_high == 13.0, "0..12 plus 'always'"

    # Two eaters: one in a thin ring (2 apples), one in a thick one (8).
    ate = np.zeros(n, np.float32); ate[0] = ate[1] = 1.0
    ring = np.zeros(n, np.float32); ring[0], ring[1] = 2.0, 8.0
    info = {"eaten_apples": jnp.asarray(ate),
            "regrow_ring_density": jnp.asarray(ring)}
    charged_at = {}
    for k in (0.0, 3.0, 9.0, 13.0):
        p = jnp.array([10.0, k])
        t = np.asarray(c.transfer_from_info(p, info))
        assert abs(t.sum()) < 1e-4, (k, t.sum())          # zero-sum at every k
        charged_at[k] = sorted(np.flatnonzero(np.asarray(c.act_from_info(p, info))))
        # A charged agent is a net PAYER: this space prices a harm.
        for i in charged_at[k]:
            assert t[i] < 0, (k, i, t[i])
    assert charged_at[0.0] == [], "k=0 must be inert -- no ring count is below 0"
    assert charged_at[3.0] == [0], "only the thin-ring eater at the regrowth cliff"
    assert charged_at[9.0] == [0, 1]
    assert charged_at[13.0] == [0, 1], "k=13 charges every harvest"

    # The null contract moves nothing whatever k says, because k alone is not a
    # contract -- and the acceptance rules all measure against that disagreement
    # point.
    for k in (0.0, 6.0, 13.0):
        p = jnp.array([0.0, k])
        assert bool(c.is_null(p)), k
        assert np.abs(np.asarray(c.transfer_from_info(p, info))).max() == 0.0, k


def test_density_contract_observation_separates_null_from_every_threshold():
    """Same argument as the scalar spaces' is_null flag: the null contract must not
    encode as a point on the theta ramp, and a threshold attached to a zero fine is
    not a fact about the episode."""
    c = make_contract("harvest_density", 7, 0.0, 10.0)
    obs = {p: np.asarray(c.to_obs(jnp.array(list(p))))
           for p in ((0.0, 0.0), (0.0, 13.0), (10.0, 0.0), (10.0, 13.0), (5.0, 6.5))}
    assert obs[(0.0, 0.0)].shape == (4,)
    # Every null contract encodes identically, whatever k rode along with it.
    assert np.allclose(obs[(0.0, 0.0)], obs[(0.0, 13.0)])
    assert obs[(0.0, 0.0)][2] == 1.0 and obs[(10.0, 0.0)][2] == -1.0
    # No input encodes as the zero vector -- the degeneracy is_null exists to kill.
    for p, v in obs.items():
        assert np.abs(v).max() > 0, p
    # Both components span [-1, 1] over their own range.
    assert np.allclose(obs[(10.0, 0.0)][:2], [1.0, -1.0])
    assert np.allclose(obs[(10.0, 13.0)][:2], [1.0, 1.0])


def test_density_contract_refuses_the_arms_that_cannot_express_it():
    """PHASE2_MODE=reinforce learns a categorical over a 1-D grid. Silently handing
    it component 0 would train a mechanism nobody asked for."""
    c = make_contract("harvest_density", 7, 0.0, 10.0)
    try:
        c.grid(11)
    except NotImplementedError as e:
        assert "2-D" in str(e) or "grid" in str(e)
    else:
        raise AssertionError("a 2-D contract must not produce a 1-D grid")


def test_density_proposal_head_widens_without_touching_the_scalar_one():
    """param_dim=1 has to stay bit-for-bit the network every existing bargaining
    checkpoint was trained with, or the 2-D space costs the 1-D runs their weights."""
    import jax as _jax
    from algorithms.MOCA import bargain
    from algorithms.MOCA.networks import BargainingActorCritic

    n = 7
    feats = bargain.bargaining_features(
        0, 4, jnp.array([0]), n, jnp.zeros((1,)), jnp.zeros((1,)),
        jnp.zeros((1,), jnp.int32), jnp.zeros((1,)), jnp.zeros((1,)),
        jnp.zeros((n, 1)), jnp.zeros((1,)), jnp.zeros((n, 1)), jnp.zeros((n, 1)),
        jnp.zeros((1,)), bargain.feature_mask("private", n))
    old = BargainingActorCritic(hidden=64).init(_jax.random.PRNGKey(0), feats[0])
    one = BargainingActorCritic(hidden=64, param_dim=1).init(
        _jax.random.PRNGKey(0), feats[0])
    assert _jax.tree.all(_jax.tree.map(
        lambda a, b: bool(np.array_equal(a, b)), old, one)), \
        "param_dim=1 changed the parameter tree"

    two = BargainingActorCritic(hidden=64, param_dim=2)
    p2 = two.init(_jax.random.PRNGKey(0), feats[0])
    assert p2["params"]["log_std"].shape == (2,)
    pi, _, _ = two.apply(p2, feats[0])
    assert pi.sample(seed=_jax.random.PRNGKey(1)).shape[-1] == 2


def test_density_arm_trains_and_reports_the_threshold():
    """The 2-D space has to run the joint bargaining arm end to end AND report what
    it bargained -- a threshold that is learned but not logged is not an experiment.
    """
    res = _run_child("harvest_common_open", "bargain:segment:density")
    series = res["series"]["metrics_joint"]
    assert "joint/contract/k_offered" in series, sorted(series)
    assert "joint/contract/theta_in_force" in series

    # Under WANDB_METRIC_SET=core the pruned set carries the threshold only when
    # there IS one, so a Clean Up view is not padded with a knob that environment
    # does not have. (Under `full` it is present and identically zero, which is how
    # tax/* and report/* already behave.)
    from algorithms.MOCA import envs as _envs
    from algorithms.MOCA.moca_cnn import joint_core_metrics

    harvest = _envs.spec_for("harvest_common_open")
    assert "density_k_offered" in joint_core_metrics(harvest, param_dim=2)
    assert "density_k_offered" not in joint_core_metrics(harvest, param_dim=1)
    assert len(joint_core_metrics(harvest, param_dim=2)) == 11


def test_median_is_refused_for_a_multi_parameter_contract():
    """The median mechanism binds the middle ask. There is no middle of a set of
    vectors -- a per-component median is a contract nobody offered."""
    from algorithms.MOCA.moca_cnn import make_train

    cfg = _bargain_cfg("harvest_common_open", "segment")
    cfg.update(CONTRACT_SPACE="harvest_density", BARGAIN_PROTOCOL="median")
    try:
        make_train(cfg)
    except ValueError as e:
        assert "median" in str(e) and "one-dimensional" in str(e), e
    else:
        raise AssertionError("median must be refused for a 2-D contract space")


def test_contract_space_must_be_one_the_environment_offers():
    """harvest_density belongs to Harvest and nowhere else: its signals are Harvest's
    and its threshold indexes Harvest's regrowth rule."""
    from algorithms.MOCA.moca_cnn import make_train

    cfg = _tiny_cfg("clean_up")
    cfg["CONTRACT_SPACE"] = "harvest_density"
    try:
        make_train(cfg)
    except ValueError as e:
        assert "does not belong" in str(e), e
    else:
        raise AssertionError("a foreign contract space must be refused")


ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--child":
    # One training run, invoked by _run_child in its own process.
    _child_main(sys.argv[2], sys.argv[3])
    sys.exit(0)

if __name__ == "__main__":
    failed = 0
    for t in ALL_TESTS:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(ALL_TESTS) - failed}/{len(ALL_TESTS)} tests passed")
    sys.exit(1 if failed else 0)
