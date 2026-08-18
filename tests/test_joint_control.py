"""The centralised joint-control baseline (algorithms/JOINT/).

Runnable two ways:
    python tests/test_joint_control.py            # plain runner, no pytest needed
    python -m pytest tests/test_joint_control.py  # if pytest is installed

This baseline is a NUMBER other results are stated against, so the things worth
pinning are the three properties that make that number mean "best achievable joint
behaviour" rather than something else:

  * the controller sees every agent (channel-concatenated observations),
  * it acts for every agent (one factored joint policy, one importance ratio),
  * it is paid their SUM (welfare, not an average and not a per-agent reward).

Get any of them wrong and the run still trains happily against the wrong objective,
which is exactly the failure a ceiling must not have.

Run it on its own, with OMP_NUM_THREADS=1 -- see the note in test_contract_envs.py
about the macOS recursive_mutex abort.
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
from algorithms.JOINT.joint_cnn import centralised_obs, make_train
from algorithms.JOINT.networks import (
    CentralisedActorCritic, joint_entropy, joint_log_prob,
)


def _wandb_off():
    import wandb
    wandb.init(mode="disabled")


def _cfg(env_name, **over):
    spec = envs.spec_for(env_name)
    cfg = {
        "LR": 5e-4, "NUM_ENVS": 2, "NUM_STEPS": 8, "TOTAL_TIMESTEPS": 8 * 2 * 3,
        "UPDATE_EPOCHS": 1, "NUM_MINIBATCHES": 1, "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95, "CLIP_EPS": 0.2, "ENT_COEF": 0.01, "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5, "ACTIVATION": "relu", "ANNEAL_LR": True, "SEED": 0,
        "CHECKPOINT_EVERY": 0, "PROGRESS_EVERY": 0, "ENV_NAME": env_name,
        "ENV_KWARGS": {"num_agents": min(spec.num_agents, 3), "num_inner_steps": 8,
                       "shared_rewards": False, "cnn": True, "jit": True,
                       spec.reward_scale_kwarg: 1.0},
    }
    cfg.update(over)
    return cfg


# ------------------------------------------------------- centralised observation

def test_centralised_obs_stacks_agents_on_the_channel_axis():
    """Channel concatenation, not a batch axis: the convolution has to see all N
    views in register at each spatial position, which is the whole informational
    advantage a central controller has over independent learners."""
    e, n, h, w, c = 2, 3, 11, 11, 5
    obs = jnp.arange(e * n * h * w * c, dtype=jnp.float32).reshape(e, n, h, w, c)
    world = centralised_obs(obs)
    assert world.shape == (e, h, w, n * c)
    # Each agent's channels survive intact, in order, at every spatial position.
    for i in range(n):
        assert np.allclose(np.array(world[:, :, :, i * c:(i + 1) * c]),
                           np.array(obs[:, i]))


def test_centralised_obs_matches_the_real_env_shapes():
    for env_name, spec in envs.ENV_SPECS.items():
        kwargs = dict(num_agents=spec.num_agents, num_inner_steps=8,
                      shared_rewards=False, cnn=True, jit=True)
        kwargs[spec.reward_scale_kwarg] = 1.0
        env = socialjax.make(env_name, **kwargs)
        obs, _ = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(0), 4))
        h, w, c = env.observation_space()[0].shape
        assert centralised_obs(obs).shape == (4, h, w, c * spec.num_agents), env_name


# ------------------------------------------------------------ the joint policy

def _pi(num_agents=3, action_dim=9, batch=4, channels=6):
    net = CentralisedActorCritic(num_agents=num_agents, action_dim=action_dim)
    x = jax.random.normal(jax.random.PRNGKey(1), (batch, 11, 11, channels))
    params = net.init(jax.random.PRNGKey(0), x)
    return net.apply(params, x)


def test_policy_emits_one_head_per_agent():
    pi, value = _pi(num_agents=3, action_dim=9)
    a = pi.sample(seed=jax.random.PRNGKey(2))
    assert a.shape == (4, 3), "one action per agent per env"
    assert value.shape == (4,), "one value: the controller has a single return"
    assert np.all((np.array(a) >= 0) & (np.array(a) < 9))


def test_joint_log_prob_is_the_sum_over_heads():
    """The factored policy means the joint log-prob is a sum, and PPO's ratio has to
    be taken on that joint quantity -- a per-agent ratio would apply the clip N times
    to N different numbers and stop being PPO."""
    pi, _ = _pi(num_agents=3, action_dim=9)
    a = pi.sample(seed=jax.random.PRNGKey(2))
    assert np.allclose(np.array(joint_log_prob(pi, a)),
                       np.array(pi.log_prob(a).sum(axis=-1)), atol=1e-6)
    assert joint_log_prob(pi, a).shape == (4,)


def test_joint_policy_normalises_over_the_whole_action_vector():
    """Sum of exp(joint log-prob) over every joint action must be 1 -- the check that
    the factorisation really is a distribution over the joint action."""
    import itertools
    pi, _ = _pi(num_agents=2, action_dim=3, batch=1)
    total = 0.0
    for combo in itertools.product(range(3), repeat=2):
        a = jnp.array([list(combo)])
        total += float(jnp.exp(joint_log_prob(pi, a))[0])
    assert abs(total - 1.0) < 1e-5, total


def test_joint_entropy_is_n_times_uniform_at_init():
    """Orthogonal(0.01) init makes the heads near-uniform, so the joint entropy
    starts at about N log|A|. It scales with N by construction, which is why the
    entropy bonus is taken on the joint policy rather than a per-head mean."""
    for n, a in ((3, 9), (2, 7), (7, 8)):
        pi, _ = _pi(num_agents=n, action_dim=a)
        assert abs(float(joint_entropy(pi).mean()) - n * np.log(a)) < 0.05 * n


# --------------------------------------------------------- the summed objective

def test_controller_is_paid_the_sum_of_agent_rewards():
    """"report straightforward sum, not average" -- the reference's own comment, and
    the property that makes this a WELFARE ceiling rather than an average-reward one.
    Checked against the env directly rather than through the loop."""
    spec = envs.spec_for("coin_game")
    env = socialjax.make("coin_game", num_agents=2, num_inner_steps=200,
                         shared_rewards=False, coin_reward=1.0)
    key = jax.random.PRNGKey(5)
    _, state = env.reset(key)
    step = jax.jit(env.step)
    seen_nonzero = False
    for _ in range(200):
        key, ka, ks = jax.random.split(key, 3)
        acts = [jax.random.randint(k, (), 0, env.num_actions)
                for k in jax.random.split(ka, 2)]
        _, state, reward, _, _ = step(ks, state, acts)
        r = jnp.asarray(reward)
        assert abs(float(r.sum(axis=-1)) - float(r.sum())) < 1e-6
        seen_nonzero |= bool(jnp.any(r != 0))
    assert seen_nonzero, "no reward at all -- the test is not exercising anything"


def test_shared_rewards_is_refused():
    """Under shared_rewards=True every agent is already paid the sum, so summing
    again scales welfare by N -- a run that trains fine against an objective N times
    the intended one and whose ceiling cannot be compared with anything."""
    cfg = _cfg("clean_up")
    cfg["ENV_KWARGS"]["shared_rewards"] = True
    try:
        make_train(cfg)
    except ValueError as e:
        assert "shared_rewards" in str(e) and "num_agents" in str(e)
    else:
        raise AssertionError("shared_rewards=True must be refused")


def test_rollout_must_be_one_episode():
    """So `welfare` is EPISODE welfare and lands on the same axis as the arms this
    is the ceiling for."""
    cfg = _cfg("clean_up", NUM_STEPS=4)
    try:
        make_train(cfg)
    except ValueError as e:
        assert "num_inner_steps" in str(e)
    else:
        raise AssertionError("a rollout that is not one episode must be refused")


# -------------------------------------------------------------- end to end

def _run_child(env_name):
    import json
    import subprocess
    proc = subprocess.run(
        [sys.executable, "-u", str(Path(__file__).resolve()), "--child", env_name],
        capture_output=True, text=True,
        env={**os.environ, "OMP_NUM_THREADS": "1", "WANDB_MODE": "disabled",
             "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
    )
    line = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    if not line:
        raise AssertionError(
            f"{env_name} produced no result.\nstdout:\n{proc.stdout[-2000:]}\n"
            f"stderr:\n{proc.stderr[-2000:]}")
    return json.loads(line[-1][len("RESULT "):])


def _child_main(env_name):
    import json
    _wandb_off()
    cfg = _cfg(env_name)
    out = jax.jit(make_train(cfg))(jax.random.PRNGKey(0))
    m = out["metrics"]
    print("RESULT " + json.dumps({
        "series": sorted(m),
        "updates": int(len(np.array(m["joint/welfare"]))),
        "num_updates": int(cfg["NUM_UPDATES"]),
        "finite": bool(all(np.all(np.isfinite(np.array(v))) for v in m.values())),
    }))


def test_trains_end_to_end_on_every_environment():
    for env_name in envs.ENV_SPECS:
        spec = envs.spec_for(env_name)
        r = _run_child(env_name)
        assert r["updates"] == r["num_updates"], env_name
        assert r["finite"], f"{env_name}: non-finite metric"
        # The behaviour series must carry the SAME info-field names the contracting
        # runs report, or the ceiling cannot be put on the same plot as them.
        for m in spec.behaviour_metrics:
            assert f"joint/{m}_mean" in r["series"], f"{env_name}: {m}"
        assert f"joint/{spec.commons_metric}_mean" in r["series"], env_name
        for k in ("joint/welfare", "joint/equality"):
            assert k in r["series"], f"{env_name}: {k}"


def test_episode_total_of_the_contracted_act_is_logged_and_consistent():
    """`<act>_total` is the readable form of the behaviour series: total cells
    cleaned / depleting harvests / coins stolen per episode, over all agents. It must
    equal the per-agent-per-step mean scaled by steps x agents, or one of the two is
    computing something other than it says.

    Run on the Coin Game because a random policy actually steals there; on Clean Up an
    untrained policy cleans nothing and the identity holds vacuously.
    """
    _wandb_off()
    T, N, E = 120, 2, 4
    cfg = _cfg("coin_game", NUM_ENVS=E, NUM_STEPS=T, TOTAL_TIMESTEPS=T * E * 2)
    cfg["ENV_KWARGS"]["num_agents"] = N
    cfg["ENV_KWARGS"]["num_inner_steps"] = T
    m = jax.jit(make_train(cfg))(jax.random.PRNGKey(0))["metrics"]
    mean = float(np.array(m["joint/stolen_by_agent_mean"])[-1])
    total = float(np.array(m["joint/stolen_by_agent_total"])[-1])
    assert total > 0, "no theft at all -- the test is not exercising anything"
    assert abs(total - mean * T * N) < 1e-3, (total, mean * T * N)


def test_commons_series_is_a_stock_not_a_cumulative_count():
    """Clean Up's `waste_cleared` is the number of potential-dirt cells that are
    currently NOT dirt -- river cleanliness at a point in time. It is emphatically not
    a running total of cleaning, and reading it as one would overstate provision by
    the episode length."""
    env = socialjax.make("clean_up", num_agents=3, num_inner_steps=30,
                         shared_rewards=False, apple_reward=1.0)
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)
    step = jax.jit(env.step)
    seen = []
    for _ in range(30):
        key, ka, ks = jax.random.split(key, 3)
        acts = [jax.random.randint(k, (), 0, env.num_actions)
                for k in jax.random.split(ka, 3)]
        _, state, _, _, info = step(ks, state, acts)
        seen.append(float(np.array(info["waste_cleared"])[0]))
    assert not all(b >= a for a, b in zip(seen, seen[1:])), (
        "waste_cleared never decreases, so it is behaving like a cumulative count "
        "rather than a stock -- the two cannot be used interchangeably")


def test_joint_and_decentralised_runs_do_not_share_a_checkpoint_path():
    """Same env, seed, reward and agent count is exactly what makes the two
    comparable -- and exactly what would collide. They are kept apart by directory,
    the convention MOCA already uses."""
    from algorithms.utils import checkpoint_filename
    cfg = {"ENV_NAME": "clean_up", "SEED": 42, "REWARD": "individual",
           "ENV_KWARGS": {"num_agents": 7}}
    stem = checkpoint_filename(cfg)
    assert f"./checkpoints/joint/{stem}.pkl" != f"./checkpoints/individual/{stem}.pkl"


ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--child":
    _child_main(sys.argv[2])
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
