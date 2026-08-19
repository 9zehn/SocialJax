"""The N-player Coin Game, and whether a contract can still be written on it.

Covers what nothing else does: that the environment's two theft signals are the two
sides of one event (which is what makes the contract's transfer zero-sum), that the
reward is the payoff matrix and nothing else, and that the observation is egocentric
in COLOUR as well as in position -- an agent reads its own coins in the same channel
whoever it is, which is what keeps the observation width independent of N.

The contracting arms themselves are covered by tests/test_contract_envs.py, which
iterates envs.ENV_SPECS and therefore picks this environment up automatically.

Run: OMP_NUM_THREADS=1 PYTHONPATH=$PWD python tests/test_coin_game_n.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

import socialjax
from algorithms.MOCA import contracts, envs
from socialjax.environments.coin_game_n.coin_game_n import CoinGameN, Items, State

N = 7


def _env(**over):
    kw = dict(num_agents=N, num_inner_steps=5000, shared_rewards=False, cnn=True,
              jit=True, coin_reward=1.0, regrow_rate=0.03)
    kw.update(over)
    return socialjax.make("coin_game_n", **kw)


def _rollout(env, steps=400, seed=0):
    """Random play, returning the per-step info dicts and rewards."""
    key = jax.random.PRNGKey(seed)
    obs, state = env.reset(key)
    out = []
    n = env.num_agents
    for _ in range(steps):
        key, sub = jax.random.split(key)
        acts = [jax.random.randint(jax.random.fold_in(sub, i), (), 0, 7)
                for i in range(n)]
        obs, state, reward, done, info = env.step_env(sub, state, acts)
        out.append((jax.tree.map(np.array, info), np.array(reward), state))
    return out


# ------------------------------------------------- the contract's preconditions

def test_every_theft_has_exactly_one_thief_and_one_victim():
    """The zero-sum property of CoinGameContract rests entirely on this: the two
    signals it differences must count the same events."""
    for info, _, _ in _rollout(_env()):
        by, frm = info["stolen_by_agent"], info["stolen_from_agent"]
        assert abs(by.sum() - frm.sum()) < 1e-5, (by, frm)


def test_a_taken_coin_is_either_your_own_or_a_theft():
    """No double counting and no gap -- otherwise `coins_taken` is not the
    denominator the behaviour metrics claim it is."""
    for info, _, _ in _rollout(_env()):
        assert np.allclose(info["eat_own_coins"] + info["stolen_by_agent"],
                           info["coins_taken"])


def test_you_are_never_recorded_as_stealing_from_yourself():
    for info, _, _ in _rollout(_env()):
        both = info["eat_own_coins"] * info["stolen_by_agent"]
        assert not both.any(), "a pickup was counted as own AND as theft"


def test_reward_is_the_payoff_matrix_and_welfare_is_own_minus_theft():
    env = _env()
    own_r, other_r, penalty = env.payoff
    for info, reward, _ in _rollout(env):
        expected = (info["eat_own_coins"] * own_r
                    + info["stolen_by_agent"] * other_r
                    + info["stolen_from_agent"] * penalty)
        assert np.allclose(reward, expected, atol=1e-4)
        # +1 for collecting your own, +1 - 2 = -1 for taking somebody else's.
        assert abs(reward.sum() - (info["eat_own_coins"].sum()
                                   - info["stolen_by_agent"].sum())) < 1e-4


def test_the_contract_is_zero_sum_on_this_environments_own_signals():
    """CoinGameContract needed no change to run at seven agents; this is the check
    that says so, driven by real rollout signals rather than synthetic ones."""
    c = contracts.make_contract("coin_game", N, low=0.0, high=2.0)
    for info, _, _ in _rollout(_env(), steps=200):
        for theta in (0.0, 0.5, 1.0, 2.0):
            t = np.array(c.compute_transfer(
                jnp.float32(theta),
                jnp.asarray(info["stolen_by_agent"]),
                jnp.asarray(info["stolen_from_agent"])))
            assert abs(t.sum()) < 1e-4, (theta, t)


def test_theta_one_makes_a_stolen_coin_worthless_to_the_thief():
    """The range [0, 2] is a property of the payoff matrix, so it must survive N."""
    c = contracts.make_contract("coin_game", N, low=0.0, high=2.0)
    by = jnp.array([1.0] + [0.0] * (N - 1))
    frm = jnp.array([0.0, 1.0] + [0.0] * (N - 2))
    t = np.array(c.compute_transfer(jnp.float32(1.0), by, frm))
    assert abs(t[0] + 1.0) < 1e-5, "the thief pays exactly its +1 gain at theta=1"
    t2 = np.array(c.compute_transfer(jnp.float32(2.0), by, frm))
    assert abs(t2[1] - 2.0) < 1e-5, "the victim is made whole at theta=2"


# -------------------------------------------------------------- the observation

def test_the_self_channel_fires_exactly_once_for_every_agent():
    """The shipped environments index the self channel with `len(Items) + i` where
    agent i sits at `len(Items) - 1 + i`, so every agent but the last reads its
    SUCCESSOR's channel and sees itself as another agent. This environment does not."""
    env = _env()
    obs, _ = env.reset(jax.random.PRNGKey(0))
    o = np.array(obs)
    self_ch = len(Items) - 1
    for i in range(N):
        assert o[i, :, :, self_ch].sum() == 1, (
            f"agent {i} sees itself {o[i, :, :, self_ch].sum()} times, expected once")


def test_an_agent_never_appears_in_its_own_other_agent_channel():
    env = _env()
    obs, state = env.reset(jax.random.PRNGKey(3))
    o = np.array(obs)
    self_ch = len(Items) - 1
    for i in range(N):
        both = o[i, :, :, self_ch] * o[i, :, :, self_ch + 1]
        assert not both.any(), f"agent {i} is its own 'other agent'"


def test_coins_are_egocentric_in_colour():
    """One coin, owned by agent 0, in view of agents 0 and 1. Agent 0 must read it in
    the MY-coin channel and agent 1 in the OTHER-coin channel -- that is what makes a
    7-agent observation the same width as a 2-agent one."""
    env = _env(jit=False)
    rows, cols = env.GRID_SIZE_ROW, env.GRID_SIZE_COL
    grid = jnp.zeros((rows, cols), jnp.int16)
    owner = jnp.full((rows, cols), -1, jnp.int8)
    # agents 0 and 1 either side of a coin owned by agent 0
    locs = [[9, 13, 0], [9, 15, 0]] + [[2, 2 * i, 0] for i in range(2, N)]
    agent_locs = jnp.array(locs, jnp.int16)
    grid = grid.at[agent_locs[:, 0], agent_locs[:, 1]].set(env._agents)
    grid = grid.at[9, 14].set(jnp.int16(Items.coin))
    owner = owner.at[9, 14].set(jnp.int8(0))
    state = State(agent_locs=agent_locs, coin_owner=owner, inner_t=0, outer_t=0,
                  grid=grid)
    o = np.array(env._get_obs_fn(state))
    mine_ch, theirs_ch = int(Items.coin) - 1, int(Items.other_coin) - 1
    assert o[0, :, :, mine_ch].sum() == 1, "the owner must see its own coin as MINE"
    assert o[0, :, :, theirs_ch].sum() == 0, "and not also as somebody else's"
    assert o[1, :, :, theirs_ch].sum() == 1, "the other agent must see it as THEIRS"
    assert o[1, :, :, mine_ch].sum() == 0, "and never as its own"


def test_observation_width_does_not_grow_with_the_agent_count():
    widths = set()
    for n in (2, 4, 7):
        env = socialjax.make("coin_game_n", num_agents=n, num_inner_steps=10,
                             shared_rewards=False, cnn=True, jit=True,
                             coin_reward=1.0)
        obs, _ = env.reset(jax.random.PRNGKey(0))
        widths.add(np.array(obs).shape[-1])
        assert np.array(obs).shape[0] == n
        assert env.observation_space()[1] == np.array(obs).shape[1:]
    assert len(widths) == 1, f"observation width varies with N: {widths}"


# ------------------------------------------------------------------- the fabric

def test_agents_never_share_a_cell():
    for _, _, state in _rollout(_env(), steps=300):
        locs = np.array(state.agent_locs)[:, :2]
        assert len(np.unique(locs, axis=0)) == N, "two agents occupy one cell"


def test_a_coin_is_removed_from_the_grid_when_it_is_taken():
    """A coin left behind would be collectable twice and would break attribution."""
    for info, _, state in _rollout(_env(), steps=200):
        grid = np.array(state.grid)
        owner = np.array(state.coin_owner)
        assert not (owner[grid != int(Items.coin)] >= 0).any(), (
            "coin_owner is set on a cell that holds no coin")
        assert (owner[grid == int(Items.coin)] >= 0).all(), (
            "a coin on the grid has no owner")


def test_every_colour_gets_coins():
    """Ownership is drawn uniformly, so over a long enough rollout no agent should be
    starved of its own colour -- otherwise cooperation is impossible for somebody."""
    env = _env()
    spawned = np.zeros(N)
    for info, _, _ in _rollout(env, steps=600):
        spawned += info["eat_own_coins"] + info["stolen_from_agent"]
    assert (spawned > 0).all(), f"some colour never appeared: {spawned}"


def test_reward_scale_defaults_to_num_agents():
    """The trap every contract range depends on: unset, the payoff matrix is N times
    too large and the [0, 2] range is N times too weak."""
    assert socialjax.make("coin_game_n", num_agents=N).coin_reward == float(N)
    assert socialjax.make("coin_game_n", num_agents=N, coin_reward=1.0).coin_reward == 1.0


def test_unported_reward_shaping_is_refused_rather_than_ignored():
    for flag in ("inequity_aversion", "svo", "interest", "enable_smooth_rewards"):
        try:
            CoinGameN(num_agents=3, **{flag: True})
        except NotImplementedError as exc:
            assert flag in str(exc)
        else:
            raise AssertionError(f"{flag}=True was silently accepted")


def test_degenerate_configurations_are_refused():
    for kwargs, needle in ((dict(num_agents=1), ">= 2"),
                           (dict(num_agents=9), "spawn")):
        try:
            CoinGameN(**kwargs)
        except ValueError as exc:
            assert needle in str(exc), (kwargs, str(exc))
        else:
            raise AssertionError(f"{kwargs} was accepted")


def test_the_spec_matches_what_the_environment_emits():
    spec = envs.spec_for("coin_game_n")
    env = _env()
    info = _rollout(env, steps=2)[0][0]
    for key in spec.behaviour_metrics + (spec.commons_metric, spec.progress_metric):
        assert key in info, f"the spec names {key}, which the env does not emit"
        assert np.array(info[key]).shape == (N,), f"{key} must be per-agent"
    commons = np.array(info[spec.commons_metric])
    assert np.allclose(commons, commons[0]), (
        "the commons metric must be uniform across agents -- the bargaining rollout "
        "reads it as info[commons][:, 0]")
    space = contracts.CONTRACT_SPACES[spec.contract_space]
    for key in space.SIGNAL_KEYS:
        assert key in info, f"the contract reads {key}, which the env does not emit"


ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

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
