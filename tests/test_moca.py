"""Tests for MOCA (formal contracting) -- contract accounting and training wiring.

Runnable two ways:
    python tests/test_moca.py            # plain runner, no pytest needed
    python -m pytest tests/test_moca.py  # if pytest is installed

Covers the invariants that, if wrong, would silently invalidate a MOCA experiment:
contracts must be ZERO-SUM (a contract redistributes welfare, it never creates it),
the cleaner must be the net receiver, the contract must reach the policy, the
two phases must split the budget as Algorithm 1 specifies, and the phase-2
REINFORCE signal must push proposals toward higher-return contracts.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

import socialjax
from algorithms.MOCA.contracts import CleanupContract, make_contract
from algorithms.MOCA.networks import ContractActorCritic, ProposalPolicy, VotingPolicy


# ------------------------------------------------------------ contract space

def test_transfer_is_zero_sum():
    c = CleanupContract(7, 0.0, 0.2)
    rng = np.random.default_rng(0)
    for _ in range(100):
        cleaned = jnp.array(rng.integers(0, 3, 7).astype(np.float32))
        theta = float(rng.uniform(0, 0.2))
        t = c.compute_transfer(jnp.float32(theta), cleaned)
        assert abs(float(jnp.sum(t))) < 1e-5, "contracts must never create/destroy welfare"


def test_cleaner_receives_others_pay():
    c = CleanupContract(7, 0.0, 0.2)
    t = np.array(c.compute_transfer(jnp.float32(0.2), jnp.array([1., 0, 0, 0, 0, 0, 0])))
    assert abs(t[0] - 0.2) < 1e-6, "cleaner receives theta per cleaned cell"
    assert np.allclose(t[1:], -0.2 / 6), "the other N-1 agents fund it evenly"


def test_uniform_cleaning_means_no_net_transfer():
    c = CleanupContract(5, 0.0, 0.2)
    t = c.compute_transfer(jnp.float32(0.2), jnp.ones(5))
    assert np.allclose(np.array(t), 0.0, atol=1e-6)


def test_transfer_batches_over_envs():
    c = CleanupContract(4, 0.0, 0.2)
    cleaned = jnp.array([[1., 0, 0, 0], [1., 1, 0, 0], [0., 0, 0, 0]])
    theta = jnp.array([0.2, 0.1, 0.15])
    t = c.compute_transfer(theta, cleaned)
    assert t.shape == (3, 4)
    assert np.allclose(np.array(jnp.sum(t, axis=-1)), 0.0, atol=1e-6)
    assert np.allclose(np.array(t[2]), 0.0), "nobody cleaned -> no transfers"


def test_null_contract_moves_nothing():
    c = CleanupContract(7, 0.0, 0.2)
    t = c.compute_transfer(jnp.float32(0.0), jnp.array([3., 1, 0, 0, 0, 0, 0]))
    assert np.allclose(np.array(t), 0.0), "theta=0 is the null contract"


def test_sampling_respects_range_and_null_prob():
    c = CleanupContract(7, 0.0, 0.2)
    s = c.sample(jax.random.PRNGKey(0), (20000,), null_prob=0.25)
    assert float(s.min()) >= 0.0 and float(s.max()) <= 0.2
    assert 0.22 < float((s == 0.0).mean()) < 0.28, "null contract drawn at ~null_prob"
    s0 = c.sample(jax.random.PRNGKey(0), (2000,), null_prob=0.0)
    assert float((s0 == 0.0).mean()) < 0.01


def test_contract_grid_spans_range():
    c = CleanupContract(7, 0.0, 0.2)
    g = c.grid(11)
    assert g.shape == (11,)
    assert abs(float(g[0])) < 1e-9 and abs(float(g[-1]) - 0.2) < 1e-6


def test_to_obs_normalises():
    c = CleanupContract(7, 0.0, 0.2)
    o = c.to_obs(jnp.array([0.0, 0.1, 0.2]))
    assert o.shape == (3, 2), "layout matches the reference impl: [params..., stage]"
    assert np.allclose(np.array(o)[:, 0], [0.0, 0.5, 1.0])
    assert np.allclose(np.array(o)[:, 1], 0.0), "stage indicator is the subgame"


def test_invalid_contract_configs_rejected():
    for bad in (dict(num_agents=1), dict(num_agents=4, low=0.5, high=0.1)):
        kw = dict(num_agents=4, low=0.0, high=0.2)
        kw.update(bad)
        try:
            CleanupContract(**kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} should raise ValueError")
    try:
        make_contract("nope", 4, 0.0, 0.2)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown contract space should raise")


# ------------------------------------------------------------------ networks

def test_policy_actually_conditions_on_contract():
    """A different contract must change the policy's output -- otherwise Phase 1
    cannot be learning the family {pi(.|s,theta)} that MOCA's whole argument rests on."""
    net = ContractActorCritic(9)
    obs = jnp.zeros((2, 11, 11, 19))
    params = net.init(jax.random.PRNGKey(0), obs, jnp.zeros((2, 2)))
    _, v_lo = net.apply(params, obs, jnp.array([[0.0, 0.0], [0.0, 0.0]]))
    _, v_hi = net.apply(params, obs, jnp.array([[1.0, 0.0], [1.0, 0.0]]))
    assert not np.allclose(np.array(v_lo), np.array(v_hi)), \
        "value must depend on the active contract"


def test_proposal_and_voting_shapes():
    prop = ProposalPolicy(11)
    p = prop.init(jax.random.PRNGKey(0))
    pi = prop.apply(p)
    assert pi.logits.shape == (11,)
    assert abs(float(jnp.sum(jax.nn.softmax(pi.logits))) - 1.0) < 1e-5

    vote = VotingPolicy(4)
    vp = vote.init(jax.random.PRNGKey(0), jnp.zeros((3, 4)), jnp.zeros((3,)))
    piv = vote.apply(vp, jax.nn.one_hot(jnp.array([0, 1, 2]), 4), jnp.array([0.0, 0.5, 1.0]))
    assert piv.logits.shape == (3, 2), "binary accept/reject per env"


def test_reinforce_pushes_toward_higher_return_contract():
    """The core Phase-2 learning signal: proposing a contract that yields a higher
    episode return must raise that contract's probability."""
    K, N, E = 5, 4, 8
    prop = ProposalPolicy(K)
    params = prop.init(jax.random.PRNGKey(0))
    proposer = jnp.zeros((E,), dtype=jnp.int32)          # agent 0 proposes everywhere
    idx = jnp.array([4, 0, 4, 0, 4, 0, 4, 0])            # alternating high/low theta
    returns = jnp.stack([jnp.array([5., 1, 5, 1, 5, 1, 5, 1])] * N)
    adv = returns - returns.mean(axis=1, keepdims=True)

    def ploss(p, i=0):
        pi = prop.apply(p)
        mask = (proposer == i).astype(jnp.float32)
        return -(pi.log_prob(idx) * adv[i] * mask).sum() / jnp.maximum(mask.sum(), 1.0)

    g = jax.grad(ploss)(params)["params"]["proposal_logits"]
    assert float(g[4]) < 0.0, "gradient descent must RAISE the high-return logit"
    assert float(g[0]) > 0.0, "and LOWER the low-return logit"


def test_contracting_policies_train_under_scan():
    """List-of-TrainState mutation inside lax.scan (the pattern phase 2 uses) really
    does propagate updates."""
    K, N, E = 5, 3, 16
    nets = [ProposalPolicy(K) for _ in range(N)]
    tx = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(0.01, eps=1e-5))
    states = [TrainState.create(apply_fn=nets[i].apply,
                                params=nets[i].init(jax.random.PRNGKey(i)), tx=tx)
              for i in range(N)]

    def step(carry, unused):
        states, rng = carry
        rng, k1, k2, k3 = jax.random.split(rng, 4)
        proposer = jax.random.randint(k1, (E,), 0, N)
        logits = jnp.stack([nets[i].apply(states[i].params).logits for i in range(N)])
        idx = jax.random.categorical(k2, logits[proposer])
        returns = jax.random.uniform(k3, (N, E)) * 5.0
        adv = returns - returns.mean(axis=1, keepdims=True)

        def ploss(p, i):
            pi = nets[i].apply(p)
            mask = (proposer == i).astype(jnp.float32)
            return -(pi.log_prob(idx) * adv[i] * mask).sum() / jnp.maximum(mask.sum(), 1.0)

        for i in range(N):
            states[i] = states[i].apply_gradients(grads=jax.grad(ploss)(states[i].params, i))
        return (states, rng), 0.0

    (states, _), _ = jax.lax.scan(step, (states, jax.random.PRNGKey(0)), None, 5)
    moved = [np.any(np.abs(np.array(states[i].params["params"]["proposal_logits"])) > 1e-6)
             for i in range(N)]
    assert any(moved), "no contracting policy received a gradient across 5 updates"


# --------------------------------------------------------------- environment

def test_apple_reward_override_and_legacy_default():
    """apple_reward=1.0 gives the unit-apple economy contracts are calibrated for;
    omitting it must preserve upstream SocialJax's num_agents scaling exactly."""
    from socialjax.environments.cleanup.clean_up import Actions
    from tests.test_pay_mechanism import _place_apple_next_to
    STAY = int(Actions.stay)

    for kwargs, expected in (({}, 7.0), ({"apple_reward": 1.0}, 1.0)):
        env = socialjax.make("clean_up", num_agents=7, shared_rewards=False, **kwargs)
        assert env.apple_reward == expected
        key = jax.random.PRNGKey(0)
        _, state = env.reset(key)
        state, move = _place_apple_next_to(state, 0)
        _, _, rewards, _, _ = env.step_env(key, state, [move] + [STAY] * 6)
        r = np.array(rewards).squeeze()
        assert abs(r[0] - expected) < 1e-5, f"harvester should get {expected}, got {r[0]}"
        assert np.allclose(r[1:], 0.0), "individual reward: only the harvester is paid"


def test_cleaned_by_agent_is_per_agent():
    """The contract conditions on per-agent cleaning credit, so it must be
    attributable to individuals -- not the grid-wide broadcast waste_cleared is."""
    from socialjax.environments.cleanup.clean_up import Actions
    STAY = int(Actions.stay)
    env = socialjax.make("clean_up", num_agents=5, shared_rewards=False, apple_reward=1.0)
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)
    _, _, _, _, info = env.step_env(key, state, [STAY] * 5)
    assert "cleaned_by_agent" in info
    c = np.array(info["cleaned_by_agent"])
    assert c.shape == (5,), f"expected per-agent (5,), got {c.shape}"
    assert np.all((c == 0) | (c == 1))
    assert np.allclose(c, 0.0), "nobody cleaned when everyone stayed put"
    # regression: the pre-existing aggregate metrics must survive
    assert "waste_cleared" in info and "cleaned_water" in info


def test_phase_split_follows_algorithm_1():
    """Algorithm 1 spends 9/10 of episodes in the subgame phase and 1/10 contracting."""
    from algorithms.MOCA.moca_cnn_cleanup import make_train
    cfg = {
        "LR": 5e-4, "NUM_ENVS": 4, "NUM_STEPS": 10, "UPDATE_EPOCHS": 1,
        "NUM_MINIBATCHES": 2, "GAMMA": 0.99, "GAE_LAMBDA": 0.95, "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01, "VF_COEF": 0.5, "MAX_GRAD_NORM": 0.5, "ACTIVATION": "relu",
        "ANNEAL_LR": True, "PARAMETER_SHARING": False, "SEED": 0,
        "CONTRACT_SPACE": "cleanup", "CONTRACT_LOW": 0.0, "CONTRACT_HIGH": 0.2,
        "NUM_CONTRACT_BINS": 5, "PHASE1_FRAC": 0.9, "NULL_CONTRACT_PROB": 0.1,
        "CONTRACT_LR": 0.01, "ENV_NAME": "clean_up",
        "TOTAL_TIMESTEPS": 10 * 4 * 100,
        "ENV_KWARGS": {"num_agents": 3, "num_inner_steps": 10, "shared_rewards": False,
                       "cnn": True, "jit": True, "apple_reward": 1.0},
    }
    make_train(cfg)
    assert cfg["NUM_UPDATES"] == 100
    assert cfg["NUM_UPDATES_PHASE1"] == 90 and cfg["NUM_UPDATES_PHASE2"] == 10


def test_rollout_must_be_one_episode():
    """A contract is agreed per episode, so NUM_STEPS != num_inner_steps is rejected
    rather than silently spanning a reset."""
    from algorithms.MOCA.moca_cnn_cleanup import make_train
    cfg = {
        "NUM_ENVS": 4, "NUM_STEPS": 7, "PARAMETER_SHARING": False,
        "TOTAL_TIMESTEPS": 1000, "NUM_MINIBATCHES": 2, "ENV_NAME": "clean_up",
        "CONTRACT_LOW": 0.0, "CONTRACT_HIGH": 0.2, "NUM_CONTRACT_BINS": 5,
        "ENV_KWARGS": {"num_agents": 3, "num_inner_steps": 10, "cnn": True, "jit": True},
    }
    try:
        make_train(cfg)
    except ValueError as e:
        assert "one episode" in str(e)
    else:
        raise AssertionError("mismatched NUM_STEPS/num_inner_steps should raise")


def test_parameter_sharing_rejected():
    from algorithms.MOCA.moca_cnn_cleanup import make_train
    cfg = {
        "NUM_ENVS": 4, "NUM_STEPS": 10, "PARAMETER_SHARING": True,
        "TOTAL_TIMESTEPS": 1000, "NUM_MINIBATCHES": 2, "ENV_NAME": "clean_up",
        "CONTRACT_LOW": 0.0, "CONTRACT_HIGH": 0.2, "NUM_CONTRACT_BINS": 5,
        "ENV_KWARGS": {"num_agents": 3, "num_inner_steps": 10, "cnn": True, "jit": True},
    }
    try:
        make_train(cfg)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("PARAMETER_SHARING=True should be rejected")


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
