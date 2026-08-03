"""Tests for the Clean Up ecology-balance parameters.

Two upstream properties made the "commons needs >= 3 simultaneous cleaners" premise
impossible to realise:

  * dirt spawned at most ONE cell per step (a hard-coded `.at[0]`), while a single
    cleaner's 4-tile beam clears up to 4 cells/step -- an 8x surplus, so one
    part-time cleaner sustained the whole river;
  * apples PERSISTED once spawned, accumulating as a stock, so harvesting stayed
    profitable long after cleaning stopped and withholding cleaning cost nobody
    anything within an episode.

`dirt_spawn_cells` and `appleDecayProbability` fix those. Both default to the
original behaviour so existing baselines are unaffected.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

import socialjax
from socialjax.environments.cleanup.clean_up import Items

STAY = 6


def _env(**kw):
    return socialjax.make("clean_up", num_agents=7, shared_rewards=False,
                          apple_reward=1.0, **kw)


def _dirt_rate(steps=12, **kw):
    """Realised dirt cells added per step, starting from a fully clean river."""
    env = _env(delayStartOfDirtSpawning=0, **kw)
    key = jax.random.PRNGKey(0)
    _, s = env.reset(key)
    n = len(s.potential_dirt_and_dirt_label)
    s = s.replace(potential_dirt_and_dirt_label=jnp.full((n,), int(Items.potential_dirt),
                                                         dtype=jnp.int16))
    rng = jax.random.PRNGKey(1)
    counts = []
    for _ in range(steps):
        rng, k = jax.random.split(rng)
        _, s, _, _, _ = env.step_env(k, s, [STAY] * 7)
        counts.append(int((np.array(s.potential_dirt_and_dirt_label) == int(Items.dirt)).sum()))
    return (counts[-1] - counts[0]) / (len(counts) - 1)


# A clean beam covers exactly 4 tiles (see cleaned_count in step_env), so this is
# the hard per-cleaner ceiling: 4 cells/step, firing every step, beam perfectly
# placed. It is what makes "how many cleaners are needed" answerable from the
# spawn rate alone, without simulating a navigating cleaner.
BEAM_TILES = 4


def test_default_dirt_rate_is_tripled_but_still_clearable():
    """1.5 dirt/step: 3x upstream, and under the beam ceiling.

    Being under 4/step is deliberate. Rates at or above it leave the river
    permanently fouled no matter how many agents clean, which collapses the dilemma
    from the other side -- with no reachable clean state, cleaning stops paying for
    anyone. So the requirement for two cleaners rests on realised throughput being
    well below the ceiling, not on geometry, and is checked on trained runs via
    stage_1 cleaned_by_agent_mean rather than asserted here.
    """
    # Measured over a long window: spawning is self-limiting, since each step only
    # considers the `dirt_spawn_cells` cleanest candidates and the eligible pool
    # shrinks as the river fouls. Short windows overshoot the nominal k*p (1.9 over
    # 12 steps), very long ones undershoot as the river saturates (0.7 over 200).
    r = _dirt_rate(steps=100)
    assert 1.3 < r < 1.7, f"expected ~1.5 dirt/step, got {r}"
    assert r < BEAM_TILES, (
        f"dirt rate {r}/step is at or above one cleaner's {BEAM_TILES}-tile ceiling, "
        f"so the river can never be cleared and the commons is unrecoverable"
    )


def test_dirt_spawns_from_the_first_step():
    """No grace window: upstream's 50-step delay let apples grow before anyone had to
    clean, which front-loads free reward and mutes the dilemma early in an episode."""
    assert _env().delayStartOfDirtSpawning == 0


def test_upstream_ecology_is_still_reachable():
    """Diverging from cooperativex/SocialJax is a choice, not a one-way door: the
    original rates must stay recoverable for a comparison against the paper's setup."""
    r = _dirt_rate(dirt_spawn_cells=1, dirtSpawnProbability=0.5)
    assert abs(r - 0.5) < 0.2, f"upstream config should give ~0.5 dirt/step, got {r}"
    assert _env(maxAppleGrowthRate=0.05).maxAppleGrowthRate == 0.05


def test_apple_growth_rate_is_below_upstream():
    """Apples are deliberately scarcer than upstream's 0.05, so a fouled river costs
    more harvest for the same cleaning effort."""
    assert _env().maxAppleGrowthRate < 0.05


def test_dirt_spawn_cells_scales_the_rate():
    """The knob has to actually lift dirt above one cleaner's throughput (~1-2/step)."""
    slow = _dirt_rate(dirt_spawn_cells=1, dirtSpawnProbability=0.9)
    fast = _dirt_rate(dirt_spawn_cells=5, dirtSpawnProbability=0.9)
    assert fast > 3.0, f"5 cells/step at p=0.9 should give ~4.5 dirt/step, got {fast}"
    assert fast > 3 * slow, "raising dirt_spawn_cells must raise the realised rate"


def _apple_trace(clean_until, steps, **kw):
    """Standing apples per step; river held clean until `clean_until`, fouled after."""
    env = _env(delayStartOfDirtSpawning=10 ** 9, **kw)   # dirt is driven manually here
    key = jax.random.PRNGKey(0)
    _, s = env.reset(key)
    n = len(s.potential_dirt_and_dirt_label)
    rng = jax.random.PRNGKey(1)
    out = []
    for t in range(steps):
        fill = int(Items.potential_dirt) if t < clean_until else int(Items.dirt)
        s = s.replace(potential_dirt_and_dirt_label=jnp.full((n,), fill, dtype=jnp.int16))
        rng, k = jax.random.split(rng)
        _, s, _, _, _ = env.step_env(k, s, [STAY] * 7)
        out.append(int((np.array(s.grid) == int(Items.apple)).sum()))
    return out


def test_apples_persist_by_default():
    """Default (decay=0) keeps the upstream stock behaviour exactly."""
    tr = _apple_trace(60, 110)
    assert tr[100] >= tr[58], "with decay=0 a fouled river must not reduce standing apples"


def test_apple_decay_couples_harvest_to_river_state():
    """With decay on, letting the river foul must visibly cost harvest inside an episode
    -- otherwise a cleaner's strike has no teeth and harvesters learn nothing."""
    tr = _apple_trace(60, 120, maxAppleGrowthRate=0.02, appleDecayProbability=0.05)
    peak = tr[58]
    assert peak > 5, f"a clean river should sustain a standing crop, got {peak}"
    assert tr[100] < 0.4 * peak, (
        f"apples should collapse after fouling: {peak} -> {tr[100]}"
    )


def test_decay_reaches_a_nonzero_equilibrium_while_clean():
    """Decay must not simply starve the map: growth/(growth+decay) should hold a crop."""
    tr = _apple_trace(200, 200, maxAppleGrowthRate=0.02, appleDecayProbability=0.05)
    tail = tr[120:]
    assert min(tail) > 5, f"clean river should hold a standing crop, got min {min(tail)}"


# ------------------------------------------------- payment observability flags

def test_observe_payment_is_off_by_default():
    """Baselines must keep their 19-channel observation, or a policy trained before
    this flag existed is neither loadable nor comparable."""
    env = _env(pay_mode="on", pay_scheme="tithe")
    assert env.observation_space()[1] == (11, 11, 19)
    o, _ = env.reset(jax.random.PRNGKey(0))
    assert o.shape[-1] == 19


def test_observe_payment_adds_share_state_channels():
    env = _env(pay_mode="on", pay_scheme="tithe", observe_payment=True)
    assert env.observation_space()[1] == (11, 11, 21)
    key = jax.random.PRNGKey(0)
    _, s = env.reset(key)
    # force agents 0 and 1 to be sharing
    s = s.replace(share_expiry_t=jnp.array([999, 999, 0, 0, 0, 0, 0], dtype=jnp.int32))
    o, _, _, _, _ = env.step_env(key, s, [STAY] * 7)
    others = np.array(o[:, 0, 0, -2])
    own = np.array(o[:, 0, 0, -1])
    assert np.allclose(own, [1, 1, 0, 0, 0, 0, 0]), "own share state channel wrong"
    # a sharer sees 1 of the other 6 sharing; a non-sharer sees 2 of 6
    assert abs(others[0] - 1 / 6) < 1e-5 and abs(others[2] - 2 / 6) < 1e-5


def test_freeze_share_state_makes_an_imposed_pattern_binding():
    """Phase 1 of two-phase training imposes payments exogenously; the policy taking
    the pay action must not be able to overwrite them -- in the stored state OR in the
    transfers resolved on that same step."""
    from socialjax.environments.cleanup.clean_up import Actions
    PAY = int(Actions.pay)
    key = jax.random.PRNGKey(0)

    frozen = _env(pay_mode="on", pay_scheme="tithe", freeze_share_state=True)
    _, s = frozen.reset(key)
    s = s.replace(share_expiry_t=jnp.zeros((7,), dtype=jnp.int32))
    _, ns, _, _, info = frozen.step_env(key, s, [PAY] * 7)
    assert np.all(np.array(ns.share_expiry_t) == 0), "imposed pattern was overwritten"
    assert np.all(np.array(info["share_active"]) == 0), "pledge moved money anyway"

    normal = _env(pay_mode="on", pay_scheme="tithe")
    _, s2 = normal.reset(key)
    s2 = s2.replace(share_expiry_t=jnp.zeros((7,), dtype=jnp.int32))
    _, ns2, _, _, _ = normal.step_env(key, s2, [PAY] * 7)
    assert np.all(np.array(ns2.share_expiry_t) > 0), "unfrozen pledges must still work"


# ----------------------------------------------------------- toggle cooldown

def test_toggle_cooldown_rate_limits_flips():
    """Blocks the "fake supporter" exploit: because the tithe only takes a slice of
    income actually earned, holding the toggle ON while idle is free, so an agent could
    look like a payer and flip OFF just before harvesting. A minimum dwell time between
    flips means a displayed pledge must be honoured across a whole window."""
    from socialjax.environments.cleanup.clean_up import Actions
    PAY = int(Actions.pay)
    env = _env(pay_mode="on", pay_scheme="tithe", toggle_cooldown=25)
    key = jax.random.PRNGKey(0)
    _, s = env.reset(key)
    flips, prev = [], 0
    for t in range(60):
        _, s, _, _, info = env.step_env(key, s, [PAY] + [STAY] * 6)
        cur = int(np.array(info["share_active"])[0])
        if cur != prev:
            flips.append(t)
        prev = cur
    assert flips == [0, 25, 50], f"expected flips every 25 steps, got {flips}"


def test_toggle_cooldown_defaults_to_pledge_semantics():
    """cooldown=0 must keep the original pledge-for-share_duration behaviour."""
    from socialjax.environments.cleanup.clean_up import Actions
    PAY = int(Actions.pay)
    env = _env(pay_mode="on", pay_scheme="tithe")
    key = jax.random.PRNGKey(0)
    _, s = env.reset(key)
    _, ns, _, _, _ = env.step_env(key, s, [PAY] + [STAY] * 6)
    assert int(np.array(ns.share_expiry_t)[0]) == env.share_duration


def test_dirt_rate_three_per_step_config():
    """The configuration used for the two-phase experiments."""
    r = _dirt_rate(dirt_spawn_cells=3, dirtSpawnProbability=1.0)
    assert abs(r - 3.0) < 0.3, f"expected ~3 dirt/step, got {r}"


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


