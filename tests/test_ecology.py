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


def test_default_dirt_rate_matches_upstream_single_cell_cap():
    """Untouched defaults must still spawn <= 1 cell/step, or old runs aren't comparable."""
    r = _dirt_rate()
    assert r <= 1.0, f"default dirt rate {r} exceeds the upstream 1 cell/step cap"


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
