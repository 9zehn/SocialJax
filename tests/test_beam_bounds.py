"""Regression tests for the zap/clean beam bounds bug.

Beam cells are built by offsetting an agent's position, so an agent at an edge facing
outward produces off-grid targets. Indexing jnp arrays with those is silently wrong:
negative indices wrap Python-style, too-large ones clamp to the last row/column. That
gave two exploits, both fixed by clip_beam_targets():

  * an agent facing a wall zapped ITSELF (the clamped cell was its own) and respawned
    at a random mid-map spawn point -- fast travel, not a game mechanic;
  * an agent at one edge stunned whoever stood at the OPPOSITE edge (wraparound).

The same flaw let a cleaning beam clear dirt across the map, which would corrupt the
per-agent cleaning credit that the tithe recipient rule and MOCA transfers depend on.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

import socialjax
from socialjax.environments.cleanup.clean_up import Actions, Items, clip_beam_targets

ZAP = int(Actions.zap_forward)
CLEAN = int(Actions.zap_clean)
STAY = int(Actions.stay)


def _env(n=2):
    return socialjax.make("clean_up", num_agents=n, shared_rewards=False, apple_reward=1.0)


def _placed(env, locs):
    """State with agents exactly at `locs` and NO stale agent ids left in the grid."""
    key = jax.random.PRNGKey(0)
    _, s = env.reset(key)
    g = jnp.where(s.grid >= len(Items), jnp.int16(Items.empty), s.grid)
    L = jnp.array(locs, dtype=jnp.int16)
    g = g.at[L[:, 0], L[:, 1]].set(env._agents)
    return s.replace(agent_locs=L, reborn_locs=L, grid=g), L


def _zap_and_settle(env, locs, actions):
    """Run the zap step plus one more: respawn is applied at the START of the next step."""
    env_ = env
    s, L = _placed(env_, locs)
    before = np.array(L)[:, :2].copy()
    key = jax.random.PRNGKey(0)
    _, s, _, _, _ = env_.step_env(key, s, actions)
    _, s, _, _, _ = env_.step_env(key, s, [STAY] * len(locs))
    after = np.array(s.agent_locs)[:, :2]
    return (after != before).any(axis=1)   # per-agent "was moved"


# ------------------------------------------------------------------- helper

def test_clip_beam_targets_flags_and_clamps():
    t = jnp.array([[-1, 5, 0], [0, -3, 0], [19, 5, 0], [5, 28, 0], [7, 7, 0]])
    clamped, valid = clip_beam_targets(t, 19, 28)
    assert list(np.array(valid)) == [False, False, False, False, True]
    assert np.all(np.array(clamped)[:, 0] >= 0) and np.all(np.array(clamped)[:, 0] <= 18)
    assert np.all(np.array(clamped)[:, 1] >= 0) and np.all(np.array(clamped)[:, 1] <= 27)
    assert list(np.array(clamped)[4]) == [7, 7, 0], "in-grid cells pass through untouched"


# ------------------------------------------------------------------ exploits

def test_facing_wall_does_not_teleport_self():
    """The reported bug: face a wall, zap, get teleported to mid-map."""
    env = _env()
    R, C = env.GRID_SIZE_ROW, env.GRID_SIZE_COL
    for name, locs in (
        ("bottom edge facing down", [[R - 1, 10, 0], [3, 25, 0]]),
        ("right edge facing right", [[8, C - 1, 1], [3, 3, 0]]),
        ("top edge facing up", [[0, 10, 2], [3, 25, 0]]),
        ("left edge facing left", [[8, 0, 3], [3, 25, 0]]),
    ):
        moved = _zap_and_settle(env, locs, [ZAP, STAY])
        assert not moved[0], f"{name}: zapper teleported itself"
        assert not moved[1], f"{name}: bystander was hit from across the map"


def test_edge_zap_does_not_hit_opposite_edge():
    """Negative indices used to wrap, so an edge zap stunned the far side of the map."""
    env = _env()
    R, C = env.GRID_SIZE_ROW, env.GRID_SIZE_COL
    moved = _zap_and_settle(env, [[0, 5, 2], [R - 1, 5, 0]], [ZAP, STAY])
    assert not moved.any(), "top-edge zap reached the bottom row"
    moved = _zap_and_settle(env, [[8, 0, 3], [8, C - 1, 0]], [ZAP, STAY])
    assert not moved.any(), "left-edge zap reached the right-hand column"


# -------------------------------------------------------- intended mechanics

def test_zap_still_stuns_agent_in_front():
    """The fix must not disarm the actual mechanic: the beam reaches 2 cells ahead."""
    env = _env()
    for dist, locs in ((1, [[8, 10, 1], [8, 11, 0]]), (2, [[8, 10, 1], [8, 12, 0]])):
        moved = _zap_and_settle(env, locs, [ZAP, STAY])
        assert moved[1], f"agent {dist} cell(s) ahead should have been stunned"
        assert not moved[0], "the zapper itself must never be stunned"


def test_zap_into_empty_space_stuns_nobody():
    env = _env()
    moved = _zap_and_settle(env, [[8, 10, 1], [3, 25, 0]], [ZAP, STAY])
    assert not moved.any()


def test_cleaning_beam_cannot_reach_across_the_map():
    """Cleaning credit drives the tithe recipient rule and MOCA transfers, so a beam
    that wrapped the map edge would mis-attribute the public good."""
    env = _env()
    R, C = env.GRID_SIZE_ROW, env.GRID_SIZE_COL
    key = jax.random.PRNGKey(0)
    for locs in ([[0, 5, 2], [3, 25, 0]], [[R - 1, 5, 0], [3, 25, 0]],
                 [[8, 0, 3], [3, 25, 0]], [[8, C - 1, 1], [3, 25, 0]]):
        s, _ = _placed(env, locs)
        _, _, _, _, info = env.step_env(key, s, [CLEAN, STAY])
        c = np.array(info["cleaned_by_agent"])
        assert c[0] == 0.0, f"agent at edge {locs[0]} cleaned dirt that is off-grid"


def test_cleaning_credit_counts_cells_not_actions():
    """The beam covers 4 tiles, so one clean action often clears several cells.

    Crediting a boolean under-reports the public good, and under a contract priced
    "per waste cell cleaned" it would pay the same for clearing 4 cells as for 1.
    """
    env = _env()
    key = jax.random.PRNGKey(0)
    # agent 0 at (8,10) facing col+1 -> beam covers exactly these four cells
    beam = [(8, 11), (8, 12), (7, 11), (9, 11)]
    for k in range(len(beam) + 1):
        s, L = _placed(env, [[8, 10, 1], [3, 25, 0]])
        g = s.grid
        for (r, c) in beam[:k]:
            g = g.at[r, c].set(jnp.int16(Items.dirt))
        g = g.at[L[:, 0], L[:, 1]].set(env._agents)
        s = s.replace(grid=g)
        before = np.array(s.grid)
        _, ns, _, _, info = env.step_env(key, s, [CLEAN, STAY])
        after = np.array(ns.grid)
        actually_cleared = sum(
            1 for (r, c) in beam[:k]
            if before[r, c] == int(Items.dirt) and after[r, c] != int(Items.dirt)
        )
        credited = float(np.array(info["cleaned_by_agent"])[0])
        assert credited == actually_cleared, (
            f"{k} dirt cells in beam: cleared {actually_cleared} but credited {credited}"
        )
        assert float(np.array(info["cleaned_by_agent"])[1]) == 0.0, "idle agent credited"


def test_cleaning_credit_is_zero_without_the_clean_action():
    env = _env()
    key = jax.random.PRNGKey(0)
    s, L = _placed(env, [[8, 10, 1], [3, 25, 0]])
    g = s.grid.at[8, 11].set(jnp.int16(Items.dirt))
    g = g.at[L[:, 0], L[:, 1]].set(env._agents)
    _, _, _, _, info = env.step_env(key, s.replace(grid=g), [STAY, STAY])
    assert float(np.array(info["cleaned_by_agent"])[0]) == 0.0




# ------------------------------------------- cleaning at the edge of the grid
# The clean beam is what MOCA contracts pay for, so a cell that is credited but
# never actually cleared is not a cosmetic glitch: it is an unbounded income
# source. Both failures below were reachable by standing at the river's edge.

def _clean_once(env, agent_rc_orient, others):
    """Fire one clean beam from a placed agent; return (grid_before, grid_after, credit)."""
    s, _ = _placed(env, [agent_rc_orient] + others)
    acts = [CLEAN] + [STAY] * len(others)
    before = np.array(s.grid).copy()
    _, s2, _, _, info = env.step_env(jax.random.PRNGKey(0), s, acts)
    credit = float(np.array(info["cleaned_by_agent"]).reshape(-1)[0])
    return before, np.array(s2.grid), credit


def _beam_cells(env, r, c, o):
    """The four beam cells, mirroring the env (out-of-grid diagonals fall back)."""
    step = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    one = (r + step[o][0], c + step[o][1])
    two = (r + 2 * step[o][0], c + 2 * step[o][1])
    rgt = (one[0] + step[(o + 1) % 4][0], one[1] + step[(o + 1) % 4][1])
    lft = (one[0] + step[(o - 1) % 4][0], one[1] + step[(o - 1) % 4][1])
    inb = lambda t: 0 <= t[0] < env.GRID_SIZE_ROW and 0 <= t[1] < env.GRID_SIZE_COL
    return [one, two, rgt if inb(rgt) else one, lft if inb(lft) else one], inb


def test_edge_clean_actually_clears_the_dirt_in_front():
    """The exploit: the off-grid two-step tile clamps back onto the dirt in front.

    Masked invalid, that slot used to re-write the cell's original value, and with
    duplicate scatter indices it could win over the legitimate clean -- so the cell
    stayed dirt while still being credited, every step, forever.
    """
    env = _env(2)
    r, c, o = 2, env.GRID_SIZE_COL - 2, 1          # facing the right edge
    cells, inb = _beam_cells(env, r, c, o)
    before, after, credit = _clean_once(env, [r, c, o], [[10, 1, 0]])
    targets = [t for t in set(cells) if inb(t) and before[t] == int(Items.dirt)]
    assert targets, "test needs dirt in front of the agent; the map changed"
    still = [t for t in targets if after[t] == int(Items.dirt)]
    assert not still, f"credited {credit} but these cells are still dirt: {still}"


def test_edge_clean_credits_distinct_cells_only():
    """Duplicate beam slots must not pay twice for one cell.

    At the top row the out-of-grid diagonal falls back onto the one-step tile, so
    three of the four slots can name two cells.
    """
    env = _env(2)
    r, c, o = 0, env.GRID_SIZE_COL - 2, 1
    cells, inb = _beam_cells(env, r, c, o)
    before, _, credit = _clean_once(env, [r, c, o], [[10, 1, 0]])
    expected = len({t for t in cells if inb(t) and before[t] == int(Items.dirt)})
    assert expected >= 1, "test needs dirt in the beam; the map changed"
    assert credit == expected, f"credited {credit} for {expected} distinct dirt cells"


def test_edge_dirt_cannot_be_farmed_indefinitely():
    """End to end: parked at the edge cleaning every step, credit must not be endless.

    With the clobber present the same cell is re-credited on every one of these
    steps; once cleared, the agent can only be paid again if dirt genuinely respawns.
    """
    env = _env(2)
    r, c, o = 2, env.GRID_SIZE_COL - 2, 1
    s, _ = _placed(env, [[r, c, o], [10, 1, 0]])
    key = jax.random.PRNGKey(0)
    front = (r, c + 1)
    credited_while_unchanged = 0
    for _ in range(25):
        was_dirt = np.array(s.grid)[front] == int(Items.dirt)
        _, s, _, _, info = env.step_env(key, s, [CLEAN, STAY])
        still_dirt = np.array(s.grid)[front] == int(Items.dirt)
        if was_dirt and still_dirt and float(np.array(info["cleaned_by_agent"]).reshape(-1)[0]) > 0:
            credited_while_unchanged += 1
    assert credited_while_unchanged == 0, (
        f"{credited_while_unchanged}/25 steps credited cleaning while the cell in "
        f"front stayed dirt -- the cell is being farmed"
    )



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
