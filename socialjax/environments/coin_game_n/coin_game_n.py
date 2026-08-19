"""The Coin Game for N players, one coin colour per agent.

The two-player Coin Game (socialjax/environments/coin_game) is red-vs-green by
construction: its payoff block reads `red_apple_matches[0]` and
`green_apple_matches[1]`, builds `jnp.zeros((2, 1))` rewards, and cannot be widened
by a config value. This is the same game for any N >= 2, written so that agent count
is a parameter rather than a rewrite.

WHAT IS THE SAME. A coin belongs to exactly one agent. Taking any coin pays the
taker +1; taking someone ELSE's additionally costs its owner -2. So collecting your
own colour adds 1 to welfare and stealing subtracts 1, which is the dilemma. Because
the harm is attributable to a single victim, this remains the cleanest contract
testbed in the project: `contracts.CoinGameContract` needs no change at any N, its
transfer theta * (stolen_from - stolen_by) is zero-sum by construction, and the
[0, 2] range still means what it meant -- theta = 1 is where stealing stops paying
the taker, theta = 2 where the owner is made whole. Above 2 a contract would pay
agents to be robbed, which is a property of the payoff matrix and not of N.

WHAT CHANGES, AND IT IS NOT A DETAIL. With two colours half the coins you meet are
yours; with seven, one in seven is. Cooperating therefore means walking past six
forbidden coins for every one you may take, so the temptation density rises sharply
with N even though per-agent income under full cooperation is held fixed (see
`regrow_rate`). The N-player game is a HARSHER dilemma, not the two-player game with
more players in it, and results should be reported that way.

Three implementation choices are worth stating because the obvious alternatives are
worse:

  * COIN OWNERSHIP LIVES IN ITS OWN PLANE, not in the `Items` enum. `state.grid`
    stores one code for "a coin is here" and `state.coin_owner` says whose. Widening
    the enum to N coin codes would instead shift the agent codes (`len(Items) + i`),
    the padding value and every index derived from `len(Items)`.

  * THE OBSERVATION IS EGOCENTRIC IN COLOUR, not just in position. Each agent's view
    recodes coins to MINE / NOT MINE, exactly as agent channels are already collapsed
    to self / other. So the observation is (OBS_SIZE, OBS_SIZE, 10) for every N -- a
    7-agent policy has the same shape as a 2-agent one -- and no agent has to learn
    "channel k is my colour". An agent does not need to know WHOSE coin it is taking:
    the payoff is the same whoever it belongs to, and the contract attributes the
    harm from the environment's own record rather than from anyone's observation.

  * COINS SPAWN IN ONE PASS WITH A RANDOM OWNER, not one pass per colour. The
    per-colour rate is then regrow_rate / N automatically, and the cost does not grow
    with N.

DEVIATION FROM THE SHIPPED ENVIRONMENTS, deliberate and load-bearing. In
coin_game.py, clean_up.py and harvest_open.py the observation's "self" channel is
built as `x[agent]` where `agent = len(Items) + i`, but agent i's one-hot index is
`len(Items) - 1 + i`. Every agent therefore reads the NEXT agent's channel, sees
itself in the "other agent" channel instead, and only the last agent is correct (its
out-of-bounds index clamps back onto itself). This file indexes the agent's own
channel. That makes observations here NOT bit-comparable with the two-player Coin
Game, which is the right trade for a new environment but has to be said out loud.
"""
from enum import IntEnum
from functools import partial
from typing import Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import numpy as onp
from flax.struct import dataclass

from socialjax.environments.multi_agent_env import MultiAgentEnv
from socialjax.environments import spaces
from socialjax.environments.common_harvest.rendering import (
    fill_coords, point_in_circle, point_in_rect, point_in_triangle, rotate_fn,
)


@dataclass
class State:
    agent_locs: jnp.ndarray      # (N, 3) row, col, direction
    #: (ROW, COL) int8. The owner of the coin in each cell, -1 where there is none.
    #: Separate from `grid` so the item vocabulary does not grow with N -- see the
    #: module docstring.
    coin_owner: jnp.ndarray
    inner_t: int
    outer_t: int
    grid: jnp.ndarray            # (ROW, COL) int16


@chex.dataclass
class EnvParams:
    payoff: chex.ArrayDevice


class Actions(IntEnum):
    turn_left = 0
    turn_right = 1
    left = 2
    right = 3
    up = 4
    down = 5
    stay = 6


class Items(IntEnum):
    """Cell contents.

    `coin` is the only coin code ever written to `state.grid`; who owns it is in
    `state.coin_owner`. `other_coin` appears ONLY in the per-agent view grid built
    for observations, where a coin belonging to somebody else is recoded to it. That
    is what keeps the observation width independent of N.
    """
    empty = 0
    wall = 1
    interact = 2
    coin = 3
    other_coin = 4


char_to_int = {'W': 1, ' ': 0, 'C': 3, 'P': 4}

ROTATIONS = jnp.array(
    [[0, 0, 1],    # turn left
     [0, 0, -1],   # turn right
     [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
    dtype=jnp.int8,
)

STEP_MOVE = jnp.array(
    [[0, 0, 0], [0, 0, 0],
     [0, 1, 0],    # left
     [0, -1, 0],   # right
     [1, 0, 0],    # up
     [-1, 0, 0],   # down
     [0, 0, 0]],
    dtype=jnp.int8,
)

#: 19 x 28 with seven player spawns -- Clean Up's grid exactly, so crowding is the
#: same in both 7-agent environments and a cross-environment comparison is not
#: confounded by how much room the agents had. Every non-spawn cell can carry a coin.
DEFAULT_MAP = [
    "CCCCCCCCCCCCCCCCCCCCCCCCCCCC",
    "CCCCCCCCCCCCCCCCCCCCCCCCCCCC",
    "CCCCPCCCCCCCCCCCCCCCCCCPCCCC",
    "CCCCCCCCCCCCCCCCCCCCCCCCCCCC",
    "CCCCCCCCCCCCCCCCCCCCCCCCCCCC",
    "CCCCCCCCCCCCCCCCCCCCCCCCCCCC",
    "CCCCCCCCCCCCCCCCCCCCCCCCCCCC",
    "CCCCCCCCCCCPCCCCCCCCCCCCCCCC",
    "CCCCCCCCCCCCCCCCCCCCCCCCCCCC",
    "CCPCCCCCCCCCCCCCCCCCCCCCPCCC",
    "CCCCCCCCCCCCCCCCCCCCCCCCCCCC",
    "CCCCCCCCCCCCCCCCCCCCCCCCCCCC",
    "CCCCCCCCCCCCCCCCCCCCCCCCCCCC",
    "CCCCCCCCCCCCCCCCCCCCCCCCCCCC",
    "CCCCCCCCCCCCCCCCCCCCCCCCCCCC",
    "CCCCCCCPCCCCCCCCCCCCPCCCCCCC",
    "CCCCCCCCCCCCCCCCCCCCCCCCCCCC",
    "CCCCCCCCCCCCCCCCCCCCCCCCCCCC",
    "CCCCCCCCCCCCCCCCCCCCCCCCCCCC",
]


def ascii_map_to_matrix(map_ascii, mapping):
    height = len(map_ascii)
    width = max(len(row) for row in map_ascii)
    matrix = onp.zeros((height, width), dtype=onp.int32)
    for i, row in enumerate(map_ascii):
        for j, char in enumerate(row):
            matrix[i, j] = mapping.get(char, 0)
    return jnp.asarray(matrix)


def generate_agent_colors(num_agents):
    """One hue per agent, evenly spaced. A coin is drawn in its owner's colour, so
    the palette is shared between agents and coins by construction."""
    import colorsys
    return [tuple(int(x * 255) for x in colorsys.hsv_to_rgb(i / num_agents, 0.8, 0.8))
            for i in range(num_agents)]


class CoinGameN(MultiAgentEnv):
    """N-player Coin Game. See the module docstring for the design."""

    def __init__(
        self,
        num_inner_steps: int = 1000,
        num_outer_steps: int = 1,
        num_agents: int = 7,
        shared_rewards: bool = False,
        #: (own, other, penalty) -- what a coin pays its taker when it is the taker's
        #: own colour, what it pays when it is not, and what it costs the owner of a
        #: coin somebody else took. The two-player game's payoff_matrix rows verbatim;
        #: a single row here because all agents are symmetric.
        payoff: Tuple[float, float, float] = (1.0, 1.0, -2.0),
        #: Per empty spawn cell per step, for ONE pass that then draws the new coin's
        #: owner uniformly. Chosen so per-agent income under full cooperation matches
        #: the two-player game rather than by tuning: that game spawns
        #: 174 cells x 0.0005 x 2 colours = 0.174 coins/step for 2 agents = 0.087 per
        #: agent, and 525 cells x 0.0012 / 7 agents = 0.09. The steady-state coin
        #: density on the grid comes out at ~9% in both.
        regrow_rate: float = 0.0012,
        coin_reward: Optional[float] = None,
        jit: bool = True,
        obs_size: int = 11,
        cnn: bool = True,
        map_ASCII=None,
        # Reward-shaping variants the two-player environment carries. They are not
        # ported: each rebuilds the reward vector itself, and a half-ported one would
        # be wrong in a way no test here would catch. Refused rather than ignored.
        inequity_aversion: bool = False,
        svo: bool = False,
        interest: bool = False,
        enable_smooth_rewards: bool = False,
    ):
        super().__init__(num_agents=num_agents)
        if num_agents < 2:
            raise ValueError(
                f"the Coin Game needs >= 2 agents (a coin needs an owner and a "
                f"thief), got {num_agents}")
        for flag, name in ((inequity_aversion, "inequity_aversion"), (svo, "svo"),
                           (interest, "interest"),
                           (enable_smooth_rewards, "enable_smooth_rewards")):
            if flag:
                raise NotImplementedError(
                    f"{name} is not implemented for coin_game_n. The two-player "
                    f"environment rebuilds the reward vector separately in each of "
                    f"these branches; porting them blind would produce a variant that "
                    f"runs and is wrong. Use socialjax.make('coin_game') for those, "
                    f"or implement the branch here deliberately.")

        self.agents = list(range(num_agents))
        self._agents = jnp.arange(num_agents, dtype=jnp.int16) + len(Items)
        self.num_inner_steps = num_inner_steps
        self.num_outer_steps = num_outer_steps
        self.shared_rewards = shared_rewards
        self.payoff = tuple(float(x) for x in payoff)
        self.regrow_rate = float(regrow_rate)
        self.cnn = cnn
        # Defaults to num_agents, exactly as clean_up's apple_reward and the
        # two-player coin_reward do, so that the individual- and shared-reward arms
        # carry equal total reward mass. Every contract range is quoted against a UNIT
        # reward, so a contracting run must set 1.0 -- algorithms/MOCA/envs.py
        # check_reward_scale is what says so at run start.
        self.coin_reward = float(num_agents if coin_reward is None else coin_reward)
        self.PLAYER_COLOURS = generate_agent_colors(num_agents)

        map_ASCII = DEFAULT_MAP if map_ASCII is None else map_ASCII
        nums_map = ascii_map_to_matrix(map_ASCII, char_to_int)
        self.GRID_SIZE_ROW = int(nums_map.shape[0])
        self.GRID_SIZE_COL = int(nums_map.shape[1])
        self.OBS_SIZE = obs_size
        self.PADDING = self.OBS_SIZE - 1
        self.SPAWNS_COIN = jnp.array(onp.argwhere(onp.asarray(nums_map) == 3))
        self.SPAWNS_PLAYERS = jnp.array(onp.argwhere(onp.asarray(nums_map) == 4))
        if len(self.SPAWNS_PLAYERS) < num_agents:
            raise ValueError(
                f"the map has {len(self.SPAWNS_PLAYERS)} player spawn points ('P') "
                f"but {num_agents} agents. Agents spawn on distinct cells, so the map "
                f"needs at least one 'P' each.")

        num_classes = num_agents + len(Items) - 1

        # ------------------------------------------------------------ movement
        def check_collision(locs):
            matcher = jax.vmap(lambda a, b: jnp.all(a[:2] == b[:2]), in_axes=(0, None))
            return jax.vmap(matcher, in_axes=(None, 0))(locs, locs)

        def fix_collisions(key, collided_moved, collision_matrix, old_locs, new_locs):
            """Agents that would share a cell are pushed back, one collision per scan
            step. Copied in structure from the two-player environment so movement is
            the same game; only the reward and observation logic differs here."""
            def one(state, _):
                key, collided_moved, collision_matrix, old_locs, new_locs = state
                return jax.lax.cond(
                    collided_moved[jnp.argmax(collided_moved)] > 0,
                    lambda: _fix(key, collided_moved, collision_matrix, old_locs,
                                 new_locs),
                    lambda: (state, new_locs),
                )

            _, ys = jax.lax.scan(
                one, (key, collided_moved, collision_matrix, old_locs, new_locs),
                jnp.arange(num_agents))
            return ys[-1]

        def _fix(key, collided_moved, collision_matrix, old_locs, new_locs):
            def random_true(key, arr):
                cumsum = jnp.cumsum(arr)
                idx = jax.random.randint(key, (1,), 0, jnp.maximum(cumsum[-1], 1))
                return jnp.argmax(cumsum > idx)

            colliders_idx = jnp.argmax(collided_moved)
            collisions = collision_matrix[colliders_idx]
            subjects = jnp.where(collisions, collided_moved, collisions)
            mask = collisions == subjects
            stayed = jnp.all(mask)
            stayed_mask = jnp.logical_and(~stayed, ~mask)
            stayed_idx = jnp.where(jnp.max(stayed_mask) > 0, jnp.argmax(stayed_mask), 0)

            k1, k2 = jax.random.split(key, 2)
            rand_idx = random_true(k1, collisions)
            revert_rand = collisions.at[rand_idx].set(False)
            locs_rand = jax.vmap(lambda p, o, n: jnp.where(p, o, n))(
                revert_rand, old_locs, new_locs)
            revert_stayed = jax.lax.select(
                jnp.max(stayed_mask) > 0, collisions.at[stayed_idx].set(False),
                revert_rand)
            locs_stayed = jax.vmap(lambda p, o, n: jnp.where(p, o, n))(
                revert_stayed, old_locs, new_locs)

            new_locs = jnp.where(stayed, locs_rand, locs_stayed)
            collided_moved = jnp.clip(collided_moved - collisions, 0, 1)
            collision_matrix = collision_matrix.at[colliders_idx].set(
                jnp.zeros_like(collisions, dtype=bool))
            return ((k2, collided_moved, collision_matrix, old_locs, new_locs),
                    new_locs)

        # --------------------------------------------------------- observation
        def _get_obs_point(agent_loc):
            x, y, direction = agent_loc
            x, y = x + self.PADDING, y + self.PADDING
            x = x - (self.OBS_SIZE // 2)
            y = y - (self.OBS_SIZE // 2)
            x = jnp.where(direction == 0, x + (self.OBS_SIZE // 2) - 1, x)
            y = jnp.where(direction == 1, y + (self.OBS_SIZE // 2) - 1, y)
            x = jnp.where(direction == 2, x - (self.OBS_SIZE // 2) + 1, x)
            y = jnp.where(direction == 3, y - (self.OBS_SIZE // 2) + 1, y)
            return x, y

        def rotate_grid(agent_loc, grid):
            for k in (1, 2, 3):
                grid = jnp.where(agent_loc[2] == k,
                                 jnp.rot90(grid, k=k, axes=(0, 1)), grid)
            return grid

        def relative_orientation(agent_idx, agent_locs, view):
            """(OBS, OBS) int: -1 where no OTHER agent stands, else that agent's
            facing relative to this one."""
            others = jnp.delete(self._agents, agent_idx, assume_unique_indices=True)
            my_dir = agent_locs[agent_idx, 2]
            idx = jnp.clip(view - len(Items), 0, num_agents - 1)
            rel = (agent_locs[idx, 2] - my_dir) % 4
            return jnp.where(jnp.isin(view, others), rel, -1)

        def combine_channels(onehot, agent_idx, angle):
            """(OBS, OBS, N+4) one-hot -> (OBS, OBS, 10) egocentric features.

            Channels: wall, interact, MY coin, OTHER's coin, self, another agent,
            and that agent's relative facing as a 4-way one-hot. Width is independent
            of N because both the coin colours and the agent identities have already
            been collapsed to self/other.
            """
            def cell(x, ang):
                # x is the one-hot of (code - 1), so slot c-1 holds grid code c and
                # agent i -- whose code is len(Items) + i -- is at slot
                # len(Items) - 1 + i. Indexing it with len(Items) + i is the
                # off-by-one in the shipped environments; see the module docstring.
                items = x[:len(Items) - 1]
                agents = x[len(Items) - 1:]
                me = agents[agent_idx].astype(jnp.int8)
                other = jnp.logical_and(jnp.any(agents > 0), me == 0).astype(jnp.int8)
                return jnp.concatenate(
                    [items, me[None], other[None], ang.astype(jnp.int8)], axis=-1)

            return jax.vmap(jax.vmap(cell))(onehot, angle)

        def _get_obs(state):
            pad = self.PADDING
            grid = jnp.pad(state.grid, ((pad, pad), (pad, pad)),
                           constant_values=Items.wall)
            owner = jnp.pad(state.coin_owner, ((pad, pad), (pad, pad)),
                            constant_values=-1)

            # Egocentric in COLOUR: another agent's coin is recoded to `other_coin`,
            # so every agent reads its own colour in the same channel.
            def view_for(i):
                theirs = (grid == Items.coin) & (owner != i)
                return jnp.where(theirs, jnp.int16(Items.other_coin), grid)

            ids = jnp.arange(num_agents)
            views = jax.vmap(view_for)(ids)
            xs, ys = jax.vmap(_get_obs_point)(state.agent_locs)
            grids = jax.vmap(
                lambda v, x, y: jax.lax.dynamic_slice(
                    v, (x, y), (self.OBS_SIZE, self.OBS_SIZE))
            )(views, xs, ys)
            grids = jax.vmap(rotate_grid)(state.agent_locs, grids)
            angles = jax.vmap(relative_orientation, in_axes=(0, None, 0))(
                ids, state.agent_locs, grids)
            angles = jax.nn.one_hot(angles, 4, dtype=jnp.int8)
            onehot = jax.nn.one_hot(grids - 1, num_classes, dtype=jnp.int8)
            return jax.vmap(combine_channels)(onehot, ids, angles)

        self._get_obs_fn = _get_obs

        # --------------------------------------------------------------- step
        def _step(key, state, actions, timestep: int = 0):
            # `timestep` is unused here but is part of the MultiAgentEnv.step_env
            # signature, which passes it positionally.
            actions = jnp.array(actions)
            key, k_spawn, k_owner, k_coll = jax.random.split(key, 4)

            # 1. Spawn. ONE pass over the spawn cells; the owner of a new coin is
            #    drawn uniformly, so the per-colour rate is regrow_rate / N with no
            #    extra pass per colour. Spawn cells are distinct, so .set() here is
            #    unambiguous (duplicate-index .set() is order-undefined in JAX).
            cells = self.SPAWNS_COIN
            here = state.grid[cells[:, 0], cells[:, 1]]
            draw = jax.random.uniform(k_spawn, (cells.shape[0],)) < self.regrow_rate
            spawn = (here == Items.empty) & draw
            owners = jax.random.randint(k_owner, (cells.shape[0],), 0, num_agents)
            grid = state.grid.at[cells[:, 0], cells[:, 1]].set(
                jnp.where(spawn, jnp.int16(Items.coin), here))
            was = state.coin_owner[cells[:, 0], cells[:, 1]]
            coin_owner = state.coin_owner.at[cells[:, 0], cells[:, 1]].set(
                jnp.where(spawn, owners.astype(jnp.int8), was))

            # 2. Move, then resolve any two agents claiming one cell.
            new = jax.vmap(lambda p, a: jnp.int16(p + ROTATIONS[a]) % jnp.array(
                [self.GRID_SIZE_ROW + 1, self.GRID_SIZE_COL + 1, 4], jnp.int16)
            )(state.agent_locs, actions)
            moved = ((actions == Actions.up) | (actions == Actions.down)
                     | (actions == Actions.right) | (actions == Actions.left))
            new = jax.vmap(lambda m, n, a: jnp.where(m, n + STEP_MOVE[a], n))(
                moved, new, actions)
            new = jax.vmap(jnp.clip, in_axes=(0, None, None))(
                new, jnp.array([0, 0, 0], jnp.int16),
                jnp.array([self.GRID_SIZE_ROW - 1, self.GRID_SIZE_COL - 1, 3],
                          jnp.int16))
            really_moved = jax.vmap(lambda n, p: jnp.any(n[:2] != p[:2]))(
                new, state.agent_locs)
            collision_matrix = check_collision(new)
            collisions = jnp.minimum(
                jnp.sum(collision_matrix, axis=-1, dtype=jnp.int8) - 1, 1)
            collided_moved = jnp.maximum(collisions - ~really_moved, 0)
            new_locs = jax.lax.cond(
                jnp.max(collided_moved) > 0,
                lambda: fix_collisions(k_coll, collided_moved, collision_matrix,
                                       state.agent_locs, new),
                lambda: new)

            # 3. Pick up. Agents occupy distinct cells after collision resolution, so
            #    each coin is taken by at most one agent and the two sides of a theft
            #    are counted exactly once each -- which is what makes the contract's
            #    transfer zero-sum.
            cell = grid[new_locs[:, 0], new_locs[:, 1]]
            owner_here = coin_owner[new_locs[:, 0], new_locs[:, 1]]
            me = jnp.arange(num_agents, dtype=jnp.int8)
            took = cell == Items.coin
            own = took & (owner_here == me)
            theft = took & (owner_here != me)
            theft_f = theft.astype(jnp.float32)
            # Scatter-ADD, never .set(): several agents can steal from the SAME owner
            # in one step, and .at[].set() with duplicate indices is order-undefined
            # in JAX (the mechanism of the e2e799e cleaning bug). add is commutative.
            stolen_from = jnp.zeros((num_agents,), jnp.float32).at[
                jnp.where(theft, owner_here, 0)].add(theft_f)

            own_reward, other_reward, penalty = self.payoff
            individual = (own.astype(jnp.float32) * own_reward
                          + theft_f * other_reward
                          + stolen_from * penalty) * self.coin_reward

            # 4. Write the grid back. Clear every OLD agent cell first, because an
            #    agent may have moved into one another just vacated; then place the
            #    agents, which also consumes whatever coin was under them.
            grid = grid.at[state.agent_locs[:, 0], state.agent_locs[:, 1]].set(
                jnp.int16(Items.empty))
            coin_owner = coin_owner.at[new_locs[:, 0], new_locs[:, 1]].set(
                jnp.int8(-1))
            grid = grid.at[new_locs[:, 0], new_locs[:, 1]].set(self._agents)

            rewards = (jnp.full((num_agents,), jnp.sum(individual))
                       if self.shared_rewards else individual)

            info = {
                # Per-agent either way, unlike the two-player environment where
                # `shaped_rewards` carries the SUM broadcast to everyone. Under
                # shared_rewards the two differ; under the individual rewards every
                # contracting run uses, they are the same number.
                "original_rewards": individual,
                "shaped_rewards": rewards,
                # The cooperative act, and the two sides of the harmful one. The
                # contract reads stolen_by_agent / stolen_from_agent; coins_taken is
                # the denominator that separates "stopped stealing" from "stopped
                # collecting".
                "eat_own_coins": own.astype(jnp.float32),
                "stolen_by_agent": theft_f,
                "stolen_from_agent": stolen_from,
                "coins_taken": took.astype(jnp.float32),
                # Grid-wide, broadcast per agent so it survives the per-agent info
                # slicing the training loops do -- the same convention clean_up's
                # waste_cleared and harvest's apple_stock use. The two-player
                # environment has no such uniform series, which is why its commons
                # metric reads only agent 0's value.
                "coins_on_grid": jnp.full(
                    (num_agents,), jnp.sum(grid == Items.coin), jnp.float32),
            }

            state_nxt = State(agent_locs=new_locs, coin_owner=coin_owner,
                              inner_t=state.inner_t + 1, outer_t=state.outer_t,
                              grid=grid)
            reset_inner = state_nxt.inner_t == num_inner_steps
            state_re = _reset_state(key).replace(outer_t=state_nxt.outer_t + 1)
            state = jax.tree.map(
                lambda x, y: jnp.where(reset_inner, x, y), state_re, state_nxt)
            reset_outer = state.outer_t == num_outer_steps
            done = {f'{a}': reset_outer for a in self.agents}
            done["__all__"] = reset_outer
            rewards = jnp.where(reset_inner, jnp.zeros_like(rewards), rewards)
            return _get_obs(state), state, rewards, done, info

        def _reset_state(key):
            k_pos, k_dir = jax.random.split(key)
            grid = jnp.zeros((self.GRID_SIZE_ROW, self.GRID_SIZE_COL), jnp.int16)
            coin_owner = jnp.full(
                (self.GRID_SIZE_ROW, self.GRID_SIZE_COL), -1, jnp.int8)
            spawn = jax.random.permutation(k_pos, self.SPAWNS_PLAYERS)[:num_agents]
            # maxval=4, so every facing is reachable. The two-player environment uses
            # maxval=3 and never spawns an agent facing direction 3.
            facing = jax.random.randint(k_dir, (num_agents,), 0, 4)
            agent_locs = jnp.stack(
                [spawn[:, 0], spawn[:, 1], facing], axis=-1).astype(jnp.int16)
            grid = grid.at[agent_locs[:, 0], agent_locs[:, 1]].set(self._agents)
            return State(agent_locs=agent_locs, coin_owner=coin_owner,
                         inner_t=0, outer_t=0, grid=grid)

        def reset(key):
            state = _reset_state(key)
            return _get_obs(state), state

        if jit:
            self.step_env = jax.jit(_step)
            self.reset = jax.jit(reset)
            self.get_obs_point = jax.jit(_get_obs_point)
        else:
            self.step_env = _step
            self.reset = reset
            self.get_obs_point = _get_obs_point

    # ------------------------------------------------------------------ spaces
    @property
    def name(self) -> str:
        return "CoinGameN"

    @property
    def num_actions(self) -> int:
        return len(Actions)

    def action_space(self, agent_id: Union[int, None] = None) -> spaces.Discrete:
        return spaces.Discrete(len(Actions))

    def observation_space(self):
        """(OBS, OBS, 10) for every N: wall, interact, my coin, other's coin, self,
        another agent, and that agent's relative facing one-hot."""
        channels = (len(Items) - 1) + 6
        shape = ((self.OBS_SIZE, self.OBS_SIZE, channels) if self.cnn
                 else (self.OBS_SIZE ** 2 * channels,))
        return spaces.Box(low=0, high=1E9, shape=shape, dtype=jnp.uint8), shape

    def state_space(self) -> spaces.Box:
        shape = (self.GRID_SIZE_ROW, self.GRID_SIZE_COL, len(Items) + 1)
        return spaces.Box(low=0, high=1, shape=shape, dtype=jnp.uint8)

    # ------------------------------------------------------------------ render
    def render_tile(self, obj, owner: int = -1, agent_dir: Optional[int] = None,
                    tile_size: int = 32):
        """One cell. A coin is drawn as a disc in ITS OWNER's colour, so who a coin
        belongs to is legible in a still frame -- with N colours a shared palette
        between agents and coins is the only way to read a recording."""
        img = onp.full((tile_size, tile_size, 3), 255, dtype=onp.uint8)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))
        code = int(obj)
        if code == Items.wall:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (127, 127, 127))
        elif code == Items.coin:
            colour = (self.PLAYER_COLOURS[owner] if 0 <= owner < self.num_agents
                      else (180, 180, 180))
            fill_coords(img, point_in_circle(0.5, 0.5, 0.31), colour)
        elif code >= len(Items):
            tri = point_in_triangle((0.12, 0.19), (0.87, 0.50), (0.12, 0.81))
            if agent_dir is not None:
                tri = rotate_fn(tri, cx=0.5, cy=0.5,
                                theta=0.5 * onp.pi * (1 - agent_dir))
            fill_coords(img, tri, self.PLAYER_COLOURS[code - len(Items)])
        return img

    def render(self, state: State, tile_size: int = 32):
        grid = onp.asarray(state.grid)
        owner = onp.asarray(state.coin_owner)
        locs = onp.asarray(state.agent_locs)
        rows, cols = grid.shape
        img = onp.zeros((rows * tile_size, cols * tile_size, 3), dtype=onp.uint8)
        for r in range(rows):
            for c in range(cols):
                code = int(grid[r, c])
                if code >= len(Items):
                    tile = self.render_tile(code, owner=code - len(Items),
                                            agent_dir=int(locs[code - len(Items), 2]),
                                            tile_size=tile_size)
                else:
                    tile = self.render_tile(code, owner=int(owner[r, c]),
                                            tile_size=tile_size)
                img[r * tile_size:(r + 1) * tile_size,
                    c * tile_size:(c + 1) * tile_size] = tile
        return img
