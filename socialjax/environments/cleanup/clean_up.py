from enum import IntEnum
import math
from typing import Any, Optional, Tuple, Union, Dict
from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as onp
from flax.struct import dataclass
import colorsys

from socialjax.environments.multi_agent_env import MultiAgentEnv
from socialjax.environments import spaces


from socialjax.environments.cleanup.rendering import (
    downsample,
    fill_coords,
    highlight_img,
    point_in_circle,
    point_in_rect,
    point_in_triangle,
    rotate_fn,
)

NUM_TYPES = 4  # empty (0), red (1), blue, red coin, blue coin, wall, interact
NUM_COIN_TYPES = 1
INTERACT_THRESHOLD = 0


@dataclass
class State:
    agent_locs: jnp.ndarray
    agent_invs: jnp.ndarray
    inner_t: int
    outer_t: int
    grid: jnp.ndarray

    apples: jnp.ndarray
    freeze: jnp.ndarray
    reborn_locs: jnp.ndarray

    potential_dirt_and_dirt_locs: jnp.ndarray
    potential_dirt_and_dirt_label: jnp.ndarray
    smooth_rewards: jnp.ndarray

    # Monetary system (pay_mode != "off"): per-agent spendable balance (cumulative
    # net reward -- apples earned +/- payments) and the inner_t at which each agent
    # last actually cleaned a dirt patch (init far-negative so none are payable at
    # reset). Present always (cheap) so State stays a fixed shape regardless of mode.
    agent_balance: jnp.ndarray
    last_clean_t: jnp.ndarray
    # pay_scheme="tithe" only: inner_t until which each agent's share-mode pledge is
    # active (exclusive). 0 at reset = inactive, since "active" means inner_t < expiry.
    share_expiry_t: jnp.ndarray
    # toggle_cooldown > 0 only: inner_t at which each agent last flipped its share
    # toggle, so flips can be rate-limited. Far-negative at reset = free to flip.
    last_toggle_t: jnp.ndarray


@chex.dataclass
class EnvParams:
    payoff_matrix: chex.ArrayDevice
    freeze_penalty: int

class Actions(IntEnum):
    turn_left = 0
    turn_right = 1
    left = 2
    right = 3
    up = 4
    down = 5
    stay = 6
    zap_forward = 7
    zap_clean = 8
    pay = 9  # pay_scheme="instant": transfer pay_amount to the most recent cleaner.
             # pay_scheme="tithe": pledge share-mode for share_duration steps, during
             # which share_fraction of each apple harvested auto-flows to the most
             # recent cleaner. (Action only exists when pay_mode != "off".)


class Items(IntEnum):
    empty = 0
    wall = 1
    interact = 2
    apple = 3
    spawn_point = 4
    inside_spawn_point = 5
    river = 6
    potential_dirt = 7
    dirt = 8
    clean_beam = 9


ROTATIONS = jnp.array(
    [
        [0, 0, 1],  # turn left
        [0, 0, -1],  # turn right
        [0, 0, 0],  # left
        [0, 0, 0],  # right
        [0, 0, 0],  # up
        [0, 0, 0],  # down
        [0, 0, 0],  # stay
        [0, 0, 0],  # zap
        [0, 0, 0],  # zap_clean
        [0, 0, 0],  # pay
    ],
    dtype=jnp.int8,
)

STEP = jnp.array(
    [
        [1, 0, 0],  # up
        [0, 1, 0],  # right
        [-1, 0, 0],  # down
        [0, -1, 0],  # left
    ],
    dtype=jnp.int8,
)

STEP_MOVE = jnp.array(
    [
        [0, 0, 0],  # turn_left
        [0, 0, 0],  # turn_right
        [0, 1, 0],  # left
        [0, -1, 0],  # right
        [1, 0, 0],  # up
        [-1, 0, 0],  # down
        [0, 0, 0],  # stay
        [0, 0, 0],  # zap_forward
        [0, 0, 0],  # zap_clean (previously relied on JAX index-clamping)
        [0, 0, 0],  # pay
    ],
    dtype=jnp.int8,
)

def compute_pay_transfers(
    actions: jnp.ndarray,
    last_clean_t: jnp.ndarray,
    current_t: int,
    balance: jnp.ndarray,
    clean_window: int,
    pay_amount: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Resolve the `pay` action into a zero-sum per-agent reward delta (Option B:
    payment goes to whoever recently cleaned the river, regardless of distance).

    An agent that takes Actions.pay sends `pay_amount` to the *most recent other
    agent that cleaned a dirt patch within the last `clean_window` steps*. The
    payment is a no-op (attempted but not executed) if the payer can't afford it
    (balance < pay_amount) or if no other agent has cleaned recently. This links
    the subsidy to cleaning -- the behaviour we want to reward -- rather than to
    proximity, and decouples payer/receiver location so a harvester in the orchard
    can pay a cleaner working the distant river.

    Kept at module level (rather than inside Clean_up.__init__'s closure) so the
    accounting can be unit-tested in isolation. Pure jnp ops only: jit/vmap-safe.

    Args:
        actions: (N,) int array of this step's actions.
        last_clean_t: (N,) int, inner_t at which each agent last cleaned dirt
            (far-negative if never), so age = current_t - last_clean_t.
        current_t: this step's inner_t counter.
        balance: (N,) float, each agent's spendable balance (cumulative net reward).
        clean_window: max age (in steps) at which a cleaner is still payable.
        pay_amount: reward units moved per executed pay action.

    Returns:
        (delta, attempted, executed, target):
            delta: (N,) float32 zero-sum reward adjustment (sender -, receiver +).
            attempted: (N,) bool, agent chose Actions.pay.
            executed: (N,) bool, pay had a valid recent-cleaner recipient AND the
                payer could afford it.
            target: (N,) int, recipient index per agent (the chosen recent cleaner);
                only meaningful where executed=True (used for the viewer's arrows).
    """
    attempted = actions == Actions.pay
    target, has_recipient = _recent_cleaner_recipient(last_clean_t, current_t, clean_window)
    can_afford = balance >= pay_amount
    executed = attempted & has_recipient & can_afford

    n = actions.shape[0]
    sent = jnp.where(executed, jnp.float32(pay_amount), 0.0)
    delta = jnp.zeros((n,), dtype=jnp.float32)
    delta = delta.at[jnp.arange(n)].add(-sent)  # payers pay...
    delta = delta.at[target].add(sent)          # ...their chosen recent cleaner receives
    return delta, attempted, executed, target


def clip_beam_targets(
    targets: jnp.ndarray, n_rows: int, n_cols: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Clamp beam target cells into the grid and report which were already inside.

    Beam cells are built by adding a direction offset to an agent's position, so an
    agent standing at an edge and facing outward produces targets outside the grid.
    Indexing a jnp array with those is silently WRONG rather than an error: negative
    indices wrap around Python-style (a beam fired off the top of the map reappears at
    the bottom) and too-large indices clamp to the last row/column (a beam fired off
    the bottom lands back on the firing agent's own cell). Both let a zap "hit"
    something it is nowhere near -- in particular an agent could stun ITSELF by facing
    a wall, and respawn at a random mid-map spawn point, which is a fast-travel exploit
    rather than a game mechanic.

    Returns (clamped_targets, valid) where `valid` is False for any cell that fell
    outside the grid. Callers must use `clamped_targets` for indexing (so the gather is
    in-range) and mask any hit/draw with `valid` (so out-of-grid cells never register).
    """
    rows = targets[:, 0]
    cols = targets[:, 1]
    valid = (rows >= 0) & (rows < n_rows) & (cols >= 0) & (cols < n_cols)
    clamped = targets.at[:, 0].set(jnp.clip(rows, 0, n_rows - 1))
    clamped = clamped.at[:, 1].set(jnp.clip(cols, 0, n_cols - 1))
    return clamped, valid


def _recent_cleaner_recipient(
    last_clean_t: jnp.ndarray, current_t: int, clean_window: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Option B recipient rule, shared by both pay schemes: for each would-be payer,
    the most recent OTHER agent that cleaned dirt within `clean_window` steps.

    Returns (target, has_recipient): (N,) recipient index per agent (only meaningful
    where has_recipient=True) and (N,) bool whether any valid other recent-cleaner
    exists. Self is masked out per-row -- an agent is never its own recipient.
    """
    n = last_clean_t.shape[0]
    # A cleaner is payable if it cleaned within the window. Recency is the sort key
    # for "most recent"; recent_key masks out non-recent agents with a floor below
    # any valid last_clean_t so they never win the argmax.
    age = current_t - last_clean_t
    recent = age <= clean_window
    FLOOR = jnp.int32(-1_000_000_000)
    recent_key = jnp.where(recent, last_clean_t.astype(jnp.int32), FLOOR)

    not_self = ~jnp.eye(n, dtype=bool)
    key_matrix = jnp.where(not_self, recent_key[None, :], FLOOR)  # row = payer, col = candidate
    target = jnp.argmax(key_matrix, axis=1)
    has_recipient = key_matrix[jnp.arange(n), target] > FLOOR
    return target, has_recipient


def _recent_cleaner_weights(
    last_clean_t: jnp.ndarray, current_t: int, clean_window: int, split: bool
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Per-payer distribution over recipient cleaners for the tithe.

    Returns (weights, has_recipient): weights is (N, N) float32 where weights[i, j] is
    the fraction of payer i's shared slice that flows to agent j (rows sum to 1 where a
    recipient exists, else 0); has_recipient is (N,) bool. Self is excluded per-row --
    an agent never tithes to itself.

    split=False ("latest", the original Option B winner-take-all rule): all mass on the
        single most-recent OTHER cleaner within the window.
    split=True: mass divided EQUALLY among every OTHER agent that cleaned within the
        window, so a harvester's tithe is spread across all current cleaners instead of
        concentrated on the freshest one. Motivation: winner-take-all structurally
        pushes toward a single specialist cleaner; splitting can support the multiple
        simultaneous cleaners the apple ecology needs.

    `split` is a static Python bool (from Clean_up.split_recipients), so the branch
    specializes at trace time -- jit/vmap-safe.
    """
    n = last_clean_t.shape[0]
    age = current_t - last_clean_t
    recent = age <= clean_window
    not_self = ~jnp.eye(n, dtype=bool)
    eligible = not_self & recent[None, :]  # [i, j]: j is a valid recipient for payer i
    has_recipient = jnp.any(eligible, axis=1)
    if split:
        counts = jnp.sum(eligible, axis=1, keepdims=True).astype(jnp.float32)
        weights = jnp.where(eligible, 1.0 / jnp.maximum(counts, 1.0), 0.0)
    else:
        FLOOR = jnp.int32(-1_000_000_000)
        key = jnp.where(eligible, last_clean_t.astype(jnp.int32)[None, :], FLOOR)
        target = jnp.argmax(key, axis=1)
        weights = jax.nn.one_hot(target, n, dtype=jnp.float32) * has_recipient[:, None].astype(jnp.float32)
    return weights.astype(jnp.float32), has_recipient


def compute_tithe_transfers(
    actions: jnp.ndarray,
    last_clean_t: jnp.ndarray,
    current_t: int,
    share_expiry_t: jnp.ndarray,
    income: jnp.ndarray,
    clean_window: int,
    share_fraction: float,
    share_duration: int,
    split_recipients: bool = False,
    last_toggle_t: jnp.ndarray = None,
    toggle_cooldown: int = 0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Resolve the pledge-style income-coupled tithe (pay_scheme="tithe").

    Actions.pay is a *pledge*: it (re)activates share-mode for the next
    `share_duration` steps (effective immediately, auto-expires, re-pledge to
    extend). While share-mode is active, `share_fraction` of the agent's positive
    income this step auto-transfers to recent cleaner(s) -- either the single most
    recent one (split_recipients=False) or split equally across all recent cleaners
    (split_recipients=True), per _recent_cleaner_weights. No balance gate is needed:
    an agent only ever shares a slice of income it is earning this very step, so
    the net-of-transfer reward on a harvest step stays positive
    (income * (1 - share_fraction)) rather than a bare -pay_amount -- the
    credit-assignment motivation for this scheme over the instant one.

    If no valid recipient exists on a harvest step, nothing is transferred (the
    harvester keeps full income); the pledge itself is free and never blocked.

    Args:
        actions: (N,) int array of this step's actions.
        last_clean_t: (N,) int, inner_t at which each agent last cleaned dirt.
        current_t: this step's inner_t counter.
        share_expiry_t: (N,) int, inner_t until which each agent's pledge is active
            (exclusive); 0 at reset means inactive.
        income: (N,) float, this step's positive reward basis (apple income).
            Callers should clip at 0 so negative rewards are never "shared".
        clean_window: max age (in steps) at which a cleaner is a valid recipient.
        share_fraction: fraction of income transferred while share-mode is active.
        share_duration: steps a pledge stays active.
        split_recipients: if True, split each sharer's slice equally across all recent
            cleaners; if False, give it all to the single most-recent cleaner.

    Returns:
        (delta, pledged, executed, target, new_expiry, active, received):
            delta: (N,) float32 zero-sum reward adjustment (sharer -, cleaner +).
            pledged: (N,) bool, agent took the pledge action this step.
            executed: (N,) bool, a transfer actually happened this step (share-mode
                active AND positive income AND valid recipient).
            target: (N,) int, the primary recipient index (the most-recent cleaner);
                exact under split_recipients=False, and just one of the recipients
                under split (use `received` for the full picture).
            new_expiry: (N,) int32, updated share_expiry_t to store in State.
            active: (N,) bool, share-mode active this step (after pledges applied).
            received: (N,) float32, amount each agent received this step (the split
                counterpart of the per-sharer `sent` slice).
    """
    pledged = actions == Actions.pay
    if toggle_cooldown > 0:
        # Rate-limited ON/OFF toggle. Without this an agent can game the
        # income-coupled design: holding the toggle ON while idle costs nothing (the
        # tithe only takes a slice of income actually earned), so it can look like a
        # payer to everyone else and flip OFF just before harvesting -- a free-riding
        # "fake cleaner-supporter". Forcing a minimum dwell time between flips means a
        # displayed pledge has to be honoured across a whole window, so it cannot be
        # timed around a harvest.
        can_toggle = (jnp.int32(current_t) - last_toggle_t.astype(jnp.int32)) >= toggle_cooldown
        currently_on = current_t < share_expiry_t
        flip = pledged & can_toggle
        FAR = jnp.int32(1 << 30)   # "on until switched off"
        new_expiry = jnp.where(
            flip,
            jnp.where(currently_on, jnp.int32(current_t), FAR),   # on->off / off->on
            share_expiry_t.astype(jnp.int32),
        )
        new_last_toggle = jnp.where(flip, jnp.int32(current_t), last_toggle_t.astype(jnp.int32))
    else:
        # Original pledge semantics: pay (re)arms share-mode for share_duration steps.
        new_expiry = jnp.where(
            pledged, jnp.int32(current_t + share_duration), share_expiry_t.astype(jnp.int32)
        )
        new_last_toggle = (
            share_expiry_t.astype(jnp.int32) * 0 if last_toggle_t is None
            else last_toggle_t.astype(jnp.int32)
        )
    active = current_t < new_expiry

    weights, has_recipient = _recent_cleaner_weights(
        last_clean_t, current_t, clean_window, split_recipients
    )

    sent = jnp.where(
        active & has_recipient & (income > 0),
        jnp.float32(share_fraction) * income.astype(jnp.float32),
        0.0,
    )
    executed = sent > 0
    received = weights.T @ sent          # each cleaner's inflow (split across sharers)
    delta = received - sent              # zero-sum: rows of weights sum to 1 where sent>0
    target = jnp.argmax(weights, axis=1)  # primary recipient (exact under "latest")
    return delta, pledged, executed, target, new_expiry, active, received, new_last_toggle


char_to_int = {
    'W': 1,
    ' ': 0,  # empty 0
    'A': 3,  # exist apple, not used in this environment
    'P': 4,  # spawn_point
    'Q': 5,  # spawn_point defence, not used in this environment
    'B': 6, # potential_apple
    'S': 7, # river
    'H': 8, # potential_dirt
    'F': 9, # actual_dirt
    '+': 0, # should be "sand", "shadow_e", "shadow_n"
    'f': 0, # should be "sand", "shadow_e", "shadow_n"
    ";": 0,
    ",": 0,
    "^": 0,
    "=": 0,
    ">": 0,
    "<": 0,
    "~": 7,
    "T": 6,
    

}

def ascii_map_to_matrix(map_ASCII, char_to_int):
    """
    Convert ASCII map to a JAX numpy matrix using the given character mapping.
    
    Args:
    map_ASCII (list): List of strings representing the ASCII map
    char_to_int (dict): Dictionary mapping characters to integer values
    
    Returns:
    jax.numpy.ndarray: 2D matrix representation of the ASCII map
    """
    # Determine matrix dimensions
    height = len(map_ASCII)
    width = max(len(row) for row in map_ASCII)
    
    # Create matrix filled with zeros
    matrix = jnp.zeros((height, width), dtype=jnp.int32)
    
    # Fill matrix with mapped values
    for i, row in enumerate(map_ASCII):
        for j, char in enumerate(row):
            matrix = matrix.at[i, j].set(char_to_int.get(char, 0))
    
    return matrix

def generate_agent_colors(num_agents):
    colors = []
    for i in range(num_agents):
        hue = i / num_agents
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.8)  # Saturation and Value set to 0.8
        colors.append(tuple(int(x * 255) for x in rgb))
    return colors

GREEN_COLOUR = (44.0, 160.0, 44.0)
RED_COLOUR = (214.0, 39.0, 40.0)
###################################################

class Clean_up(MultiAgentEnv):

    # used for caching
    tile_cache: Dict[Tuple[Any, ...], Any] = {}

    def __init__(
        self,
        num_inner_steps=1000,
        num_outer_steps=1,
        num_agents=7,
        shared_rewards=True,
        inequity_aversion=False,
        inequity_aversion_target_agents=None,
        inequity_aversion_alpha=5,
        inequity_aversion_beta=0.05,
        svo=False,
        svo_target_agents=None,
        svo_w=0.5,
        svo_ideal_angle_degrees=45,
        enable_smooth_rewards=False,
        interest=False,
        s_interest = 0.5,
        s_interest_schedule=None,
        s_interest_change_every=30000000,
        cf=False,
        cf_alpha=1,
        # Upstream (cooperativex/SocialJax) uses 0.05. Lowered so apples are a
        # scarcer flow: the same cleaning effort now buys less harvest, which widens
        # the gap between a fouled and a maintained river without touching the
        # depletion threshold that gates growth entirely.
        maxAppleGrowthRate=0.04,
        thresholdDepletion=0.4,  # 0.4
        thresholdRestoration=0.0,
        # Upstream uses 0.5. See dirt_spawn_cells below: the pair sets the dirt rate.
        dirtSpawnProbability=1.0,
        delayStartOfDirtSpawning=50, # 50
        # --- ecology balance -------------------------------------------------
        # How many candidate cells may turn to dirt per step; expected dirt per step
        # is dirt_spawn_cells * dirtSpawnProbability.
        #
        # Upstream hard-codes 1 cell and p=0.5, i.e. 0.5 dirt/step. A clean beam
        # covers exactly 4 tiles (see cleaned_count in step_env), so ONE cleaner can
        # clear up to 4 cells/step -- an 8x surplus. One part-time cleaner sustained
        # the whole river, and no amount of tuning made cooperation necessary.
        #
        # 5 cells at p=1.0 gives 5.0 dirt/step, which is chosen against the beam
        # geometry rather than by feel:
        #   * 5 > 4  -> a single cleaner CANNOT keep up even firing every step with a
        #              perfectly placed beam. This is a hard ceiling, not a guess
        #              about duty cycle, so "one cleaner suffices" is ruled out by
        #              construction.
        #   * 5 < 8  -> two cleaners can hold the river at ~63% duty each, leaving
        #              room to reposition. Demanding but reachable, so the second
        #              cleaner is genuinely pivotal rather than the third or fourth.
        # p=1.0 fixes the COUNT per step; which cells foul is still random (the
        # candidate ordering is noise-perturbed before sorting).
        #
        # Pass dirt_spawn_cells=1, dirtSpawnProbability=0.5 to recover upstream.
        dirt_spawn_cells=5,
        # Add payment-state channels to the observation (see _get_obs). OFF by default:
        # turning it on changes the observation SHAPE, so a policy trained with it is
        # not loadable by, or comparable to, a baseline trained without it.
        observe_payment=False,
        # When True the `pay` action no longer changes an agent's pledge state, so an
        # externally imposed share pattern stays binding for the episode. Used by
        # two-phase training, whose first phase forces payments exogenously and must
        # not let the policy overwrite them.
        freeze_share_state=False,
        # tithe scheme only. 0 (default) keeps the original pledge-with-expiry
        # semantics. >0 switches the pay action to an explicit ON/OFF toggle that can
        # only be flipped every `toggle_cooldown` steps, so a displayed pledge must be
        # honoured for a whole window instead of being switched off moments before a
        # harvest (see compute_tithe_transfers).
        toggle_cooldown=0,
        # Per-step probability that a standing apple disappears uneaten. 0.0 (default)
        # reproduces upstream, where apples PERSIST forever and therefore accumulate as
        # a stock: harvesting stays profitable long after cleaning stops, which
        # decouples reward from current river state. With decay > 0 apples become a
        # flow whose equilibrium stock is growth/(growth+decay), so letting the river
        # foul visibly costs harvest within the same episode.
        appleDecayProbability=0.0,
        jit=True,

        # Monetary system: an agent's `pay` action wires reward to whoever recently
        # cleaned the river (Option B -- subsidy linked to cleaning, not proximity).
        #   "off"  -> 9 actions, no pay (original baseline)
        #   "noop" -> 10 actions, pay selectable but transfers nothing (placebo
        #             control: isolates action-space-size effects from the money itself)
        #   "on"   -> 10 actions, pay actually moves reward
        pay_mode="off",
        # What the pay action does when pay_mode != "off":
        #   "instant" -> pay moves pay_amount immediately, gated by the payer's balance
        #                (the original mechanism; a bare -pay_amount cost signal).
        #   "tithe"   -> pay is a PLEDGE: activates share-mode for share_duration steps,
        #                during which share_fraction of each apple harvested auto-flows
        #                to the most recent cleaner. Cost is income-coupled (net reward
        #                on a harvest step stays positive), which is the point: the
        #                instant scheme's certain, immediate -pay_amount is near-worst-
        #                case for policy-gradient credit assignment.
        pay_scheme="instant",
        # Reward for harvesting one apple. None keeps the ORIGINAL upstream SocialJax
        # behaviour of num_agents: in shared_rewards mode every agent receives the sum
        # over all agents (1 apple -> everyone +1 -> total mass N), so scaling the
        # individual-mode reward by N makes the total reward mass identical across the
        # two arms, keeping their value/gradient scales comparable. Set apple_reward=1.0
        # for a plain "one apple = one reward to whoever ate it" economy (what the
        # formal-contracting literature assumes).
        apple_reward=None,
        pay_amount=1.0,       # instant scheme only
        share_fraction=0.5,   # tithe scheme only: slice of income shared while pledged
        share_duration=50,    # tithe scheme only: steps a pledge stays active
        pay_clean_window=50,  # a cleaner stays payable for this many steps after cleaning
        # tithe scheme only: if True, split each sharer's slice equally across ALL recent
        # cleaners instead of giving it to the single most-recent one (winner-take-all).
        split_recipients=False,
        pay_radius=None,      # deprecated (Option A only); accepted but unused under Option B

        obs_size=11,
        cnn=True,

        map_ASCII = [
                'HFFFHFFHFHFHFHFHFHFHHFHFFFHF',
                'HFHFHFFHFHFHFHFHFHFHHFHFFFHF',
                'HFFHFFHHFHFHFHFHFHFHHFHFFFHF',
                'HFHFHFFHFHFHFHFHFHFHHFHFFFHF',
                'HFFFFFFHFHFHFHFHFHFHHFHFFFHF',
                '==============+~FHHHHHHf====',
                '   P    P      ===+~SSf     ',
                '     P     P   P  <~Sf  P   ',
                '             P   P<~S>      ',
                '   P    P         <~S>   P  ',
                '               P  <~S>P     ',
                '     P           P<~S>      ',
                '           P      <~S> P    ',
                '  P             P <~S>      ',
                '^T^T^T^T^T^T^T^T^T;~S,^T^T^T',
                'BBBBBBBBBBBBBBBBBBBssBBBBBBB',
                'BBBBBBBBBBBBBBBBBBBBBBBBBBBB',
                'BBBBBBBBBBBBBBBBBBBBBBBBBBBB',
                'BBBBBBBBBBBBBBBBBBBBBBBBBBBB',
            ]
    ):

        super().__init__(num_agents=num_agents)

        self.maxAppleGrowthRate = maxAppleGrowthRate
        self.dirt_spawn_cells = int(dirt_spawn_cells)
        self.observe_payment = bool(observe_payment)
        self.freeze_share_state = bool(freeze_share_state)
        self.toggle_cooldown = int(toggle_cooldown)
        self.appleDecayProbability = float(appleDecayProbability)
        self.thresholdDepletion = thresholdDepletion
        self.thresholdRestoration = thresholdRestoration
        self.dirtSpawnProbability = dirtSpawnProbability
        self.delayStartOfDirtSpawning = delayStartOfDirtSpawning
        self.shared_rewards = shared_rewards
        self.inequity_aversion = inequity_aversion
        self.inequity_aversion_target_agents = inequity_aversion_target_agents
        self.inequity_aversion_alpha = inequity_aversion_alpha
        self.inequity_aversion_beta = inequity_aversion_beta
        self.svo = svo
        self.svo_target_agents = svo_target_agents
        self.svo_w = svo_w
        self.svo_ideal_angle_degrees = svo_ideal_angle_degrees
        self.smooth_rewards = enable_smooth_rewards
        self.interest = interest
        self.s_interest = s_interest
        # Convert schedule to JAX array for JIT compatibility
        if s_interest_schedule is not None:
            self.s_interest_schedule = jnp.array(s_interest_schedule)
        else:
            self.s_interest_schedule = None
        self.s_interest_change_every = s_interest_change_every
        self.cnn = cnn
        self.num_inner_steps = num_inner_steps
        self.num_outer_steps = num_outer_steps
        self.cf = cf
        self.cf_alpha = cf_alpha

        if pay_mode not in ("off", "noop", "on"):
            raise ValueError(f"pay_mode must be 'off', 'noop', or 'on', got {pay_mode!r}")
        if pay_scheme not in ("instant", "tithe"):
            raise ValueError(f"pay_scheme must be 'instant' or 'tithe', got {pay_scheme!r}")
        if not (0.0 < share_fraction <= 1.0):
            raise ValueError(f"share_fraction must be in (0, 1], got {share_fraction!r}")
        self.apple_reward = float(num_agents if apple_reward is None else apple_reward)
        self.pay_mode = pay_mode
        self.pay_scheme = pay_scheme
        self.pay_amount = pay_amount
        self.share_fraction = share_fraction
        self.share_duration = share_duration
        self.pay_clean_window = pay_clean_window
        self.split_recipients = bool(split_recipients)
        self.pay_radius = (obs_size // 2) if pay_radius is None else pay_radius
        # "off" keeps the original 9-action space so old baselines stay comparable;
        # "noop"/"on" expose Actions.pay as a 10th action.
        self._num_actions = len(Actions) if pay_mode != "off" else len(Actions) - 1
        self.agents = list(range(num_agents))#, dtype=jnp.int16)
        self._agents = jnp.array(self.agents, dtype=jnp.int16) + len(Items)

        self.PLAYER_COLOURS = generate_agent_colors(num_agents)
        self.GRID_SIZE_ROW = len(map_ASCII)
        self.GRID_SIZE_COL = len(map_ASCII[0])
        self.OBS_SIZE = obs_size
        self.PADDING = self.OBS_SIZE - 1

        GRID = jnp.zeros(
            (self.GRID_SIZE_ROW + 2 * self.PADDING, self.GRID_SIZE_COL + 2 * self.PADDING),
            dtype=jnp.int16,
        )

        # First layer of padding is Wall
        GRID = GRID.at[self.PADDING - 1, :].set(5)
        GRID = GRID.at[self.GRID_SIZE_ROW + self.PADDING, :].set(5)
        GRID = GRID.at[:, self.PADDING - 1].set(5)
        self.GRID = GRID.at[:, self.GRID_SIZE_COL + self.PADDING].set(5)

        def find_positions(grid_array, letter):
            a_positions = jnp.array(jnp.where(grid_array == letter)).T
            return a_positions

        nums_map = ascii_map_to_matrix(map_ASCII, char_to_int)
        self.POTENTIAL_APPLE = find_positions(nums_map, char_to_int['B'])

        self.SPAWNS_PLAYER_IN = find_positions(nums_map, char_to_int['Q'])
        self.SPAWNS_PLAYERS = find_positions(nums_map, char_to_int['P'])
        self.SPAWNS_WALL = find_positions(nums_map, char_to_int['W'])
        self.RIVER = find_positions(nums_map, char_to_int['S'])
        self.POTENTIAL_DIRT = find_positions(nums_map, char_to_int['H'])
        self.DIRT = find_positions(nums_map, char_to_int['F'])


        
        def check_collision(
                new_agent_locs: jnp.ndarray
            ) -> jnp.ndarray:
            '''
            Function to check agent collisions.
            
            Args:
                - new_agent_locs: jnp.ndarray, the agent locations at the 
                current time step.
                
            Returns:
                - jnp.ndarray matrix of bool of agents in collision.
            '''
            matcher = jax.vmap(
                lambda x,y: jnp.all(x[:2] == y[:2]),
                in_axes=(0, None)
            )

            collisions = jax.vmap(
                matcher,
                in_axes=(None, 0)
            )(new_agent_locs, new_agent_locs)

            return collisions
        
        # first attempt at func - needs improvement
        # inefficient due to double-checking collisions
        
        def fix_collisions(
            key: jnp.ndarray,
            collided_moved: jnp.ndarray,
            collision_matrix: jnp.ndarray,
            agent_locs: jnp.ndarray,
            new_agent_locs: jnp.ndarray
        ) -> jnp.ndarray:
            """
            Function defining multi-collision logic.

            Args:
                - key: jax key for randomisation
                - collided_moved: jnp.ndarray, the agents which moved in the
                last time step and caused collisions.
                - collision_matrix: jnp.ndarray, the agents currently in
                collisions
                - agent_locs: jnp.ndarray, the agent locations at the previous
                time step.
                - new_agent_locs: jnp.ndarray, the agent locations at the
                current time step.

            Returns:
                - jnp.ndarray of the final positions after collisions are
                managed.
            """
            def scan_fn(
                    state,
                    idx
            ):
                key, collided_moved, collision_matrix, agent_locs, new_agent_locs = state

                return jax.lax.cond(
                    collided_moved[idx] > 0,
                    lambda: _fix_collisions(
                        key,
                        collided_moved,
                        collision_matrix,
                        agent_locs,
                        new_agent_locs
                    ),
                    lambda: (state, new_agent_locs)
                )

            _, ys = jax.lax.scan(
                scan_fn,
                (key, collided_moved, collision_matrix, agent_locs, new_agent_locs),
                jnp.arange(self.num_agents)
            )

            final_locs = ys[-1]

            return final_locs

        def _fix_collisions(
            key: jnp.ndarray,
            collided_moved: jnp.ndarray,
            collision_matrix: jnp.ndarray,
            agent_locs: jnp.ndarray,
            new_agent_locs: jnp.ndarray
        ) -> Tuple[Tuple, jnp.ndarray]:
            def select_random_true_index(key, array):
                # Calculate the cumulative sum of True values
                cumsum_array = jnp.cumsum(array)

                # Count the number of True values
                true_count = cumsum_array[-1]

                # Generate a random index in the range of the number of True
                # values
                rand_index = jax.random.randint(
                    key,
                    (1,),
                    0,
                    true_count
                )

                # Find the position of the random index within the cumulative
                # sum
                chosen_index = jnp.argmax(cumsum_array > rand_index)

                return chosen_index
            # Pick one from all who collided & moved
            colliders_idx = jnp.argmax(collided_moved)

            collisions = collision_matrix[colliders_idx]

            # Check whether any of collision participants didn't move
            collision_subjects = jnp.where(
                collisions,
                collided_moved,
                collisions
            )
            collision_mask = collisions == collision_subjects
            stayed = jnp.all(collision_mask)
            stayed_mask = jnp.logical_and(~stayed, ~collision_mask)
            stayed_idx = jnp.where(
                jnp.max(stayed_mask) > 0,
                jnp.argmax(stayed_mask),
                0
            )

            # Prepare random agent selection
            k1, k2 = jax.random.split(key, 2)
            rand_idx = select_random_true_index(k1, collisions)
            collisions_rand = collisions.at[rand_idx].set(False) # <<<< PROBLEM LINE        
            new_locs_rand = jax.vmap(
                lambda p, l, n: jnp.where(p, l, n)
            )(
                collisions_rand,
                agent_locs,
                new_agent_locs
            )

            collisions_stayed = jax.lax.select(
                jnp.max(stayed_mask) > 0,
                collisions.at[stayed_idx].set(False),
                collisions_rand
            )
            new_locs_stayed = jax.vmap(
                lambda p, l, n: jnp.where(p, l, n)
            )(
                collisions_stayed,
                agent_locs,
                new_agent_locs
            )

            # Choose between the two scenarios - revert positions if
            # non-mover exists, otherwise choose random agent if all moved
            new_agent_locs = jnp.where(
                stayed,
                new_locs_rand,
                new_locs_stayed
            )

            # Update move bools to reflect the post-collision positions
            collided_moved = jnp.clip(collided_moved - collisions, 0, 1)
            collision_matrix = collision_matrix.at[colliders_idx].set(
                [False] * collisions.shape[0]
            )
            return ((k2, collided_moved, collision_matrix, agent_locs, new_agent_locs), new_agent_locs)

        def to_dict(
                agent: int,
                obs: jnp.ndarray,
                agent_invs: jnp.ndarray,
                agent_pickups: jnp.ndarray,
                inv_to_show: jnp.ndarray
            ) -> dict:
            '''
            Function to produce observation/state dictionaries.
            
            Args:
                - agent: int, number identifying agent
                - obs: jnp.ndarray, the combined grid observations for each
                agent
                - agent_invs: jnp.ndarray of current agents' inventories
                - agent_pickups: boolean indicators of interaction
                - inv_to_show: jnp.ndarray inventory to show to other agents
                
            Returns:
                - dictionary of full state observation.
            '''
            idx = agent - len(Items)
            state_dict = {
                "observation": obs,
                "inventory": {
                    "agent_inv": agent_invs,
                    "agent_pickups": agent_pickups,
                    "invs_to_show": jnp.delete(
                        inv_to_show,
                        idx,
                        assume_unique_indices=True
                    )
                }
            }

            return state_dict
        
        def combine_channels(
                grid: jnp.ndarray,
                agent: int,
                angles: jnp.ndarray,
                agent_pickups: jnp.ndarray,
                state: State,
            ):

            def move_and_collapse(
                    x: jnp.ndarray,
                    angle: jnp.ndarray,
                ) -> jnp.ndarray:

                # get agent's one-hot
                agent_element = jnp.array([jnp.int8(x[agent])])

                # mask to check if any other agent exists there
                mask = x[len(Items)-1:] > 0

                # does an agent exist which is not the subject?
                other_agent = jnp.int8(
                    jnp.logical_and(
                        jnp.any(mask),
                        jnp.logical_not(
                            agent_element
                        )
                    )
                )

                # what is the class of the item in cell
                item_idx = jnp.where(
                    x,
                    size=1
                )[0]

                # check if agent is frozen and can observe inventories
                show_inv_bool = jnp.logical_and(
                        state.freeze[
                            agent-len(Items)
                        ].max(axis=-1) > 0,
                        item_idx >= len(Items)
                )

                show_inv_idxs = jnp.where(
                    state.freeze[agent],
                    size=12, # since, in a setting where simultaneous interac-
                    fill_value=-1 # -tions can happen, only a max of 12 can
                )[0] # happen at once (zap logic), regardless of pop size

                inv_to_show = jnp.where(
                    jnp.logical_or(
                        jnp.logical_and(
                            show_inv_bool,
                            jnp.isin(item_idx-len(Items), show_inv_idxs),
                        ),
                        agent_element
                    ),
                    state.agent_invs[item_idx - len(Items)],
                    jnp.array([0, 0], dtype=jnp.int8)
                )[0]

                # check if agent is not the subject & is frozen & therefore
                # not possible to interact with
                frozen = jnp.where(
                    other_agent,
                    state.freeze[
                        item_idx-len(Items)
                    ].max(axis=-1) > 0,
                    0
                )

                # get pickup/inv info
                pick_up_idx = jnp.where(
                    jnp.any(mask),
                    jnp.nonzero(mask, size=1)[0],
                    jnp.int8(-1)
                )
                picked_up = jnp.where(
                    pick_up_idx > -1,
                    agent_pickups[pick_up_idx],
                    jnp.int8(0)
                )

                # build extension
                extension = jnp.concatenate(
                    [
                        agent_element,
                        other_agent,
                        angle,
                        picked_up,
                        inv_to_show,
                        frozen
                    ],
                    axis=-1
                )

                # build final feature vector
                final_vec = jnp.concatenate(
                    [x[:len(Items)-1], extension],
                    axis=-1
                )

                return final_vec

            new_grid = jax.vmap(
                jax.vmap(
                    move_and_collapse
                )
            )(grid, angles)
            return new_grid
        
        def check_relative_orientation(
                agent: int,
                agent_locs: jnp.ndarray,
                grid: jnp.ndarray
            ) -> jnp.ndarray:
            '''
            Check's relative orientations of all other agents in view of
            current agent.
            
            Args:
                - agent: int, an index indicating current agent number
                - agent_locs: jax ndarray of agent locations (x, y, direction)
                - grid: jax ndarray of current agent's obs grid
                
            Returns:
                - grid with 1) int -1 in places where no agent exists, or
                where the agent is the current agent, and 2) int in range
                0-3 in cells of opposing agents indicating relative
                orientation to current agent.
            '''
            # we decrement by num of Items when indexing as we incremented by
            # 5 in constructor call (due to 5 non-agent Items enum & locations
            # are indexed from 0)
            idx = agent - len(Items)
            agents = jnp.delete(
                self._agents,
                idx,
                assume_unique_indices=True
            )
            curr_agent_dir = agent_locs[idx, 2]

            def calc_relative_direction(cell):
                cell_agent = cell - len(Items)
                cell_direction = agent_locs[cell_agent, 2]
                return (cell_direction - curr_agent_dir) % 4

            angle = jnp.where(
                jnp.isin(grid, agents),
                jax.vmap(calc_relative_direction)(grid),
                -1
            )

            return angle
        
        def rotate_grid(agent_loc: jnp.ndarray, grid: jnp.ndarray) -> jnp.ndarray:
            '''
            Rotates agent's observation grid k * 90 degrees, depending on agent's
            orientation.

            Args:
                - agent_loc: jax ndarray of agent's x, y, direction
                - grid: jax ndarray of agent's obs grid

            Returns:
                - jnp.ndarray of new rotated grid.

            '''
            grid = jnp.where(
                agent_loc[2] == 1,
                jnp.rot90(grid, k=1, axes=(0, 1)),
                grid,
            )
            grid = jnp.where(
                agent_loc[2] == 2,
                jnp.rot90(grid, k=2, axes=(0, 1)),
                grid,
            )
            grid = jnp.where(
                agent_loc[2] == 3,
                jnp.rot90(grid, k=3, axes=(0, 1)),
                grid,
            )

            return grid

        def _get_obs_point(agent_loc: jnp.ndarray) -> jnp.ndarray:
            '''
            Obtain the position of top-left corner of obs map using
            agent's current location & orientation.

            Args: 
                - agent_loc: jnp.ndarray, agent x, y, direction.
            Returns:
                - x, y: ints of top-left corner of agent's obs map.
            '''
            
            x, y, direction = agent_loc

            x, y = x + self.PADDING, y + self.PADDING

            x = x - (self.OBS_SIZE // 2)
            y = y - (self.OBS_SIZE // 2)


            x = jnp.where(direction == 0, x + (self.OBS_SIZE//2)-1, x)
            y = jnp.where(direction == 0, y, y)

            x = jnp.where(direction == 1, x, x)
            y = jnp.where(direction == 1, y + (self.OBS_SIZE//2)-1, y)


            x = jnp.where(direction == 2, x - (self.OBS_SIZE//2)+1, x)
            y = jnp.where(direction == 2, y, y)


            x = jnp.where(direction == 3, x, x)
            y = jnp.where(direction == 3, y - (self.OBS_SIZE//2)+1, y)
            return x, y

        def _get_obs(state: State) -> jnp.ndarray:
            '''
            Obtain the agent's observation of the grid.

            Args: 
                - state: State object containing env state.
            Returns:
                - jnp.ndarray of grid observation.
            '''
            # create state
            grid = jnp.pad(
                state.grid,
                ((self.PADDING, self.PADDING), (self.PADDING, self.PADDING)),
                constant_values=Items.wall,
            )

            # obtain all agent obs-points
            agent_start_idxs = jax.vmap(_get_obs_point)(state.agent_locs)

            dynamic_slice = partial(
                jax.lax.dynamic_slice,
                operand=grid,
                slice_sizes=(self.OBS_SIZE, self.OBS_SIZE)
            )

            # obtain agent obs grids
            grids = jax.vmap(dynamic_slice)(start_indices=agent_start_idxs)

            # rotate agent obs grids
            grids = jax.vmap(rotate_grid)(state.agent_locs, grids)

            angles = jax.vmap(
                check_relative_orientation,
                in_axes=(0, None, 0)
            )(
                self._agents,
                state.agent_locs,
                grids
            )

            angles = jax.nn.one_hot(angles, 4)

            # one-hot (drop first channel as its empty blocks)
            grids = jax.nn.one_hot(
                grids - 1,
                num_agents + len(Items) - 1, # will be collapsed into a
                dtype=jnp.int8 # [Items, self, other, extra features] representation
            )

            # check agents that can interact
            inventory_sum = jnp.sum(state.agent_invs, axis=-1)
            agent_pickups = jnp.where(
                inventory_sum > INTERACT_THRESHOLD,
                True,
                False
            )

            # make index len(Item) always the current agent
            # and sum all others into an "other" agent
            grids = jax.vmap(
                combine_channels,
                in_axes=(0, 0, 0, None, None)
            )(
                grids,
                self._agents,
                angles,
                agent_pickups,
                state
            )

            if self.observe_payment:
                # Payment observability (opt-in; OFF by default so baselines keep their
                # 19-channel observation and stay comparable).
                #
                # Without this an agent cannot see the incentive landscape at all -- the
                # observation is purely a local tile view -- so "clean only while I am
                # being paid" is not merely unlearned but UNREPRESENTABLE. Two scalars,
                # broadcast across the spatial dims so the existing CNN consumes them:
                #   ch0: fraction of the OTHER agents currently sharing. Aggregate is
                #        enough here because withholding cleaning is inherently
                #        collective -- you cannot clean the river "for" one agent -- so
                #        payer identity would add nothing a strike could act on.
                #   ch1: this agent's own share state, so it knows its own pledge.
                active = (state.share_expiry_t > state.inner_t).astype(jnp.float32)  # (N,)
                others_frac = (
                    (jnp.sum(active) - active) / jnp.maximum(self.num_agents - 1, 1)
                )
                feats = jnp.stack([others_frac, active], axis=-1)          # (N, 2)
                feats = jnp.broadcast_to(
                    feats[:, None, None, :],
                    (self.num_agents, self.OBS_SIZE, self.OBS_SIZE, 2),
                )
                grids = jnp.concatenate([grids.astype(jnp.float32), feats], axis=-1)

            return grids

        def get_current_s_interest(timestep):
            """Calculate current s_interest based on timestep and schedule."""
            if self.s_interest_schedule is None:
                return self.s_interest

            # Calculate which phase we're in using JAX operations
            phase = timestep // self.s_interest_change_every
            phase_idx = phase % self.s_interest_schedule.shape[0]
            return self.s_interest_schedule[phase_idx]

        def _interact_fire_zapping(
            key: jnp.ndarray, state: State, actions: jnp.ndarray
        ) -> Tuple[jnp.ndarray, jnp.ndarray, State, jnp.ndarray]:
            '''
            Main interaction logic entry point.

            Args:
                - key: jax key for randomisation.
                - state: State env state object.
                - actions: jnp.ndarray of actions taken by agents.
            Returns:
                - (jnp.ndarray, State, jnp.ndarray) - Tuple where index 0 is
                the array of rewards obtained, index 2 is the new env State,
                and index 3 is the new freeze penalty matrix.
            '''
            # if interact
            zaps = jnp.isin(actions,
                jnp.array(
                    [
                        Actions.zap_forward,
                        # Actions.zap_ahead
                    ]
                )
            )

            interact_idx = jnp.int16(Items.interact)

            # remove old interacts
            state = state.replace(grid=jnp.where(
                state.grid == interact_idx, jnp.int16(Items.empty), state.grid
            ))

            state = state.replace(grid=jnp.where(
                state.grid == Items.clean_beam, jnp.int16(Items.empty), state.grid
            ))

            # calculate pickups
            # agent_pickups = state.agent_invs.sum(axis=-1) > -100

            one_step_targets = jax.vmap(
                lambda p: p + STEP[p[2]]
            )(state.agent_locs)

            # check 2 ahead
            two_step_targets = jax.vmap(
                lambda p: p + 2*STEP[p[2]]
            )(state.agent_locs)


            target_right = jax.vmap(
                lambda p: p + STEP[p[2]] + STEP[(p[2] + 1) % 4]
            )(state.agent_locs)

            right_oob_check = jax.vmap(
                lambda t: jnp.logical_or(
                    jnp.logical_or((t[0] > self.GRID_SIZE_ROW - 1).any(), (t[1] > self.GRID_SIZE_COL - 1).any()),
                    (t < 0).any(),
                )
            )(target_right)

            target_right = jnp.where(
                right_oob_check[:, None],
                one_step_targets,
                target_right
            )


            target_left = jax.vmap(
                lambda p: p + STEP[p[2]] + STEP[(p[2] - 1) % 4]
            )(state.agent_locs)

            left_oob_check = jax.vmap(
                lambda t: jnp.logical_or(
                    jnp.logical_or((t[0] > self.GRID_SIZE_ROW - 1).any(), (t[1] > self.GRID_SIZE_COL - 1).any()),
                    (t < 0).any(),
                )
            )(target_left)

            target_left = jnp.where(
                left_oob_check[:, None],
                one_step_targets,
                target_left
            )

            # Beam cells that fall outside the grid must not hit or draw anything.
            # Without this an agent facing a wall zaps ITSELF (off-grid indices clamp
            # back onto its own cell) or an agent at the far edge (negative indices wrap
            # around), and the victim respawns mid-map -- a fast-travel exploit, not a
            # game mechanic. Clamping keeps every gather in range; the mask is what
            # actually suppresses the phantom hit.
            R_, C_ = self.GRID_SIZE_ROW, self.GRID_SIZE_COL
            one_step_targets, one_valid = clip_beam_targets(one_step_targets, R_, C_)
            two_step_targets, two_valid = clip_beam_targets(two_step_targets, R_, C_)
            target_right, right_valid = clip_beam_targets(target_right, R_, C_)
            target_left, left_valid = clip_beam_targets(target_left, R_, C_)

            all_zaped_locs = jnp.concatenate((one_step_targets, two_step_targets, target_right, target_left), 0)
            # zaps_3d = jnp.stack([zaps, zaps, zaps], axis=-1)

            zaps_4_locs = jnp.concatenate((zaps, zaps, zaps, zaps), 0)
            all_zaped_valid = jnp.concatenate((one_valid, two_valid, right_valid, left_valid), 0)

            # all_zaped_locs = jax.vmap(filter_zaped_locs)(all_zaped_locs)

            def zaped_gird(a, z, v):
                return jnp.where(z & v, state.grid[a[0], a[1]], -1)

            all_zaped_gird = jax.vmap(zaped_gird)(all_zaped_locs, zaps_4_locs, all_zaped_valid)
            # jax.debug.print("all_zaped_gird {all_zaped_gird} 🤯", all_zaped_gird=all_zaped_gird)

            def check_reborn_player(a):
                return jnp.isin(a, all_zaped_gird)
            
            reborn_players = jax.vmap(check_reborn_player)(self._agents)

            aux_grid = jnp.copy(state.grid)

            o_items = jnp.where(
                        state.grid[
                            one_step_targets[:, 0],
                            one_step_targets[:, 1]
                        ],
                        state.grid[
                            one_step_targets[:, 0],
                            one_step_targets[:, 1]
                        ],
                        interact_idx
                    )

            t_items = jnp.where(
                        state.grid[
                            two_step_targets[:, 0],
                            two_step_targets[:, 1]
                        ],
                        state.grid[
                            two_step_targets[:, 0],
                            two_step_targets[:, 1]
                        ],
                        interact_idx
                    )

            r_items = jnp.where(
                        state.grid[
                            target_right[:, 0],
                            target_right[:, 1]
                        ],
                        state.grid[
                            target_right[:, 0],
                            target_right[:, 1]
                        ],
                        interact_idx
                    )

            l_items = jnp.where(
                        state.grid[
                            target_left[:, 0],
                            target_left[:, 1]
                        ],
                        state.grid[
                            target_left[:, 0],
                            target_left[:, 1]
                        ],
                        interact_idx
                    )

            qualified_to_zap = zaps.squeeze()
            # jax.debug.print("qualified_to_zap {qualified_to_zap} 🤯", qualified_to_zap=qualified_to_zap)
            # update grid
            # `valid` keeps the beam graphic from being drawn on a clamped (off-grid)
            # cell, which would otherwise paint a stray beam onto the map edge.
            def update_grid(a_i, t, i, grid, valid):
                return grid.at[t[:, 0], t[:, 1]].set(
                    jax.vmap(jnp.where)(
                        a_i & valid,
                        i,
                        aux_grid[t[:, 0], t[:, 1]]
                    )
                )
            # def update_grid(a_i, t, i, grid):
            #     return grid.at[t[:, 0], t[:, 1]].set(2)


            # jax.debug.print("one_step_targets {one_step_targets} 🤯", one_step_targets=one_step_targets)
            aux_grid = update_grid(qualified_to_zap, one_step_targets, o_items, aux_grid, one_valid)
            aux_grid = update_grid(qualified_to_zap, two_step_targets, t_items, aux_grid, two_valid)
            aux_grid = update_grid(qualified_to_zap, target_right, r_items, aux_grid, right_valid)
            aux_grid = update_grid(qualified_to_zap, target_left, l_items, aux_grid, left_valid)

            # jax.debug.print("aux_grid {aux_grid} 🤯", aux_grid=aux_grid)
            state = state.replace(
                grid=jnp.where(
                    jnp.any(zaps),
                    aux_grid,
                    state.grid
                )
            )
            return reborn_players, state
        
        def _interact_fire_cleaning(
            key: jnp.ndarray, state: State, actions: jnp.ndarray
        ) -> Tuple[jnp.ndarray, jnp.ndarray, State, jnp.ndarray]:
            '''
            Main interaction logic entry point.

            Args:
                - key: jax key for randomisation.
                - state: State env state object.
                - actions: jnp.ndarray of actions taken by agents.
            Returns:
                - (jnp.ndarray, State, jnp.ndarray) - Tuple where index 0 is
                the array of rewards obtained, index 2 is the new env State,
                and index 3 is the new freeze penalty matrix.
            '''
            # if interact
            zaps = jnp.isin(actions,
                jnp.array(
                    [
                        Actions.zap_clean,
                    ]
                )
            )

            interact_idx = jnp.int16(Items.clean_beam)

            # remove old interacts

            state = state.replace(grid=jnp.where(
                state.grid == interact_idx, jnp.int16(Items.empty), state.grid
            ))


            one_step_targets = jax.vmap(
                lambda p: p + STEP[p[2]]
            )(state.agent_locs)

            two_step_targets = jax.vmap(
                lambda p: p + 2*STEP[p[2]]
            )(state.agent_locs)

            target_right = jax.vmap(
                lambda p: p + STEP[p[2]] + STEP[(p[2] + 1) % 4]
            )(state.agent_locs)

            right_oob_check = jax.vmap(
                lambda t: jnp.logical_or(
                    jnp.logical_or((t[0] > self.GRID_SIZE_ROW - 1).any(), (t[1] > self.GRID_SIZE_COL - 1).any()),
                    (t < 0).any(),
                )
            )(target_right)

            target_right = jnp.where(
                right_oob_check[:, None],
                one_step_targets,
                target_right
            )

            target_left = jax.vmap(
                lambda p: p + STEP[p[2]] + STEP[(p[2] - 1) % 4]
            )(state.agent_locs)

            left_oob_check = jax.vmap(
                lambda t: jnp.logical_or(
                    jnp.logical_or((t[0] > self.GRID_SIZE_ROW - 1).any(), (t[1] > self.GRID_SIZE_COL - 1).any()),
                    (t < 0).any(),
                )
            )(target_left)

            target_left = jnp.where(
                left_oob_check[:, None],
                one_step_targets,
                target_left
            )


            # Same off-grid hazard as the zap beam: without clamping+masking, an agent
            # at an edge would "clean" dirt on the opposite side of the map (negative
            # indices wrap) or re-clean its own cell (large indices clamp). That would
            # also corrupt cleaned_dirt, which credits cleaning per agent and therefore
            # drives both the tithe recipient rule and MOCA contract transfers.
            R_, C_ = self.GRID_SIZE_ROW, self.GRID_SIZE_COL
            one_step_targets, one_valid = clip_beam_targets(one_step_targets, R_, C_)
            two_step_targets, two_valid = clip_beam_targets(two_step_targets, R_, C_)
            target_right, right_valid = clip_beam_targets(target_right, R_, C_)
            target_left, left_valid = clip_beam_targets(target_left, R_, C_)

            all_zaped_locs = jnp.concatenate((one_step_targets, two_step_targets, target_right, target_left), 0)
            all_zaped_valid = jnp.concatenate((one_valid, two_valid, right_valid, left_valid), 0)
            # zaps_3d = jnp.stack([zaps, zaps, zaps], axis=-1)

            # Per-agent "actually cleaned a dirt patch this step": took zap_clean AND
            # at least one of its 4 IN-GRID beam tiles was Items.dirt in the (pre-clean) grid.
            # all_zaped_locs is laid out as [one_step(all agents), two_step(all), right(all),
            # left(all)], so reshaping to (4, N, 2) groups the 4 tiles per agent on axis 0.
            n_ag = self.num_agents
            # all_zaped_locs rows are [row, col, orient] (3 cols), grouped as
            # [one_step(all agents), two_step(all), right(all), left(all)].
            _beam_tiles = all_zaped_locs.reshape(4, n_ag, all_zaped_locs.shape[-1])
            _beam_valid = all_zaped_valid.reshape(4, n_ag)
            _tile_is_dirt = (
                state.grid[_beam_tiles[:, :, 0], _beam_tiles[:, :, 1]] == Items.dirt
            ) & _beam_valid
            # HOW MANY dirt cells this agent's beam cleared this step (0..4). The beam
            # covers 4 tiles, so a single clean action routinely clears several at once;
            # crediting only a boolean would under-count the public good and, under a
            # contract priced "per waste cell cleaned", pay the same for clearing 4
            # cells as for 1. Kept as a count for that reason.
            # Caveat: if two agents' beams cover the same dirt cell on the same step the
            # cell is cleared once but both are credited -- beam overlap is inherently
            # ambiguous to attribute, and the env has always resolved it this way.
            cleaned_count = jnp.where(
                zaps.reshape(-1), jnp.sum(_tile_is_dirt, axis=0), 0
            ).astype(jnp.float32)
            # Boolean form ("did this agent clean at all"), which is what last_clean_t
            # and the recent-cleaner recipient rules need.
            cleaned_dirt = cleaned_count > 0

            zaps_4_locs_judge = jnp.concatenate((zaps, zaps, zaps, zaps), 0)


            # all_zaped_locs = jax.vmap(filter_zaped_locs)(all_zaped_locs)

            potential_dirt_all_zap = jnp.repeat(jnp.array(Items.potential_dirt), len(all_zaped_locs))
            # make clean gird
            def clean_gird(a, judge):
                return state.grid.at[a[:, 0], a[:, 1]].set(
                    jax.vmap(jnp.where)(
                        ((judge == True) & (state.grid[a[:, 0], a[:, 1]] == Items.dirt)),
                        potential_dirt_all_zap,
                        state.grid[a[:, 0], a[:, 1]]
                    )
                )


            grid_clean = clean_gird(
                all_zaped_locs, (zaps_4_locs_judge.squeeze() & all_zaped_valid)
            )
            state = state.replace(grid=grid_clean)

            # refresh label

            def renew_dirt_label(locs, labels):
                return jnp.where((grid_clean[locs[0], locs[1]] == Items.dirt) | (grid_clean[locs[0], locs[1]] == Items.potential_dirt), grid_clean[locs[0], locs[1]], labels)


            renew_label = jax.vmap(renew_dirt_label)(state.potential_dirt_and_dirt_locs, state.potential_dirt_and_dirt_label)


            state = state.replace(
                potential_dirt_and_dirt_label=renew_label
            )

            
            aux_grid = jnp.copy(state.grid)

            o_items = jnp.where(
                        state.grid[
                            one_step_targets[:, 0],
                            one_step_targets[:, 1]
                        ],
                        state.grid[
                            one_step_targets[:, 0],
                            one_step_targets[:, 1]
                        ],
                        interact_idx
                    )

            t_items = jnp.where(
                        state.grid[
                            two_step_targets[:, 0],
                            two_step_targets[:, 1]
                        ],
                        state.grid[
                            two_step_targets[:, 0],
                            two_step_targets[:, 1]
                        ],
                        interact_idx
                    )

            r_items = jnp.where(
                        state.grid[
                            target_right[:, 0],
                            target_right[:, 1]
                        ],
                        state.grid[
                            target_right[:, 0],
                            target_right[:, 1]
                        ],
                        interact_idx
                    )

            l_items = jnp.where(
                        state.grid[
                            target_left[:, 0],
                            target_left[:, 1]
                        ],
                        state.grid[
                            target_left[:, 0],
                            target_left[:, 1]
                        ],
                        interact_idx
                    )

            qualified_to_zap = zaps.squeeze()


            # update grid
            def update_grid(a_i, t, i, grid):
                return grid.at[t[:, 0], t[:, 1]].set(
                    jax.vmap(jnp.where)(
                        a_i,
                        i,
                        aux_grid[t[:, 0], t[:, 1]]
                    )
                )



            aux_grid = update_grid(qualified_to_zap, one_step_targets, o_items, aux_grid)
            aux_grid = update_grid(qualified_to_zap, two_step_targets, t_items, aux_grid)
            aux_grid = update_grid(qualified_to_zap, target_right, r_items, aux_grid)
            aux_grid = update_grid(qualified_to_zap, target_left, l_items, aux_grid)


            state = state.replace(
                grid=jnp.where(
                    jnp.any(zaps),
                    aux_grid,
                    state.grid
                )
            )
            return state, cleaned_dirt, cleaned_count


        def _step(
            key: chex.PRNGKey,
            state: State,
            actions: jnp.ndarray,
            timestep: int = 0
        ):
            """Step the environment."""

            # regrowth of apply
            grid_apple = state.grid
            dirtCount = jnp.sum(state.potential_dirt_and_dirt_label == Items.dirt)
            dirtFraction = dirtCount / (len(state.potential_dirt_and_dirt_locs) + len(self.RIVER))
            depletion = self.thresholdDepletion
            restoration = self.thresholdRestoration
            interpolation = (dirtFraction - depletion) / (restoration - depletion)

            interpolation = jnp.clip(interpolation, -jnp.inf, 1.0)
            probability = self.maxAppleGrowthRate * interpolation
            key, key_decay = jax.random.split(key)
            decay_draw = jax.random.uniform(key_decay, shape=(len(self.POTENTIAL_APPLE),))

            def regrow_apple(apple_locs, p, d):
                is_empty = grid_apple[apple_locs[0], apple_locs[1]] == Items.empty
                is_apple = grid_apple[apple_locs[0], apple_locs[1]] == Items.apple
                # A standing apple survives unless it decays; decay=0 keeps the original
                # "apples persist forever" behaviour exactly.
                survives = is_apple & (d >= self.appleDecayProbability)
                return jnp.where((is_empty & (p < probability)) | survives,
                                 Items.apple, Items.empty)
            prob = jax.random.uniform(key, shape=(len(self.POTENTIAL_APPLE),))
            new_apple = jax.vmap(regrow_apple)(self.POTENTIAL_APPLE, prob, decay_draw)

            new_apple_grid = grid_apple.at[self.POTENTIAL_APPLE[:, 0], self.POTENTIAL_APPLE[:, 1]].set(new_apple)
            state = state.replace(grid=new_apple_grid)

            # DirtSpawning update the grid and potential_dirt_and_dirt_label
            grid_dirt = state.grid

            noise = jax.random.uniform(key, shape=(len(state.potential_dirt_and_dirt_label),)) * 1e-4
            label_with_noise = state.potential_dirt_and_dirt_label + noise

            label_with_noise_rank = jnp.sort(label_with_noise)
            unstable_indices = jnp.argsort(label_with_noise)

            unstable_sorted_locs = state.potential_dirt_and_dirt_locs[unstable_indices]
            
            # Sorting puts potential_dirt (7) ahead of dirt (8), so the first
            # `dirt_spawn_cells` entries are the cleanest candidates. Upstream only ever
            # converted entry 0, capping dirt growth at 1 cell/step; attempting the
            # first k (each with its own draw) makes the expected rate
            # k * dirtSpawnProbability, so the commons can actually outpace one cleaner.
            k = max(int(self.dirt_spawn_cells), 1)
            k = min(k, int(label_with_noise_rank.shape[0]))
            p = jax.random.uniform(key, shape=(k,))
            cand = unstable_sorted_locs[:k]
            spawn = (
                (grid_dirt[cand[:, 0], cand[:, 1]] == Items.potential_dirt)
                & (p < self.dirtSpawnProbability)
                & (state.inner_t > self.delayStartOfDirtSpawning)
            )
            new_piece_dirt = jnp.where(
                spawn, jnp.float32(Items.dirt), label_with_noise_rank[:k]
            )

            label_with_noise_rank_new = label_with_noise_rank.at[:k].set(new_piece_dirt)

            label_rank_new = jnp.round(label_with_noise_rank_new).astype(jnp.int16)
            

            state = state.replace(potential_dirt_and_dirt_label=label_rank_new)
            state = state.replace(potential_dirt_and_dirt_locs=unstable_sorted_locs)
            actions = jnp.array(actions)

            new_grid = state.grid.at[
                state.agent_locs[:, 0],
                state.agent_locs[:, 1]
            ].set(
                jnp.int16(Items.empty)
            )

            new_grid = new_grid.at[state.potential_dirt_and_dirt_locs[:, 0], state.potential_dirt_and_dirt_locs[:, 1]].set(state.potential_dirt_and_dirt_label)
            
            new_grid = new_grid.at[self.RIVER[:, 0], self.RIVER[:, 1]].set(Items.river)



            x, y = state.reborn_locs[:, 0], state.reborn_locs[:, 1]
            new_grid = new_grid.at[x, y].set(self._agents)
            state = state.replace(grid=new_grid)
            state = state.replace(agent_locs=state.reborn_locs)

            key, subkey = jax.random.split(key)
            all_new_locs = jax.vmap(lambda p, a: jnp.int16(p + ROTATIONS[a]) % jnp.array([self.GRID_SIZE_ROW + 1, self.GRID_SIZE_COL + 1, 4], dtype=jnp.int16))(p=state.agent_locs, a=actions).squeeze()

            agent_move = (actions == Actions.up) | (actions == Actions.down) | (actions == Actions.right) | (actions == Actions.left)
            all_new_locs = jax.vmap(lambda m, n, p: jnp.where(m, n + STEP_MOVE[p], n))(m=agent_move, n=all_new_locs, p=actions)
            
            all_new_locs = jax.vmap(
                jnp.clip,
                in_axes=(0, None, None)
            )(
                all_new_locs,
                jnp.array([0, 0, 0], dtype=jnp.int16),
                jnp.array(
                    [self.GRID_SIZE_ROW - 1, self.GRID_SIZE_COL - 1, 3],
                    dtype=jnp.int16
                ),
            ).squeeze()

            # if you bounced back to your original space,
            # change your move to stay (for collision logic)
            agents_move = jax.vmap(lambda n, p: jnp.any(n[:2] != p[:2]))(n=all_new_locs, p=state.agent_locs)

            # generate bool mask for agents colliding
            collision_matrix = check_collision(all_new_locs)

            # sum & subtract "self-collisions"
            collisions = jnp.sum(
                collision_matrix,
                axis=-1,
                dtype=jnp.int8
            ) - 1
            collisions = jnp.minimum(collisions, 1)

            # identify which of those agents made wrong moves
            collided_moved = jnp.maximum(
                collisions - ~agents_move,
                0
            )

            # fix collisions at the correct indices
            new_locs = jax.lax.cond(
                jnp.max(collided_moved) > 0,
                lambda: fix_collisions(
                    key,
                    collided_moved,
                    collision_matrix,
                    state.agent_locs,
                    all_new_locs
                ),
                lambda: all_new_locs
            )

            # get apples
            def coin_matcher(p: jnp.ndarray) -> jnp.ndarray:
                c_matches = jnp.array([
                    state.grid[p[0], p[1]] == Items.apple
                    ])
                return c_matches
            
            apple_matches = jax.vmap(coin_matcher)(p=new_locs)

            # # individual rewards
            # rewards = jnp.zeros((self.num_agents, 1))
            # rewards = jnp.where(apple_matches, 1, rewards)

            # # single reward or sum reward

            # rewards_sum_all_agents = jnp.zeros((self.num_agents, 1))
            # rewards_sum = jnp.sum(rewards)
            # rewards_sum_all_agents += rewards_sum
            # rewards = rewards_sum_all_agents

            new_invs = state.agent_invs + apple_matches

            state = state.replace(
                agent_invs=new_invs
            )

            # update grid
            old_grid = state.grid

            new_grid = old_grid.at[
                state.agent_locs[:, 0],
                state.agent_locs[:, 1]
            ].set(
                jnp.int16(Items.empty)
            )

            new_grid = new_grid.at[state.potential_dirt_and_dirt_locs[:, 0], state.potential_dirt_and_dirt_locs[:, 1]].set(state.potential_dirt_and_dirt_label)

            new_grid = new_grid.at[self.RIVER[:, 0], self.RIVER[:, 1]].set(Items.river)
            x, y = new_locs[:, 0], new_locs[:, 1]
            new_grid = new_grid.at[x, y].set(self._agents)
            state = state.replace(grid=new_grid)

            # update agent locations
            state = state.replace(agent_locs=new_locs)

            reborn_players, state = _interact_fire_zapping(key, state, actions)

            state, cleaned_dirt, cleaned_count = _interact_fire_cleaning(key, state, actions)

            # Record when each agent last actually cleaned dirt (used to decide who is
            # a payable "recent cleaner"). state.inner_t is this step's counter.
            new_last_clean_t = jnp.where(
                cleaned_dirt, jnp.int32(state.inner_t), state.last_clean_t
            )
            state = state.replace(last_clean_t=new_last_clean_t)

            reborn_players_3d = jnp.stack([reborn_players, reborn_players, reborn_players], axis=-1)

            # jax.debug.print("reborn_players_3d {reborn_players_3d} 🤯", reborn_players_3d=reborn_players_3d)

            re_agents_pos = jax.random.permutation(subkey, self.SPAWNS_PLAYERS)[:num_agents]

            player_dir = jax.random.randint(
                subkey, shape=(
                    num_agents,
                    ), minval=0, maxval=3, dtype=jnp.int8
            )

            re_agent_locs = jnp.array(
                [re_agents_pos[:, 0], re_agents_pos[:, 1], player_dir],
                dtype=jnp.int16
            ).T

            new_re_locs = jnp.where(reborn_players_3d == False, new_locs, re_agent_locs)
            new_re_locs = jnp.where(reborn_players.any(), new_re_locs, state.agent_locs)
            state = state.replace(reborn_locs=new_re_locs)

            if self.shared_rewards:
                rewards = jnp.zeros((self.num_agents, 1))
                original_rewards = jnp.where(apple_matches, 1, rewards)

                rewards_sum_all_agents = jnp.zeros((self.num_agents, 1))
                rewards_sum = jnp.sum(original_rewards)
                rewards_sum_all_agents += rewards_sum
                rewards = rewards_sum_all_agents
                info = {
                    "original_rewards": original_rewards.squeeze(),
                    "shaped_rewards": rewards.squeeze(),
                }
            elif self.inequity_aversion:
                rewards = jnp.zeros((self.num_agents, 1))
                original_rewards = jnp.where(apple_matches, 1, rewards) * self.apple_reward
                if self.smooth_rewards:
                    should_smooth = (state.inner_t % 1) == 0
                    new_smooth_rewards = 0.99 * 0.01* state.smooth_rewards + original_rewards
                    rewards, disadvantageous, advantageous = self.get_inequity_aversion_rewards_immediate(new_smooth_rewards, state.inner_t, self.inequity_aversion_target_agents, self.inequity_aversion_alpha, self.inequity_aversion_beta)
                    state = state.replace(smooth_rewards=new_smooth_rewards)
                    info = {
                    "original_rewards": original_rewards.squeeze(),
                    "smooth_rewards": state.smooth_rewards.squeeze(),
                    "shaped_rewards": rewards.squeeze(),
                }
                else:
                    rewards, disadvantageous, advantageous = self.get_inequity_aversion_rewards_immediate(original_rewards, state.inner_t, self.inequity_aversion_target_agents, self.inequity_aversion_alpha, self.inequity_aversion_beta)
                    info = {
                    "original_rewards": original_rewards.squeeze(),
                    "shaped_rewards": rewards.squeeze(),
                }
            elif self.svo:
                rewards = jnp.zeros((self.num_agents, 1))
                original_rewards = jnp.where(apple_matches, 1, rewards) * self.apple_reward
                rewards, theta = self.get_svo_rewards(original_rewards, self.svo_w, self.svo_ideal_angle_degrees, self.svo_target_agents)
                info = {
                    "original_rewards": original_rewards.squeeze(),
                    "svo_theta": theta.squeeze(),
                    "shaped_rewards": rewards.squeeze(),
                }
            elif self.interest:
                rewards = jnp.zeros((self.num_agents, 1))
                original_rewards = jnp.where(apple_matches, 1, rewards) * self.apple_reward
                original_flat = original_rewards.squeeze()

                # Calculate current s_interest based on timestep
                current_s_interest = get_current_s_interest(timestep)

                # Each agent gets s * their_reward + (1-s)/(n-1) * sum_of_others
                total_reward = jnp.sum(original_flat)
                others_reward = total_reward - original_flat  # sum of all other agents' rewards

                rewards = (current_s_interest * original_flat +
                        (1 - current_s_interest) / (self.num_agents - 1) * others_reward).reshape(-1, 1)

                info = {
                    "original_rewards": original_rewards.squeeze(),
                    "shaped_rewards": rewards.squeeze(),
                    "s_interest": current_s_interest,
                }
            elif self.cf:
                rewards = jnp.zeros((self.num_agents, 1))
                original_rewards = jnp.where(apple_matches, 1, rewards) * self.apple_reward
                rewards, theta = self.get_cf_rewards(original_rewards, self.cf_w, self.cf_ideal_angle_degrees, self.cf_target_agents)
                info = {
                    "original_rewards": original_rewards.squeeze(),
                    "cf_theta": theta.squeeze(),
                    "shaped_rewards": rewards.squeeze(),
                }
            else:
                rewards = jnp.zeros((self.num_agents, 1))
                rewards = jnp.where(apple_matches, 1, rewards) * self.apple_reward
                info = {
                    "original_rewards": rewards.squeeze(),
                    "shaped_rewards": rewards.squeeze(),
                }

            # Monetary system: resolve pay actions into a zero-sum reward transfer.
            # self.pay_mode / self.pay_scheme are static Python strs, so these branches
            # specialize at trace time (jit-safe; "off" compiles to the original graph).
            if self.pay_mode != "off":
                if self.pay_scheme == "tithe":
                    # income basis: this step's positive reward only, so a sharer
                    # never "shares" a negative reward into a refund
                    tithe_income = jnp.maximum(rewards.squeeze(), 0.0)
                    # freeze_share_state: neutralise the pay action before it is
                    # resolved, so an externally imposed share pattern governs BOTH the
                    # stored pledge state and this step's transfers. (Suppressing only
                    # the state write would still let a pledge move money on the step it
                    # was taken.) self.freeze_share_state is a static Python bool, so
                    # this branch specialises at trace time.
                    pay_actions = (
                        jnp.full_like(jnp.asarray(actions), jnp.asarray(Actions.stay))
                        if self.freeze_share_state else actions
                    )
                    pay_delta, pay_attempted, pay_executed, pay_target, new_expiry, share_active, pay_received, new_last_toggle = (
                        compute_tithe_transfers(
                            pay_actions, state.last_clean_t, state.inner_t,
                            state.share_expiry_t, tithe_income,
                            self.pay_clean_window, self.share_fraction, self.share_duration,
                            self.split_recipients, state.last_toggle_t, self.toggle_cooldown,
                        )
                    )
                    # Pledge state advances in BOTH on and noop: the placebo keeps the
                    # action's dynamics identical and only withholds the money.
                    state = state.replace(share_expiry_t=new_expiry,
                                          last_toggle_t=new_last_toggle)
                    info["share_active"] = jnp.float32(share_active).squeeze()
                    # Per-agent amount actually sent this step (sender side): the tithe
                    # slice share_fraction*income on executed shares. For share_fraction
                    # 0.5 this is a *fraction* of income, not a flat unit -- the point of
                    # the scheme -- so it's exposed for the viewer to label arrows with.
                    pay_sent = jnp.where(
                        pay_executed, jnp.float32(self.share_fraction) * tithe_income, 0.0
                    )
                else:  # "instant"
                    pay_delta, pay_attempted, pay_executed, pay_target = compute_pay_transfers(
                        actions, state.last_clean_t, state.inner_t,
                        state.agent_balance, self.pay_clean_window, self.pay_amount,
                    )
                    pay_sent = jnp.where(pay_executed, jnp.float32(self.pay_amount), 0.0)
                    # Instant is single-recipient, so received is just sent scattered to
                    # each payer's target (keeps info["pay_received"] present in all modes).
                    pay_received = jnp.zeros((self.num_agents,), dtype=jnp.float32).at[pay_target].add(pay_sent)
                if self.pay_mode == "on":
                    rewards = rewards + pay_delta[:, None]
                # "noop" (placebo): action exists, recipient/affordability/pledging are
                # computed and logged identically, but no reward actually moves.
                # Total sent this step (from the sender side, not the net delta, which
                # can mix sends and receipts for the same agent).
                pay_volume_on = jnp.sum(pay_sent)
                info["pay_attempts"] = jnp.float32(pay_attempted).squeeze()
                info["pay_executed"] = jnp.float32(pay_executed).squeeze()
                # Per-agent amount sent this step (only nonzero where pay_executed);
                # lets the viewer show the transfer size on each arrow.
                info["pay_sent"] = jnp.float32(pay_sent).squeeze()
                # Per-agent amount RECEIVED this step. Under split_recipients this fans
                # one sharer's slice across several cleaners, so received is the only way
                # to see the split; the viewer uses sent+received to draw the arrows.
                info["pay_received"] = jnp.float32(pay_received).squeeze()
                # Payer -> recipient (chosen recent cleaner) index, valid only where
                # pay_executed=True (used by the viewer to draw a "who paid whom" arrow).
                info["pay_target"] = jnp.int32(pay_target).squeeze()
                pay_volume = pay_volume_on if self.pay_mode == "on" else jnp.float32(0.0)
                info["pay_volume"] = jnp.broadcast_to(pay_volume, (self.num_agents,)).squeeze()
                # "shaped_rewards" was captured before this block ran (same pattern the
                # SVO/inequity_aversion/interest branches above use: original = raw apple
                # pickups, shaped = final transformed reward) -- keep that pairing correct
                # by refreshing it now that pay has been applied. original_rewards is left
                # untouched on purpose, so the two together show how much of an agent's
                # final reward came from payments vs. its own pickups.
                info["shaped_rewards"] = rewards.squeeze()

            # Balance = running cumulative net reward, so it equals what an agent has
            # available to pay with. Updated in BOTH pay modes (off leaves it tracking
            # pure apple income, harmless) using the final post-transfer reward, so
            # received payments credit the recipient's balance and payments made debit
            # the payer's. The gate above reads this step's incoming balance, so an
            # agent spends only what it accumulated on prior steps.
            state = state.replace(
                agent_balance=state.agent_balance + rewards.squeeze().astype(jnp.float32)
            )

            info["clean_action_info"] = jnp.where(actions == Actions.zap_clean, 1, 0).squeeze()
            # PER-AGENT successful cleaning this step (1.0 if this agent's clean beam
            # actually hit dirt, else 0.0). Distinct from "clean_action_info", which only
            # says the agent TRIED to clean, and from "waste_cleared", which is a single
            # grid-wide count broadcast to every agent. This is the per-agent credit the
            # formal-contracting literature calls "cleaned_squares" -- a contract
            # conditions transfers on it, so it has to be attributable to an individual.
            info["cleaned_by_agent"] = jnp.float32(cleaned_count).squeeze()
            info["cleaned_water"] = jnp.array([len(state.potential_dirt_and_dirt_label) - dirtCount] * self.num_agents).squeeze()
            info["waste_cleared"] = jnp.array([len(state.potential_dirt_and_dirt_label) - dirtCount] * self.num_agents).squeeze() 
            
            state_nxt = State(
                agent_locs=state.agent_locs,
                agent_invs=state.agent_invs,
                inner_t=state.inner_t + 1,
                outer_t=state.outer_t,
                grid=state.grid,
                apples=state.apples,
                freeze=state.freeze,
                reborn_locs=state.reborn_locs,
                potential_dirt_and_dirt_locs=state.potential_dirt_and_dirt_locs,
                potential_dirt_and_dirt_label=state.potential_dirt_and_dirt_label,
                smooth_rewards=state.smooth_rewards,
                agent_balance=state.agent_balance,
                last_clean_t=state.last_clean_t,
                share_expiry_t=state.share_expiry_t,
                last_toggle_t=state.last_toggle_t,
            )

            # now calculate if done for inner or outer episode
            inner_t = state_nxt.inner_t
            outer_t = state_nxt.outer_t
            reset_inner = inner_t == num_inner_steps

            # if inner episode is done, return start state for next game
            state_re = _reset_state(key)

            state_re = state_re.replace(outer_t=outer_t + 1)
            state = jax.tree.map(
                lambda x, y: jnp.where(reset_inner, x, y),
                state_re,
                state_nxt,
            )
            outer_t = state.outer_t
            reset_outer = outer_t == num_outer_steps
            done = {f'{a}': reset_outer for a in self.agents}
            # done = [reset_outer for _ in self.agents]
            done["__all__"] = reset_outer

            obs = _get_obs(state)
            rewards = jnp.where(
                reset_inner,
                jnp.zeros_like(rewards, dtype=jnp.int16),
                rewards
            )

            # mean_inv = state.agent_invs.mean(axis=0)
            return (
                obs,
                state,
                rewards.squeeze(),
                done,
                info,
            )

        def _reset_state(
            key: jnp.ndarray
        ) -> State:
            key, subkey = jax.random.split(key)

            # Find the free spaces in the grid
            grid = jnp.zeros((self.GRID_SIZE_ROW, self.GRID_SIZE_COL), jnp.int16)


            inside_players_pos = jax.random.permutation(subkey, self.SPAWNS_PLAYER_IN)
            player_positions = jnp.concatenate((inside_players_pos, self.SPAWNS_PLAYERS))
            agent_pos = jax.random.permutation(subkey, player_positions)[:num_agents]
            wall_pos = self.SPAWNS_WALL
            apple_pos = self.POTENTIAL_APPLE

            river = self.RIVER
            potential_dirt = self.POTENTIAL_DIRT
            dirt = self.DIRT

            potential_dirt_label = jnp.zeros((len(potential_dirt)), dtype=jnp.int16) +Items.potential_dirt
            dirt_label = jnp.zeros((len(dirt)), dtype=jnp.int16) + Items.dirt

            potential_dirt_and_dirt = jnp.concatenate((potential_dirt, dirt))
            potential_dirt_and_dirt_label = jnp.concatenate((potential_dirt_label, dirt_label))


            # set wall
            grid = grid.at[
                wall_pos[:, 0],
                wall_pos[:, 1]
            ].set(jnp.int16(Items.wall))

            # set dirt
            grid = grid.at[dirt[:, 0],
                           dirt[:, 1]
                           ].set(jnp.int16(Items.dirt))
            
            # set river
            grid = grid.at[river[:, 0],
                            river[:, 1]
                            ].set(jnp.int16(Items.river))
            
            # set potential dirt
            grid = grid.at[potential_dirt[:, 0],
                            potential_dirt[:, 1]
                            ].set(jnp.int16(Items.potential_dirt))
            


            player_dir = jax.random.randint(
                subkey, shape=(
                    num_agents,
                    ), minval=0, maxval=3, dtype=jnp.int8
            )

            agent_locs = jnp.array(
                [agent_pos[:, 0], agent_pos[:, 1], player_dir],
                dtype=jnp.int16
            ).T

            grid = grid.at[
                agent_locs[:, 0],
                agent_locs[:, 1]
            ].set(jnp.int16(self._agents))

            freeze = jnp.array(
                [[-1]*num_agents]*num_agents,
            dtype=jnp.int16
            )

            return State(
                agent_locs=agent_locs,
                agent_invs=jnp.array([(0,0)]*num_agents, dtype=jnp.int8),
                inner_t=0,
                outer_t=0,
                grid=grid,
                apples=apple_pos,

                freeze=freeze,
                reborn_locs=agent_locs,
                potential_dirt_and_dirt_locs=potential_dirt_and_dirt,
                potential_dirt_and_dirt_label=potential_dirt_and_dirt_label,
                smooth_rewards=jnp.zeros((self.num_agents, 1)),
                agent_balance=jnp.zeros((self.num_agents,), dtype=jnp.float32),
                last_clean_t=jnp.full((self.num_agents,), -10_000, dtype=jnp.int32),
                share_expiry_t=jnp.zeros((self.num_agents,), dtype=jnp.int32),
                last_toggle_t=jnp.full((self.num_agents,), -1_000_000, dtype=jnp.int32),
            )

        def reset(
            key: jnp.ndarray
        ) -> Tuple[jnp.ndarray, State]:
            state = _reset_state(key)
            obs = _get_obs(state)
            return obs, state
        
        ################################################################################
        # if you want to test whether it can run on gpu, activate following code
        # overwrite Gymnax as it makes single-agent assumptions
        if jit:
            self.step_env = jax.jit(_step)
            self.reset = jax.jit(reset)
            self.get_obs_point = jax.jit(_get_obs_point)
        else:
            # if you want to see values whilst debugging, don't jit
            self.step_env = _step
            self.reset = reset
            self.get_obs_point = _get_obs_point
        ################################################################################

    @property
    def name(self) -> str:
        """Environment name."""
        return "MGinTheGrid"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self._num_actions

    def action_space(
        self, agent_id: Union[int, None] = None
    ) -> spaces.Discrete:
        """Action space of the environment (9, or 10 when the pay action is exposed)."""
        return spaces.Discrete(self._num_actions)

    def observation_space(self) -> spaces.Dict:
        """Observation space of the environment."""
        n_ch = (len(Items) - 1) + 10 + (2 if self.observe_payment else 0)
        _shape_obs = (
            (self.OBS_SIZE, self.OBS_SIZE, n_ch)
            if self.cnn
            else (self.OBS_SIZE**2 * n_ch,)
        )

        return spaces.Box(
                low=0, high=1E9, shape=_shape_obs, dtype=jnp.uint8
            ), _shape_obs
    
    def state_space(self) -> spaces.Dict:
        """State space of the environment."""
        _shape = (
            (self.GRID_SIZE_ROW, self.GRID_SIZE_COL, NUM_TYPES + 4)
            if self.cnn
            else (self.GRID_SIZE_ROW* self.GRID_SIZE_COL * (NUM_TYPES + 4),)
        )
        return spaces.Box(low=0, high=1, shape=_shape, dtype=jnp.uint8)
    
    def render_tile(
        self,
        obj: int,
        agent_dir: Union[int, None] = None,
        agent_hat: bool = False,
        highlight: bool = False,
        tile_size: int = 32,
        subdivs: int = 3,
    ) -> onp.ndarray:
        """
        Render a tile and cache the result
        """

        # Hash map lookup key for the cache
        key: tuple[Any, ...] = (agent_dir, agent_hat, highlight, tile_size)
        if obj:
            key = (obj, 0, 0, 0) + key if obj else key

        if key in self.tile_cache:
            return self.tile_cache[key]

        img = onp.full(
                shape=(tile_size * subdivs, tile_size * subdivs, 3),
                fill_value=(190, 170, 120),
                dtype=onp.uint8,
            )

    # class Items(IntEnum):

        if obj in self._agents:
            # Draw the agent
            agent_color = self.PLAYER_COLOURS[obj-len(Items)]
        elif obj == Items.apple:
            # Draw the red coin as GREEN COOPERATE
            fill_coords(
                img, point_in_circle(0.5, 0.5, 0.31), (214.0, 39.0, 40.0)
            )
        
        # elif obj == Items.blue_coin:
        #     # Draw the blue coin as DEFECT/ RED COIN
        #     fill_coords(
        #         img, point_in_circle(0.5, 0.5, 0.31), (214.0, 39.0, 40.0)
        #     )

        elif obj == Items.river:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (40.0, 80.0, 214.0))
        elif obj == Items.potential_dirt:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (40.0, 80.0, 214.0))
        elif obj == Items.dirt:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (40.0, 80.0, 80.0))


        elif obj == Items.wall:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (127.0, 127.0, 127.0))

        elif obj == Items.interact:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (188.0, 189.0, 34.0))

        elif obj == Items.clean_beam:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (170, 220, 255))

        elif obj == 99:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (44.0, 160.0, 44.0))

        elif obj == 100:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (214.0, 39.0, 40.0))

        elif obj == 101:
            # white square
            fill_coords(img, point_in_rect(0, 1, 0, 1), (255.0, 255.0, 255.0))

        # Overlay the agent on top
        if agent_dir is not None:
            if agent_hat:
                tri_fn = point_in_triangle(
                    (0.12, 0.19),
                    (0.87, 0.50),
                    (0.12, 0.81),
                    0.3,
                )

                # Rotate the agent based on its direction
                tri_fn = rotate_fn(
                    tri_fn,
                    cx=0.5,
                    cy=0.5,
                    theta=0.5 * math.pi * (1 - agent_dir),
                )
                fill_coords(img, tri_fn, (255.0, 255.0, 255.0))

            tri_fn = point_in_triangle(
                (0.12, 0.19),
                (0.87, 0.50),
                (0.12, 0.81),
                0.0,
            )

            # Rotate the agent based on its direction
            tri_fn = rotate_fn(
                tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * (1 - agent_dir)
            )
            fill_coords(img, tri_fn, agent_color)

        # # Highlight the cell if needed
        if highlight:
            highlight_img(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        self.tile_cache[key] = img
        return img

    def render(
        self,
        state: State,
    ) -> onp.ndarray:
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """
        tile_size = 32
        highlight_mask = onp.zeros_like(onp.array(self.GRID))

        # Compute the total grid size
        width_px = self.GRID.shape[1] * tile_size
        height_px = self.GRID.shape[0] * tile_size

        img = onp.zeros(shape=(height_px, width_px, 3), dtype=onp.uint8)

        grid = onp.array(state.grid)
        # print(onp.argwhere(grid == Items.clean_beam))
        grid = onp.pad(
            grid, ((self.PADDING, self.PADDING), (self.PADDING, self.PADDING)), constant_values=Items.wall
        )
        for a in range(self.num_agents):
            startx, starty = self.get_obs_point(
                state.agent_locs[a]
            )
            highlight_mask[
                startx : startx + self.OBS_SIZE, starty : starty + self.OBS_SIZE
            ] = True

        # Render the grid
        for j in range(0, grid.shape[1]):
            for i in range(0, grid.shape[0]):
                cell = grid[i, j]
                if cell == 0:
                    cell = None
                agent_here = []
                for a in self._agents:
                    agent_here.append(cell == a)
                # if cell in [1,2]:
                #     print(f'coordinates: {i},{j}')
                #     print(cell)

                agent_dir = None
                for a in range(self.num_agents):
                    agent_dir = (
                        state.agent_locs[a,2].item()
                        if agent_here[a]
                        else agent_dir
                    )
                
                agent_hat = False
                # for a in range(self.num_agents):
                #     agent_hat = (
                #         bool(state.agent_invs[a].sum() > INTERACT_THRESHOLD)
                #         if agent_here[a]
                #         else agent_hat
                #     )

                tile_img = self.render_tile(
                    cell,
                    agent_dir=agent_dir,
                    agent_hat=agent_hat,
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size,
                )

                ymin = i * tile_size
                ymax = (i + 1) * tile_size
                xmin = j * tile_size
                xmax = (j + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img
        
        img = onp.rot90(
            img[
                (self.PADDING - 1) * tile_size : -(self.PADDING - 1) * tile_size,
                (self.PADDING - 1) * tile_size : -(self.PADDING - 1) * tile_size,
                :,
            ],
            2,
        )
        # time = self.render_time(state, img.shape[1])
        # img = onp.concatenate((img, time), axis=0)
        return img



    def render_time(self, state, width_px) -> onp.array:
        inner_t = state.inner_t
        outer_t = state.outer_t
        tile_height = 32
        img = onp.zeros(shape=(2 * tile_height, width_px, 3), dtype=onp.uint8)
        tile_width = width_px // (self.num_inner_steps)
        j = 0
        for i in range(0, inner_t):
            ymin = j * tile_height
            ymax = (j + 1) * tile_height
            xmin = i * tile_width
            xmax = (i + 1) * tile_width
            img[ymin:ymax, xmin:xmax, :] = onp.int8(255)
        tile_width = width_px // (self.num_outer_steps)
        j = 1
        for i in range(0, outer_t):
            ymin = j * tile_height
            ymax = (j + 1) * tile_height
            xmin = i * tile_width
            xmax = (i + 1) * tile_width
            img[ymin:ymax, xmin:xmax, :] = onp.int8(255)
        return img
    
    def get_inequity_aversion_rewards_immediate(self, array, inner_t, target_agents=None, alpha=5, beta=0.05):
        """
        Calculate inequity aversion rewards using immediate rewards, based on equation (3) in the paper
        
        Args:
            array: shape: [num_agents, 1] immediate rewards r_i^t for each agent
            target_agents: list of agent indices to apply inequity aversion
            alpha: inequity aversion coefficient (when other agents' rewards are greater than self)
            beta: inequity aversion coefficient (when self's rewards are greater than others)
        Returns:
            subjective_rewards: adjusted subjective rewards u_i^t after inequity aversion
        """
        # Ensure correct input shape
        assert array.shape == (self.num_agents, 1), f"Expected shape ({self.num_agents}, 1), got {array.shape}"
        
        # Calculate inequality using immediate rewards
        r_i = array  # [num_agents, 1]
        r_j = jnp.transpose(array)  # [1, num_agents]
        
        # Calculate inequality
        disadvantageous = jnp.maximum(r_j - r_i, 0)  # when other agents' rewards are higher
        advantageous = jnp.maximum(r_i - r_j, 0)     # when self's rewards are higher
        
        # Create mask to exclude self-comparison
        mask = 1 - jnp.eye(self.num_agents)
        disadvantageous = disadvantageous * mask
        advantageous = advantageous * mask
        
        # Calculate inequality penalty
        n_others = self.num_agents - 1
        inequity_penalty = (alpha * jnp.sum(disadvantageous, axis=1, keepdims=True) +
                           beta * jnp.sum(advantageous, axis=1, keepdims=True)) / n_others

        # Calculate subjective rewards u_i^t = r_i^t - inequality penalty
        subjective_rewards = array - inequity_penalty

        subjective_rewards = jnp.where(jnp.all(array == 0), -(alpha + beta) * n_others, subjective_rewards)
        
        # Apply inequity aversion only to target agents if specified
        if target_agents is not None:
            target_agents_array = jnp.array(target_agents)
            agent_mask = jnp.zeros(self.num_agents, dtype=bool)
            agent_mask = agent_mask.at[target_agents_array].set(True)
            agent_mask = agent_mask.reshape(-1, 1)  # [num_agents, 1]
            return jnp.where(agent_mask, subjective_rewards, array),jnp.sum(disadvantageous, axis=1, keepdims=True),jnp.sum(advantageous, axis=1, keepdims=True)
        else:
            return subjective_rewards,jnp.sum(disadvantageous, axis=1, keepdims=True),jnp.sum(advantageous, axis=1, keepdims=True)

    def get_svo_rewards(self, array, w=0.5, ideal_angle_degrees=45, target_agents=None):
        """
        Reward shaping function based on Social Value Orientation (SVO)
        
        Args:
            array: shape: [num_agents, 1] immediate rewards r_i for each agent
            w: SVO weight to balance self-reward and social value (0 <= w <= 1)
               w=0 means completely selfish, w=1 means completely altruistic
            ideal_angle_degrees: ideal angle in degrees
               - 45 degrees means complete equality
               - 0 degrees means completely selfish
               - 90 degrees means completely altruistic
            target_agents: list of agent indices to apply SVO
        
        Returns:
            shaped_rewards: rewards adjusted by SVO
            theta: reward angle in radians
        """
        # Ensure correct input shape
        assert array.shape == (self.num_agents, 1), f"Expected shape ({self.num_agents}, 1), got {array.shape}"
        
        # Convert ideal angle from degrees to radians
        ideal_angle = (ideal_angle_degrees * jnp.pi) / 180.0
        
        # Calculate group average reward r_j (excluding self)
        mask = 1 - jnp.eye(self.num_agents)  # [num_agents, num_agents]
        # Modified: use matrix multiplication to calculate other agents' rewards
        others_rewards = jnp.matmul(mask, array)  # [num_agents, 1]
        mean_others = others_rewards / (self.num_agents - 1)  # divide by number of other agents
        
        # Calculate reward angle θ(R) = arctan(r_j / r_i)
        r_i = array  # [num_agents, 1]
        r_j = mean_others  # [num_agents, 1]
        theta = jnp.arctan2(r_j, r_i)
        
        # Calculate social value oriented utility
        # U(r_i, r_j) = r_i - w * |θ(R) - ideal_angle|
        angle_deviation = jnp.abs(theta - ideal_angle)
        svo_utility = r_i - self.num_agents * w * angle_deviation

        # Apply SVO only to target agents if specified
        if target_agents is not None:
            target_agents_array = jnp.array(target_agents)
            agent_mask = jnp.zeros(self.num_agents, dtype=bool)
            agent_mask = agent_mask.at[target_agents_array].set(True)
            agent_mask = agent_mask.reshape(-1, 1)  # [num_agents, 1]
            return jnp.where(agent_mask, svo_utility, array), theta
        else:
            return svo_utility, theta

    def get_standardized_svo_rewards(self, array, w=0.5, ideal_angle_degrees=45, target_agents=None):
        """
        Reward shaping function based on Social Value Orientation (SVO)
        """
        # Ensure correct input shape
        assert array.shape == (self.num_agents, 1), f"Expected shape ({self.num_agents}, 1), got {array.shape}"
        
        # Convert ideal angle from degrees to radians
        ideal_angle = (ideal_angle_degrees * jnp.pi) / 180.0
        
        # Calculate group average reward r_j (excluding self)
        mask = 1 - jnp.eye(self.num_agents)
        others_rewards = jnp.matmul(mask, array)
        mean_others = others_rewards / (self.num_agents - 1)
        
        # Calculate reward angle θ(R) = arctan(r_j / r_i)
        r_i = array
        r_j = mean_others
        theta = jnp.arctan2(r_j, r_i)
        
        # Convert angle to [0, 2π] range
        theta = (theta + 2 * jnp.pi) % (2 * jnp.pi)
        
        # Calculate angle deviation and normalize to [0, 1] range
        angle_deviation = jnp.abs(theta - ideal_angle)
        angle_deviation = jnp.minimum(angle_deviation, 2 * jnp.pi - angle_deviation)  # take minimum deviation
        normalized_deviation = angle_deviation / jnp.pi  # normalize to [0, 1]
        
        # Use multiplicative form of penalty instead of subtraction
        svo_utility = r_i * (1 - w * normalized_deviation)
        
        # Apply SVO only to target agents if specified
        if target_agents is not None:
            target_agents_array = jnp.array(target_agents)
            agent_mask = jnp.zeros(self.num_agents, dtype=bool)
            agent_mask = agent_mask.at[target_agents_array].set(True)
            agent_mask = agent_mask.reshape(-1, 1)
            return jnp.where(agent_mask, svo_utility, array), theta
        else:
            return svo_utility, theta

    def get_cf_regret(self, cf_rewards, actions):
        """
        计算每个智能体的cf regret（反事实遗憾）。
        Args:
            cf_rewards: jnp.ndarray, 形状为[num_agents, num_actions]，每个智能体每个动作的反事实奖励
            actions: jnp.ndarray, 形状为[num_agents]，每个智能体的实际动作
        Returns:
            cf_regret: jnp.ndarray, 形状为[num_agents]，每个智能体的cf regret
        """
        # 1. 对每个智能体，找到最大反事实奖励
        max_cf_reward = jnp.max(cf_rewards, axis=1)  # [num_agents]
        # 2. 取实际动作下的反事实奖励
        actual_cf_reward = cf_rewards[jnp.arange(self.num_agents), actions]  # [num_agents]
        # 3. 计算cf regret
        cf_regret = max_cf_reward - actual_cf_reward
        return cf_regret

    def get_cf_regret_from_state(self, key, state, actions):
        """
        计算每个智能体的cf regret（反事实遗憾），通过枚举每个agent的所有可能动作，其他agent动作不变，调用环境获得奖励。
        Args:
            key: jax.random.PRNGKey
            state: 当前环境状态
            actions: jnp.ndarray, 形状为[num_agents]，每个智能体的实际动作
        Returns:
            cf_regret: jnp.ndarray, 形状为[num_agents]，每个智能体的cf regret
        """
        num_agents = self.num_agents
        num_actions = self.num_actions

        def agent_cf_rewards(agent_id):
            def single_action_cf(a_cf):
                # 构造反事实动作
                cf_actions = actions.at[agent_id].set(a_cf)
                # 调用环境获得奖励
                _, _, rewards, _, _ = self.step_env(key, state, cf_actions)
                return rewards[agent_id]
            # 对该agent所有动作枚举
            return jax.vmap(single_action_cf)(jnp.arange(num_actions))  # [num_actions]

        # 对所有agent批量计算cf_rewards
        cf_rewards = jax.vmap(agent_cf_rewards)(jnp.arange(num_agents))  # [num_agents, num_actions]
        # 计算cf regret
        cf_regret = self.get_cf_regret(cf_rewards, actions)
        return cf_regret

    def get_simple_cf_regret(self, rewards):
        """
        近似cf regret，只基于当前reward array。
        Args:
            rewards: jnp.ndarray, 形状为[num_agents, 1]
        Returns:
            regret: jnp.ndarray, 形状为[num_agents]
        """
        max_reward = jnp.max(rewards)  # 全体agent中最大即时奖励
        regret = max_reward - rewards.squeeze()
        return regret

