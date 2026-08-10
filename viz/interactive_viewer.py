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
from algorithms.utils.io_utils import load_params



def rollout(env, params, num_steps, seed, contract=None, theta=None):
    """Step the env and collect raw states (fast, sequential — each step depends on the last).

    Rendering is deferred to render_states() so it can be parallelized separately.

    Returns (states, extras). extras["cleaned"] is the per-step (N,) count of dirt
    cells each agent cleared, and extras["transfers"] the per-step (N,) zero-sum
    contract transfer (all zeros unless a MOCA `contract` and `theta` are supplied,
    in which case the gameplay policy is the contract-conditioned network).
    """
    use_contract = contract is not None and theta is not None
    rng = jax.random.PRNGKey(seed)
    rng, rng_reset = jax.random.split(rng)
    obs, state = env.reset(rng_reset)

    states = [state]
    transfers_per_step = [np.zeros(env.num_agents)]
    cleaned_per_step = [np.zeros(env.num_agents, dtype=np.float32)]

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

        cleaned = np.atleast_1d(np.array(info["cleaned_by_agent"], dtype=np.float32)) \
            if "cleaned_by_agent" in info else np.zeros(env.num_agents, dtype=np.float32)
        cleaned_per_step.append(cleaned)

        if use_contract:
            tr = np.array(contract.compute_transfer(jnp.float32(theta), jnp.asarray(cleaned)))
            transfers_per_step.append(tr)
        else:
            transfers_per_step.append(np.zeros(env.num_agents))

        if bool(done["__all__"]):
            break

    return states, {"transfers": transfers_per_step, "cleaned": cleaned_per_step}



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


def recording_meta(env, contract_info=None):
    """Everything render_recording() needs to rasterise without the env or JAX."""
    from socialjax.environments.cleanup.clean_up import Items

    return {
        "version": 1,
        "n_items": len(Items),
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
    """The panel needs per-agent balances/colors, which only clean_up exposes."""
    return (
        len(states) > 0
        and hasattr(states[0], "agent_balance")
        and getattr(env, "PLAYER_COLOURS", None) is not None
    )


def collect_panel_data(states, env, transfers=None, cleaned=None):
    """Per-step, per-agent stats for the info panel, aligned with `states`.

    Returns a list (one entry per state) of dicts:
        balance:   (N,) cumulative net reward = each agent's spendable balance.
        clean:     (N,) running count of dirt CELLS the agent has cleared. Taken from
                   the env's per-step info["cleaned_by_agent"] when `cleaned` is
                   supplied. The fallback (last_clean_t changing) can only count
                   cleaning STEPS, and the beam covers 4 tiles, so it under-reports
                   whenever an agent clears more than one cell in a single action.
        share:     (N,) bool, share-mode toggle currently ON (tithe scheme only;
                   all-False otherwise).
        transfer:  (N,) CUMULATIVE contract transfer received (negative = net
                   funder). Only meaningful under MOCA; zeros otherwise.

    Note the contract case needs `transfers` passed in: contract transfers are
    computed by the viewer during the rollout, not stored in env State the way the
    pay mechanism's balance is.
    """
    n = env.num_agents
    clean_counts = np.zeros(n, dtype=int)
    cum_transfer = np.zeros(n)
    prev_lct = None
    out = []
    for idx, s in enumerate(states):
        if cleaned is not None:
            if idx < len(cleaned):
                clean_counts = clean_counts + np.asarray(cleaned[idx]).reshape(-1).astype(int)
        else:
            lct = np.array(s.last_clean_t) if hasattr(s, "last_clean_t") else None
            if lct is not None and prev_lct is not None:
                clean_counts = clean_counts + (lct != prev_lct).astype(int)
            prev_lct = lct
        balance = np.array(s.agent_balance) if hasattr(s, "agent_balance") else np.zeros(n)
        if hasattr(s, "share_expiry_t"):
            share = np.array(s.share_expiry_t) > int(s.inner_t)
        else:
            share = np.zeros(n, dtype=bool)
        if transfers is not None and idx < len(transfers):
            cum_transfer = cum_transfer + np.asarray(transfers[idx]).reshape(-1)
        out.append({"balance": np.asarray(balance).reshape(-1),
                    "clean": clean_counts.copy(),
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
        return f"{reward} · contract θ={theta:g}/cell{(' ' + src) if src else ''}"
    mode = getattr(env, "pay_mode", "off")
    if mode == "off":
        return f"{reward} · no payments"
    scheme = getattr(env, "pay_scheme", "?")
    tag = "" if mode == "on" else " [placebo]"
    if scheme == "tithe":
        who = "→all cleaners" if getattr(env, "split_recipients", False) else "→latest cleaner"
        return f"{reward} · tithe {getattr(env, 'share_fraction', 0.5):g} {who}{tag}"
    return f"{reward} · instant pays {getattr(env, 'pay_amount', 1.0):g}{tag}"


def render_info_panel(height, datum, colors, env, step, width, contract_info=None):
    """Render one info-panel frame (RGB, height x width) for a single step.

    First column is each agent's color (a swatch + a color bar down the row edge)
    so every stat reads back to the matching agent in the grid.
    Columns: color/label | Reward (balance) | Clean (count) | and then either
    Share (ON/OFF) for the tithe toggle, or Transfer (cumulative net contract
    transfer, + receiver / - funder) under a MOCA contract.
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
    draw.text((pad, y), _scheme_caption(env, contract_info), font=head_f, fill=_PANEL_MUTED)
    total_reward = float(np.sum(datum["balance"]))
    total_clean = int(np.sum(datum["clean"]))
    y += int(head_f.size * 1.5)
    totals = f"total reward {total_reward:.1f}    total clean {total_clean}"
    if show_contract:
        # Under a zero-sum contract the transfers must cancel; showing the moved
        # volume instead makes "is the contract doing anything" readable at a glance.
        moved = float(np.sum(np.maximum(datum["transfer"], 0.0)))
        totals += f"    moved {moved:.2f}"
    draw.text((pad, y), totals, font=head_f, fill=_PANEL_MUTED)

    # Column x anchors (right-aligned value columns). The rightmost column (Share's
    # dot + ON/OFF, or the signed Transfer figure) needs the widest slot; leave a
    # clear gap between it and the Clean number so they never run together.
    x_last = width - pad                                             # rightmost
    has_last = show_share or show_contract
    x_clean = (x_last - int(width * 0.26)) if has_last else (width - pad)
    x_reward = x_clean - int(width * 0.20)

    # Column header row.
    y_head = y + int(head_f.size * 1.9)
    draw.text((x_reward, y_head), "Reward", font=head_f, fill=_PANEL_MUTED, anchor="ra")
    draw.text((x_clean, y_head), "Clean", font=head_f, fill=_PANEL_MUTED, anchor="ra")
    if show_share:
        draw.text((x_last, y_head), "Share", font=head_f, fill=_PANEL_MUTED, anchor="ra")
    elif show_contract:
        draw.text((x_last, y_head), "Transfer", font=head_f, fill=_PANEL_MUTED, anchor="ra")
    x_share = x_last

    # Rows.
    top = y_head + int(head_f.size * 1.5)
    avail = height - top - pad
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

        # Reward + clean values.
        draw.text((x_reward, cy), f"{float(datum['balance'][i]):.1f}", font=cell_f,
                  fill=_PANEL_FG, anchor="rm")
        draw.text((x_clean, cy), f"{int(datum['clean'][i])}", font=cell_f,
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

        # Cumulative contract transfer: green = net receiver (was subsidised for
        # cleaning), red = net funder. Reading this column tells you at a glance
        # whether the contract actually moved money toward the cleaners.
        elif show_contract:
            t = float(datum["transfer"][i])
            if t > 1e-9:
                fill, txt = _PANEL_ON, f"+{t:.2f}"
            elif t < -1e-9:
                fill, txt = _PANEL_OFF, f"{t:.2f}"
            else:
                fill, txt = _PANEL_MUTED, "0.00"
            draw.text((x_share, cy), txt, font=cell_f, fill=fill, anchor="rm")

    return np.array(img)


def add_info_panels(frames, panel_data, colors, env, width=None, contract_info=None):
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
        panel = render_info_panel(h, panel_data[i], colors, env, i, width,
                                  contract_info=contract_info)
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

    contracts = sorted(glob.glob(f"{stem}_contract_*.pkl"))
    if not contracts:  # runs written before the role suffix was disambiguated
        contracts = sorted(glob.glob(f"{stem}_negotiate_*.pkl"))
    if contracts:
        return {"mode": "negotiate", "contract_paths": contracts, "gameplay": gameplay}

    # Solver runs save no contracting policy, so the stem is the only evidence.
    base = os.path.basename(stem)
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


def _infer_env_kwargs_from_checkpoint(checkpoint_arg):
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

    Returns a dict of env kwargs (possibly empty).
    """
    import glob
    import re

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
    if "pay_noop" in name:
        kw["pay_mode"] = "noop"
    elif "pay_on" in name:
        kw["pay_mode"] = "on"
    else:
        kw["pay_mode"] = "off"

    # Scheme + its off-default knobs only matter when pay is active.
    if kw["pay_mode"] in ("on", "noop"):
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
        kw["apple_reward"] = 1.0
        kw["pay_mode"] = "off"
    return kw


def _report_env_config(env, checkpoint_arg):
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
    extra = ""
    if pay_mode != "off" and pay_scheme == "tithe":
        extra = (f", recipients={'split-all' if split else 'latest'}, "
                 f"clean_window={getattr(env, 'pay_clean_window', '?')}")
    print(
        f"Env config: reward={'shared/common' if shared else 'individual'}, "
        f"pay_mode={pay_mode}, pay_scheme={pay_scheme}{extra}, "
        f"apple_reward={getattr(env, 'apple_reward', '?')}, num_agents={env.num_agents}"
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True, help="e.g. clean_up, coop_mining, coin_game, harvest_common_open")
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
    parser.add_argument("--contract-low", type=float, default=0.2,
                        help="MOCA only: lower bound of the NON-NULL contract range the run was "
                             "trained on (not encoded in the filename; must match CONTRACT_LOW). "
                             "The null contract theta=0 is always available and is unaffected by "
                             "these bounds. Default tracks moca_base.yaml; pass 0.0 for runs "
                             "trained before the range excluded weak contracts")
    parser.add_argument("--solver-samples", type=int, default=50,
                        help="PHASE2_MODE=solver: contracts sampled and scored by the "
                             "frozen critics at reset (training default: 50)")
    parser.add_argument("--solver-rule", default="majority", choices=("majority", "max"),
                        help="PHASE2_MODE=solver: decision rule (training default: majority)")
    parser.add_argument("--contract-high", type=float, default=1.0,
                        help="MOCA only: upper bound of the contract space (must match the run's "
                             "CONTRACT_HIGH; the bounds are not encoded in the filename, and a "
                             "mismatch silently rescales theta). Default tracks moca_base.yaml; "
                             "pass 0.2 for runs on the paper's original range")
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

    # Recover the training config from the checkpoint name, then let explicit
    # --env-kwarg / --num_agents win over it (inference is a convenience, not a lock).
    inferred = {}
    if args.checkpoint and not args.no_autoconfig:
        inferred = _infer_env_kwargs_from_checkpoint(args.checkpoint)
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
    env = socialjax.make(args.env, **env_kwargs)

    _warn_if_fixed_agent_count(env, args.env, args.num_agents)

    params = _load_checkpoint(args.checkpoint) if args.checkpoint else None

    _report_env_config(env, args.checkpoint)

    # MOCA: pick the contract to replay under. The learned proposal distribution is
    # the run's actual result, so its mode is the default -- what the agents ended up
    # wanting to sign -- with --contract-theta available to replay a counterfactual
    # (notably 0, the null contract, to see what the same policies do unsubsidised).
    contract, theta, contract_info = None, None, None
    moca = detect_moca(args.checkpoint) if args.checkpoint else None
    if moca is not None:
        from algorithms.MOCA.contracts import CleanupContract

        contract = CleanupContract(env.num_agents, args.contract_low, args.contract_high)
        mode = moca["mode"]
        print(f"MOCA run detected (PHASE2_MODE={mode})")
        print(f"  contract space: theta in [{args.contract_low}, {args.contract_high}]"
              + (" (continuous)" if mode != "reinforce" else ""))

        learned = None
        if mode == "reinforce":
            info = load_contract_policies(moca, args.contract_low, args.contract_high)
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

        if args.contract_theta is not None:
            theta, source = float(args.contract_theta), "(manual)"
        else:
            theta, source = learned, "(learned)"
        contract_info = {"theta": theta, "source": source}
        print(f"  replaying under theta={theta:g} {source}")

    states, extras = rollout(
        env, params, args.steps, args.seed, contract=contract, theta=theta
    )
    print(f"Rolled out {len(states) - 1} steps ({'trained checkpoint' if params else 'random policy'}).")

    traces = [_agent_snapshot(s) for s in states]
    t0 = time.time()
    if args.slow_render:
        frames = render_states(env, args.env, env_kwargs, states, workers=args.render_workers)
    else:
        # Vectorised path: one palette lookup + block upscale per frame instead of a
        # Python loop over ~1900 padded tiles (each of which forced a device sync).
        from viz.recording import render_recording

        meta = recording_meta(env, contract_info)
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
                                        cleaned=extras["cleaned"])

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
            recording_meta(env, contract_info),
        )
        size_kb = Path(args.record).stat().st_size / 1024
        print(f"Recorded {len(states)} steps to {args.record} ({size_kb:.0f} KB) "
              f"-- reopen with --replay {args.record}")

    if not args.no_panel and panel_data is not None:
        frames = add_info_panels(frames, panel_data, colors, env, contract_info=contract_info)
        print("Added per-agent info panel.")

    if args.gif:
        save_gif(frames, args.gif)
        print(f"Saved GIF to {args.gif}")

    if not args.no_interactive:
        interactive_view(frames, traces)


if __name__ == "__main__":
    main()
