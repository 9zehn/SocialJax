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
import math
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw

import socialjax
from algorithms.utils.io_utils import load_params


def _extract_pay_events(info, prev_state):
    """(sender, receiver, sender_loc, receiver_loc) for this step's executed payments.

    Positions are taken from prev_state (before the step), matching what
    compute_pay_transfers itself used to decide who's in range -- the receiver may
    have since moved by the time a later frame displays this event, but the arrow
    is a fixed annotation of where the payment happened, not a tracker.
    Returns [] for envs/modes with no pay mechanism (info won't have these keys).
    """
    if "pay_executed" not in info or "pay_target" not in info:
        return []
    executed = np.atleast_1d(np.array(info["pay_executed"])).astype(bool)
    target = np.atleast_1d(np.array(info["pay_target"]))
    locs = np.array(prev_state.agent_locs)
    events = []
    for sender in np.nonzero(executed)[0]:
        receiver = int(target[sender])
        events.append((int(sender), receiver, tuple(locs[sender, :2]), tuple(locs[receiver, :2])))
    return events


def rollout(env, params, num_steps, seed):
    """Step the env and collect raw states (fast, sequential — each step depends on the last).

    Rendering is deferred to render_states() so it can be parallelized separately.
    Also returns pay_events, aligned with states: pay_events[i] is the list of
    (sender, receiver, sender_loc, receiver_loc) payments that produced states[i]
    (pay_events[0] is always empty -- states[0] is the initial reset).
    """
    rng = jax.random.PRNGKey(seed)
    rng, rng_reset = jax.random.split(rng)
    obs, state = env.reset(rng_reset)

    states = [state]
    pay_events = [[]]

    network = None
    if params is not None:
        from algorithms.utils.networks import ActorCritic
        network = ActorCritic(action_dim=env.action_space().n, activation="relu")
        if isinstance(params, list) and len(params) != env.num_agents:
            raise SystemExit(
                f"got {len(params)} per-agent checkpoints but env has {env.num_agents} agents"
            )

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
                pi, _ = network.apply(params[i], obs[a][None, ...])
                actions.append(int(pi.sample(seed=act_keys[i]).squeeze()))
        else:
            obs_batch = jnp.stack([obs[a] for a in env.agents])
            pi, _ = network.apply(params, obs_batch)
            sampled = pi.sample(seed=rng_act)
            actions = [int(sampled[i]) for i in range(env.num_agents)]

        prev_state = state
        obs, state, reward, done, info = env.step(rng_step, state, actions)
        states.append(state)
        pay_events.append(_extract_pay_events(info, prev_state))

        if bool(done["__all__"]):
            break

    return states, pay_events


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


def _agent_pixel_center(env, row, col, frame_height):
    """Map an unpadded (row, col) grid position to a pixel (x, y) center in the
    image env.render() produces.

    Specific to clean_up.py's render() pipeline: it pads the grid by env.PADDING,
    draws tile_size=32px tiles, then crops (PADDING-1)*tile_size off each edge and
    rotates 180 degrees. Working through that transform: a cell at unpadded row r
    ends up centered at pixel row tile_size*(GRID_SIZE_ROW - r + 0.5) (symmetric
    for columns) -- derived once here rather than re-deriving per call. Returns
    None for envs without GRID_SIZE_ROW/COL (i.e. this is a no-op there).
    """
    grid_rows = getattr(env, "GRID_SIZE_ROW", None)
    grid_cols = getattr(env, "GRID_SIZE_COL", None)
    if grid_rows is None or grid_cols is None:
        return None
    tile_size = frame_height / (grid_rows + 2)
    cy = tile_size * (grid_rows - row + 0.5)
    cx = tile_size * (grid_cols - col + 0.5)
    return cx, cy


def _draw_arrow(draw, p1, p2, color=(255, 215, 0), width=3, head_len=12):
    draw.line([p1, p2], fill=color, width=width)
    angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    for offset in (math.radians(150), math.radians(-150)):
        hx = p2[0] + head_len * math.cos(angle + offset)
        hy = p2[1] + head_len * math.sin(angle + offset)
        draw.line([p2, (hx, hy)], fill=color, width=width)


def draw_pay_arrows(frames, pay_events, env, persist_frames=3, color=(255, 215, 0)):
    """Overlay a payer -> payee arrow for `persist_frames` frames starting at the
    frame each payment occurred on. No-op (returns frames unchanged) if there are
    no events at all, or if the env doesn't expose GRID_SIZE_ROW/COL.
    """
    if not any(pay_events) or _agent_pixel_center(env, 0, 0, frames[0].shape[0]) is None:
        return frames

    n = len(frames)
    active = [[] for _ in range(n)]
    for i, events in enumerate(pay_events):
        for event in events:
            for j in range(i, min(i + persist_frames, n)):
                active[j].append(event)

    out = []
    for i, frame in enumerate(frames):
        if not active[i]:
            out.append(frame)
            continue
        img = Image.fromarray(frame).convert("RGB")
        draw = ImageDraw.Draw(img)
        h = frame.shape[0]
        for _sender, _receiver, sender_loc, receiver_loc in active[i]:
            p1 = _agent_pixel_center(env, sender_loc[0], sender_loc[1], h)
            p2 = _agent_pixel_center(env, receiver_loc[0], receiver_loc[1], h)
            _draw_arrow(draw, p1, p2, color=color)
        out.append(np.array(img))
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

    fig, ax = plt.subplots(figsize=(6, 7))
    plt.subplots_adjust(bottom=0.22)
    im = ax.imshow(frames[0])
    ax.axis("off")
    ax.set_title(_frame_title(0, traces), fontsize=8)

    ax_slider = plt.axes([0.15, 0.14, 0.7, 0.03])
    slider = Slider(ax_slider, "Step", 0, len(frames) - 1, valinit=0, valstep=1)

    def show_frame(t):
        im.set_data(frames[t])
        ax.set_title(_frame_title(t, traces), fontsize=8)
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
            ax.set_title(f"{_frame_title(t, traces)}  [saved -> {path.name}]", fontsize=8)
            fig.canvas.draw_idle()
        except Exception as exc:  # surface errors instead of letting the GUI swallow them
            print(f"Save failed: {exc}", flush=True)

    button_specs = [
        ("▶", start_playing),    # play
        ("||", stop_playing),         # pause
        ("◀◀", lambda e: set_speed(1.5)),      # slower
        ("▶▶", lambda e: set_speed(1 / 1.5)),  # faster
        ("↻", toggle_loop),      # loop
        ("Save", save_frame),
    ]
    n = len(button_specs)
    width, gap = 0.13, 0.015
    total = n * width + (n - 1) * gap
    x0 = (1 - total) / 2
    buttons = []
    for i, (label, callback) in enumerate(button_specs):
        ax_btn = plt.axes([x0 + i * (width + gap), 0.05, width, 0.05])
        btn = Button(ax_btn, label)
        btn.on_clicked(callback)
        buttons.append(btn)
    play_button, pause_button, slower_button, faster_button, loop_button, save_button = buttons
    loop_button.ax.set_facecolor("honeydew")

    speed_label = fig.text(0.5, 0.115, f"{1000 / state['interval']:.1f} fps", fontsize=8, ha="center")

    plt.show()


def _frame_title(t, traces):
    locs = traces[t].get("locs")
    return f"step {t}" if locs is None else f"step {t}  agent_locs={locs.tolist()}"


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


def _load_checkpoint(arg):
    """Load --checkpoint: single .pkl -> shared params; glob with N matches -> per-agent list."""
    import glob

    matches = sorted(glob.glob(arg))
    if not matches:
        raise SystemExit(f"no checkpoint file matches {arg!r}")
    if len(matches) == 1:
        return load_params(matches[0])
    print(f"Loading {len(matches)} per-agent checkpoints (sorted -> agent 0..{len(matches) - 1})")
    return [load_params(m) for m in matches]


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
    parser.add_argument("--pay-arrow-frames", type=int, default=3,
                        help="how many frames a payer->payee arrow stays visible for (0 disables)")
    args = parser.parse_args()

    env_kwargs = {}
    for kv in args.env_kwarg:
        key, _, raw = kv.partition("=")
        if not _:
            raise SystemExit(f"--env-kwarg expects KEY=VALUE, got {kv!r}")
        env_kwargs[key] = _parse_env_kwarg_value(raw)
    if args.no_jit:
        env_kwargs["jit"] = False
    if args.num_agents is not None:
        env_kwargs["num_agents"] = args.num_agents
    env = socialjax.make(args.env, **env_kwargs)

    _warn_if_fixed_agent_count(env, args.env, args.num_agents)

    params = _load_checkpoint(args.checkpoint) if args.checkpoint else None

    states, pay_events = rollout(env, params, args.steps, args.seed)
    print(f"Rolled out {len(states) - 1} steps ({'trained checkpoint' if params else 'random policy'}).")

    traces = [_agent_snapshot(s) for s in states]
    frames = render_states(env, args.env, env_kwargs, states, workers=args.render_workers)
    print(f"Rendered {len(frames)} frames.")

    if args.pay_arrow_frames > 0:
        n_events = sum(len(e) for e in pay_events)
        if n_events:
            frames = draw_pay_arrows(frames, pay_events, env, persist_frames=args.pay_arrow_frames)
            print(f"Drew {n_events} payment arrow(s).")

    if args.gif:
        save_gif(frames, args.gif)
        print(f"Saved GIF to {args.gif}")

    if not args.no_interactive:
        interactive_view(frames, traces)


if __name__ == "__main__":
    main()
