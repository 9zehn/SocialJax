"""Compact rollout recordings and a fast vectorised renderer.

Two problems with rendering straight from the env:

* `Clean_up.render()` loops in Python over the PADDED grid -- 39x48 = 1872 tiles per
  frame -- and inside that loop calls `state.agent_locs[a, 2].item()`, forcing a
  device sync per agent per tile. That is why a 300-step episode takes minutes.
* Nothing is kept afterwards, so reviewing a rollout again means re-running the whole
  simulation (and, with per-agent networks, re-running the policies too).

This module fixes both. A *recording* stores the per-step grid plus the panel stats --
the objects in the world, not pixels -- in a single .npz, which is tiny (the grid is
one byte per cell, ~0.5 KB per step) and can be reopened later without JAX, the
policies, or the env. `render_recording()` then rasterises it with numpy only:
palette lookup + block upscale + a handful of stamped sprites, no per-tile Python.
"""
import json

import numpy as np

RECORDING_VERSION = 1

# Colours copied from Clean_up.render_tile so recordings look like the env's own
# renderer. Keyed by the Items enum values in clean_up.py.
_BACKGROUND = (190, 170, 120)
_ITEM_COLOURS = {
    1: (127, 127, 127),    # wall
    2: (188, 189, 34),     # interact / zap beam
    3: _BACKGROUND,        # apple -- drawn as a circle on the background, see below
    4: _BACKGROUND,        # spawn point
    5: _BACKGROUND,        # inside spawn point
    6: (40, 80, 214),      # river
    7: (40, 80, 214),      # potential dirt
    8: (40, 80, 80),       # dirt
    9: (170, 220, 255),    # clean beam
}
_APPLE_COLOUR = (214, 39, 40)
_APPLE_CODE = 3


def save_recording(path, grids, agent_locs, panel_data, meta):
    """Write a rollout to `path` (.npz).

    Args:
        grids: (T, R, C) int array of item codes -- the whole world state per step.
        agent_locs: (T, N, 3) int array of (row, col, orientation).
        panel_data: list of per-step dicts from collect_panel_data (may be None).
        meta: JSON-serialisable dict (env config, colours, contract info, ...).
    """
    arrays = {
        "grids": np.asarray(grids, dtype=np.int16),
        "agent_locs": np.asarray(agent_locs, dtype=np.int16),
        "meta": np.array(json.dumps(meta)),
    }
    if panel_data:
        for key in ("balance", "clean", "transfer", "share"):
            if key in panel_data[0]:
                arrays[f"panel_{key}"] = np.stack(
                    [np.asarray(d[key]) for d in panel_data]
                )
    np.savez_compressed(path, **arrays)
    return path


def load_recording(path):
    """Read a recording written by save_recording()."""
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    if meta.get("version") != RECORDING_VERSION:
        raise SystemExit(
            f"{path}: recording version {meta.get('version')} != {RECORDING_VERSION}; "
            "re-record with the current viewer"
        )
    panel = {k[len("panel_"):]: z[k] for k in z.files if k.startswith("panel_")}
    panel_data = None
    if panel:
        n_steps = len(z["grids"])
        panel_data = [{k: v[t] for k, v in panel.items()} for t in range(n_steps)]
    return {
        "grids": z["grids"],
        "agent_locs": z["agent_locs"],
        "panel_data": panel_data,
        "meta": meta,
    }


def _triangle_mask(tile_size, direction):
    """Boolean sprite for an agent facing `direction`, matching the env's triangle.

    The env draws a triangle then rotates it by 0.5*pi*(1 - dir) about the tile centre;
    we rasterise the same shape once per direction and reuse it for every agent.
    """
    ts = tile_size
    ys, xs = np.mgrid[0:ts, 0:ts]
    x = (xs + 0.5) / ts
    y = (ys + 0.5) / ts
    # base triangle (0.12,0.19) (0.87,0.50) (0.12,0.81) in tile coords
    theta = 0.5 * np.pi * (1 - direction)
    cx = cy = 0.5
    ct, st = np.cos(-theta), np.sin(-theta)
    xr = cx + (x - cx) * ct - (y - cy) * st
    yr = cy + (x - cx) * st + (y - cy) * ct
    ax, ay = 0.12, 0.19
    bx, by = 0.87, 0.50
    cx3, cy3 = 0.12, 0.81

    def side(px, py, qx, qy):
        return (xr - px) * (qy - py) - (yr - py) * (qx - px)

    d1 = side(ax, ay, bx, by)
    d2 = side(bx, by, cx3, cy3)
    d3 = side(cx3, cy3, ax, ay)
    neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(neg & pos)


def _circle_mask(tile_size, radius=0.31):
    ts = tile_size
    ys, xs = np.mgrid[0:ts, 0:ts]
    x = (xs + 0.5) / ts - 0.5
    y = (ys + 0.5) / ts - 0.5
    return (x * x + y * y) <= radius * radius


def render_recording(grids, agent_locs, meta, tile_size=32):
    """Rasterise a whole recording to RGB frames, using numpy only.

    Mirrors Clean_up.render()'s geometry (pad with wall, crop (PADDING-1) tiles off
    each edge, rotate 180) so frames line up with the env renderer and with the
    viewer's existing pixel-coordinate maths for arrows.

    One deliberate difference: the env renderer tints every cell inside an agent's
    observation window (highlight_img, alpha 0.30 toward white). That is a debug aid
    for field-of-view, it washes out large parts of the map, and reproducing it means
    re-deriving the jitted, rotation-dependent obs-window geometry. It is omitted here,
    which is why frames look cleaner (and differ pixel-wise) from the env renderer.
    Terrain, items, agents and their positions/colours are identical; use the viewer's
    --slow-render if you specifically need the env's own output.
    """
    grids = np.asarray(grids)
    agent_locs = np.asarray(agent_locs)
    n_items = meta["n_items"]
    padding = meta["padding"]
    colours = [tuple(c) for c in meta["player_colours"]]
    n_agents = len(colours)

    # Palette: item codes, then one entry per agent id (agents are drawn as sprites on
    # the background, so their palette entry is just the ground colour).
    max_code = n_items + n_agents
    palette = np.zeros((max_code + 1, 3), dtype=np.uint8)
    palette[:] = _BACKGROUND
    for code, col in _ITEM_COLOURS.items():
        if code <= max_code:
            palette[code] = col
    palette[0] = _BACKGROUND

    tri = np.stack([_triangle_mask(tile_size, d) for d in range(4)])
    circ = _circle_mask(tile_size)

    frames = []
    for t in range(len(grids)):
        g = np.pad(grids[t], padding, constant_values=1)  # 1 == Items.wall
        # Palette lookup + block upscale: the whole grid in two vectorised ops.
        small = palette[np.clip(g, 0, max_code)]
        img = np.repeat(np.repeat(small, tile_size, axis=0), tile_size, axis=1)

        # Apples are circles rather than filled cells.
        apple_big = np.repeat(np.repeat(g == _APPLE_CODE, tile_size, 0), tile_size, 1)
        circ_big = np.tile(circ, g.shape)
        img[apple_big & circ_big] = _APPLE_COLOUR

        # Agents: a handful of stamps, so a small loop is fine here.
        for a in range(n_agents):
            r, c, d = agent_locs[t, a]
            rr, cc = int(r) + padding, int(c) + padding
            y0, x0 = rr * tile_size, cc * tile_size
            sub = img[y0:y0 + tile_size, x0:x0 + tile_size]
            if sub.shape[:2] == (tile_size, tile_size):
                sub[tri[int(d) % 4]] = colours[a]

        crop = (padding - 1) * tile_size
        frames.append(np.rot90(img[crop:-crop, crop:-crop, :], 2).copy())
    return frames
