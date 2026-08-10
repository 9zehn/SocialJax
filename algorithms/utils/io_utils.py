"""
Input/Output utilities for saving and loading model parameters.

This module provides functions for persisting and restoring trained model parameters
across different MARL algorithms. All functions use pickle serialization and JAX
tree mapping for efficient parameter conversion.
"""

import os
import pickle
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState


def checkpoint_filename(config: Dict[str, Any], latest: bool = False) -> str:
    """Deterministic checkpoint filename stem (no extension, no per-agent suffix).

    Shared by the final save (_runner.py) and the periodic rolling save
    (ippo_cnn_<env>.py's checkpoint_callback) so the two can't drift apart.

    Encodes every config value that would otherwise let two different runs
    silently collide onto the same path: originally this only included
    ENV_NAME/SEED/REWARD, so e.g. three pay_mode conditions (off/noop/on) run
    with the same SEED and reward=individual -- exactly the setup a controlled
    comparison needs -- would all resolve to one identical filename, with each
    run silently overwriting the last one's checkpoint with no error at all.

    Args:
        config: the run's Hydra config dict.
        latest: True for the periodic rolling checkpoint (adds a "_latest"
            marker so it's visually distinct from, and never confused with,
            the final end-of-training checkpoint at the same base name).
    """
    reward = config.get("REWARD")
    suffix = f"_reward_{reward}" if reward else ""

    env_kwargs = config.get("ENV_KWARGS", {})
    pay_mode = env_kwargs.get("pay_mode")
    if pay_mode and pay_mode != "off":  # "off" is clean_up's own default; keep it unmarked
        suffix += f"_pay_{pay_mode}"
        # Sub-parameters are only meaningful when pay_mode is active, and only
        # marked when swept away from clean_up's own defaults, so existing
        # pay_on/pay_noop runs at default settings keep their current filenames.
        pay_scheme = env_kwargs.get("pay_scheme")
        if pay_scheme and pay_scheme != "instant":  # "instant" is the default scheme
            suffix += f"_{pay_scheme}"
            # tithe's own knobs (fraction/duration), marked only off-default too
            share_fraction = env_kwargs.get("share_fraction")
            if share_fraction is not None and share_fraction != 0.5:
                suffix += f"_f{share_fraction}"
            share_duration = env_kwargs.get("share_duration")
            if share_duration is not None and share_duration != 50:
                suffix += f"_d{share_duration}"
            # recipient rule: split the tithe across all recent cleaners vs. the single
            # most-recent one (False, clean_up's default) -- marked so split runs get
            # their own checkpoint path and never overwrite winner-take-all ones.
            if env_kwargs.get("split_recipients"):
                suffix += "_split"
        pay_clean_window = env_kwargs.get("pay_clean_window")
        if pay_clean_window and pay_clean_window != 50:
            suffix += f"_win{pay_clean_window}"
    num_agents = env_kwargs.get("num_agents")
    if num_agents:
        suffix += f"_agents{num_agents}"

    # MOCA's phase-2 mode. The three modes are meant to be run against each other
    # at the same seed and reward -- the controlled comparison this function's
    # whole purpose is to keep from colliding -- and without this every one of
    # them resolves to the same path. Marked for all modes rather than only
    # off-default ones, so no MOCA run can overwrite another. Absent from every
    # other algorithm's config, so nothing outside MOCA is affected.
    phase2_mode = config.get("PHASE2_MODE")
    if phase2_mode:
        suffix += f"_{phase2_mode}"
        # Solver arms differ only in the decision rule, and are meant to be run
        # against one shared phase-1 policy at the same seed -- so without the rule
        # in the name every arm of that comparison lands on one path.
        if phase2_mode == "solver":
            rule = config.get("SOLVER_DECISION_RULE")
            if rule:
                suffix += f"_{rule}"
        elif phase2_mode == "negotiate":
            # nu changes who gates the contract and so what gets learned; a nu=2 and
            # a nu=all run at the same seed would otherwise overwrite each other.
            nu = config.get("NEGOTIATE_NU")
            if nu:
                suffix += f"_nu{nu}"

    name = f'{config["ENV_NAME"]}_seed{config["SEED"]}{suffix}'
    return f"{name}_latest" if latest else name


def save_params(train_state: TrainState, save_path: str) -> None:
    """
    Save model parameters to disk.

    This function extracts parameters from a Flax TrainState, converts them
    to numpy arrays for serialization, and saves them as a pickle file.

    Args:
        train_state: Flax TrainState containing the model parameters to save.
        save_path: Path where the parameters will be saved (typically .pkl file).
                   Parent directories will be created if they don't exist.

    Example:
        >>> train_state = TrainState.create(...)
        >>> save_params(train_state, "./checkpoints/model_seed42.pkl")

    Note:
        The function creates the directory structure if it doesn't exist.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    params = jax.tree_util.tree_map(lambda x: np.array(x), train_state.params)

    with open(save_path, 'wb') as f:
        pickle.dump(params, f)


def load_params(load_path: str) -> Dict[str, Any]:
    """
    Load model parameters from disk.

    This function loads parameters from a pickle file and converts them
    back to JAX arrays for use in model inference or further training.

    Args:
        load_path: Path to the saved parameters file (.pkl file).

    Returns:
        Dictionary containing the loaded model parameters as JAX arrays,
        structured according to the original model architecture.

    Example:
        >>> params = load_params("./checkpoints/model_seed42.pkl")
        >>> network = ActorCritic(...)
        >>> pi, value = network.apply(params, obs)

    Note:
        The returned parameters are in JAX array format and ready for use
        with network.apply() or other JAX operations.
    """
    with open(load_path, 'rb') as f:
        params = pickle.load(f)
    return jax.tree_util.tree_map(lambda x: jnp.array(x), params)


def save_train_state(train_state: TrainState, update_step: int, save_path: str) -> None:
    """Save enough of a TrainState to properly *resume* training later.

    save_params() only keeps .params, which is enough for evaluation/inference
    but NOT enough to resume from: restarting with a fresh optimizer state resets
    Adam's moment estimates and the LR schedule's internal step count back to
    zero, silently changing training dynamics rather than truly continuing. This
    additionally saves .opt_state, .step (the optimizer's own step counter, which
    the LR schedule reads from), and update_step (this repo's own outer-loop
    counter, used to resume progress/checkpoint logging and to compute how many
    scan iterations remain).

    Args:
        train_state: the TrainState to snapshot (params + optimizer state).
        update_step: this run's current outer-loop update counter (distinct from
            train_state.step, the optimizer's internal gradient-step counter).
        save_path: where to write the pickle; parent dirs created if needed.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    payload = {
        "params": jax.tree_util.tree_map(lambda x: np.array(x), train_state.params),
        "opt_state": jax.tree_util.tree_map(lambda x: np.array(x), train_state.opt_state),
        "step": int(train_state.step),
        "update_step": int(update_step),
    }
    with open(save_path, 'wb') as f:
        pickle.dump(payload, f)


def load_train_state(load_path: str) -> Dict[str, Any]:
    """Load a save_train_state() payload.

    Returns {"params", "opt_state", "step", "update_step"} with params/opt_state
    as JAX arrays, ready to splice into a freshly-created TrainState via
    `train_state.replace(params=loaded["params"], opt_state=loaded["opt_state"],
    step=loaded["step"])` -- apply_fn/tx aren't serialized here since they're
    reconstructed fresh from the (assumed-matching) network/optimizer config.
    """
    with open(load_path, 'rb') as f:
        payload = pickle.load(f)
    return {
        "params": jax.tree_util.tree_map(lambda x: jnp.array(x), payload["params"]),
        "opt_state": jax.tree_util.tree_map(lambda x: jnp.array(x), payload["opt_state"]),
        "step": payload["step"],
        "update_step": payload["update_step"],
    }
