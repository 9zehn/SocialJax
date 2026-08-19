"""
Input/Output utilities for saving and loading model parameters.

This module provides functions for persisting and restoring trained model parameters
across different MARL algorithms. All functions use pickle serialization and JAX
tree mapping for efficient parameter conversion.
"""

import datetime
import os
import pickle
import re
import subprocess
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import yaml
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

    # Harvest's zap beam. Marked for BOTH settings rather than only the off-default
    # one, the same rule BARGAIN_BINDING follows and for the same reason: with or
    # without the beam are different games at the same seed, and an unmarked name
    # would be ambiguous between "the default of the day it was written" (upstream
    # had the beam) and "the default of the day it is read" (this repo does not).
    # Absent from every other environment's ENV_KWARGS, so no other name changes.
    if "enable_zap" in env_kwargs:
        suffix += "_zap" if env_kwargs["enable_zap"] else "_nozap"

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

    # Rubinstein bargaining. Segment length sets the cost of a rejection, and the
    # proposer rule and quorum are the two design axes the arms differ on -- so all
    # three have to be in the name or the comparison collides onto one path. Marked
    # only for bargain runs, so no existing filename changes.
    if phase2_mode == "bargain":
        suffix += f"_seg{config.get('BARGAIN_SEGMENT')}"
        # How long an accepted contract binds. This is the axis the renegotiation
        # experiment turns on -- "the first offer to carry binds for the episode" vs
        # "renegotiated every segment" are different games, run at the same seed and
        # segment length and otherwise identical, so without it in the name the
        # second silently overwrites the first. Marked for ALL bindings rather than
        # only off-default ones, the same rule PHASE2_MODE follows and for the same
        # reason: an unmarked name would be ambiguous between the default of the day
        # it was written and the default of the day it is read.
        binding = config.get("BARGAIN_BINDING")
        if binding:
            suffix += f"_{binding}"
        protocol = config.get("BARGAIN_PROTOCOL")
        if protocol and protocol != "alternating":
            # A simultaneous protocol has no proposer and no quorum, so the
            # protocol tag REPLACES those tags rather than joining them -- and a
            # median run at the same seed must never collide with an alternating
            # one, since the checkpoints are different mechanisms entirely.
            suffix += f"_{protocol}"
        else:
            proposer = config.get("BARGAIN_PROPOSER")
            if proposer and proposer != "rotate":
                suffix += f"_{proposer}"
            elif config.get("BARGAIN_ROTATE_START") == "fixed":
                # Only meaningful under rotation, and it changes who captures the
                # first-mover premium -- so the ablation needs its own path.
                suffix += "_fixedstart"
            quorum = config.get("BARGAIN_QUORUM")
            if quorum and quorum != "all":
                suffix += f"_q{quorum}"
        features = config.get("BARGAIN_FEATURES")
        if features and features != "private":
            suffix += f"_{features}"
    # How the contracting stage was TRAINED. This is the axis the whole
    # MOCA-vs-vanilla-contracting comparison turns on -- same environment, same
    # protocol, same seed, same range, differing only here -- so without it the two
    # arms resolve to one path and the second silently overwrites the first.
    # "two_phase" stays unmarked so every existing MOCA filename is unchanged.
    training_mode = config.get("TRAINING_MODE")
    if training_mode in ("joint", "combined"):
        suffix += f"_{training_mode}"

    # Phase-1 null-contract mass. This shapes the GAMEPLAY policy, so unlike the
    # phase-2 knobs above it is not neutralised by PHASE1_ONLY / PHASE1_FROM: two
    # PHASE1_ONLY runs differing only in null mass -- precisely the controlled
    # comparison this variable is swept for -- otherwise resolve to one path and the
    # second silently overwrites the first. Marked only when off the reference's 0.1.
    null_frac = config.get("NULL_CONTRACT_FRAC")
    if null_frac is not None and abs(float(null_frac) - 0.1) > 1e-9:
        suffix += f"_null{null_frac}"

    name = f'{config["ENV_NAME"]}_seed{config["SEED"]}{suffix}'
    return f"{name}_latest" if latest else name


# ---------------------------------------------------------------------------
# Run-config sidecars.
#
# checkpoint_filename encodes only what it must to keep two runs from colliding.
# Plenty of settings that change what a checkpoint MEANS are absent from it --
# CONTRACT_LOW/HIGH above all, since replaying a policy under a different range
# rescales theta through both the contract observation the policy reads and the
# unsquash of the proposal it emits, silently producing numbers for a mechanism
# that was never trained. Hydra does write the resolved config, but to
# outputs/<date>/<time>/.hydra/, which is gitignored and, for a Colab run, on a
# VM that no longer exists by the time the .pkl files are downloaded.
#
# So the config is also written NEXT TO the checkpoints, travelling with them
# into runs/ whenever the directory is copied. Written at run start rather than
# at the end, because the checkpoints that actually get analysed here are
# usually rolling `_latest` snapshots of a run that was still going.
# ---------------------------------------------------------------------------

RUN_CONFIG_EXT = ".run.yaml"

# Role/rolling markers appended AFTER the run stem. Stripped right-to-left so
# every checkpoint of a run -- gameplay, contracting, resume, rolling -- resolves
# to the one sidecar written for that run.
_CKPT_ROLE_TOKENS = ("contract", "proposal", "voting", "resume", "latest")


def run_stem(checkpoint_path: str) -> str:
    """The run stem shared by every checkpoint of one run.

    Accepts anything that names a run's files -- `..._joint_0.pkl`,
    `..._joint_[0-9].pkl`, `..._joint_latest_contract_3.pkl`, and the
    `..._latest_0 (1).pkl` form a browser leaves behind on a repeat download --
    and strips the per-agent index and any role/rolling markers.
    """
    stem = re.sub(r" \(\d+\)(?=\.pkl$)", "", str(checkpoint_path))
    stem = re.sub(r"_(?:\d+|\[0-9\]|\*|\?)\.pkl$", "", stem)
    stem = re.sub(r"\.pkl$", "", stem)
    changed = True
    while changed:  # e.g. "_latest_contract" needs two passes
        changed = False
        for token in _CKPT_ROLE_TOKENS:
            if stem.endswith(f"_{token}"):
                stem = stem[: -len(token) - 1]
                changed = True
    return stem


def run_config_path(checkpoint_path: str) -> str:
    """Sidecar path for a checkpoint path, glob, or bare stem."""
    return run_stem(checkpoint_path) + RUN_CONFIG_EXT


def _git_provenance() -> Dict[str, Any]:
    """Commit/branch/dirty state of the working tree, best-effort."""
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def git(*args):
        try:
            done = subprocess.run(("git", *args), cwd=repo, capture_output=True,
                                  text=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    if commit is None:  # no git, or not a checkout (a bare Colab copy, say)
        return {}
    status = git("status", "--porcelain")
    return {
        "git_commit": commit,
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        # A dirty tree means the commit alone does not identify the code that
        # ran, so the flag is the difference between provenance and a guess.
        "git_dirty": bool(status),
    }


def save_run_config(config: Dict[str, Any], stem_path: str, **provenance: Any) -> str:
    """Write a run's resolved config beside its checkpoints. Returns the path.

    Args:
        config: the resolved config dict (post-make_train, so derived values
            like NEGOTIATE_NU and NUM_UPDATES_* are the ones actually used).
        stem_path: checkpoint path stem, without the `_<agent>.pkl` suffix --
            i.e. exactly the f"{dir}/{checkpoint_filename(config)}" the saves use.
        **provenance: extra fields recorded alongside git/timestamp.
    """
    path = stem_path + RUN_CONFIG_EXT
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    meta = {
        "saved_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": "recorded-at-run-start",
        **_git_provenance(),
        **provenance,
    }
    with open(path, "w") as f:
        yaml.safe_dump({"_provenance": meta, **config}, f,
                       default_flow_style=False, sort_keys=True)
    return path


def load_run_config(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """Read the sidecar for a checkpoint, or None if the run predates them.

    Returning None rather than raising is deliberate: every checkpoint in runs/
    older than this mechanism has no sidecar, and callers should fall back to
    their flags (loudly) rather than refusing to run at all.
    """
    path = run_config_path(checkpoint_path)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return yaml.safe_load(f)


def contract_range(checkpoint_path: str, low=None, high=None, fallback=(0.2, 1.0)):
    """Resolve a contracting run's theta space to (low, high, source).

    Precedence: explicit override > sidecar > fallback. `source` is "flag",
    "sidecar" or "fallback" so callers can WARN when they are guessing --
    guessing silently is the exact failure this whole mechanism exists to stop.
    """
    cfg = load_run_config(checkpoint_path) or {}
    recorded = (cfg.get("CONTRACT_LOW"), cfg.get("CONTRACT_HIGH"))
    if recorded[0] is None or recorded[1] is None:
        source = "fallback"
        resolved = [float(fallback[0]), float(fallback[1])]
    else:
        source = "sidecar"
        resolved = [float(recorded[0]), float(recorded[1])]
    for i, override in enumerate((low, high)):
        if override is not None:
            resolved[i] = float(override)
            source = "flag"
    return resolved[0], resolved[1], source


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
