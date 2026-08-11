"""Drift guard for the existing MOCA arms.

The Rubinstein bargaining work adds a second rollout path alongside the original
one. The hazard is not deletion -- it is SILENT drift: restructuring anything on
the shared path changes how PRNG keys are split and consumed, so an existing arm
produces different trajectories at an identical config while still looking fine.
Every number already collected then becomes incomparable, with nothing failing.

So: pin a checksum of a tiny deterministic run per arm. If one of these fails and
you did not intend to change that arm's behaviour, stop -- the baselines moved.

These are checksums, not correctness claims. Regenerate with --update after a
DELIBERATE change (or a JAX/platform upgrade), and say so in the commit message:

    python tests/test_golden_arms.py --update

Runnable two ways:
    python tests/test_golden_arms.py
    python -m pytest tests/test_golden_arms.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT)

import jax
import numpy as np

GOLDEN_PATH = Path(__file__).with_suffix(".json")

# Deliberately tiny: this measures reproducibility, not learning. Two updates is
# enough to exercise a rollout, a GAE and an optimiser step on both phases.
BASE_CONFIG = {
    "LR": 5e-4, "NUM_ENVS": 4, "NUM_STEPS": 8, "TOTAL_TIMESTEPS": 64,
    "UPDATE_EPOCHS": 1, "NUM_MINIBATCHES": 2, "GAMMA": 0.99, "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2, "ENT_COEF": 0.01, "VF_COEF": 0.5, "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "relu", "ANNEAL_LR": False, "PARAMETER_SHARING": False,
    "CONTRACT_SPACE": "cleanup", "CONTRACT_LOW": 0.2, "CONTRACT_HIGH": 1.0,
    "NUM_CONTRACT_BINS": 5, "PHASE1_FRAC": 0.5, "NULL_CONTRACT_FRAC": 0.25,
    "CONTRACT_MINIBATCHES": 2, "CONTRACT_LR": 0.1, "VOTER_SAMPLE_NU": 2,
    "ENV_NAME": "clean_up", "SEED": 7, "NUM_SEEDS": 1, "EVALUATE": False,
    "CHECKPOINT_EVERY": 0, "PROGRESS_EVERY": 10_000,
    "ENV_KWARGS": {"num_agents": 3, "num_inner_steps": 8, "shared_rewards": False,
                   "cnn": True, "jit": True, "apple_reward": 1.0},
    "ENTITY": "", "PROJECT": "socialjax", "WANDB_MODE": "disabled",
    "WANDB_RUN_NAME": None, "REWARD": "individual",
}

ARMS = {
    "solver": {"PHASE2_MODE": "solver", "SOLVER_SAMPLES": 3,
               "SOLVER_DECISION_RULE": "majority"},
    "negotiate": {"PHASE2_MODE": "negotiate", "NEGOTIATE_UPDATE_EPOCHS": 1},
    "reinforce": {"PHASE2_MODE": "reinforce"},
}


def _signature(arm_overrides):
    """A few scalars that depend on the whole pipeline: rollout, GAE, optimiser."""
    import wandb

    from algorithms.MOCA.moca_cnn_cleanup import make_train

    wandb.init(mode="disabled")
    config = json.loads(json.dumps(BASE_CONFIG))     # deep copy
    config.update(arm_overrides)
    out = jax.jit(make_train(config))(jax.random.PRNGKey(config["SEED"]))

    sig = {}
    for phase in ("metrics_phase1", "metrics_phase2"):
        if phase not in out:
            continue
        for key in sorted(out[phase]):
            v = np.asarray(out[phase][key])
            # Mean over updates: one number per series, order-independent.
            sig[f"{phase}/{key}"] = float(np.asarray(v, dtype=np.float64).mean())
    # Final gameplay weights, so a change in the update itself is caught even if
    # every logged metric happens to coincide.
    params = out["runner_state"][0][0].params
    leaves = jax.tree_util.tree_leaves(params)
    sig["params_checksum"] = float(sum(float(np.asarray(l, np.float64).sum())
                                       for l in leaves))
    return sig


def _signature_isolated(arm):
    """`_signature` in a fresh interpreter.

    Building more than one training function per process reliably trips a JAX
    recursive_mutex abort on macOS -- unrelated to anything here, but it makes a
    three-arm guard impossible in-process. One subprocess per arm sidesteps it and
    costs a few seconds.
    """
    code = (
        "import json, sys; sys.path.insert(0, %r);"
        "from tests.test_golden_arms import _signature, ARMS;"
        "print('@@' + json.dumps(_signature(ARMS[%r])))" % (ROOT, arm)
    )
    env = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONPATH": ROOT}
    p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, env=env)
    line = next((l for l in p.stdout.splitlines() if l.startswith("@@")), None)
    if line is None:
        raise RuntimeError(f"{arm}: signature subprocess failed\n"
                           f"{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
    return json.loads(line[2:])


def _load():
    if not GOLDEN_PATH.exists():
        raise AssertionError(
            f"no golden file at {GOLDEN_PATH}. Generate it with:\n"
            f"    python tests/test_golden_arms.py --update"
        )
    return json.loads(GOLDEN_PATH.read_text())


def _compare(arm, got, want):
    missing = sorted(set(want) - set(got))
    assert not missing, f"{arm}: series vanished from the run: {missing}"
    bad = []
    for k, expected in want.items():
        actual = got[k]
        # Loose enough to survive harmless float reassociation, tight enough that
        # a different trajectory cannot slip through.
        if not np.isclose(actual, expected, rtol=1e-4, atol=1e-6):
            bad.append(f"    {k}: {actual!r} != {expected!r}")
    assert not bad, (
        f"{arm}: behaviour drifted from the pinned baseline.\n" + "\n".join(bad) +
        "\n  If this change was deliberate, regenerate with --update and say so "
        "in the commit message."
    )


def test_solver_arm_is_unchanged():
    _compare("solver", _signature_isolated("solver"), _load()["solver"])


def test_negotiate_arm_is_unchanged():
    _compare("negotiate", _signature_isolated("negotiate"), _load()["negotiate"])


def test_reinforce_arm_is_unchanged():
    _compare("reinforce", _signature_isolated("reinforce"), _load()["reinforce"])


def _update():
    golden = {name: _signature_isolated(name) for name in ARMS}
    GOLDEN_PATH.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN_PATH} ({sum(len(v) for v in golden.values())} values)")


ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    if "--update" in sys.argv:
        _update()
        sys.exit(0)
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
