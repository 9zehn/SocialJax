"""Tests for save_train_state/load_train_state -- the resume-capable checkpoint
functions (unlike save_params/load_params, these also carry optimizer state and
the update counter, both required to properly continue training rather than
silently restarting the LR schedule and Adam's moment estimates from scratch).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import optax
from flax.training.train_state import TrainState

from algorithms.utils.io_utils import save_train_state, load_train_state


def _dummy_train_state(seed=0):
    rng = np.random.default_rng(seed)
    params = {"w": rng.normal(size=(4, 3)).astype("float32"), "b": rng.normal(size=(3,)).astype("float32")}
    tx = optax.adam(1e-3)
    ts = TrainState.create(apply_fn=lambda p, x: x, params=params, tx=tx)
    # simulate a few optimizer steps so opt_state/step are non-trivial
    grads = {"w": np.ones_like(params["w"]), "b": np.ones_like(params["b"])}
    for _ in range(3):
        ts = ts.apply_gradients(grads=grads)
    return ts


def test_round_trip_preserves_params_opt_state_and_step(tmp_path=Path("/tmp/test_resume_ckpt.pkl")):
    ts = _dummy_train_state()
    save_train_state(ts, update_step=7, save_path=str(tmp_path))
    loaded = load_train_state(str(tmp_path))

    assert np.allclose(np.array(loaded["params"]["w"]), np.array(ts.params["w"]))
    assert np.allclose(np.array(loaded["params"]["b"]), np.array(ts.params["b"]))
    assert loaded["step"] == int(ts.step) == 3, "optimizer's own step counter must round-trip exactly"
    assert loaded["update_step"] == 7, "this repo's outer-loop update counter must round-trip exactly"

    # opt_state must round-trip too (Adam's mu/nu moment estimates), not just params
    orig_leaves = [np.array(x) for x in __import__("jax").tree_util.tree_leaves(ts.opt_state)]
    loaded_leaves = [np.array(x) for x in __import__("jax").tree_util.tree_leaves(loaded["opt_state"])]
    assert len(orig_leaves) == len(loaded_leaves)
    for a, b in zip(orig_leaves, loaded_leaves):
        assert np.allclose(a, b), "opt_state must round-trip exactly (else Adam's moments reset on resume)"

    tmp_path.unlink()


def test_resumed_train_state_can_continue_training():
    """A TrainState rebuilt from a saved snapshot must produce IDENTICAL updates
    to what the original, uninterrupted TrainState would have produced next --
    the actual guarantee "resume" needs to provide."""
    ts = _dummy_train_state()
    path = Path("/tmp/test_resume_ckpt2.pkl")
    save_train_state(ts, update_step=3, save_path=str(path))
    loaded = load_train_state(str(path))

    tx = optax.adam(1e-3)
    fresh = TrainState.create(apply_fn=lambda p, x: x, params=loaded["params"], tx=tx)
    resumed = fresh.replace(opt_state=loaded["opt_state"], step=loaded["step"])

    grads = {"w": np.ones_like(ts.params["w"]) * 2, "b": np.ones_like(ts.params["b"]) * 2}
    ts_next = ts.apply_gradients(grads=grads)
    resumed_next = resumed.apply_gradients(grads=grads)

    assert np.allclose(np.array(ts_next.params["w"]), np.array(resumed_next.params["w"]))
    assert np.allclose(np.array(ts_next.params["b"]), np.array(resumed_next.params["b"]))
    assert int(ts_next.step) == int(resumed_next.step)

    path.unlink()


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
